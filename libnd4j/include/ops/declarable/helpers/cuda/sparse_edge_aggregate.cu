/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// csr_edge_aggregate / csr_edge_aggregate_bp — CUDA implementation.
//
// Forward kernel: one thread per (row, feature).
//   tid = blockIdx.x * blockDim.x + threadIdx.x  covers [0, rows*F).
//   row = tid / F;  feat = tid % F.
//   mode 0 (SUM):  out[row, feat] = Σ_{e in row} edgeMsg[e, feat]
//   mode 1 (MEAN): same / degree (0 for empty rows)
//   mode 2 (MAX):  max_{e in row} edgeMsg[e, feat]; 0 for empty rows
//   All reads stride-aware via specialBuffer() + strideAt().
//   No atomics in forward (each thread owns its output cell).
//
// Backward kernel: one thread per (row, feature).
//   mode 0/1: route gradOut[row, feat] to all edges in row (÷ deg for MEAN).
//             each edge belongs to exactly one row → no race condition.
//   mode 2:   recompute argmax edge per (row, feat); route gradient there.
//             argmax edge belongs to one row → no race condition.
//   dEdgeMsg is zero-filled via cudaMemsetAsync before the kernel.
//
// No raw cudaMalloc; no host-compute shim.
// prepareSpecialUse / registerSpecialUse bracket each kernel launch.
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_edge_aggregate.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Forward kernel: one thread per (row, feature)
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrEdgeAggregateKernel(
    const I* rowPtr,   // [rows+1]
    const X* emBuf,    // [nnz, F]  edgeMsg, stride-aware
    X*       oBuf,     // [rows, F] out, stride-aware
    LongType rows,
    LongType F,
    LongType ems0,     // edgeMsg strides
    LongType ems1,
    LongType os0,      // out strides
    LongType os1,
    int      mode)
{
  const LongType tid  = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= rows * F) return;

  const LongType row  = tid / F;
  const LongType feat = tid % F;

  const LongType eStart = static_cast<LongType>(rowPtr[row]);
  const LongType eEnd   = static_cast<LongType>(rowPtr[row + 1]);
  const LongType deg    = eEnd - eStart;

  X* o = oBuf + row * os0 + feat * os1;

  if (mode == 2) {
    // MAX
    if (deg == 0) { *o = static_cast<X>(0); return; }
    X m = emBuf[eStart * ems0 + feat * ems1];
    for (LongType e = eStart + 1; e < eEnd; ++e) {
      X v = emBuf[e * ems0 + feat * ems1];
      if (v > m) m = v;
    }
    *o = m;
  } else {
    // SUM / MEAN
    X acc = static_cast<X>(0);
    for (LongType e = eStart; e < eEnd; ++e) {
      acc += emBuf[e * ems0 + feat * ems1];
    }
    if (mode == 1 && deg > 0) {
      acc /= static_cast<X>(deg);
    }
    *o = acc;
  }
}

template <typename X, typename I>
static void csrEdgeAggregateCuda_(NDArray& rowPtr, NDArray& edgeMsg, NDArray& out,
                                   sd::LongType rows, int mode) {
  auto* stream = edgeMsg.getContext()->getCudaStream();

  const I* rpBuf  = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const X* emBuf  = reinterpret_cast<const X*>(edgeMsg.specialBuffer());
  X*       oBuf   = reinterpret_cast<X*>(out.specialBuffer());

  const LongType F    = edgeMsg.sizeAt(1);
  const LongType ems0 = edgeMsg.strideAt(0);
  const LongType ems1 = edgeMsg.strideAt(1);
  const LongType os0  = out.strideAt(0);
  const LongType os1  = out.strideAt(1);

  const LongType total = rows * F;
  if (total == 0) return;

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((total + blockSize - 1) / blockSize);

  csrEdgeAggregateKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      rpBuf, emBuf, oBuf, rows, F, ems0, ems1, os0, os1, mode);
}

void csr_edge_aggregate(NDArray& rowPtr, NDArray& edgeMsg, NDArray& out,
                         sd::LongType rows, int mode) {
  NDArray::prepareSpecialUse({&out}, {&rowPtr, &edgeMsg});

  BUILD_DOUBLE_SELECTOR(edgeMsg.dataType(), rowPtr.dataType(), csrEdgeAggregateCuda_,
                        (rowPtr, edgeMsg, out, rows, mode),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&out}, {&rowPtr, &edgeMsg});
}

// ────────────────────────────────────────────────────────────────────────────
// Backward kernel: one thread per (row, feature)
//
//   Each edge belongs to exactly one row, so:
//   SUM/MEAN: thread (row, feat) writes directly to its edge range — no races.
//   MAX:      argmax edge is unique per row per feat — no races.
//   dEdgeMsg is pre-zeroed on the stream so non-argmax edges stay zero (MAX).
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrEdgeAggregateBpKernel(
    const I* rowPtr,   // [rows+1]
    const X* emBuf,    // [nnz, F]  forward edgeMsg (for MAX argmax recompute)
    const X* goBuf,    // [rows, F] gradOut
    X*       deBuf,    // [nnz, F]  dEdgeMsg (pre-zeroed)
    LongType rows,
    LongType F,
    LongType ems0,     // edgeMsg strides
    LongType ems1,
    LongType gos0,     // gradOut strides
    LongType gos1,
    LongType des0,     // dEdgeMsg strides
    LongType des1,
    int      mode)
{
  const LongType tid  = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= rows * F) return;

  const LongType row  = tid / F;
  const LongType feat = tid % F;

  const LongType eStart = static_cast<LongType>(rowPtr[row]);
  const LongType eEnd   = static_cast<LongType>(rowPtr[row + 1]);
  const LongType deg    = eEnd - eStart;
  if (deg == 0) return;

  const X g = goBuf[row * gos0 + feat * gos1];

  if (mode == 2) {
    // MAX backward: recompute argmax edge for this (row, feat) cell
    LongType argmaxE = eStart;
    X        maxVal  = emBuf[eStart * ems0 + feat * ems1];
    for (LongType e = eStart + 1; e < eEnd; ++e) {
      X v = emBuf[e * ems0 + feat * ems1];
      if (v > maxVal) { maxVal = v; argmaxE = e; }
    }
    // No race: argmax edge is in this row's exclusive range
    deBuf[argmaxE * des0 + feat * des1] = g;
  } else if (mode == 1) {
    // MEAN backward
    const X scale = g / static_cast<X>(deg);
    for (LongType e = eStart; e < eEnd; ++e) {
      deBuf[e * des0 + feat * des1] = scale;
    }
  } else {
    // SUM backward
    for (LongType e = eStart; e < eEnd; ++e) {
      deBuf[e * des0 + feat * des1] = g;
    }
  }
}

template <typename X, typename I>
static void csrEdgeAggregateBpCuda_(NDArray& rowPtr, NDArray& edgeMsg,
                                     NDArray& gradOut, NDArray& dEdgeMsg,
                                     sd::LongType rows, int mode) {
  auto* stream = edgeMsg.getContext()->getCudaStream();

  const I* rpBuf  = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const X* emBuf  = reinterpret_cast<const X*>(edgeMsg.specialBuffer());
  const X* goBuf  = reinterpret_cast<const X*>(gradOut.specialBuffer());
  X*       deBuf  = reinterpret_cast<X*>(dEdgeMsg.specialBuffer());

  const LongType F    = edgeMsg.sizeAt(1);
  const LongType ems0 = edgeMsg.strideAt(0);
  const LongType ems1 = edgeMsg.strideAt(1);
  const LongType gos0 = gradOut.strideAt(0);
  const LongType gos1 = gradOut.strideAt(1);
  const LongType des0 = dEdgeMsg.strideAt(0);
  const LongType des1 = dEdgeMsg.strideAt(1);

  // Zero dEdgeMsg on the context stream before writes
  cudaMemsetAsync(deBuf, 0, sizeof(X) * static_cast<size_t>(dEdgeMsg.lengthOf()), *stream);

  const LongType total = rows * F;
  if (total == 0) return;

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((total + blockSize - 1) / blockSize);

  csrEdgeAggregateBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      rpBuf, emBuf, goBuf, deBuf,
      rows, F, ems0, ems1, gos0, gos1, des0, des1, mode);
}

void csr_edge_aggregate_bp(NDArray& rowPtr, NDArray& edgeMsg, NDArray& gradOut,
                            NDArray& dEdgeMsg, sd::LongType rows, int mode) {
  NDArray::prepareSpecialUse({&dEdgeMsg}, {&rowPtr, &edgeMsg, &gradOut});

  BUILD_DOUBLE_SELECTOR(edgeMsg.dataType(), rowPtr.dataType(), csrEdgeAggregateBpCuda_,
                        (rowPtr, edgeMsg, gradOut, dEdgeMsg, rows, mode),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dEdgeMsg}, {&rowPtr, &edgeMsg, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
