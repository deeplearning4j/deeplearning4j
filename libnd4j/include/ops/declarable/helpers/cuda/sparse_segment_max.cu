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
// csr_segment_max / csr_segment_max_bp — CUDA implementation.
//
// Forward kernel:  one thread per (row, feature).
//   tid = blockIdx.x * blockDim.x + threadIdx.x  covers [0, rows*f).
//   row = tid / f;  feat = tid % f.
//   Scans neighbors k in [rowPtr[row], rowPtr[row+1]); takes max of X values.
//   Empty row → 0.  X accessed stride-aware via specialBuffer() + stridesOf().
//   No atomics (each thread owns its own output cell).
//
// Backward kernel: one thread per (row, feature).
//   Recomputes argmax k* over neighbors; atomicAdd into dX[colIdx[k*], feat].
//   Multiple rows may map to the same (node, feat) → sd::math::atomics::sd_atomicAdd.
//   dX is zero-filled on the context stream via cudaMemsetAsync before the kernel.
//   dX accessed stride-aware.
//
// No raw cudaMalloc; no host-compute shim.
// prepareSpecialUse / registerSpecialUse bracket each kernel launch.
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_segment_max.h>
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
static SD_KERNEL void csrSegmentMaxKernel(
    const I*  colIdx,   // [nnz]
    const I*  rowPtr,   // [rows+1]
    const X*  xBuf,     // [n*f]  device buffer of X, stride-aware
    X*        outBuf,   // [rows*f] device buffer of out
    LongType  rows,
    LongType  f,
    LongType  xs0,      // X strides
    LongType  xs1,
    LongType  os0,      // out strides
    LongType  os1)
{
  const LongType tid = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= rows * f) return;

  const LongType row  = tid / f;
  const LongType feat = tid % f;

  const LongType eStart = static_cast<LongType>(rowPtr[row]);
  const LongType eEnd   = static_cast<LongType>(rowPtr[row + 1]);

  X* o = outBuf + row * os0 + feat * os1;

  if (eStart == eEnd) {
    *o = static_cast<X>(0);
    return;
  }

  // Initialise to the first neighbor value, then scan
  LongType firstNode = static_cast<LongType>(colIdx[eStart]);
  X m = xBuf[firstNode * xs0 + feat * xs1];

  for (LongType e = eStart + 1; e < eEnd; ++e) {
    LongType node = static_cast<LongType>(colIdx[e]);
    X v = xBuf[node * xs0 + feat * xs1];
    if (v > m) m = v;
  }
  *o = m;
}

template <typename X, typename I>
static void csrSegmentMaxCuda_(NDArray& colIdx, NDArray& rowPtr, NDArray& xArr,
                                NDArray& out, sd::LongType rows) {
  auto* stream = xArr.getContext()->getCudaStream();

  const I* ciBuf  = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpBuf  = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const X* xBuf   = reinterpret_cast<const X*>(xArr.specialBuffer());
  X*       outBuf = reinterpret_cast<X*>(out.specialBuffer());

  const LongType f   = xArr.sizeAt(1);
  const LongType xs0 = xArr.strideAt(0);
  const LongType xs1 = xArr.strideAt(1);
  const LongType os0 = out.strideAt(0);
  const LongType os1 = out.strideAt(1);

  const LongType total = rows * f;
  if (total == 0) return;

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((total + blockSize - 1) / blockSize);

  csrSegmentMaxKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      ciBuf, rpBuf, xBuf, outBuf, rows, f, xs0, xs1, os0, os1);
}

void csr_segment_max(NDArray& colIdx, NDArray& rowPtr, NDArray& xArr, NDArray& out,
                     sd::LongType rows) {
  NDArray::prepareSpecialUse({&out}, {&colIdx, &rowPtr, &xArr});

  BUILD_DOUBLE_SELECTOR(xArr.dataType(), colIdx.dataType(), csrSegmentMaxCuda_,
                        (colIdx, rowPtr, xArr, out, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&out}, {&colIdx, &rowPtr, &xArr});
}

// ────────────────────────────────────────────────────────────────────────────
// Backward kernel: one thread per (row, feature)
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrSegmentMaxBpKernel(
    const I*  colIdx,    // [nnz]
    const I*  rowPtr,    // [rows+1]
    const X*  xBuf,      // [n*f] device buffer of X
    const X*  goBuf,     // [rows*f] upstream gradient
    X*        dxBuf,     // [n*f] output gradient (must be pre-zeroed)
    LongType  rows,
    LongType  f,
    LongType  xs0,       // X strides
    LongType  xs1,
    LongType  gs0,       // gradOut strides
    LongType  gs1,
    LongType  ds0,       // dX strides
    LongType  ds1)
{
  const LongType tid = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= rows * f) return;

  const LongType row  = tid / f;
  const LongType feat = tid % f;

  const LongType eStart = static_cast<LongType>(rowPtr[row]);
  const LongType eEnd   = static_cast<LongType>(rowPtr[row + 1]);
  if (eStart == eEnd) return;  // empty row — no contribution

  // Recompute argmax k*
  LongType argmaxNode = static_cast<LongType>(colIdx[eStart]);
  X        maxVal     = xBuf[argmaxNode * xs0 + feat * xs1];

  for (LongType e = eStart + 1; e < eEnd; ++e) {
    LongType node = static_cast<LongType>(colIdx[e]);
    X v = xBuf[node * xs0 + feat * xs1];
    if (v > maxVal) {
      maxVal     = v;
      argmaxNode = node;
    }
  }

  // Scatter-accumulate: multiple rows may target the same (node, feat)
  sd::math::atomics::sd_atomicAdd(
      dxBuf + argmaxNode * ds0 + feat * ds1,
      goBuf[row * gs0 + feat * gs1]);
}

template <typename X, typename I>
static void csrSegmentMaxBpCuda_(NDArray& colIdx, NDArray& rowPtr, NDArray& xArr,
                                  NDArray& gradOut, NDArray& dX,
                                  sd::LongType rows) {
  auto* stream = xArr.getContext()->getCudaStream();

  const I* ciBuf  = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpBuf  = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const X* xBuf   = reinterpret_cast<const X*>(xArr.specialBuffer());
  const X* goBuf  = reinterpret_cast<const X*>(gradOut.specialBuffer());
  X*       dxBuf  = reinterpret_cast<X*>(dX.specialBuffer());

  const LongType f    = xArr.sizeAt(1);
  const LongType xs0  = xArr.strideAt(0);
  const LongType xs1  = xArr.strideAt(1);
  const LongType gs0  = gradOut.strideAt(0);
  const LongType gs1  = gradOut.strideAt(1);
  const LongType ds0  = dX.strideAt(0);
  const LongType ds1  = dX.strideAt(1);

  // Zero dX on the context stream before scatter-accumulation
  cudaMemsetAsync(dxBuf, 0, sizeof(X) * static_cast<size_t>(dX.lengthOf()), *stream);

  const LongType total = rows * f;
  if (total == 0) return;

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((total + blockSize - 1) / blockSize);

  csrSegmentMaxBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      ciBuf, rpBuf, xBuf, goBuf, dxBuf,
      rows, f, xs0, xs1, gs0, gs1, ds0, ds1);
}

void csr_segment_max_bp(NDArray& colIdx, NDArray& rowPtr, NDArray& xArr,
                         NDArray& gradOut, NDArray& dX, sd::LongType rows) {
  NDArray::prepareSpecialUse({&dX}, {&colIdx, &rowPtr, &xArr, &gradOut});

  BUILD_DOUBLE_SELECTOR(xArr.dataType(), colIdx.dataType(), csrSegmentMaxBpCuda_,
                        (colIdx, rowPtr, xArr, gradOut, dX, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dX}, {&colIdx, &rowPtr, &xArr, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
