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
// csr_edge_gather / csr_edge_gather_bp — CUDA implementation.
//
// Forward kernel: one thread per (edge, feature).
//   tid = blockIdx.x * blockDim.x + threadIdx.x  covers [0, nnz*F).
//   e    = tid / F;  f = tid % F.
//   edgeFeat[e, f] = X[colIdx[e], f]    (stride-aware read/write)
//   No atomics (each thread owns its own output element).
//
// Backward kernel: one thread per (edge, feature).
//   Scatter-accumulate upstream gradient into dX:
//   dX[colIdx[e], f] += gradEdge[e, f]
//   Multiple edges may share the same colIdx → sd::math::atomics::sd_atomicAdd.
//   dX is zero-filled on the context stream via cudaMemsetAsync before the kernel.
//
// No raw cudaMalloc; no host-compute shim.
// prepareSpecialUse / registerSpecialUse bracket each kernel launch.
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_edge_gather.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Forward kernel: one thread per (edge, feature)
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrEdgeGatherKernel(
    const I* colIdx,   // [nnz]
    const X* xBuf,     // [n, F]  device buffer, stride-aware
    X*       efBuf,    // [nnz, F] device buffer of edgeFeat
    LongType nnz,
    LongType F,
    LongType xs0,      // X strides
    LongType xs1,
    LongType es0,      // edgeFeat strides
    LongType es1)
{
  const LongType tid = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= nnz * F) return;

  const LongType e    = tid / F;
  const LongType f    = tid % F;
  const LongType node = static_cast<LongType>(colIdx[e]);

  efBuf[e * es0 + f * es1] = xBuf[node * xs0 + f * xs1];
}

template <typename X, typename I>
static void csrEdgeGatherCuda_(NDArray& colIdx, NDArray& xArr, NDArray& edgeFeat) {
  auto* stream = xArr.getContext()->getCudaStream();

  const I* ciBuf  = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const X* xBuf   = reinterpret_cast<const X*>(xArr.specialBuffer());
  X*       efBuf  = reinterpret_cast<X*>(edgeFeat.specialBuffer());

  const LongType nnz  = colIdx.lengthOf();
  const LongType F    = xArr.sizeAt(1);
  const LongType xs0  = xArr.strideAt(0);
  const LongType xs1  = xArr.strideAt(1);
  const LongType es0  = edgeFeat.strideAt(0);
  const LongType es1  = edgeFeat.strideAt(1);

  const LongType total = nnz * F;
  if (total == 0) return;

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((total + blockSize - 1) / blockSize);

  csrEdgeGatherKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      ciBuf, xBuf, efBuf, nnz, F, xs0, xs1, es0, es1);
}

void csr_edge_gather(NDArray& colIdx, NDArray& xArr, NDArray& edgeFeat) {
  NDArray::prepareSpecialUse({&edgeFeat}, {&colIdx, &xArr});

  BUILD_DOUBLE_SELECTOR(xArr.dataType(), colIdx.dataType(), csrEdgeGatherCuda_,
                        (colIdx, xArr, edgeFeat),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&edgeFeat}, {&colIdx, &xArr});
}

// ────────────────────────────────────────────────────────────────────────────
// Backward kernel: one thread per (edge, feature)
//   Scatter-accumulates upstream gradient; multiple edges can share a node →
//   sd_atomicAdd.  dX is pre-zeroed on the stream before this kernel.
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrEdgeGatherBpKernel(
    const I* colIdx,   // [nnz]
    const X* geBuf,    // [nnz, F]  gradEdge device buffer
    X*       dxBuf,    // [n, F]   dX device buffer (pre-zeroed)
    LongType nnz,
    LongType F,
    LongType ges0,     // gradEdge strides
    LongType ges1,
    LongType ds0,      // dX strides
    LongType ds1)
{
  const LongType tid = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= nnz * F) return;

  const LongType e    = tid / F;
  const LongType f    = tid % F;
  const LongType node = static_cast<LongType>(colIdx[e]);

  // Multiple edges may reference the same node → use sd_atomicAdd
  sd::math::atomics::sd_atomicAdd(
      dxBuf + node * ds0 + f * ds1,
      geBuf[e * ges0 + f * ges1]);
}

template <typename X, typename I>
static void csrEdgeGatherBpCuda_(NDArray& colIdx, NDArray& xArr, NDArray& gradEdge, NDArray& dX) {
  // xArr is passed for its shape (n = xArr.sizeAt(0), F = xArr.sizeAt(1));
  // its values are NOT read — dX is already shaped [n, F] by DECLARE_SHAPE_FN.
  auto* stream = gradEdge.getContext()->getCudaStream();

  const I* ciBuf  = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const X* geBuf  = reinterpret_cast<const X*>(gradEdge.specialBuffer());
  X*       dxBuf  = reinterpret_cast<X*>(dX.specialBuffer());

  const LongType nnz  = colIdx.lengthOf();
  const LongType F    = dX.sizeAt(1);
  const LongType ges0 = gradEdge.strideAt(0);
  const LongType ges1 = gradEdge.strideAt(1);
  const LongType ds0  = dX.strideAt(0);
  const LongType ds1  = dX.strideAt(1);

  // Zero dX on the context stream before scatter-accumulation
  cudaMemsetAsync(dxBuf, 0, sizeof(X) * static_cast<size_t>(dX.lengthOf()), *stream);

  const LongType total = nnz * F;
  if (total == 0) return;

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((total + blockSize - 1) / blockSize);

  csrEdgeGatherBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      ciBuf, geBuf, dxBuf, nnz, F, ges0, ges1, ds0, ds1);
}

void csr_edge_gather_bp(NDArray& colIdx, NDArray& xArr, NDArray& gradEdge, NDArray& dX) {
  NDArray::prepareSpecialUse({&dX}, {&colIdx, &xArr, &gradEdge});

  BUILD_DOUBLE_SELECTOR(gradEdge.dataType(), colIdx.dataType(), csrEdgeGatherBpCuda_,
                        (colIdx, xArr, gradEdge, dX),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dX}, {&colIdx, &xArr, &gradEdge});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
