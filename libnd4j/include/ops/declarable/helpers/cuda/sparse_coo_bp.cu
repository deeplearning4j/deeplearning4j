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

// No raw cudaMalloc anywhere in this file.
// All output zeroing uses cudaMemsetAsync (byte-zero, not an allocation).

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_coo_bp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ============================================================================
// dense_to_coo_bp — CUDA
//
// Scatter upstream gradient gradValues[k] into dDense at the position
// (indices[k,0], indices[k,1]).  dDense is zeroed with cudaMemsetAsync before
// the kernel runs (IEEE 0x00 == 0.0 for all float types).
// One thread per COO entry k; no atomics (unique indices guaranteed).
// ============================================================================

template <typename X, typename I>
__global__ void denseToCoo_bp_kernel(const I* __restrict__ indices,    // [nnz, 2]
                                     const X* __restrict__ gradValues,  // [nnz]
                                           X* __restrict__ dDense,      // [rows, cols] pre-zeroed
                                     sd::LongType rowStride,
                                     sd::LongType colStride,
                                     sd::LongType nnz) {
  const sd::LongType k = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= nnz) return;
  const auto r = static_cast<sd::LongType>(indices[k * 2 + 0]);
  const auto c = static_cast<sd::LongType>(indices[k * 2 + 1]);
  dDense[r * rowStride + c * colStride] = gradValues[k];
}

template <typename X, typename I>
static void denseToCoo_bpCuda_(sd::LaunchContext* context,
                               NDArray* indices, NDArray* gradValues, NDArray* dDense,
                               sd::LongType /*rows*/, sd::LongType /*cols*/) {
  const sd::LongType nnz    = gradValues->lengthOf();
  const cudaStream_t stream = *context->getCudaStream();

  // Zero-initialise the dense gradient on device.
  // For all IEEE 754 float types, 0x00 byte pattern == 0.0.
  cudaMemsetAsync(dDense->specialBuffer(), 0,
                  static_cast<size_t>(dDense->lengthOf()) * dDense->sizeOfT(),
                  stream);

  if (nnz == 0) return;

  // Row/col strides from the host shape info (fresh contiguous output: rowStr=cols, colStr=1).
  const sd::LongType rowStride = dDense->stridesOf()[0];
  const sd::LongType colStride = dDense->stridesOf()[1];

  const I* dIdx  = reinterpret_cast<const I*>(indices->specialBuffer());
  const X* dGrad = reinterpret_cast<const X*>(gradValues->specialBuffer());
        X* dOut  = reinterpret_cast<X*>(dDense->specialBuffer());

  const int threads = 256;
  const int blocks  = static_cast<int>((nnz + threads - 1) / threads);

  denseToCoo_bp_kernel<X, I><<<blocks, threads, 0, stream>>>(
      dIdx, dGrad, dOut, rowStride, colStride, nnz);
}

// ============================================================================
// coo_to_csr_bp — CUDA
//
// One thread per COO entry k.
// Binary-searches csrColIdx[ csrRowPtr[row] .. csrRowPtr[row+1] ) for col j.
// dCooValues[k] = gradCsrValues[foundPos]  (or 0.0 if not found).
// No atomics: each COO entry maps to exactly one CSR slot.
// ============================================================================

template <typename X, typename I>
__global__ void cooToCsr_bp_kernel(const I*       __restrict__ cooIndices,    // [coo_nnz, 2]
                                   const int32_t* __restrict__ csrColIdx,      // [csr_nnz]
                                   const int32_t* __restrict__ csrRowPtr,      // [rows + 1]
                                   const X*       __restrict__ gradCsrValues,  // [csr_nnz]
                                         X*       __restrict__ dCooValues,     // [coo_nnz]
                                   sd::LongType cooNnz) {
  const sd::LongType k = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= cooNnz) return;

  const int32_t row = static_cast<int32_t>(cooIndices[k * 2 + 0]);
  const int32_t col = static_cast<int32_t>(cooIndices[k * 2 + 1]);

  const int32_t rowBegin = csrRowPtr[row];
  const int32_t rowEnd   = csrRowPtr[row + 1];

  // Binary search for column j in [rowBegin, rowEnd).
  int32_t lo = rowBegin, hi = rowEnd;
  int32_t foundPos = -1;
  while (lo < hi) {
    const int32_t mid = lo + (hi - lo) / 2;
    const int32_t v   = csrColIdx[mid];
    if      (v == col) { foundPos = mid; break; }
    else if (v < col)  lo = mid + 1;
    else               hi = mid;
  }

  dCooValues[k] = (foundPos >= 0) ? gradCsrValues[foundPos] : static_cast<X>(0);
}

template <typename X, typename I>
static void cooToCsr_bpCuda_(sd::LaunchContext* context,
                             NDArray* cooIndices, NDArray* csrColIdx, NDArray* csrRowPtr,
                             NDArray* gradCsrValues, NDArray* dCooValues,
                             sd::LongType /*rows*/, sd::LongType /*cols*/) {
  const sd::LongType cooNnz  = dCooValues->lengthOf();
  const cudaStream_t stream  = *context->getCudaStream();

  if (cooNnz == 0) return;

  const I*       dIdx    = reinterpret_cast<const I*>(cooIndices->specialBuffer());
  const int32_t* dColIdx = reinterpret_cast<const int32_t*>(csrColIdx->specialBuffer());
  const int32_t* dRowPtr = reinterpret_cast<const int32_t*>(csrRowPtr->specialBuffer());
  const X*       dGrad   = reinterpret_cast<const X*>(gradCsrValues->specialBuffer());
        X*       dOut    = reinterpret_cast<X*>(dCooValues->specialBuffer());

  const int threads = 256;
  const int blocks  = static_cast<int>((cooNnz + threads - 1) / threads);

  cooToCsr_bp_kernel<X, I><<<blocks, threads, 0, stream>>>(
      dIdx, dColIdx, dRowPtr, dGrad, dOut, cooNnz);
}

// ============================================================================
// Public CUDA entry points
// ============================================================================

void denseToCoo_bp(sd::LaunchContext* context,
                   NDArray* indices, NDArray* gradValues, NDArray* dDense,
                   sd::LongType rows, sd::LongType cols) {
  NDArray::prepareSpecialUse({dDense}, {indices, gradValues});

  BUILD_DOUBLE_SELECTOR(gradValues->dataType(), indices->dataType(), denseToCoo_bpCuda_,
                        (context, indices, gradValues, dDense, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({dDense}, {indices, gradValues});
}

void cooToCsr_bp(sd::LaunchContext* context,
                 NDArray* cooIndices, NDArray* csrColIdx, NDArray* csrRowPtr,
                 NDArray* gradCsrValues, NDArray* dCooValues,
                 sd::LongType rows, sd::LongType cols) {
  NDArray::prepareSpecialUse({dCooValues},
                             {cooIndices, csrColIdx, csrRowPtr, gradCsrValues});

  BUILD_DOUBLE_SELECTOR(gradCsrValues->dataType(), cooIndices->dataType(), cooToCsr_bpCuda_,
                        (context, cooIndices, csrColIdx, csrRowPtr,
                         gradCsrValues, dCooValues, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({dCooValues},
                              {cooIndices, csrColIdx, csrRowPtr, gradCsrValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
