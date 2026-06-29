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
// CUDA backward-pass helpers for csr_to_dense, dense_to_csr, and csr_to_csc.
//
// csr_to_dense_bp:
//   One CUDA thread per nnz entry.  Each thread binary-searches rowPtr to find
//   its row, then reads gradDense[row, colIdx[e]] → dValues[e].
//   Pure gather — no atomics.
//
// dense_to_csr_bp:
//   One CUDA thread per nnz entry.  Each thread binary-searches rowPtr to find
//   its row, then writes gradValues[e] → dDense[row, colIdx[e]].
//   Scatter with unique (row,col) pairs → no atomics.
//   dDense is OUTPUT_NULLIFIED (pre-zeroed by the framework on device).
//
// csr_to_csc_bp:
//   Calls the existing csr_to_csc CUDA helper with dimensions swapped:
//     csr_to_csc(gradCscValues, cscRowIdx, cscColPtr,
//                dAValues, tempRowIdx, tempColPtr,
//                rows_new=cols, cols_new=rows)
//   Two temporary INT32 device arrays (tempRowIdx[nnz], tempColPtr[rows+1])
//   are allocated via NDArrayFactory::create(ctx) — no raw cudaMalloc.
//

#include <cuda_runtime.h>
#include <array/NDArrayFactory.h>
#include <memory>
#include <ops/declarable/helpers/sparse_convert_bp.h>
#include <ops/declarable/helpers/sparse_csc.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ═══════════════════════════════════════════════════════════════════════════
// Section A: csr_to_dense_bp — CUDA kernel
//
// One thread per non-zero entry e (0-based).
// Binary-searches rowPtr[0..rows] for the row i such that rowPtr[i] <= e < rowPtr[i+1].
// Reads dValues[e] = gradDense[i * cols + colIdx[e]]   (row-major dense).
// ═══════════════════════════════════════════════════════════════════════════

template <typename X, typename I>
static SD_KERNEL void csrToDenseBpKernel(const X* gradDense, const I* colIdx, const I* rowPtr,
                                          X* dValues,
                                          LongType nnz, LongType rows, LongType cols,
                                          LongType gdStride0, LongType gdStride1) {
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;

  // upper_bound: find smallest pos such that rowPtr[pos] > e
  I lo = 0, hi = static_cast<I>(rows) + 1;
  while (lo < hi) {
    const I mid = lo + (hi - lo) / 2;
    if (rowPtr[mid] <= static_cast<I>(e)) lo = mid + 1;
    else                                  hi = mid;
  }
  const I i = lo - 1;       // row of entry e
  const I j = colIdx[e];    // column of entry e

  // stride-aware: gradDense[i, j] using actual array strides
  dValues[e] = gradDense[static_cast<LongType>(i) * gdStride0 + static_cast<LongType>(j) * gdStride1];
}

template <typename X, typename I>
static void csrToDenseBpCuda_(NDArray& colIdx, NDArray& rowPtr, NDArray& gradDense,
                               NDArray& dValues, LongType rows, LongType cols) {
  const LongType nnz = colIdx.lengthOf();

  const X* gBuf  = reinterpret_cast<const X*>(gradDense.specialBuffer());
  const I* ciBuf = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpBuf = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  X*       dvBuf = reinterpret_cast<X*>(dValues.specialBuffer());

  // Read strides on the host side; pass as scalars into the kernel.
  const LongType gdS0 = gradDense.stridesOf()[0];
  const LongType gdS1 = gradDense.stridesOf()[1];

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);

  auto stream = dValues.getContext()->getCudaStream();
  csrToDenseBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      gBuf, ciBuf, rpBuf, dvBuf, nnz, rows, cols, gdS0, gdS1);
}

void csr_to_dense_bp(NDArray& colIdx, NDArray& rowPtr, NDArray& gradDense,
                     NDArray& dValues, LongType rows, LongType cols) {
  NDArray::prepareSpecialUse({&dValues}, {&colIdx, &rowPtr, &gradDense});

  BUILD_DOUBLE_SELECTOR(gradDense.dataType(), colIdx.dataType(), csrToDenseBpCuda_,
                        (colIdx, rowPtr, gradDense, dValues, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dValues}, {&colIdx, &rowPtr, &gradDense});
}

// ═══════════════════════════════════════════════════════════════════════════
// Section B: dense_to_csr_bp — CUDA kernel
//
// One thread per non-zero entry e (0-based).
// Binary-searches rowPtr[0..rows] for row i of entry e, then:
//   dDense[i * cols + colIdx[e]] = gradValues[e]
// dDense is OUTPUT_NULLIFIED (device-zeroed before the kernel), so unwritten
// entries remain zero.  The (i,j) pairs in a valid CSR are unique → no atomics.
// ═══════════════════════════════════════════════════════════════════════════

template <typename X, typename I>
static SD_KERNEL void denseToCsrBpKernel(const X* gradValues, const I* colIdx, const I* rowPtr,
                                          X* dDense,
                                          LongType nnz, LongType rows, LongType cols,
                                          LongType ddStride0, LongType ddStride1) {
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;

  // upper_bound search — identical to csr_to_dense_bp
  I lo = 0, hi = static_cast<I>(rows) + 1;
  while (lo < hi) {
    const I mid = lo + (hi - lo) / 2;
    if (rowPtr[mid] <= static_cast<I>(e)) lo = mid + 1;
    else                                  hi = mid;
  }
  const I i = lo - 1;
  const I j = colIdx[e];

  // stride-aware: dDense[i, j] using actual array strides
  dDense[static_cast<LongType>(i) * ddStride0 + static_cast<LongType>(j) * ddStride1] = gradValues[e];
}

template <typename X, typename I>
static void denseToCsrBpCuda_(NDArray& colIdx, NDArray& rowPtr, NDArray& gradValues,
                               NDArray& dDense, LongType rows, LongType cols) {
  const LongType nnz = colIdx.lengthOf();

  const X* gvBuf = reinterpret_cast<const X*>(gradValues.specialBuffer());
  const I* ciBuf = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpBuf = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  X*       ddBuf = reinterpret_cast<X*>(dDense.specialBuffer());

  // Read strides on the host side; pass as scalars into the kernel.
  const LongType ddS0 = dDense.stridesOf()[0];
  const LongType ddS1 = dDense.stridesOf()[1];

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);

  auto stream = dDense.getContext()->getCudaStream();
  denseToCsrBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      gvBuf, ciBuf, rpBuf, ddBuf, nnz, rows, cols, ddS0, ddS1);
}

void dense_to_csr_bp(NDArray& colIdx, NDArray& rowPtr, NDArray& gradValues,
                     NDArray& dDense, LongType rows, LongType cols) {
  NDArray::prepareSpecialUse({&dDense}, {&colIdx, &rowPtr, &gradValues});

  BUILD_DOUBLE_SELECTOR(gradValues.dataType(), colIdx.dataType(), denseToCsrBpCuda_,
                        (colIdx, rowPtr, gradValues, dDense, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dDense}, {&colIdx, &rowPtr, &gradValues});
}

// ═══════════════════════════════════════════════════════════════════════════
// Section C: csr_to_csc_bp — reuse csr_to_csc helper
//
// The forward csr_to_csc permutation is self-inverse in the following sense:
//   csc_of_A = csr_of_A^T    (shape [cols, rows])
// Applying csr_to_csc again to (gradCscValues, cscRowIdx, cscColPtr) with
// dimensions (cols, rows) transposes back, producing values in the original
// CSR(A) [rows, cols] order.  This is the gradient of the forward permutation.
//
// Temporaries:
//   tempRowIdx [nnz]       INT32 — receives the re-transposed column indices
//   tempColPtr [rows+1]    INT32 — receives the re-transposed row pointers
// Both are allocated with NDArrayFactory::create(ctx) — no raw cudaMalloc.
// Both are discarded after the call.
// ═══════════════════════════════════════════════════════════════════════════

void csr_to_csc_bp(NDArray& cscRowIdx, NDArray& cscColPtr,
                   NDArray& gradCscValues, NDArray& dAValues,
                   LongType rows, LongType cols) {
  const LongType nnz = gradCscValues.lengthOf();

  // Allocate temporaries on the same device as the output.
  auto* ctx = dAValues.getContext();
  // NDArrayFactory::create returns NDArray* — own it via unique_ptr (sparse-helper idiom).
  std::unique_ptr<NDArray> tempRowIdx(NDArrayFactory::create('c', {nnz},      sd::DataType::INT32, ctx));
  std::unique_ptr<NDArray> tempColPtr(NDArrayFactory::create('c', {rows + 1}, sd::DataType::INT32, ctx));

  // csr_to_csc handles prepareSpecialUse / registerSpecialUse internally.
  csr_to_csc(gradCscValues, cscRowIdx, cscColPtr,
              dAValues,     *tempRowIdx, *tempColPtr,
              cols, rows);
  // tempRowIdx and tempColPtr are discarded; dAValues is the result.
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
