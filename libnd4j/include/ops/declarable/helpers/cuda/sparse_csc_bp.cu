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
// CUDA backward-pass helpers for CSC (Compressed Sparse Column) operations.
//
// csc_to_dense_bp:
//   One CUDA thread per nonzero entry k (0-based).
//   Binary-searches cscColPtr[0..cols] to find the column c containing entry k,
//   then: dCscValues[k] = gradDense[cscRowIdx[k], c].
//   Pure gather — no atomics.
//
// dense_to_csc_bp:
//   One CUDA thread per nonzero entry k (0-based).
//   Binary-searches cscColPtr[0..cols] to find the column c containing entry k,
//   then: dDense[cscRowIdx[k], c] = gradCscValues[k].
//   dDense is OUTPUT_NULLIFIED (pre-zeroed by the framework on device).
//   In a valid CSC pattern every (row, col) pair is unique → no atomics.
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_csc_bp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ─── device utility ──────────────────────────────────────────────────────────

// Binary-search cscColPtr[0..cols] to find the column c such that
//   cscColPtr[c] <= k < cscColPtr[c+1].
// This is the standard upper_bound / predecessor search.
template <typename I>
static SD_INLINE SD_DEVICE LongType cscFindCol(const I* cscColPtr, LongType cols, LongType k) {
  LongType lo = 0, hi = cols - 1, col = 0;
  while (lo <= hi) {
    const LongType mid = (lo + hi) / 2;
    if (static_cast<LongType>(cscColPtr[mid]) <= k &&
        k < static_cast<LongType>(cscColPtr[mid + 1])) {
      col = mid;
      break;
    } else if (static_cast<LongType>(cscColPtr[mid]) > k) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  return col;
}

// ═══════════════════════════════════════════════════════════════════════════
// Section A: csc_to_dense_bp
// ═══════════════════════════════════════════════════════════════════════════

template <typename X, typename I>
static SD_KERNEL void cscToDenseBpKernel(
    const I* cscRowIdx, const I* cscColPtr,
    const X* gradDense, X* dCscValues,
    LongType nnz, LongType cols,
    LongType gdStride0, LongType gdStride1) {
  const LongType k = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= nnz) return;

  const LongType c   = cscFindCol(cscColPtr, cols, k);
  const LongType row = static_cast<LongType>(cscRowIdx[k]);
  dCscValues[k] = gradDense[row * gdStride0 + c * gdStride1];
}

template <typename X, typename I>
static void cscToDenseBpCuda_(
    NDArray& cscRowIdx, NDArray& cscColPtr,
    NDArray& gradDense, NDArray& dCscValues,
    LongType rows, LongType cols) {
  const LongType nnz = cscRowIdx.lengthOf();

  const I* riB = reinterpret_cast<const I*>(cscRowIdx.specialBuffer());
  const I* cpB = reinterpret_cast<const I*>(cscColPtr.specialBuffer());
  const X* gD  = reinterpret_cast<const X*>(gradDense.specialBuffer());
  X*       dCV = reinterpret_cast<X*>(dCscValues.specialBuffer());

  const LongType gdS0 = gradDense.stridesOf()[0];
  const LongType gdS1 = gradDense.stridesOf()[1];

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);

  auto* stream = dCscValues.getContext()->getCudaStream();
  cscToDenseBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      riB, cpB, gD, dCV, nnz, cols, gdS0, gdS1);
}

void csc_to_dense_bp(
    NDArray& cscRowIdx, NDArray& cscColPtr,
    NDArray& gradDense, NDArray& dCscValues,
    LongType rows, LongType cols) {
  NDArray::prepareSpecialUse({&dCscValues}, {&cscRowIdx, &cscColPtr, &gradDense});

  BUILD_DOUBLE_SELECTOR(gradDense.dataType(), cscRowIdx.dataType(), cscToDenseBpCuda_,
                        (cscRowIdx, cscColPtr, gradDense, dCscValues, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dCscValues}, {&cscRowIdx, &cscColPtr, &gradDense});
}

// ═══════════════════════════════════════════════════════════════════════════
// Section B: dense_to_csc_bp
// ═══════════════════════════════════════════════════════════════════════════

template <typename X, typename I>
static SD_KERNEL void denseToCscBpKernel(
    const I* cscRowIdx, const I* cscColPtr,
    const X* gradCscValues, X* dDense,
    LongType nnz, LongType cols,
    LongType ddStride0, LongType ddStride1) {
  const LongType k = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= nnz) return;

  const LongType c   = cscFindCol(cscColPtr, cols, k);
  const LongType row = static_cast<LongType>(cscRowIdx[k]);
  dDense[row * ddStride0 + c * ddStride1] = gradCscValues[k];
}

template <typename X, typename I>
static void denseToCscBpCuda_(
    NDArray& cscRowIdx, NDArray& cscColPtr,
    NDArray& gradCscValues, NDArray& dDense,
    LongType rows, LongType cols) {
  const LongType nnz = cscRowIdx.lengthOf();

  const I* riB = reinterpret_cast<const I*>(cscRowIdx.specialBuffer());
  const I* cpB = reinterpret_cast<const I*>(cscColPtr.specialBuffer());
  const X* gCV = reinterpret_cast<const X*>(gradCscValues.specialBuffer());
  X*       dD  = reinterpret_cast<X*>(dDense.specialBuffer());

  const LongType ddS0 = dDense.stridesOf()[0];
  const LongType ddS1 = dDense.stridesOf()[1];

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);

  auto* stream = dDense.getContext()->getCudaStream();
  denseToCscBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      riB, cpB, gCV, dD, nnz, cols, ddS0, ddS1);
}

void dense_to_csc_bp(
    NDArray& cscRowIdx, NDArray& cscColPtr,
    NDArray& gradCscValues, NDArray& dDense,
    LongType rows, LongType cols) {
  NDArray::prepareSpecialUse({&dDense}, {&cscRowIdx, &cscColPtr, &gradCscValues});

  BUILD_DOUBLE_SELECTOR(gradCscValues.dataType(), cscRowIdx.dataType(), denseToCscBpCuda_,
                        (cscRowIdx, cscColPtr, gradCscValues, dDense, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dDense}, {&cscRowIdx, &cscColPtr, &gradCscValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
