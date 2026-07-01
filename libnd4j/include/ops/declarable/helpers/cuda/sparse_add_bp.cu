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
// CUDA backward-pass helper for csr_add: C = A + B (sorted-merge union sparsity).
//
// csr_add_bp:
//   Two kernels, one per input matrix (A and B).
//   Kernel for A: one thread per nonzero k in A.
//     1. Binary-search aRowPtr to find row r of entry k.
//     2. Column col = aColIdx[k].
//     3. Binary-search C's column list for row r to find pos of col in gradCValues.
//     4. dAValues[k] = gradCValues[pos].
//   Kernel for B: identical structure using bColIdx / bRowPtr.
//   Both are pure gathers — no atomics required (union sparsity guarantees every
//   A / B entry is present in C).
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_add_bp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ─── device utilities ────────────────────────────────────────────────────────

// Binary-search rowPtr[0..rows] to find the row i such that
//   rowPtr[i] <= k < rowPtr[i+1].
// Identical logic to sparse_blas_bp.cu's findRow, scoped to this TU.
template <typename I>
static SD_INLINE SD_DEVICE LongType addBpFindRow(const I* rowPtr, LongType rows, LongType k) {
  LongType lo = 0, hi = rows - 1, row = 0;
  while (lo <= hi) {
    const LongType mid = (lo + hi) / 2;
    if (static_cast<LongType>(rowPtr[mid]) <= k &&
        k < static_cast<LongType>(rowPtr[mid + 1])) {
      row = mid;
      break;
    } else if (static_cast<LongType>(rowPtr[mid]) > k) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  return row;
}

// Lower-bound search: find the leftmost position p in [clo, chi) such that
//   cColIdx[p] == col.
// Caller guarantees col is present (union sparsity).
template <typename I>
static SD_INLINE SD_DEVICE LongType addBpFindCol(const I* cColIdx, I clo, I chi, I col) {
  while (clo < chi) {
    const I cmid = clo + (chi - clo) / 2;
    if (cColIdx[cmid] < col) clo = cmid + 1;
    else chi = cmid;
  }
  return static_cast<LongType>(clo);
}

// ─── Kernel A ────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrAddBpAKernel(
    const I* aColIdx, const I* aRowPtr,
    const I* cColIdx, const I* cRowPtr,
    const X* gradCValues, X* dAValues,
    LongType nnzA, LongType m) {
  const LongType k = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= nnzA) return;

  const LongType r   = addBpFindRow(aRowPtr, m, k);
  const I        col = aColIdx[k];
  const LongType pos = addBpFindCol(cColIdx, cRowPtr[r], cRowPtr[r + 1], col);
  dAValues[k] = gradCValues[pos];
}

// ─── Kernel B ────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrAddBpBKernel(
    const I* bColIdx, const I* bRowPtr,
    const I* cColIdx, const I* cRowPtr,
    const X* gradCValues, X* dBValues,
    LongType nnzB, LongType m) {
  const LongType k = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= nnzB) return;

  const LongType r   = addBpFindRow(bRowPtr, m, k);
  const I        col = bColIdx[k];
  const LongType pos = addBpFindCol(cColIdx, cRowPtr[r], cRowPtr[r + 1], col);
  dBValues[k] = gradCValues[pos];
}

// ─── typed launcher ──────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrAddBpCuda_(
    NDArray& aColIdx, NDArray& aRowPtr,
    NDArray& bColIdx, NDArray& bRowPtr,
    NDArray& cColIdx, NDArray& cRowPtr,
    NDArray& gradCValues,
    NDArray& dAValues, NDArray& dBValues,
    LongType m, LongType n) {
  const LongType nnzA = aColIdx.lengthOf();
  const LongType nnzB = bColIdx.lengthOf();

  const I* aCI = reinterpret_cast<const I*>(aColIdx.specialBuffer());
  const I* aRP = reinterpret_cast<const I*>(aRowPtr.specialBuffer());
  const I* bCI = reinterpret_cast<const I*>(bColIdx.specialBuffer());
  const I* bRP = reinterpret_cast<const I*>(bRowPtr.specialBuffer());
  const I* cCI = reinterpret_cast<const I*>(cColIdx.specialBuffer());
  const I* cRP = reinterpret_cast<const I*>(cRowPtr.specialBuffer());
  const X* gCV = reinterpret_cast<const X*>(gradCValues.specialBuffer());
  X*       dAV = reinterpret_cast<X*>(dAValues.specialBuffer());
  X*       dBV = reinterpret_cast<X*>(dBValues.specialBuffer());

  auto*          stream    = dAValues.getContext()->getCudaStream();
  const int      blockSize = 256;

  if (nnzA > 0) {
    const int gridA = static_cast<int>((nnzA + blockSize - 1) / blockSize);
    csrAddBpAKernel<X, I><<<gridA, blockSize, 0, *stream>>>(
        aCI, aRP, cCI, cRP, gCV, dAV, nnzA, m);
  }
  if (nnzB > 0) {
    const int gridB = static_cast<int>((nnzB + blockSize - 1) / blockSize);
    csrAddBpBKernel<X, I><<<gridB, blockSize, 0, *stream>>>(
        bCI, bRP, cCI, cRP, gCV, dBV, nnzB, m);
  }
}

// ─── public entry point ──────────────────────────────────────────────────────

void csr_add_bp(
    NDArray& aColIdx, NDArray& aRowPtr,
    NDArray& bColIdx, NDArray& bRowPtr,
    NDArray& cColIdx, NDArray& cRowPtr,
    NDArray& gradCValues,
    NDArray& dAValues, NDArray& dBValues,
    sd::LongType m, sd::LongType n) {
  NDArray::prepareSpecialUse({&dAValues, &dBValues},
                              {&aColIdx, &aRowPtr, &bColIdx, &bRowPtr,
                               &cColIdx, &cRowPtr, &gradCValues});

  BUILD_DOUBLE_SELECTOR(gradCValues.dataType(), aColIdx.dataType(), csrAddBpCuda_,
                        (aColIdx, aRowPtr, bColIdx, bRowPtr, cColIdx, cRowPtr,
                         gradCValues, dAValues, dBValues, m, n),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dAValues, &dBValues},
                               {&aColIdx, &aRowPtr, &bColIdx, &bRowPtr,
                                &cColIdx, &cRowPtr, &gradCValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
