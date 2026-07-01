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
// CPU backward-pass helpers for BSR sparse operations.
//
// bsr_to_dense_bp : gather gradDense at the BSR block sparsity pattern → dBsrValues.
// bsr_spmm_bp     : gradient of C = A_bsr * B with respect to A values and B.
// csr_to_bsr_bp   : gather gradBsrValues at the CSR→BSR mapping → dCsrValues.
//

#include <ops/declarable/helpers/sparse_bsr_bp.h>
#include <system/op_boilerplate.h>

#include <algorithm>  // std::lower_bound

namespace sd {
namespace ops {
namespace helpers {

// ===========================================================================
// bsr_to_dense_bp — CPU
//
// Forward bsr_to_dense places block k (block-row br, block-col bc = bsrColIdx[k])
// into dense[br*bd + lr, bc*bd + lc] for lr,lc in [0,bd).
//
// Backward is a pure gather from the same dense positions:
//   dBsrValues[k*bd*bd + lr*bd + lc] = gradDense[br*bd + lr, bc*bd + lc]
// ===========================================================================

template <typename X, typename I>
static void bsrToDenseBp_(
    NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& gradDense, NDArray& dBsrValues,
    LongType rows, LongType /*cols*/, LongType blockDim) {
  const LongType mb = rows / blockDim;
  const LongType bd = blockDim;

  const auto* bciB = bsrColIdx.bufferAsT<I>();
  const auto* brpB = bsrRowPtr.bufferAsT<I>();
  const auto* gBuf = gradDense.bufferAsT<X>();
  auto*       dvBuf = dBsrValues.bufferAsT<X>();

  const LongType gStride0 = gradDense.stridesOf()[0];
  const LongType gStride1 = gradDense.stridesOf()[1];

  for (LongType bi = 0; bi < mb; ++bi) {
    const I blkStart = brpB[bi];
    const I blkEnd   = brpB[bi + 1];
    for (I blk = blkStart; blk < blkEnd; ++blk) {
      const LongType bj      = static_cast<LongType>(bciB[blk]);
      const LongType blkBase = static_cast<LongType>(blk) * bd * bd;
      for (LongType lr = 0; lr < bd; ++lr) {
        for (LongType lc = 0; lc < bd; ++lc) {
          const LongType dense_row = bi * bd + lr;
          const LongType dense_col = bj * bd + lc;
          dvBuf[blkBase + lr * bd + lc] =
              gBuf[dense_row * gStride0 + dense_col * gStride1];
        }
      }
    }
  }
}

void bsr_to_dense_bp(
    NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& gradDense, NDArray& dBsrValues,
    LongType rows, LongType cols, LongType blockDim) {
  NDArray::preparePrimaryUse({&dBsrValues}, {&bsrColIdx, &bsrRowPtr, &gradDense});

  BUILD_DOUBLE_SELECTOR(gradDense.dataType(), bsrColIdx.dataType(), bsrToDenseBp_,
      (bsrColIdx, bsrRowPtr, gradDense, dBsrValues, rows, cols, blockDim),
      SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dBsrValues}, {&bsrColIdx, &bsrRowPtr, &gradDense});
}

// ===========================================================================
// bsr_spmm_bp — CPU
//
// Forward: C[rows, n] = A_bsr[rows, cols] * B[cols, n]
// For block k at (br, bc):
//   C[br*bd+lr, j] += sum_lc A_block[lr, lc] * B[bc*bd+lc, j]
//
// Backward (gradC [rows, n] given):
//
// dBsrValues[k*bd*bd + lr*bd + lc]
//   = sum_{j=0}^{n-1} gradC[(br*bd+lr), j] * B[(bc*bd+lc), j]
//
// dB = A_bsr^T * gradC, i.e. for block k at (br, bc):
//   dB[(bc*bd+lc), j] += A_block[lr, lc] * gradC[(br*bd+lr), j]
//   for all lr, lc, j.
//
// Both dBsrValues and dB are OUTPUT_NULLIFIED (pre-zeroed by the framework).
// ===========================================================================

template <typename X, typename I>
static void bsrSpmmBp_(
    NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& B, NDArray& gradC,
    NDArray& dBsrValues, NDArray& dB,
    LongType rows, LongType /*cols*/, LongType blockDim) {
  const LongType mb = rows / blockDim;
  const LongType bd = blockDim;
  const LongType n  = B.sizeAt(1);

  const auto* bvBuf  = bsrValues.bufferAsT<X>();
  const auto* bciB   = bsrColIdx.bufferAsT<I>();
  const auto* brpB   = bsrRowPtr.bufferAsT<I>();
  const auto* bBuf   = B.bufferAsT<X>();
  const auto* gcBuf  = gradC.bufferAsT<X>();
  auto*       dvBuf  = dBsrValues.bufferAsT<X>();
  auto*       dbBuf  = dB.bufferAsT<X>();

  const LongType bStride0  = B.stridesOf()[0];
  const LongType bStride1  = B.stridesOf()[1];
  const LongType gcStride0 = gradC.stridesOf()[0];
  const LongType gcStride1 = gradC.stridesOf()[1];
  const LongType dbStride0 = dB.stridesOf()[0];
  const LongType dbStride1 = dB.stridesOf()[1];

  for (LongType bi = 0; bi < mb; ++bi) {
    const I blkStart = brpB[bi];
    const I blkEnd   = brpB[bi + 1];
    for (I blk = blkStart; blk < blkEnd; ++blk) {
      const LongType bj      = static_cast<LongType>(bciB[blk]);
      const LongType blkBase = static_cast<LongType>(blk) * bd * bd;

      for (LongType lr = 0; lr < bd; ++lr) {
        const LongType rowC = bi * bd + lr;

        for (LongType lc = 0; lc < bd; ++lc) {
          const LongType rowB = bj * bd + lc;
          X dv = static_cast<X>(0);
          const X aVal = bvBuf[blkBase + lr * bd + lc];

          for (LongType j = 0; j < n; ++j) {
            // dBsrValues: inner product of gradC row (br*bd+lr) with B row (bc*bd+lc)
            dv += gcBuf[rowC * gcStride0 + j * gcStride1]
                * bBuf[rowB * bStride0  + j * bStride1];
            // dB: A^T * gradC accumulation
            dbBuf[rowB * dbStride0 + j * dbStride1]
                += aVal * gcBuf[rowC * gcStride0 + j * gcStride1];
          }
          dvBuf[blkBase + lr * bd + lc] = dv;
        }
      }
    }
  }
}

void bsr_spmm_bp(
    NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& B, NDArray& gradC,
    NDArray& dBsrValues, NDArray& dB,
    LongType rows, LongType cols, LongType blockDim) {
  NDArray::preparePrimaryUse({&dBsrValues, &dB},
      {&bsrValues, &bsrColIdx, &bsrRowPtr, &B, &gradC});

  BUILD_DOUBLE_SELECTOR(bsrValues.dataType(), bsrColIdx.dataType(), bsrSpmmBp_,
      (bsrValues, bsrColIdx, bsrRowPtr, B, gradC, dBsrValues, dB, rows, cols, blockDim),
      SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dBsrValues, &dB},
      {&bsrValues, &bsrColIdx, &bsrRowPtr, &B, &gradC});
}

// ===========================================================================
// csr_to_bsr_bp — CPU
//
// Forward csr_to_bsr maps each CSR entry k at global (row r, col c) into
// BSR block (br = r/bd, bc = c/bd), local position (lr = r%bd, lc = c%bd).
// The block index within bsrColIdx is found via binary search in
// bsrColIdx[bsrRowPtr[br]..bsrRowPtr[br+1]).
//
// Backward is a pure gather:
//   dCsrValues[k] = gradBsrValues[block_k * bd*bd + lr*bd + lc]
// ===========================================================================

template <typename X, typename I>
static void csrToBsrBp_(
    NDArray& csrColIdx, NDArray& csrRowPtr,
    NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& gradBsrValues, NDArray& dCsrValues,
    LongType rows, LongType /*cols*/, LongType blockDim) {
  const LongType bd = blockDim;

  const auto* ciBuf  = csrColIdx.bufferAsT<I>();
  const auto* rpBuf  = csrRowPtr.bufferAsT<I>();
  const auto* bciB   = bsrColIdx.bufferAsT<I>();
  const auto* brpB   = bsrRowPtr.bufferAsT<I>();
  const auto* gbvBuf = gradBsrValues.bufferAsT<X>();
  auto*       dcvBuf = dCsrValues.bufferAsT<X>();

  for (LongType r = 0; r < rows; ++r) {
    const LongType br = r / bd;
    const LongType lr = r % bd;
    const I bsrRowStart = brpB[br];
    const I bsrRowEnd   = brpB[br + 1];

    const I start = rpBuf[r];
    const I end   = rpBuf[r + 1];
    for (I k = start; k < end; ++k) {
      const LongType c  = static_cast<LongType>(ciBuf[k]);
      const LongType bc = c / bd;
      const LongType lc = c % bd;

      // Binary search for bc in bsrColIdx[bsrRowStart..bsrRowEnd)
      I lo = bsrRowStart, hi = bsrRowEnd;
      while (lo < hi) {
        const I mid = lo + (hi - lo) / 2;
        if (static_cast<LongType>(bciB[mid]) < bc) lo = mid + 1; else hi = mid;
      }
      // lo is the block index within bsrColIdx (guaranteed to be found for valid csr_to_bsr)
      const LongType blkBase = static_cast<LongType>(lo) * bd * bd;
      dcvBuf[k] = gbvBuf[blkBase + lr * bd + lc];
    }
  }
}

void csr_to_bsr_bp(
    NDArray& csrColIdx, NDArray& csrRowPtr,
    NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& gradBsrValues, NDArray& dCsrValues,
    LongType rows, LongType cols, LongType blockDim) {
  NDArray::preparePrimaryUse({&dCsrValues},
      {&csrColIdx, &csrRowPtr, &bsrColIdx, &bsrRowPtr, &gradBsrValues});

  BUILD_DOUBLE_SELECTOR(gradBsrValues.dataType(), csrColIdx.dataType(), csrToBsrBp_,
      (csrColIdx, csrRowPtr, bsrColIdx, bsrRowPtr, gradBsrValues, dCsrValues, rows, cols, blockDim),
      SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dCsrValues},
      {&csrColIdx, &csrRowPtr, &bsrColIdx, &bsrRowPtr, &gradBsrValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
