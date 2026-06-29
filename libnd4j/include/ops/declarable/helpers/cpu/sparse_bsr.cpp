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

#include <ops/declarable/helpers/sparse_bsr.h>
#include <system/op_boilerplate.h>

#include <algorithm>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// ── csr_to_bsr CPU ──────────────────────────────────────────────────────────
// Algorithm:
//   1. Scan CSR row-by-row; per block-row bi track which block-cols bj appear.
//   2. Build bsrRowPtr via prefix sum over non-empty block counts.
//   3. Fill bsrColIdx (sorted by bj within each block-row).
//   4. Scatter each CSR entry into the correct slot of bsrValues.

template <typename X, typename I>
static void csrToBsr_(NDArray& csrValues, NDArray& csrColIdx, NDArray& csrRowPtr,
                      NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                      LongType rows, LongType cols, LongType blockDim) {
  const LongType mb   = rows / blockDim;
  const LongType nb   = cols / blockDim;
  const LongType bd   = blockDim;
  const LongType nnzb = bsrColIdx.lengthOf();  // pre-computed by DECLARE_SHAPE_FN

  const auto* vBuf  = csrValues.bufferAsT<X>();
  const auto* ciBuf = csrColIdx.bufferAsT<I>();
  const auto* rpBuf = csrRowPtr.bufferAsT<I>();

  auto* bvBuf  = bsrValues.bufferAsT<X>();
  auto* bciOut = bsrColIdx.bufferAsT<int32_t>();
  auto* brpOut = bsrRowPtr.bufferAsT<int32_t>();

  // Zero bsrValues (outputs may not be pre-zeroed for non-NULLIFIED outputs)
  const LongType totalVals = nnzb * bd * bd;
  for (LongType i = 0; i < totalVals; ++i) bvBuf[i] = static_cast<X>(0);

  // ── Step 1: for each block-row, collect sorted list of non-empty block-cols ──
  // Visited sentinel: visited[bj] == bi means bj was already seen for block-row bi.
  // Since bi is strictly increasing from 0 .. mb-1, initializing to -1 suffices.
  std::vector<LongType> visited(static_cast<size_t>(nb), static_cast<LongType>(-1));
  std::vector<std::vector<int32_t>> bjPerRow(static_cast<size_t>(mb));

  brpOut[0] = 0;
  for (LongType bi = 0; bi < mb; ++bi) {
    for (LongType r = bi * bd; r < (bi + 1) * bd; ++r) {
      const I start = rpBuf[r];
      const I end   = rpBuf[r + 1];
      for (I k = start; k < end; ++k) {
        const LongType bj = static_cast<LongType>(ciBuf[k]) / bd;
        if (visited[static_cast<size_t>(bj)] != bi) {
          visited[static_cast<size_t>(bj)] = bi;
          bjPerRow[static_cast<size_t>(bi)].push_back(static_cast<int32_t>(bj));
        }
      }
    }
    std::sort(bjPerRow[static_cast<size_t>(bi)].begin(),
              bjPerRow[static_cast<size_t>(bi)].end());
    brpOut[bi + 1] =
        brpOut[bi] + static_cast<int32_t>(bjPerRow[static_cast<size_t>(bi)].size());
  }

  // ── Step 2: fill bsrColIdx ──
  LongType blkOffset = 0;
  for (LongType bi = 0; bi < mb; ++bi) {
    for (int32_t bj : bjPerRow[static_cast<size_t>(bi)]) {
      bciOut[blkOffset++] = bj;
    }
  }

  // ── Step 3: scatter CSR entries into bsrValues ──
  // For each CSR entry at (r, c): bi = r/bd, bj = c/bd
  // block index within bsrColIdx = brpOut[bi] + position of bj in bjPerRow[bi]  (binary search)
  for (LongType bi = 0; bi < mb; ++bi) {
    const auto& bjList       = bjPerRow[static_cast<size_t>(bi)];
    const int32_t blockBase  = brpOut[bi];

    for (LongType r = bi * bd; r < (bi + 1) * bd; ++r) {
      const I start = rpBuf[r];
      const I end   = rpBuf[r + 1];
      const LongType ri = r - bi * bd;  // row within block (0..bd-1)

      for (I k = start; k < end; ++k) {
        const LongType c  = static_cast<LongType>(ciBuf[k]);
        const LongType bj = c / bd;
        const LongType ci = c % bd;  // col within block (0..bd-1)

        auto it = std::lower_bound(bjList.begin(), bjList.end(), static_cast<int32_t>(bj));
        const LongType posInRow = it - bjList.begin();
        const LongType blkIdx   = blockBase + posInRow;

        bvBuf[blkIdx * bd * bd + ri * bd + ci] = vBuf[k];
      }
    }
  }
}

void csr_to_bsr(NDArray& csrValues, NDArray& csrColIdx, NDArray& csrRowPtr,
                NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                LongType rows, LongType cols, LongType blockDim) {
  NDArray::preparePrimaryUse({&bsrValues, &bsrColIdx, &bsrRowPtr},
                             {&csrValues, &csrColIdx, &csrRowPtr});

  BUILD_DOUBLE_SELECTOR(csrValues.dataType(), csrColIdx.dataType(), csrToBsr_,
                        (csrValues, csrColIdx, csrRowPtr, bsrValues, bsrColIdx, bsrRowPtr, rows,
                         cols, blockDim),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&bsrValues, &bsrColIdx, &bsrRowPtr},
                              {&csrValues, &csrColIdx, &csrRowPtr});
}

// ── bsr_to_dense CPU ─────────────────────────────────────────────────────────

template <typename X, typename I>
static void bsrToDense_(NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                        NDArray& output, LongType rows, LongType cols, LongType blockDim) {
  const LongType mb = rows / blockDim;
  const LongType bd = blockDim;

  const auto* bvBuf = bsrValues.bufferAsT<X>();
  const auto* bciB  = bsrColIdx.bufferAsT<I>();
  const auto* brpB  = bsrRowPtr.bufferAsT<I>();

  auto*       oBuf  = output.bufferAsT<X>();
  const auto* oStr  = output.stridesOf();

  for (LongType bi = 0; bi < mb; ++bi) {
    const I blkStart = brpB[bi];
    const I blkEnd   = brpB[bi + 1];
    for (I blk = blkStart; blk < blkEnd; ++blk) {
      const LongType bj      = static_cast<LongType>(bciB[blk]);
      const LongType blkBase = static_cast<LongType>(blk) * bd * bd;
      for (LongType r = 0; r < bd; ++r) {
        for (LongType c = 0; c < bd; ++c) {
          const LongType row = bi * bd + r;
          const LongType col = bj * bd + c;
          oBuf[row * oStr[0] + col * oStr[1]] = bvBuf[blkBase + r * bd + c];
        }
      }
    }
  }
}

void bsr_to_dense(NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                  NDArray& output, LongType rows, LongType cols, LongType blockDim) {
  NDArray::preparePrimaryUse({&output}, {&bsrValues, &bsrColIdx, &bsrRowPtr});

  BUILD_DOUBLE_SELECTOR(bsrValues.dataType(), bsrColIdx.dataType(), bsrToDense_,
                        (bsrValues, bsrColIdx, bsrRowPtr, output, rows, cols, blockDim),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&output}, {&bsrValues, &bsrColIdx, &bsrRowPtr});
}

// ── bsr_spmm CPU ─────────────────────────────────────────────────────────────
// C[rows, n] = A_bsr[rows, cols] * B[cols, n]
// For each block-row bi, for each non-empty block at (bi, bj):
//   C[bi*bd:(bi+1)*bd, :] += blockMat[bd, bd] * B[bj*bd:(bj+1)*bd, :]

template <typename X, typename I>
static void bsrSpmm_(NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                     NDArray& B, NDArray& C,
                     LongType rows, LongType cols, LongType blockDim) {
  const LongType mb = rows / blockDim;
  const LongType bd = blockDim;
  const LongType n  = B.sizeAt(1);

  const auto* bvBuf = bsrValues.bufferAsT<X>();
  const auto* bciB  = bsrColIdx.bufferAsT<I>();
  const auto* brpB  = bsrRowPtr.bufferAsT<I>();
  const auto* bBuf  = B.bufferAsT<X>();
  auto*       cBuf  = C.bufferAsT<X>();

  const LongType bStride0 = B.stridesOf()[0];
  const LongType bStride1 = B.stridesOf()[1];
  const LongType cStride0 = C.stridesOf()[0];
  const LongType cStride1 = C.stridesOf()[1];

  for (LongType bi = 0; bi < mb; ++bi) {
    const I blkStart = brpB[bi];
    const I blkEnd   = brpB[bi + 1];
    for (I blk = blkStart; blk < blkEnd; ++blk) {
      const LongType bj      = static_cast<LongType>(bciB[blk]);
      const LongType blkBase = static_cast<LongType>(blk) * bd * bd;
      // block (bi, bj): bd x bd matrix at bvBuf[blkBase..]
      // C[bi*bd+r, j] += sum_c bsrBlock[r,c] * B[bj*bd+c, j]
      for (LongType r = 0; r < bd; ++r) {
        const LongType rowC = bi * bd + r;
        for (LongType c = 0; c < bd; ++c) {
          const LongType rowB = bj * bd + c;
          const X bVal = bvBuf[blkBase + r * bd + c];
          for (LongType j = 0; j < n; ++j) {
            cBuf[rowC * cStride0 + j * cStride1] +=
                bVal * bBuf[rowB * bStride0 + j * bStride1];
          }
        }
      }
    }
  }
}

void bsr_spmm(NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
              NDArray& B, NDArray& C,
              LongType rows, LongType cols, LongType blockDim) {
  NDArray::preparePrimaryUse({&C}, {&bsrValues, &bsrColIdx, &bsrRowPtr, &B});

  BUILD_DOUBLE_SELECTOR(bsrValues.dataType(), bsrColIdx.dataType(), bsrSpmm_,
                        (bsrValues, bsrColIdx, bsrRowPtr, B, C, rows, cols, blockDim),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&C}, {&bsrValues, &bsrColIdx, &bsrRowPtr, &B});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
