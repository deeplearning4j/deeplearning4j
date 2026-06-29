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

// csr_to_bsr: CSR [rows,cols] → BSR [mb,nb] with blockDim bd
// in[0]=csrValues[nnz] float, in[1]=csrColIdx[nnz] int, in[2]=csrRowPtr[rows+1] int
// IArgs: [0]=rows, [1]=cols, [2]=blockDim
// out[0]=bsrValues[nnzb*bd*bd] float, out[1]=bsrColIdx[nnzb] INT32, out[2]=bsrRowPtr[mb+1] INT32

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_to_bsr)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_bsr.h>

#include <vector>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_to_bsr, 3, 3, false, 0, 3) {
  auto csrValues = INPUT_VARIABLE(0);
  auto csrColIdx = INPUT_VARIABLE(1);
  auto csrRowPtr = INPUT_VARIABLE(2);

  const LongType rows     = INT_ARG(0);
  const LongType cols     = INT_ARG(1);
  const LongType blockDim = INT_ARG(2);

  REQUIRE_TRUE(csrValues->rankOf() == 1, 0, "csr_to_bsr: csrValues must be 1D");
  REQUIRE_TRUE(csrColIdx->rankOf() == 1, 0, "csr_to_bsr: csrColIdx must be 1D");
  REQUIRE_TRUE(csrRowPtr->rankOf() == 1, 0, "csr_to_bsr: csrRowPtr must be 1D");
  REQUIRE_TRUE(rows % blockDim == 0, 0,
               "csr_to_bsr: rows (%lld) must be divisible by blockDim (%lld)",
               (long long)rows, (long long)blockDim);
  REQUIRE_TRUE(cols % blockDim == 0, 0,
               "csr_to_bsr: cols (%lld) must be divisible by blockDim (%lld)",
               (long long)cols, (long long)blockDim);
  REQUIRE_TRUE(csrValues->lengthOf() == csrColIdx->lengthOf(), 0,
               "csr_to_bsr: csrValues and csrColIdx must have the same length");
  REQUIRE_TRUE(csrRowPtr->lengthOf() == rows + 1, 0,
               "csr_to_bsr: csrRowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)csrRowPtr->lengthOf());

  auto bsrValues = OUTPUT_VARIABLE(0);
  auto bsrColIdx = OUTPUT_VARIABLE(1);
  auto bsrRowPtr = OUTPUT_VARIABLE(2);

  if (csrValues->lengthOf() == 0) {
    // Use nullify(), NOT assign(0): the literal 0 binds to assign(NDArray*=nullptr) -> SIGSEGV.
    bsrRowPtr->nullify();
    return sd::Status::OK;
  }

  sd::ops::helpers::csr_to_bsr(*csrValues, *csrColIdx, *csrRowPtr, *bsrValues, *bsrColIdx,
                                *bsrRowPtr, rows, cols, blockDim);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_to_bsr) {
  auto csrValues = INPUT_VARIABLE(0);
  auto csrColIdx = INPUT_VARIABLE(1);
  auto csrRowPtr = INPUT_VARIABLE(2);

  const LongType rows     = INT_ARG(0);
  const LongType cols     = INT_ARG(1);
  const LongType blockDim = INT_ARG(2);

  const LongType mb = rows / blockDim;
  const LongType nb = cols / blockDim;

  // Count distinct (block-row, block-col) pairs = nnzb
  // visited[bi * nb + bj] == bi means block (bi, bj) already counted for block-row bi.
  // Since bi is strictly increasing, the sentinel trick eliminates per-block-row reinit.
  const LongType nbSz = static_cast<size_t>(mb * nb);
  std::vector<LongType> visited(static_cast<size_t>(nbSz), static_cast<LongType>(-1));
  LongType nnzb = 0;

  for (LongType bi = 0; bi < mb; ++bi) {
    const LongType rowStart = bi * blockDim;
    const LongType rowEnd   = rowStart + blockDim;
    for (LongType r = rowStart; r < rowEnd; ++r) {
      const LongType start = csrRowPtr->e<LongType>(r);
      const LongType end   = csrRowPtr->e<LongType>(r + 1);
      for (LongType k = start; k < end; ++k) {
        const LongType bj  = csrColIdx->e<LongType>(k) / blockDim;
        const LongType idx = bi * nb + bj;
        if (visited[static_cast<size_t>(idx)] != bi) {
          visited[static_cast<size_t>(idx)] = bi;
          ++nnzb;
        }
      }
    }
  }

  const auto floatType = csrValues->dataType();
  const auto idxType   = sd::DataType::INT32;

  auto valShape = ConstantShapeHelper::getInstance().createShapeInfo(
      floatType, 'c', {nnzb * blockDim * blockDim});
  auto ciShape  = ConstantShapeHelper::getInstance().createShapeInfo(idxType, 'c', {nnzb});
  auto rpShape  = ConstantShapeHelper::getInstance().createShapeInfo(idxType, 'c', {mb + 1});

  return SHAPELIST(valShape, ciShape, rpShape);
}

DECLARE_TYPES(csr_to_bsr) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})   // csrValues
      ->setAllowedInputTypes(1, {ALL_INTS})     // csrColIdx
      ->setAllowedInputTypes(2, {ALL_INTS})     // csrRowPtr
      ->setAllowedOutputTypes(0, {ALL_FLOATS})  // bsrValues
      ->setAllowedOutputTypes(1, {ALL_INTS})    // bsrColIdx
      ->setAllowedOutputTypes(2, {ALL_INTS});   // bsrRowPtr
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_to_bsr)
