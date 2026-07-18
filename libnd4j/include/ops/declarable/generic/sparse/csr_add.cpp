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
// csr_add: C = A + B, A and B both CSR [m, n], output C CSR [m, n]
//
// The sparsity of C is the column-set UNION of A and B per row;
// overlapping column entries are summed.
//
// Inputs:
//   [0] aValues  [annz]  float   — non-zero values of A
//   [1] aColIdx  [annz]  int     — column indices of A  (sorted per row)
//   [2] aRowPtr  [m+1]   int     — row pointers of A
//   [3] bValues  [bnnz]  float   — non-zero values of B
//   [4] bColIdx  [bnnz]  int     — column indices of B  (sorted per row)
//   [5] bRowPtr  [m+1]   int     — row pointers of B
// IArgs:
//   [0] m — number of rows
//   [1] n — number of columns
// Outputs:
//   [0] cValues  [cnnz]  float   — non-zero values of C  (same dtype as aValues)
//   [1] cColIdx  [cnnz]  INT32   — column indices of C
//   [2] cRowPtr  [m+1]   INT32   — row pointers of C
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_add)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_add.h>

#include <vector>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_add, 6, 3, false, 0, 2) {
  auto aValues = INPUT_VARIABLE(0);
  auto aColIdx = INPUT_VARIABLE(1);
  auto aRowPtr = INPUT_VARIABLE(2);
  auto bValues = INPUT_VARIABLE(3);
  auto bColIdx = INPUT_VARIABLE(4);
  auto bRowPtr = INPUT_VARIABLE(5);

  const LongType m = INT_ARG(0);
  const LongType n = INT_ARG(1);

  REQUIRE_TRUE(aValues->rankOf() == 1, 0,
               "csr_add: aValues must be 1D, got rank %d", aValues->rankOf());
  REQUIRE_TRUE(aColIdx->rankOf() == 1, 0,
               "csr_add: aColIdx must be 1D, got rank %d", aColIdx->rankOf());
  REQUIRE_TRUE(aRowPtr->rankOf() == 1, 0,
               "csr_add: aRowPtr must be 1D, got rank %d", aRowPtr->rankOf());
  REQUIRE_TRUE(bValues->rankOf() == 1, 0,
               "csr_add: bValues must be 1D, got rank %d", bValues->rankOf());
  REQUIRE_TRUE(bColIdx->rankOf() == 1, 0,
               "csr_add: bColIdx must be 1D, got rank %d", bColIdx->rankOf());
  REQUIRE_TRUE(bRowPtr->rankOf() == 1, 0,
               "csr_add: bRowPtr must be 1D, got rank %d", bRowPtr->rankOf());

  REQUIRE_TRUE(aValues->lengthOf() == aColIdx->lengthOf(), 0,
               "csr_add: aValues and aColIdx must have the same length");
  REQUIRE_TRUE(bValues->lengthOf() == bColIdx->lengthOf(), 0,
               "csr_add: bValues and bColIdx must have the same length");

  REQUIRE_TRUE(aRowPtr->lengthOf() == m + 1, 0,
               "csr_add: aRowPtr length must be m+1=%lld, got %lld",
               (long long)(m + 1), (long long)aRowPtr->lengthOf());
  REQUIRE_TRUE(bRowPtr->lengthOf() == m + 1, 0,
               "csr_add: bRowPtr length must be m+1=%lld, got %lld",
               (long long)(m + 1), (long long)bRowPtr->lengthOf());

  REQUIRE_TRUE(aColIdx->dataType() == aRowPtr->dataType(), 0,
               "csr_add: aColIdx and aRowPtr must have the same integer dtype");
  REQUIRE_TRUE(bColIdx->dataType() == bRowPtr->dataType(), 0,
               "csr_add: bColIdx and bRowPtr must have the same integer dtype");
  REQUIRE_TRUE(aColIdx->dataType() == bColIdx->dataType(), 0,
               "csr_add: A and B index arrays must have the same integer dtype");
  REQUIRE_TRUE(aValues->dataType() == bValues->dataType(), 0,
               "csr_add: aValues and bValues must have the same float dtype");

  auto cValues = OUTPUT_VARIABLE(0);
  auto cColIdx = OUTPUT_VARIABLE(1);
  auto cRowPtr = OUTPUT_VARIABLE(2);

  sd::ops::helpers::csr_add(*aValues, *aColIdx, *aRowPtr,
                             *bValues, *bColIdx, *bRowPtr,
                             *cValues, *cColIdx, *cRowPtr,
                             m, n);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_add) {
  auto aValues = INPUT_VARIABLE(0);
  auto aColIdx = INPUT_VARIABLE(1);
  auto aRowPtr = INPUT_VARIABLE(2);
  auto bColIdx = INPUT_VARIABLE(4);
  auto bRowPtr = INPUT_VARIABLE(5);

  const LongType m = INT_ARG(0);
  // n is not needed for the symbolic pass (columns only appear in colIdx arrays)

  // Symbolic union pass: for each row i count the number of distinct columns
  // in (A row i) ∪ (B row i).  Both per-row colIdx sequences are sorted in
  // ascending order (standard CSR invariant), so we use a linear merge-count.
  LongType cnnz = 0;

  for (LongType i = 0; i < m; ++i) {
    LongType aS = aRowPtr->e<LongType>(i);
    LongType aE = aRowPtr->e<LongType>(i + 1);
    LongType bS = bRowPtr->e<LongType>(i);
    LongType bE = bRowPtr->e<LongType>(i + 1);

    LongType a = aS, b = bS;
    while (a < aE && b < bE) {
      const LongType ca = aColIdx->e<LongType>(a);
      const LongType cb = bColIdx->e<LongType>(b);
      if (ca < cb) {
        ++a;
      } else if (ca > cb) {
        ++b;
      } else {
        // same column: both advance, one entry in C
        ++a;
        ++b;
      }
      ++cnnz;
    }
    // Remaining entries in whichever side has leftover all go into C
    cnnz += (aE - a) + (bE - b);
  }

  // Output dtypes: cValues inherits float dtype from aValues;
  // cColIdx and cRowPtr are always INT32.
  const auto floatType = aValues->dataType();
  const auto idxType   = sd::DataType::INT32;

  auto cValShape = ConstantShapeHelper::getInstance().createShapeInfo(floatType, 'c', {cnnz});
  auto cCiShape  = ConstantShapeHelper::getInstance().createShapeInfo(idxType,   'c', {cnnz});
  auto cRpShape  = ConstantShapeHelper::getInstance().createShapeInfo(idxType,   'c', {m + 1});

  return SHAPELIST(cValShape, cCiShape, cRpShape);
}

DECLARE_TYPES(csr_add) {

  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})  // aValues
      ->setAllowedInputTypes(1, {ALL_INTS})    // aColIdx
      ->setAllowedInputTypes(2, {ALL_INTS})    // aRowPtr
      ->setAllowedInputTypes(3, {ALL_FLOATS})  // bValues
      ->setAllowedInputTypes(4, {ALL_INTS})    // bColIdx
      ->setAllowedInputTypes(5, {ALL_INTS})    // bRowPtr
      ->setAllowedOutputTypes(0, {ALL_FLOATS}) // cValues
      ->setAllowedOutputTypes(1, {ALL_INTS})   // cColIdx
      ->setAllowedOutputTypes(2, {ALL_INTS});  // cRowPtr

  getOpDescriptor()->addTraits(OP_TRAIT_DATA_DEPENDENT | OP_TRAIT_DYNAMIC_OUTPUT_SIZE);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_add)
