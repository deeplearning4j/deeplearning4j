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
// csr_spgemm: C = A · B, A CSR [m,k], B CSR [k,n], OUTPUT C CSR [m,n]
//
// Inputs:
//   [0] aValues  [annz]  float   — non-zero values of A
//   [1] aColIdx  [annz]  int     — column indices of A
//   [2] aRowPtr  [m+1]   int     — row pointers of A
//   [3] bValues  [bnnz]  float   — non-zero values of B
//   [4] bColIdx  [bnnz]  int     — column indices of B
//   [5] bRowPtr  [k+1]   int     — row pointers of B
// IArgs:
//   [0] m  — rows in A / rows in C
//   [1] k  — cols in A / rows in B
//   [2] n  — cols in B / cols in C
// Outputs:
//   [0] cValues  [cnnz]  float   — non-zero values of C (same dtype as aValues)
//   [1] cColIdx  [cnnz]  INT32   — column indices of C
//   [2] cRowPtr  [m+1]   INT32   — row pointers of C
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_spgemm)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_spgemm.h>

#include <vector>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_spgemm, 6, 3, false, 0, 3) {
  auto aValues = INPUT_VARIABLE(0);
  auto aColIdx = INPUT_VARIABLE(1);
  auto aRowPtr = INPUT_VARIABLE(2);
  auto bValues = INPUT_VARIABLE(3);
  auto bColIdx = INPUT_VARIABLE(4);
  auto bRowPtr = INPUT_VARIABLE(5);

  const LongType m = INT_ARG(0);
  const LongType k = INT_ARG(1);
  const LongType n = INT_ARG(2);

  REQUIRE_TRUE(aValues->rankOf() == 1, 0,
               "csr_spgemm: aValues must be 1D, got rank %d", aValues->rankOf());
  REQUIRE_TRUE(aColIdx->rankOf() == 1, 0,
               "csr_spgemm: aColIdx must be 1D, got rank %d", aColIdx->rankOf());
  REQUIRE_TRUE(aRowPtr->rankOf() == 1, 0,
               "csr_spgemm: aRowPtr must be 1D, got rank %d", aRowPtr->rankOf());
  REQUIRE_TRUE(bValues->rankOf() == 1, 0,
               "csr_spgemm: bValues must be 1D, got rank %d", bValues->rankOf());
  REQUIRE_TRUE(bColIdx->rankOf() == 1, 0,
               "csr_spgemm: bColIdx must be 1D, got rank %d", bColIdx->rankOf());
  REQUIRE_TRUE(bRowPtr->rankOf() == 1, 0,
               "csr_spgemm: bRowPtr must be 1D, got rank %d", bRowPtr->rankOf());

  REQUIRE_TRUE(aValues->lengthOf() == aColIdx->lengthOf(), 0,
               "csr_spgemm: aValues and aColIdx must have the same length");
  REQUIRE_TRUE(bValues->lengthOf() == bColIdx->lengthOf(), 0,
               "csr_spgemm: bValues and bColIdx must have the same length");

  REQUIRE_TRUE(aRowPtr->lengthOf() == m + 1, 0,
               "csr_spgemm: aRowPtr length must be m+1=%lld, got %lld",
               (long long)(m + 1), (long long)aRowPtr->lengthOf());
  REQUIRE_TRUE(bRowPtr->lengthOf() == k + 1, 0,
               "csr_spgemm: bRowPtr length must be k+1=%lld, got %lld",
               (long long)(k + 1), (long long)bRowPtr->lengthOf());

  REQUIRE_TRUE(aColIdx->dataType() == aRowPtr->dataType(), 0,
               "csr_spgemm: aColIdx and aRowPtr must have the same integer dtype");
  REQUIRE_TRUE(bColIdx->dataType() == bRowPtr->dataType(), 0,
               "csr_spgemm: bColIdx and bRowPtr must have the same integer dtype");
  REQUIRE_TRUE(aColIdx->dataType() == bColIdx->dataType(), 0,
               "csr_spgemm: A and B index arrays must have the same integer dtype");
  REQUIRE_TRUE(aValues->dataType() == bValues->dataType(), 0,
               "csr_spgemm: aValues and bValues must have the same float dtype");

  auto cValues = OUTPUT_VARIABLE(0);
  auto cColIdx = OUTPUT_VARIABLE(1);
  auto cRowPtr = OUTPUT_VARIABLE(2);

  sd::ops::helpers::csr_spgemm(*aValues, *aColIdx, *aRowPtr,
                                *bValues, *bColIdx, *bRowPtr,
                                *cValues, *cColIdx, *cRowPtr,
                                m, k, n);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_spgemm) {
  auto aValues = INPUT_VARIABLE(0);
  auto aColIdx = INPUT_VARIABLE(1);
  auto aRowPtr = INPUT_VARIABLE(2);
  auto bColIdx = INPUT_VARIABLE(4);
  auto bRowPtr = INPUT_VARIABLE(5);

  const LongType m = INT_ARG(0);
  const LongType k = INT_ARG(1);
  const LongType n = INT_ARG(2);

  // Symbolic Gustavson pass: count distinct output non-zeros (cnnz).
  // mask[j] == i  means column j was already recorded for row i.
  // Initialise to -1 (no valid row index equals -1).
  std::vector<LongType> mask(static_cast<size_t>(n), static_cast<LongType>(-1));
  LongType cnnz = 0;

  for (LongType i = 0; i < m; ++i) {
    const LongType aStart = aRowPtr->e<LongType>(i);
    const LongType aEnd   = aRowPtr->e<LongType>(i + 1);

    for (LongType ka = aStart; ka < aEnd; ++ka) {
      const LongType kcol  = aColIdx->e<LongType>(ka);
      const LongType bStart = bRowPtr->e<LongType>(kcol);
      const LongType bEnd   = bRowPtr->e<LongType>(kcol + 1);

      for (LongType kb = bStart; kb < bEnd; ++kb) {
        const LongType bcol = bColIdx->e<LongType>(kb);
        if (mask[static_cast<size_t>(bcol)] != i) {
          mask[static_cast<size_t>(bcol)] = i;
          ++cnnz;
        }
      }
    }
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

DECLARE_TYPES(csr_spgemm) {
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
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_spgemm)
