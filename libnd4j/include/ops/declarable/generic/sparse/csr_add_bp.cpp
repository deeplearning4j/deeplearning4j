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
// csr_add_bp — backward pass for csr_add: C = A + B (CSR union sparsity).
//
// Inputs:
//   [0] aColIdx     [nnzA]  INT   — column indices of A (sorted per row)
//   [1] aRowPtr     [m+1]   INT   — row pointers of A
//   [2] bColIdx     [nnzB]  INT   — column indices of B (sorted per row)
//   [3] bRowPtr     [m+1]   INT   — row pointers of B
//   [4] cColIdx     [nnzC]  INT   — column indices of C (forward output[1])
//   [5] cRowPtr     [m+1]   INT   — row pointers of C (forward output[2])
//   [6] gradCValues [nnzC]  float — upstream gradient w.r.t. C values
// IArgs:
//   [0] m  — number of rows
//   [1] n  — number of columns
// Outputs:
//   [0] dAValues [nnzA] float — gradient w.r.t. A values
//   [1] dBValues [nnzB] float — gradient w.r.t. B values
//
// Math: each A entry at (row r, col c) maps to a unique C entry via binary
// search in C's sorted per-row column list — pure gather, no atomics.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_add_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_add_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_add_bp, 7, 2, false, 0, 2) {
  auto aColIdx     = INPUT_VARIABLE(0);
  auto aRowPtr     = INPUT_VARIABLE(1);
  auto bColIdx     = INPUT_VARIABLE(2);
  auto bRowPtr     = INPUT_VARIABLE(3);
  auto cColIdx     = INPUT_VARIABLE(4);
  auto cRowPtr     = INPUT_VARIABLE(5);
  auto gradCValues = INPUT_VARIABLE(6);

  const LongType m = INT_ARG(0);
  const LongType n = INT_ARG(1);

  REQUIRE_TRUE(aColIdx->rankOf() == 1, 0,
               "csr_add_bp: aColIdx must be 1D, got rank %d", aColIdx->rankOf());
  REQUIRE_TRUE(aRowPtr->rankOf() == 1, 0,
               "csr_add_bp: aRowPtr must be 1D, got rank %d", aRowPtr->rankOf());
  REQUIRE_TRUE(bColIdx->rankOf() == 1, 0,
               "csr_add_bp: bColIdx must be 1D, got rank %d", bColIdx->rankOf());
  REQUIRE_TRUE(bRowPtr->rankOf() == 1, 0,
               "csr_add_bp: bRowPtr must be 1D, got rank %d", bRowPtr->rankOf());
  REQUIRE_TRUE(cColIdx->rankOf() == 1, 0,
               "csr_add_bp: cColIdx must be 1D, got rank %d", cColIdx->rankOf());
  REQUIRE_TRUE(cRowPtr->rankOf() == 1, 0,
               "csr_add_bp: cRowPtr must be 1D, got rank %d", cRowPtr->rankOf());
  REQUIRE_TRUE(gradCValues->rankOf() == 1, 0,
               "csr_add_bp: gradCValues must be 1D, got rank %d", gradCValues->rankOf());
  REQUIRE_TRUE(aRowPtr->lengthOf() == m + 1, 0,
               "csr_add_bp: aRowPtr length must be m+1=%lld, got %lld",
               (long long)(m + 1), (long long)aRowPtr->lengthOf());
  REQUIRE_TRUE(bRowPtr->lengthOf() == m + 1, 0,
               "csr_add_bp: bRowPtr length must be m+1=%lld, got %lld",
               (long long)(m + 1), (long long)bRowPtr->lengthOf());
  REQUIRE_TRUE(cRowPtr->lengthOf() == m + 1, 0,
               "csr_add_bp: cRowPtr length must be m+1=%lld, got %lld",
               (long long)(m + 1), (long long)cRowPtr->lengthOf());
  REQUIRE_TRUE(gradCValues->lengthOf() == cColIdx->lengthOf(), 0,
               "csr_add_bp: gradCValues length must equal cColIdx length %lld, got %lld",
               (long long)cColIdx->lengthOf(), (long long)gradCValues->lengthOf());

  auto dAValues = OUTPUT_VARIABLE(0);
  auto dBValues = OUTPUT_VARIABLE(1);

  sd::ops::helpers::csr_add_bp(
      *aColIdx, *aRowPtr,
      *bColIdx, *bRowPtr,
      *cColIdx, *cRowPtr,
      *gradCValues,
      *dAValues, *dBValues,
      m, n);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_add_bp) {
  auto aColIdx     = INPUT_VARIABLE(0);
  auto bColIdx     = INPUT_VARIABLE(2);
  auto gradCValues = INPUT_VARIABLE(6);

  const LongType nnzA = aColIdx->lengthOf();
  const LongType nnzB = bColIdx->lengthOf();
  auto dtype = gradCValues->dataType();

  auto shapeA = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {nnzA});
  auto shapeB = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {nnzB});
  return SHAPELIST(shapeA, shapeB);
}

DECLARE_TYPES(csr_add_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})     // aColIdx
      ->setAllowedInputTypes(1, {ALL_INTS})     // aRowPtr
      ->setAllowedInputTypes(2, {ALL_INTS})     // bColIdx
      ->setAllowedInputTypes(3, {ALL_INTS})     // bRowPtr
      ->setAllowedInputTypes(4, {ALL_INTS})     // cColIdx
      ->setAllowedInputTypes(5, {ALL_INTS})     // cRowPtr
      ->setAllowedInputTypes(6, {ALL_FLOATS})   // gradCValues
      ->setAllowedOutputTypes(0, {ALL_FLOATS})  // dAValues
      ->setAllowedOutputTypes(1, {ALL_FLOATS}); // dBValues
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_add_bp)
