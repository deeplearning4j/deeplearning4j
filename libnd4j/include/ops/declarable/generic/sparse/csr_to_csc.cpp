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
// CSR sparse → CSC sparse conversion op.
// (CSC of A == CSR of Aᵀ relabelled; this op therefore also gives the
//  transpose of a sparse matrix.)
//
// Inputs:
//   [0] csrValues  [nnz],     float dtype
//   [1] csrColIdx  [nnz],     INT32 or INT64
//   [2] csrRowPtr  [rows+1],  same INT dtype as csrColIdx
// IArgs:
//   [0] rows
//   [1] cols
// Outputs:
//   [0] cscValues  [nnz],     same float dtype as csrValues
//   [1] cscRowIdx  [nnz],     INT32
//   [2] cscColPtr  [cols+1],  INT32
//
// nnz == csrValues.lengthOf() — not data-dependent, so DECLARE_SHAPE_FN
// can emit exact shapes without scanning.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_to_csc)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_csc.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_to_csc, 3, 3, false, 0, 2) {
  auto csrValues = INPUT_VARIABLE(0);
  auto csrColIdx = INPUT_VARIABLE(1);
  auto csrRowPtr = INPUT_VARIABLE(2);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(csrValues->rankOf() == 1, 0,
               "csr_to_csc: csrValues must be 1D, got rank %d", csrValues->rankOf());
  REQUIRE_TRUE(csrColIdx->rankOf() == 1, 0,
               "csr_to_csc: csrColIdx must be 1D, got rank %d", csrColIdx->rankOf());
  REQUIRE_TRUE(csrRowPtr->rankOf() == 1, 0,
               "csr_to_csc: csrRowPtr must be 1D, got rank %d", csrRowPtr->rankOf());
  REQUIRE_TRUE(csrRowPtr->lengthOf() == rows + 1, 0,
               "csr_to_csc: csrRowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)csrRowPtr->lengthOf());
  REQUIRE_TRUE(csrValues->lengthOf() == csrColIdx->lengthOf(), 0,
               "csr_to_csc: csrValues and csrColIdx must have the same length");
  REQUIRE_TRUE(csrColIdx->dataType() == csrRowPtr->dataType(), 0,
               "csr_to_csc: csrColIdx and csrRowPtr must have the same integer dtype");

  auto cscValues = OUTPUT_VARIABLE(0);
  auto cscRowIdx = OUTPUT_VARIABLE(1);
  auto cscColPtr = OUTPUT_VARIABLE(2);

  if (csrValues->lengthOf() == 0) {
    // Empty input: colPtr is all zeros (already done by framework), done.
    // Use nullify(), NOT assign(0): the literal 0 binds to assign(NDArray*=nullptr) -> SIGSEGV.
    cscColPtr->nullify();
    return sd::Status::OK;
  }

  sd::ops::helpers::csr_to_csc(*csrValues, *csrColIdx, *csrRowPtr,
                                *cscValues, *cscRowIdx, *cscColPtr, rows, cols);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_to_csc) {
  auto csrValues = INPUT_VARIABLE(0);

  const LongType nnz  = csrValues->lengthOf();
  const LongType cols = INT_ARG(1);

  auto dtype   = csrValues->dataType();
  auto idxType = sd::DataType::INT32;

  auto valShape  = ConstantShapeHelper::getInstance().createShapeInfo(dtype,   'c', {nnz});
  auto rowShape  = ConstantShapeHelper::getInstance().createShapeInfo(idxType, 'c', {nnz});
  auto colPShape = ConstantShapeHelper::getInstance().createShapeInfo(idxType, 'c', {cols + 1});

  return SHAPELIST(valShape, rowShape, colPShape);
}

DECLARE_TYPES(csr_to_csc) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})   // csrValues
      ->setAllowedInputTypes(1, {ALL_INTS})     // csrColIdx
      ->setAllowedInputTypes(2, {ALL_INTS})     // csrRowPtr
      ->setAllowedOutputTypes(0, {ALL_FLOATS})  // cscValues
      ->setAllowedOutputTypes(1, {ALL_INTS})    // cscRowIdx
      ->setAllowedOutputTypes(2, {ALL_INTS});   // cscColPtr
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_to_csc)
