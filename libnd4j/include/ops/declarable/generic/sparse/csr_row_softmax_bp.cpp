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
// csr_row_softmax_bp — backward pass of csr_row_softmax.
//
// Per row i, let dot_i = sum_{k in row i} alpha[k] * gradOut[k].
// dValues[k] = alpha[k] * (gradOut[k] - dot_i)
//
// Inputs:
//   [0] alpha    [nnz]    float  — forward output (softmax weights)
//   [1] rowPtr   [rows+1] int    — CSR row pointer array
//   [2] gradOut  [nnz]    float  — upstream gradient
// IArgs:
//   [0] rows
// Outputs:
//   [0] dValues  [nnz]  float  — gradient w.r.t. input values
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_row_softmax_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_row_softmax.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_row_softmax_bp, 3, 1, false, 0, 1) {
  auto alpha   = INPUT_VARIABLE(0);  // [nnz]    float
  auto rowPtr  = INPUT_VARIABLE(1);  // [rows+1] int
  auto gradOut = INPUT_VARIABLE(2);  // [nnz]    float

  const sd::LongType rows = INT_ARG(0);

  REQUIRE_TRUE(alpha->rankOf() == 1, 0,
               "csr_row_softmax_bp: alpha must be rank-1, got %d", alpha->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_row_softmax_bp: rowPtr must be rank-1, got %d", rowPtr->rankOf());
  REQUIRE_TRUE(gradOut->rankOf() == 1, 0,
               "csr_row_softmax_bp: gradOut must be rank-1, got %d", gradOut->rankOf());
  REQUIRE_TRUE(gradOut->lengthOf() == alpha->lengthOf(), 0,
               "csr_row_softmax_bp: gradOut length %lld != alpha length %lld",
               (long long)gradOut->lengthOf(), (long long)alpha->lengthOf());
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_row_softmax_bp: rowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());

  auto dValues = OUTPUT_NULLIFIED(0);  // [nnz] float

  sd::ops::helpers::csr_row_softmax_bp(*alpha, *rowPtr, *gradOut, *dValues, rows);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_row_softmax_bp) {
  auto alpha = INPUT_VARIABLE(0);
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      alpha->dataType(), 'c', {alpha->lengthOf()});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_row_softmax_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})  // alpha
      ->setAllowedInputTypes(1, {ALL_INTS})    // rowPtr
      ->setAllowedInputTypes(2, {ALL_FLOATS})  // gradOut
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_row_softmax_bp)
