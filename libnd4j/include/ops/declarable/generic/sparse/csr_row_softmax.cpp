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
// csr_row_softmax — numerically-stable per-row softmax over the non-zero
// entries of a CSR matrix (GAT edge-softmax).
//
// Inputs:
//   [0] values  [nnz]    float  — edge scores
//   [1] rowPtr  [rows+1] int    — CSR row pointer array
// IArgs:
//   [0] rows  — number of rows in the sparse matrix
// Outputs:
//   [0] alpha  [nnz]  float  — softmax weights (same dtype & shape as values)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_row_softmax)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_row_softmax.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_row_softmax, 2, 1, false, 0, 1) {
  auto values = INPUT_VARIABLE(0);  // [nnz]    float
  auto rowPtr = INPUT_VARIABLE(1);  // [rows+1] int

  const sd::LongType rows = INT_ARG(0);

  REQUIRE_TRUE(values->rankOf() == 1, 0,
               "csr_row_softmax: values must be rank-1, got %d", values->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_row_softmax: rowPtr must be rank-1, got %d", rowPtr->rankOf());
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_row_softmax: rowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());

  auto alpha = OUTPUT_NULLIFIED(0);  // [nnz] float

  sd::ops::helpers::csr_row_softmax(*values, *rowPtr, *alpha, rows);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_row_softmax) {
  auto values = INPUT_VARIABLE(0);
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      values->dataType(), 'c', {values->lengthOf()});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_row_softmax) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})  // values
      ->setAllowedInputTypes(1, {ALL_INTS})    // rowPtr
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_row_softmax)
