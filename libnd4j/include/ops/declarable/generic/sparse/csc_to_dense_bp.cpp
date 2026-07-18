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
// csc_to_dense_bp — backward pass for csc_to_dense.
//
// Forward:  CSC entry k (row = cscRowIdx[k], col c found via cscColPtr) →
//           dense[row, col] = cscValues[k]
//
// Backward: gather from gradient:
//   dCscValues[k] = gradDense[cscRowIdx[k], col]
//
// Inputs:
//   [0] cscRowIdx  [nnz]       INT   — CSC row indices
//   [1] cscColPtr  [cols+1]    INT   — CSC column pointers
//   [2] gradDense  [rows,cols] float — upstream gradient w.r.t. dense output
// IArgs:
//   [0] rows
//   [1] cols
// Output:
//   [0] dCscValues [nnz] float — gradient w.r.t. CSC values
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csc_to_dense_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_csc_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csc_to_dense_bp, 3, 1, false, 0, 2) {
  auto cscRowIdx = INPUT_VARIABLE(0);
  auto cscColPtr = INPUT_VARIABLE(1);
  auto gradDense = INPUT_VARIABLE(2);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(cscRowIdx->rankOf() == 1, 0,
               "csc_to_dense_bp: cscRowIdx must be 1D, got rank %d", cscRowIdx->rankOf());
  REQUIRE_TRUE(cscColPtr->rankOf() == 1, 0,
               "csc_to_dense_bp: cscColPtr must be 1D, got rank %d", cscColPtr->rankOf());
  REQUIRE_TRUE(gradDense->rankOf() == 2, 0,
               "csc_to_dense_bp: gradDense must be 2D, got rank %d", gradDense->rankOf());
  REQUIRE_TRUE(cscColPtr->lengthOf() == cols + 1, 0,
               "csc_to_dense_bp: cscColPtr length must be cols+1=%lld, got %lld",
               (long long)(cols + 1), (long long)cscColPtr->lengthOf());
  REQUIRE_TRUE(gradDense->sizeAt(0) == rows && gradDense->sizeAt(1) == cols, 0,
               "csc_to_dense_bp: gradDense shape must be [rows=%lld, cols=%lld], got [%lld, %lld]",
               (long long)rows, (long long)cols,
               (long long)gradDense->sizeAt(0), (long long)gradDense->sizeAt(1));

  auto dCscValues = OUTPUT_VARIABLE(0);

  const LongType nnz = cscRowIdx->lengthOf();
  if (nnz > 0) {
    sd::ops::helpers::csc_to_dense_bp(
        *cscRowIdx, *cscColPtr,
        *gradDense, *dCscValues,
        rows, cols);
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csc_to_dense_bp) {
  auto cscRowIdx = INPUT_VARIABLE(0);
  auto gradDense = INPUT_VARIABLE(2);

  const LongType nnz = cscRowIdx->lengthOf();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      gradDense->dataType(), 'c', {nnz});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csc_to_dense_bp) {

  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})     // cscRowIdx
      ->setAllowedInputTypes(1, {ALL_INTS})     // cscColPtr
      ->setAllowedInputTypes(2, {ALL_FLOATS})   // gradDense
      ->setAllowedOutputTypes(0, {ALL_FLOATS}); // dCscValues

  getOpDescriptor()->addTraits(OP_TRAIT_BACKWARD);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csc_to_dense_bp)
