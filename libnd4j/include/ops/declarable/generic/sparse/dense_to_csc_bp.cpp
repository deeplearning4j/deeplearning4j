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
// dense_to_csc_bp — backward pass for dense_to_csc.
//
// Forward: dense[rows, cols] → (cscValues[nnz], cscRowIdx[nnz], cscColPtr[cols+1])
// selecting entries where |x| > threshold (column-major scan order).
//
// Backward: scatter gradient back to the dense positions:
//   dDense[cscRowIdx[k], col] = gradCscValues[k]
// where col is determined by binary search in cscColPtr.
// Each (row, col) in a valid CSC pattern is unique — no atomics required.
//
// Inputs:
//   [0] cscRowIdx      [nnz]     INT   — CSC row indices (forward output[1])
//   [1] cscColPtr      [cols+1]  INT   — CSC column pointers (forward output[2])
//   [2] gradCscValues  [nnz]     float — upstream gradient w.r.t. CSC values
// IArgs:
//   [0] rows
//   [1] cols
// Output:
//   [0] dDense [rows, cols] float — gradient w.r.t. dense input (pre-zeroed)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dense_to_csc_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_csc_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dense_to_csc_bp, 3, 1, false, 0, 2) {
  auto cscRowIdx     = INPUT_VARIABLE(0);
  auto cscColPtr     = INPUT_VARIABLE(1);
  auto gradCscValues = INPUT_VARIABLE(2);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(cscRowIdx->rankOf() == 1, 0,
               "dense_to_csc_bp: cscRowIdx must be 1D, got rank %d", cscRowIdx->rankOf());
  REQUIRE_TRUE(cscColPtr->rankOf() == 1, 0,
               "dense_to_csc_bp: cscColPtr must be 1D, got rank %d", cscColPtr->rankOf());
  REQUIRE_TRUE(gradCscValues->rankOf() == 1, 0,
               "dense_to_csc_bp: gradCscValues must be 1D, got rank %d", gradCscValues->rankOf());
  REQUIRE_TRUE(cscColPtr->lengthOf() == cols + 1, 0,
               "dense_to_csc_bp: cscColPtr length must be cols+1=%lld, got %lld",
               (long long)(cols + 1), (long long)cscColPtr->lengthOf());
  REQUIRE_TRUE(gradCscValues->lengthOf() == cscRowIdx->lengthOf(), 0,
               "dense_to_csc_bp: gradCscValues and cscRowIdx must have the same length");

  auto dDense = OUTPUT_NULLIFIED(0);

  const LongType nnz = cscRowIdx->lengthOf();
  if (nnz > 0) {
    sd::ops::helpers::dense_to_csc_bp(
        *cscRowIdx, *cscColPtr,
        *gradCscValues, *dDense,
        rows, cols);
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(dense_to_csc_bp) {
  auto gradCscValues = INPUT_VARIABLE(2);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      gradCscValues->dataType(), 'c', {rows, cols});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(dense_to_csc_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})     // cscRowIdx
      ->setAllowedInputTypes(1, {ALL_INTS})     // cscColPtr
      ->setAllowedInputTypes(2, {ALL_FLOATS})   // gradCscValues
      ->setAllowedOutputTypes(0, {ALL_FLOATS}); // dDense
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_dense_to_csc_bp)
