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
// CSC sparse → dense conversion op.
//
// Inputs:
//   [0] cscValues  [nnz],     float dtype
//   [1] cscRowIdx  [nnz],     INT32 or INT64
//   [2] cscColPtr  [cols+1],  same INT dtype as cscRowIdx
// IArgs:
//   [0] rows
//   [1] cols
// Output:
//   [0] dense [rows, cols], same float dtype as cscValues
//
// The dense output is OUTPUT_NULLIFIED (pre-zeroed); the helper scatters
// only the non-zero entries.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csc_to_dense)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_csc.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csc_to_dense, 3, 1, false, 0, 2) {
  auto cscValues = INPUT_VARIABLE(0);
  auto cscRowIdx = INPUT_VARIABLE(1);
  auto cscColPtr = INPUT_VARIABLE(2);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(cscValues->rankOf() == 1, 0,
               "csc_to_dense: cscValues must be 1D, got rank %d", cscValues->rankOf());
  REQUIRE_TRUE(cscRowIdx->rankOf() == 1, 0,
               "csc_to_dense: cscRowIdx must be 1D, got rank %d", cscRowIdx->rankOf());
  REQUIRE_TRUE(cscColPtr->rankOf() == 1, 0,
               "csc_to_dense: cscColPtr must be 1D, got rank %d", cscColPtr->rankOf());
  REQUIRE_TRUE(cscColPtr->lengthOf() == cols + 1, 0,
               "csc_to_dense: cscColPtr length must be cols+1=%lld, got %lld",
               (long long)(cols + 1), (long long)cscColPtr->lengthOf());
  REQUIRE_TRUE(cscValues->lengthOf() == cscRowIdx->lengthOf(), 0,
               "csc_to_dense: cscValues and cscRowIdx must have the same length");
  REQUIRE_TRUE(cscRowIdx->dataType() == cscColPtr->dataType(), 0,
               "csc_to_dense: cscRowIdx and cscColPtr must have the same integer dtype");

  // OUTPUT_NULLIFIED guarantees the output is pre-zeroed; the helper only
  // writes the non-zero positions.
  auto output = OUTPUT_NULLIFIED(0);

  if (cscValues->lengthOf() > 0) {
    sd::ops::helpers::csc_to_dense(*cscValues, *cscRowIdx, *cscColPtr, *output);
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csc_to_dense) {
  auto cscValues = INPUT_VARIABLE(0);
  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  auto dtype    = cscValues->dataType();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {rows, cols});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csc_to_dense) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})   // cscValues
      ->setAllowedInputTypes(1, {ALL_INTS})     // cscRowIdx
      ->setAllowedInputTypes(2, {ALL_INTS})     // cscColPtr
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csc_to_dense)
