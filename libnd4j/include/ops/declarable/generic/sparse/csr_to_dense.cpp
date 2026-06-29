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
// CSR sparse → dense conversion op
// Inputs:  [0] values (1D [nnz], float dtype)
//          [1] colIdx (1D [nnz], INT32/INT64)
//          [2] rowPtr (1D [rows+1], same INT dtype)
// IArgs:   [0] rows, [1] cols
// Output:  [0] dense [rows, cols]
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_to_dense)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_csr.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_to_dense, 3, 1, false, 0, 2) {
  auto values = INPUT_VARIABLE(0);
  auto colIdx = INPUT_VARIABLE(1);
  auto rowPtr = INPUT_VARIABLE(2);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(values->rankOf() == 1, 0,
               "csr_to_dense: values must be 1D, got rank %d", values->rankOf());
  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_to_dense: colIdx must be 1D, got rank %d", colIdx->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_to_dense: rowPtr must be 1D, got rank %d", rowPtr->rankOf());
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_to_dense: rowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(values->lengthOf() == colIdx->lengthOf(), 0,
               "csr_to_dense: values and colIdx must have the same length");
  REQUIRE_TRUE(colIdx->dataType() == rowPtr->dataType(), 0,
               "csr_to_dense: colIdx and rowPtr must have the same integer dtype");

  auto output = OUTPUT_NULLIFIED(0);

  if (values->lengthOf() > 0) {
    sd::ops::helpers::csr_to_dense(*values, *colIdx, *rowPtr, *output);
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_to_dense) {
  auto values = INPUT_VARIABLE(0);
  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  auto dtype = values->dataType();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {rows, cols});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_to_dense) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})   // values
      ->setAllowedInputTypes(1, {ALL_INTS})         // colIdx
      ->setAllowedInputTypes(2, {ALL_INTS})         // rowPtr
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_to_dense)
