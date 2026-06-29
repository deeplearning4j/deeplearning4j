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

// bsr_to_dense: BSR → dense [rows, cols]
// in[0]=bsrValues[nnzb*bd*bd], in[1]=bsrColIdx[nnzb], in[2]=bsrRowPtr[mb+1]
// IArgs: [0]=rows, [1]=cols, [2]=blockDim
// out[0]=dense[rows,cols] OUTPUT_NULLIFIED

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_bsr_to_dense)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_bsr.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(bsr_to_dense, 3, 1, false, 0, 3) {
  auto bsrValues = INPUT_VARIABLE(0);
  auto bsrColIdx = INPUT_VARIABLE(1);
  auto bsrRowPtr = INPUT_VARIABLE(2);

  const LongType rows     = INT_ARG(0);
  const LongType cols     = INT_ARG(1);
  const LongType blockDim = INT_ARG(2);

  REQUIRE_TRUE(bsrValues->rankOf() == 1, 0, "bsr_to_dense: bsrValues must be 1D");
  REQUIRE_TRUE(bsrColIdx->rankOf() == 1, 0, "bsr_to_dense: bsrColIdx must be 1D");
  REQUIRE_TRUE(bsrRowPtr->rankOf() == 1, 0, "bsr_to_dense: bsrRowPtr must be 1D");

  auto output = OUTPUT_NULLIFIED(0);

  if (bsrValues->lengthOf() == 0) {
    return sd::Status::OK;
  }

  sd::ops::helpers::bsr_to_dense(*bsrValues, *bsrColIdx, *bsrRowPtr, *output, rows, cols,
                                  blockDim);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(bsr_to_dense) {
  auto bsrValues  = INPUT_VARIABLE(0);
  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  auto dtype    = bsrValues->dataType();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {rows, cols});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(bsr_to_dense) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setAllowedInputTypes(2, {ALL_INTS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_bsr_to_dense)
