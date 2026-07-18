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

// bsr_spmm: C = A_bsr [rows,cols] * B dense [cols,n] → C [rows,n]
// in[0]=bsrValues[nnzb*bd*bd], in[1]=bsrColIdx[nnzb], in[2]=bsrRowPtr[mb+1], in[3]=B[cols,n]
// IArgs: [0]=rows, [1]=cols, [2]=blockDim
// out[0]=C[rows,n]

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_bsr_spmm)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_bsr.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(bsr_spmm, 4, 1, false, 0, 3) {
  auto bsrValues = INPUT_VARIABLE(0);
  auto bsrColIdx = INPUT_VARIABLE(1);
  auto bsrRowPtr = INPUT_VARIABLE(2);
  auto B         = INPUT_VARIABLE(3);

  const LongType rows     = INT_ARG(0);
  const LongType cols     = INT_ARG(1);
  const LongType blockDim = INT_ARG(2);

  REQUIRE_TRUE(bsrValues->rankOf() == 1, 0, "bsr_spmm: bsrValues must be 1D");
  REQUIRE_TRUE(bsrColIdx->rankOf() == 1, 0, "bsr_spmm: bsrColIdx must be 1D");
  REQUIRE_TRUE(bsrRowPtr->rankOf() == 1, 0, "bsr_spmm: bsrRowPtr must be 1D");
  REQUIRE_TRUE(B->rankOf() == 2, 0, "bsr_spmm: B must be 2D, got rank %d", (int)B->rankOf());
  REQUIRE_TRUE(B->sizeAt(0) == cols, 0,
               "bsr_spmm: B must have %lld rows (= cols of A), got %lld",
               (long long)cols, (long long)B->sizeAt(0));

  auto C = OUTPUT_NULLIFIED(0);

  sd::ops::helpers::bsr_spmm(*bsrValues, *bsrColIdx, *bsrRowPtr, *B, *C, rows, cols, blockDim);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(bsr_spmm) {
  auto bsrValues = INPUT_VARIABLE(0);
  auto B         = INPUT_VARIABLE(3);

  const LongType rows = INT_ARG(0);
  const LongType n    = B->sizeAt(1);

  auto dtype    = bsrValues->dataType();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {rows, n});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(bsr_spmm) {

  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setAllowedInputTypes(2, {ALL_INTS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});

  getOpDescriptor()->addTraits(OP_TRAIT_DATA_DEPENDENT | OP_TRAIT_DYNAMIC_OUTPUT_SIZE);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_bsr_spmm)
