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
// bsr_to_dense_bp — backward pass for bsr_to_dense.
//
// Forward: BSR block at (br, bc = bsrColIdx[k]) is scattered into
//   dense[br*bd + lr, bc*bd + lc] = bsrValues[k*bd*bd + lr*bd + lc]
//
// Backward: gather from same region of the gradient:
//   dBsrValues[k*bd*bd + lr*bd + lc] = gradDense[br*bd + lr, bc*bd + lc]
//
// Inputs:
//   [0] bsrColIdx [nnzb]        INT   — BSR block column indices
//   [1] bsrRowPtr [mb+1]        INT   — BSR block row pointers (mb = rows/blockDim)
//   [2] gradDense [rows, cols]  float — upstream gradient w.r.t. dense output
// IArgs:
//   [0] rows
//   [1] cols
//   [2] blockDim
// Output:
//   [0] dBsrValues [nnzb*blockDim*blockDim] float — gradient w.r.t. BSR values
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_bsr_to_dense_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_bsr_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(bsr_to_dense_bp, 3, 1, false, 0, 3) {
  auto bsrColIdx = INPUT_VARIABLE(0);
  auto bsrRowPtr = INPUT_VARIABLE(1);
  auto gradDense = INPUT_VARIABLE(2);

  const LongType rows     = INT_ARG(0);
  const LongType cols     = INT_ARG(1);
  const LongType blockDim = INT_ARG(2);

  REQUIRE_TRUE(bsrColIdx->rankOf() == 1, 0,
               "bsr_to_dense_bp: bsrColIdx must be 1D, got rank %d", bsrColIdx->rankOf());
  REQUIRE_TRUE(bsrRowPtr->rankOf() == 1, 0,
               "bsr_to_dense_bp: bsrRowPtr must be 1D, got rank %d", bsrRowPtr->rankOf());
  REQUIRE_TRUE(gradDense->rankOf() == 2, 0,
               "bsr_to_dense_bp: gradDense must be 2D, got rank %d", gradDense->rankOf());
  REQUIRE_TRUE(gradDense->sizeAt(0) == rows && gradDense->sizeAt(1) == cols, 0,
               "bsr_to_dense_bp: gradDense shape must be [rows=%lld, cols=%lld], got [%lld, %lld]",
               (long long)rows, (long long)cols,
               (long long)gradDense->sizeAt(0), (long long)gradDense->sizeAt(1));

  const LongType nnzb = bsrColIdx->lengthOf();
  auto dBsrValues = OUTPUT_VARIABLE(0);

  if (nnzb > 0) {
    sd::ops::helpers::bsr_to_dense_bp(
        *bsrColIdx, *bsrRowPtr,
        *gradDense, *dBsrValues,
        rows, cols, blockDim);
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(bsr_to_dense_bp) {
  auto bsrColIdx = INPUT_VARIABLE(0);
  auto gradDense = INPUT_VARIABLE(2);

  const LongType nnzb     = bsrColIdx->lengthOf();
  const LongType blockDim = INT_ARG(2);
  const LongType len      = nnzb * blockDim * blockDim;
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      gradDense->dataType(), 'c', {len});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(bsr_to_dense_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})     // bsrColIdx
      ->setAllowedInputTypes(1, {ALL_INTS})     // bsrRowPtr
      ->setAllowedInputTypes(2, {ALL_FLOATS})   // gradDense
      ->setAllowedOutputTypes(0, {ALL_FLOATS}); // dBsrValues
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_bsr_to_dense_bp)
