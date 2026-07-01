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
// csr_to_bsr_bp — backward pass for csr_to_bsr.
//
// Forward: CSR entry k at (row r, col c) is placed into BSR block
//   (br = r/bd, bc = c/bd) at local position (lr = r%bd, lc = c%bd):
//   bsrValues[block_k*bd*bd + lr*bd + lc] = csrValues[k]
//
// Backward: pure gather — each CSR entry maps to exactly one BSR slot:
//   dCsrValues[k] = gradBsrValues[block_k*bd*bd + lr*bd + lc]
// where block_k is determined by binary search in bsrRowPtr/bsrColIdx.
//
// Inputs:
//   [0] csrColIdx      [nnz]          INT   — CSR column indices
//   [1] csrRowPtr      [rows+1]       INT   — CSR row pointers
//   [2] bsrColIdx      [nnzb]         INT   — BSR block column indices (forward output[1])
//   [3] bsrRowPtr      [mb+1]         INT   — BSR block row pointers (forward output[2])
//   [4] gradBsrValues  [nnzb*bd*bd]   float — upstream gradient w.r.t. BSR values
// IArgs:
//   [0] rows
//   [1] cols
//   [2] blockDim
// Output:
//   [0] dCsrValues [nnz] float — gradient w.r.t. CSR values
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_to_bsr_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_bsr_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_to_bsr_bp, 5, 1, false, 0, 3) {
  auto csrColIdx     = INPUT_VARIABLE(0);
  auto csrRowPtr     = INPUT_VARIABLE(1);
  auto bsrColIdx     = INPUT_VARIABLE(2);
  auto bsrRowPtr     = INPUT_VARIABLE(3);
  auto gradBsrValues = INPUT_VARIABLE(4);

  const LongType rows     = INT_ARG(0);
  const LongType cols     = INT_ARG(1);
  const LongType blockDim = INT_ARG(2);

  REQUIRE_TRUE(csrColIdx->rankOf() == 1, 0,
               "csr_to_bsr_bp: csrColIdx must be 1D, got rank %d", csrColIdx->rankOf());
  REQUIRE_TRUE(csrRowPtr->rankOf() == 1, 0,
               "csr_to_bsr_bp: csrRowPtr must be 1D, got rank %d", csrRowPtr->rankOf());
  REQUIRE_TRUE(bsrColIdx->rankOf() == 1, 0,
               "csr_to_bsr_bp: bsrColIdx must be 1D, got rank %d", bsrColIdx->rankOf());
  REQUIRE_TRUE(bsrRowPtr->rankOf() == 1, 0,
               "csr_to_bsr_bp: bsrRowPtr must be 1D, got rank %d", bsrRowPtr->rankOf());
  REQUIRE_TRUE(gradBsrValues->rankOf() == 1, 0,
               "csr_to_bsr_bp: gradBsrValues must be 1D, got rank %d", gradBsrValues->rankOf());
  REQUIRE_TRUE(csrRowPtr->lengthOf() == rows + 1, 0,
               "csr_to_bsr_bp: csrRowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)csrRowPtr->lengthOf());

  auto dCsrValues = OUTPUT_VARIABLE(0);

  sd::ops::helpers::csr_to_bsr_bp(
      *csrColIdx, *csrRowPtr,
      *bsrColIdx, *bsrRowPtr,
      *gradBsrValues, *dCsrValues,
      rows, cols, blockDim);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_to_bsr_bp) {
  // dCsrValues has the same shape as csrColIdx (input[0])
  // dtype from gradBsrValues (input[4])
  auto csrColIdx     = INPUT_VARIABLE(0);
  auto gradBsrValues = INPUT_VARIABLE(4);

  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      gradBsrValues->dataType(), 'c', {csrColIdx->lengthOf()});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_to_bsr_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})     // csrColIdx
      ->setAllowedInputTypes(1, {ALL_INTS})     // csrRowPtr
      ->setAllowedInputTypes(2, {ALL_INTS})     // bsrColIdx
      ->setAllowedInputTypes(3, {ALL_INTS})     // bsrRowPtr
      ->setAllowedInputTypes(4, {ALL_FLOATS})   // gradBsrValues
      ->setAllowedOutputTypes(0, {ALL_FLOATS}); // dCsrValues
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_to_bsr_bp)
