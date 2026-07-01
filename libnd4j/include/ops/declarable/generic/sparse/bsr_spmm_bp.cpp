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
// bsr_spmm_bp — backward pass for bsr_spmm: C = A_bsr * B
//
// Given upstream gradient gradC [rows, n], computes:
//   dBsrValues[k, lr, lc] = gradC[br*bd+lr, :] · B[bc*bd+lc, :]
//   dB = A_bsr^T * gradC   (sparse transpose multiply)
//
// Inputs:
//   [0] bsrValues [nnzb*bd*bd] float — BSR non-zero block values of A
//   [1] bsrColIdx [nnzb]       INT   — BSR block column indices
//   [2] bsrRowPtr [mb+1]       INT   — BSR block row pointers
//   [3] B         [cols, n]    float — dense right factor (forward input)
//   [4] gradC     [rows, n]    float — upstream gradient w.r.t. C
// IArgs:
//   [0] rows
//   [1] cols
//   [2] blockDim
// Outputs:
//   [0] dBsrValues [nnzb*bd*bd] float — gradient w.r.t. BSR values (pre-zeroed)
//   [1] dB         [cols, n]    float — gradient w.r.t. B (pre-zeroed)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_bsr_spmm_bp)

#include <ops/declarable/headers/common.h>
#include <ops/declarable/helpers/sparse_bsr_bp.h>

namespace sd {
namespace ops {

DECLARE_CUSTOM_OP(bsr_spmm_bp, 5, 2, false, 0, 3);

CUSTOM_OP_IMPL(bsr_spmm_bp, 5, 2, false, 0, 3) {
  auto bsrValues = INPUT_VARIABLE(0);
  auto bsrColIdx = INPUT_VARIABLE(1);
  auto bsrRowPtr = INPUT_VARIABLE(2);
  auto B         = INPUT_VARIABLE(3);
  auto gradC     = INPUT_VARIABLE(4);

  const LongType rows     = INT_ARG(0);
  const LongType cols     = INT_ARG(1);
  const LongType blockDim = INT_ARG(2);

  REQUIRE_TRUE(bsrValues->rankOf() == 1, 0, "bsr_spmm_bp: bsrValues must be 1D");
  REQUIRE_TRUE(bsrColIdx->rankOf() == 1, 0, "bsr_spmm_bp: bsrColIdx must be 1D");
  REQUIRE_TRUE(bsrRowPtr->rankOf() == 1, 0, "bsr_spmm_bp: bsrRowPtr must be 1D");
  REQUIRE_TRUE(B->rankOf() == 2,          0, "bsr_spmm_bp: B must be 2D");
  REQUIRE_TRUE(gradC->rankOf() == 2,      0, "bsr_spmm_bp: gradC must be 2D");

  auto dBsrValues = OUTPUT_NULLIFIED(0);
  auto dB         = OUTPUT_NULLIFIED(1);

  sd::ops::helpers::bsr_spmm_bp(
      *bsrValues, *bsrColIdx, *bsrRowPtr,
      *B, *gradC,
      *dBsrValues, *dB,
      rows, cols, blockDim);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(bsr_spmm_bp) {
  // dBsrValues same shape as bsrValues (input[0])
  // dB same shape as B (input[3])
  return SHAPELIST(inputShape->at(0), inputShape->at(3));
}

DECLARE_TYPES(bsr_spmm_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(sd::DataType::ANY)
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_bsr_spmm_bp)
