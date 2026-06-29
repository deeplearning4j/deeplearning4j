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
// Backward pass for sddmm (sampled dense-dense matrix multiplication).
//
// Forward: out[k at (i,j)] = sum_l D1[i,l] * D2[j,l]   (D1·D2^T at pattern)
//
// Inputs:
//   [0] rowPtr   – 1D [rows+1], INT  (CSR row pointers)
//   [1] colIdx   – 1D [nnz],   INT   (CSR column indices)
//   [2] D1       – 2D [rows, p]  left dense factor
//   [3] D2       – 2D [cols, p]  right dense factor
//   [4] gradOut  – 1D [nnz]  upstream gradient
// IArgs:
//   [0] rows
//   [1] cols
// Outputs:
//   [0] dD1  – 2D [rows, p]: gradient w.r.t. D1 (pre-zeroed)
//   [1] dD2  – 2D [cols, p]: gradient w.r.t. D2 (pre-zeroed)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_sddmm_bp)

#include <ops/declarable/headers/common.h>
#include <ops/declarable/helpers/sparse_blas_bp.h>

namespace sd {
namespace ops {

DECLARE_CUSTOM_OP(sddmm_bp, 5, 2, false, 0, 2);

CUSTOM_OP_IMPL(sddmm_bp, 5, 2, false, 0, 2) {
  auto rowPtr  = INPUT_VARIABLE(0);
  auto colIdx  = INPUT_VARIABLE(1);
  auto D1      = INPUT_VARIABLE(2);
  auto D2      = INPUT_VARIABLE(3);
  auto gradOut = INPUT_VARIABLE(4);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(rowPtr->rankOf() == 1,  0, "sddmm_bp: rowPtr must be 1D");
  REQUIRE_TRUE(colIdx->rankOf() == 1,  0, "sddmm_bp: colIdx must be 1D");
  REQUIRE_TRUE(D1->rankOf() == 2,      0, "sddmm_bp: D1 must be 2D [rows, p]");
  REQUIRE_TRUE(D2->rankOf() == 2,      0, "sddmm_bp: D2 must be 2D [cols, p]");
  REQUIRE_TRUE(gradOut->rankOf() == 1, 0, "sddmm_bp: gradOut must be 1D [nnz]");
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "sddmm_bp: rowPtr length must equal rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(gradOut->lengthOf() == colIdx->lengthOf(), 0,
               "sddmm_bp: gradOut and colIdx must have the same length (nnz)");
  REQUIRE_TRUE(D1->sizeAt(1) == D2->sizeAt(1), 0,
               "sddmm_bp: D1 and D2 must share the inner dimension p");

  auto dD1 = OUTPUT_NULLIFIED(0);
  auto dD2 = OUTPUT_NULLIFIED(1);

  sd::ops::helpers::sddmm_bp(*rowPtr, *colIdx, *D1, *D2, *gradOut,
                               *dD1, *dD2, rows, cols);
  return sd::Status::OK;
}

DECLARE_SHAPE_FN(sddmm_bp) {
  // dD1 has the same shape as D1 (input[2])
  // dD2 has the same shape as D2 (input[3])
  return SHAPELIST(inputShape->at(2), inputShape->at(3));
}

DECLARE_TYPES(sddmm_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(sd::DataType::ANY)
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_sddmm_bp)
