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
// Backward pass for csr_spmm: C = op(A) * B
//
// Inputs:
//   [0] values   – 1D [nnz], float   (CSR non-zero values of A)
//   [1] colIdx   – 1D [nnz], INT     (column indices)
//   [2] rowPtr   – 1D [rows+1], INT  (row pointers)
//   [3] B        – 2D dense matrix   (forward input)
//   [4] gradC    – 2D upstream gradient (same shape as C)
// IArgs:
//   [0] rows
//   [1] cols
//   [2] transposeA  (0 = forward was A*B, 1 = forward was A^T*B)
// Outputs:
//   [0] dAValues – 1D [nnz]: gradient w.r.t. values (pre-zeroed)
//   [1] dB       – 2D:       gradient w.r.t. B      (pre-zeroed)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_spmm_bp)

#include <ops/declarable/headers/common.h>
#include <ops/declarable/helpers/sparse_blas_bp.h>

namespace sd {
namespace ops {

DECLARE_CUSTOM_OP(csr_spmm_bp, 5, 2, false, 0, 3);

CUSTOM_OP_IMPL(csr_spmm_bp, 5, 2, false, 0, 3) {
  auto values  = INPUT_VARIABLE(0);
  auto colIdx  = INPUT_VARIABLE(1);
  auto rowPtr  = INPUT_VARIABLE(2);
  auto B       = INPUT_VARIABLE(3);
  auto gradC   = INPUT_VARIABLE(4);

  const LongType rows       = INT_ARG(0);
  const LongType cols       = INT_ARG(1);
  const int      transposeA = static_cast<int>(INT_ARG(2));

  REQUIRE_TRUE(values->rankOf() == 1, 0, "csr_spmm_bp: values must be 1D");
  REQUIRE_TRUE(colIdx->rankOf() == 1, 0, "csr_spmm_bp: colIdx must be 1D");
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0, "csr_spmm_bp: rowPtr must be 1D");
  REQUIRE_TRUE(B->rankOf() == 2,      0, "csr_spmm_bp: B must be 2D");
  REQUIRE_TRUE(gradC->rankOf() == 2,  0, "csr_spmm_bp: gradC must be 2D");
  REQUIRE_TRUE(values->lengthOf() == colIdx->lengthOf(), 0,
               "csr_spmm_bp: values and colIdx must have the same length");
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_spmm_bp: rowPtr length must equal rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());

  auto dAValues = OUTPUT_NULLIFIED(0);
  auto dB       = OUTPUT_NULLIFIED(1);

  sd::ops::helpers::csr_spmm_bp(*values, *colIdx, *rowPtr, *B, *gradC,
                                  *dAValues, *dB, rows, cols, transposeA);
  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_spmm_bp) {
  // dAValues has the same shape as values (input[0])
  // dB has the same shape as B (input[3])
  return SHAPELIST(inputShape->at(0), inputShape->at(3));
}

DECLARE_TYPES(csr_spmm_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(sd::DataType::ANY)
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_spmm_bp)
