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
// csr_sddmm_sparse_bp — backward pass for csr_sddmm_sparse.
//
// Forward: for each target nonzero t at (row p_t, col q_t = targetColIdx[t]):
//   outValues[t] = dot(L row p_t, M row q_t)
//                = sum_{c in L-row-p AND M-row-q} Lvalues[kL] * Mvalues[kM]
//
// Backward: given gradOut [tnnz]:
//   dLvalues[k@(p,c)] += sum_{t: row=p, c in M-row-q_t} gradOut[t] * Mvalues[k'@(q_t,c)]
//   dMvalues[k@(q,c)] += sum_{t: col=q, c in L-row-p_t} gradOut[t] * Lvalues[k'@(p_t,c)]
// Uses sd_atomicAdd because multiple targets can contribute to each L/M entry.
//
// Inputs:
//   [0] targetRowPtr [P+1]   INT   — target sparsity row pointers
//   [1] targetColIdx [tnnz]  INT   — target sparsity column indices
//   [2] LcolIdx      [Lnnz]  INT   — column indices of L [P, R]
//   [3] LrowPtr      [P+1]   INT   — row pointers of L
//   [4] McolIdx      [Mnnz]  INT   — column indices of M [Q, R]
//   [5] MrowPtr      [Q+1]   INT   — row pointers of M
//   [6] Lvalues      [Lnnz]  float — non-zero values of L
//   [7] Mvalues      [Mnnz]  float — non-zero values of M
//   [8] gradOut      [tnnz]  float — upstream gradient w.r.t. outValues
// IArgs:
//   [0] P — rows in L and in target
//   [1] Q — rows in M (cols in target)
//   [2] R — inner dimension (cols in both L and M)
// Outputs:
//   [0] dLvalues [Lnnz] float — gradient w.r.t. Lvalues (pre-zeroed)
//   [1] dMvalues [Mnnz] float — gradient w.r.t. Mvalues (pre-zeroed)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_sddmm_sparse_bp)

#include <ops/declarable/headers/common.h>
#include <ops/declarable/helpers/sparse_sddmm_sparse_bp.h>

namespace sd {
namespace ops {

DECLARE_CUSTOM_OP(csr_sddmm_sparse_bp, 9, 2, false, 0, 3);

CUSTOM_OP_IMPL(csr_sddmm_sparse_bp, 9, 2, false, 0, 3) {
  auto targetRowPtr = INPUT_VARIABLE(0);
  auto targetColIdx = INPUT_VARIABLE(1);
  auto LcolIdx      = INPUT_VARIABLE(2);
  auto LrowPtr      = INPUT_VARIABLE(3);
  auto McolIdx      = INPUT_VARIABLE(4);
  auto MrowPtr      = INPUT_VARIABLE(5);
  auto Lvalues      = INPUT_VARIABLE(6);
  auto Mvalues      = INPUT_VARIABLE(7);
  auto gradOut      = INPUT_VARIABLE(8);

  const LongType P = INT_ARG(0);
  const LongType Q = INT_ARG(1);
  const LongType R = INT_ARG(2);

  REQUIRE_TRUE(targetRowPtr->rankOf() == 1, 0,
               "csr_sddmm_sparse_bp: targetRowPtr must be 1D");
  REQUIRE_TRUE(targetColIdx->rankOf() == 1, 0,
               "csr_sddmm_sparse_bp: targetColIdx must be 1D");
  REQUIRE_TRUE(LcolIdx->rankOf() == 1, 0,
               "csr_sddmm_sparse_bp: LcolIdx must be 1D");
  REQUIRE_TRUE(LrowPtr->rankOf() == 1, 0,
               "csr_sddmm_sparse_bp: LrowPtr must be 1D");
  REQUIRE_TRUE(McolIdx->rankOf() == 1, 0,
               "csr_sddmm_sparse_bp: McolIdx must be 1D");
  REQUIRE_TRUE(MrowPtr->rankOf() == 1, 0,
               "csr_sddmm_sparse_bp: MrowPtr must be 1D");
  REQUIRE_TRUE(Lvalues->rankOf() == 1, 0,
               "csr_sddmm_sparse_bp: Lvalues must be 1D");
  REQUIRE_TRUE(Mvalues->rankOf() == 1, 0,
               "csr_sddmm_sparse_bp: Mvalues must be 1D");
  REQUIRE_TRUE(gradOut->rankOf() == 1, 0,
               "csr_sddmm_sparse_bp: gradOut must be 1D");
  REQUIRE_TRUE(targetRowPtr->lengthOf() == P + 1, 0,
               "csr_sddmm_sparse_bp: targetRowPtr length must be P+1=%lld, got %lld",
               (long long)(P + 1), (long long)targetRowPtr->lengthOf());
  REQUIRE_TRUE(LrowPtr->lengthOf() == P + 1, 0,
               "csr_sddmm_sparse_bp: LrowPtr length must be P+1=%lld, got %lld",
               (long long)(P + 1), (long long)LrowPtr->lengthOf());
  REQUIRE_TRUE(MrowPtr->lengthOf() == Q + 1, 0,
               "csr_sddmm_sparse_bp: MrowPtr length must be Q+1=%lld, got %lld",
               (long long)(Q + 1), (long long)MrowPtr->lengthOf());
  REQUIRE_TRUE(Lvalues->lengthOf() == LcolIdx->lengthOf(), 0,
               "csr_sddmm_sparse_bp: Lvalues and LcolIdx must have the same length");
  REQUIRE_TRUE(Mvalues->lengthOf() == McolIdx->lengthOf(), 0,
               "csr_sddmm_sparse_bp: Mvalues and McolIdx must have the same length");
  REQUIRE_TRUE(gradOut->lengthOf() == targetColIdx->lengthOf(), 0,
               "csr_sddmm_sparse_bp: gradOut and targetColIdx must have the same length");

  auto dLvalues = OUTPUT_NULLIFIED(0);
  auto dMvalues = OUTPUT_NULLIFIED(1);

  sd::ops::helpers::csr_sddmm_sparse_bp(
      *targetRowPtr, *targetColIdx,
      *LcolIdx, *LrowPtr,
      *McolIdx, *MrowPtr,
      *Lvalues, *Mvalues,
      *gradOut,
      *dLvalues, *dMvalues,
      P, Q, R);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_sddmm_sparse_bp) {
  auto LcolIdx = INPUT_VARIABLE(2);
  auto McolIdx = INPUT_VARIABLE(4);
  auto Lvalues = INPUT_VARIABLE(6);
  auto Mvalues = INPUT_VARIABLE(7);

  const LongType Lnnz = LcolIdx->lengthOf();
  const LongType Mnnz = McolIdx->lengthOf();

  auto shapeL = ConstantShapeHelper::getInstance().createShapeInfo(
      Lvalues->dataType(), 'c', {Lnnz});
  auto shapeM = ConstantShapeHelper::getInstance().createShapeInfo(
      Mvalues->dataType(), 'c', {Mnnz});
  return SHAPELIST(shapeL, shapeM);
}

DECLARE_TYPES(csr_sddmm_sparse_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})     // targetRowPtr
      ->setAllowedInputTypes(1, {ALL_INTS})     // targetColIdx
      ->setAllowedInputTypes(2, {ALL_INTS})     // LcolIdx
      ->setAllowedInputTypes(3, {ALL_INTS})     // LrowPtr
      ->setAllowedInputTypes(4, {ALL_INTS})     // McolIdx
      ->setAllowedInputTypes(5, {ALL_INTS})     // MrowPtr
      ->setAllowedInputTypes(6, {ALL_FLOATS})   // Lvalues
      ->setAllowedInputTypes(7, {ALL_FLOATS})   // Mvalues
      ->setAllowedInputTypes(8, {ALL_FLOATS})   // gradOut
      ->setAllowedOutputTypes(0, {ALL_FLOATS})  // dLvalues
      ->setAllowedOutputTypes(1, {ALL_FLOATS}); // dMvalues
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_sddmm_sparse_bp)
