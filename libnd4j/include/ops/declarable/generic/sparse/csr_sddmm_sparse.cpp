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
// csr_sddmm_sparse — sampled sparse×sparseᵀ dot product (gradient kernel for
// differentiable SpGEMM).
//
// Given a target sparsity pattern over [P, Q] (in CSR) and two CSR matrices
// L[P, R] and M[Q, R], for each target nonzero at position t (row p, column
// q = targetColIdx[t]) computes:
//
//   outValues[t] = dot(L row p, M row q)
//                = sum_{c in BOTH L-row-p AND M-row-q} L[p,c] * M[q,c]
//
// i.e. out = (L · Mᵀ) sampled at the target CSR sparsity pattern.
//
// Inputs:
//   [0] targetRowPtr  [P+1]   int    — target sparsity row pointers
//   [1] targetColIdx  [tnnz]  int    — target sparsity column indices
//   [2] Lvalues       [Lnnz]  float  — non-zero values of L
//   [3] LcolIdx       [Lnnz]  int    — column indices of L
//   [4] LrowPtr       [P+1]   int    — row pointers of L
//   [5] Mvalues       [Mnnz]  float  — non-zero values of M
//   [6] McolIdx       [Mnnz]  int    — column indices of M
//   [7] MrowPtr       [Q+1]   int    — row pointers of M
// IArgs:
//   [0] P   — rows in L and in the target
//   [1] Q   — rows in M and cols in the target
//   [2] R   — inner dimension (cols in both L and M)
// Output:
//   [0] outValues [tnnz]  float  — same dtype as Lvalues
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_sddmm_sparse)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_sddmm_sparse.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_sddmm_sparse, 8, 1, false, 0, 3) {
  auto targetRowPtr = INPUT_VARIABLE(0);
  auto targetColIdx = INPUT_VARIABLE(1);
  auto Lvalues      = INPUT_VARIABLE(2);
  auto LcolIdx      = INPUT_VARIABLE(3);
  auto LrowPtr      = INPUT_VARIABLE(4);
  auto Mvalues      = INPUT_VARIABLE(5);
  auto McolIdx      = INPUT_VARIABLE(6);
  auto MrowPtr      = INPUT_VARIABLE(7);

  const LongType P = INT_ARG(0);
  const LongType Q = INT_ARG(1);
  const LongType R = INT_ARG(2);

  REQUIRE_TRUE(targetRowPtr->rankOf() == 1, 0,
               "csr_sddmm_sparse: targetRowPtr must be 1D, got rank %d",
               targetRowPtr->rankOf());
  REQUIRE_TRUE(targetColIdx->rankOf() == 1, 0,
               "csr_sddmm_sparse: targetColIdx must be 1D, got rank %d",
               targetColIdx->rankOf());
  REQUIRE_TRUE(Lvalues->rankOf() == 1, 0,
               "csr_sddmm_sparse: Lvalues must be 1D, got rank %d",
               Lvalues->rankOf());
  REQUIRE_TRUE(LcolIdx->rankOf() == 1, 0,
               "csr_sddmm_sparse: LcolIdx must be 1D, got rank %d",
               LcolIdx->rankOf());
  REQUIRE_TRUE(LrowPtr->rankOf() == 1, 0,
               "csr_sddmm_sparse: LrowPtr must be 1D, got rank %d",
               LrowPtr->rankOf());
  REQUIRE_TRUE(Mvalues->rankOf() == 1, 0,
               "csr_sddmm_sparse: Mvalues must be 1D, got rank %d",
               Mvalues->rankOf());
  REQUIRE_TRUE(McolIdx->rankOf() == 1, 0,
               "csr_sddmm_sparse: McolIdx must be 1D, got rank %d",
               McolIdx->rankOf());
  REQUIRE_TRUE(MrowPtr->rankOf() == 1, 0,
               "csr_sddmm_sparse: MrowPtr must be 1D, got rank %d",
               MrowPtr->rankOf());

  REQUIRE_TRUE(targetRowPtr->lengthOf() == P + 1, 0,
               "csr_sddmm_sparse: targetRowPtr length must be P+1=%lld, got %lld",
               (long long)(P + 1), (long long)targetRowPtr->lengthOf());
  REQUIRE_TRUE(LrowPtr->lengthOf() == P + 1, 0,
               "csr_sddmm_sparse: LrowPtr length must be P+1=%lld, got %lld",
               (long long)(P + 1), (long long)LrowPtr->lengthOf());
  REQUIRE_TRUE(MrowPtr->lengthOf() == Q + 1, 0,
               "csr_sddmm_sparse: MrowPtr length must be Q+1=%lld, got %lld",
               (long long)(Q + 1), (long long)MrowPtr->lengthOf());

  REQUIRE_TRUE(Lvalues->lengthOf() == LcolIdx->lengthOf(), 0,
               "csr_sddmm_sparse: Lvalues and LcolIdx must have the same length");
  REQUIRE_TRUE(Mvalues->lengthOf() == McolIdx->lengthOf(), 0,
               "csr_sddmm_sparse: Mvalues and McolIdx must have the same length");

  REQUIRE_TRUE(Lvalues->dataType() == Mvalues->dataType(), 0,
               "csr_sddmm_sparse: Lvalues and Mvalues must have the same float dtype");
  REQUIRE_TRUE(targetRowPtr->dataType() == targetColIdx->dataType(), 0,
               "csr_sddmm_sparse: targetRowPtr and targetColIdx must have the same integer dtype");
  REQUIRE_TRUE(LcolIdx->dataType() == LrowPtr->dataType(), 0,
               "csr_sddmm_sparse: LcolIdx and LrowPtr must have the same integer dtype");
  REQUIRE_TRUE(McolIdx->dataType() == MrowPtr->dataType(), 0,
               "csr_sddmm_sparse: McolIdx and MrowPtr must have the same integer dtype");
  REQUIRE_TRUE(targetColIdx->dataType() == LcolIdx->dataType(), 0,
               "csr_sddmm_sparse: all index arrays must have the same integer dtype");

  auto outValues = OUTPUT_NULLIFIED(0);

  sd::ops::helpers::csr_sddmm_sparse(*targetRowPtr, *targetColIdx,
                                      *Lvalues, *LcolIdx, *LrowPtr,
                                      *Mvalues, *McolIdx, *MrowPtr,
                                      *outValues, P, Q, R);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_sddmm_sparse) {
  // Output shape is [tnnz] = targetColIdx.lengthOf() — trivially known from
  // input shapes without any data-dependent computation.
  auto targetColIdx = INPUT_VARIABLE(1);
  auto Lvalues      = INPUT_VARIABLE(2);

  const LongType tnnz = targetColIdx->lengthOf();
  const auto dtype    = Lvalues->dataType();

  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {tnnz});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_sddmm_sparse) {

  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})    // targetRowPtr
      ->setAllowedInputTypes(1, {ALL_INTS})    // targetColIdx
      ->setAllowedInputTypes(2, {ALL_FLOATS})  // Lvalues
      ->setAllowedInputTypes(3, {ALL_INTS})    // LcolIdx
      ->setAllowedInputTypes(4, {ALL_INTS})    // LrowPtr
      ->setAllowedInputTypes(5, {ALL_FLOATS})  // Mvalues
      ->setAllowedInputTypes(6, {ALL_INTS})    // McolIdx
      ->setAllowedInputTypes(7, {ALL_INTS})    // MrowPtr
      ->setAllowedOutputTypes({ALL_FLOATS});

  getOpDescriptor()->addTraits(OP_TRAIT_DATA_DEPENDENT | OP_TRAIT_DYNAMIC_OUTPUT_SIZE);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_sddmm_sparse)
