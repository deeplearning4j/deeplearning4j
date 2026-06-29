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
// csr_spgemm_bp: backward pass for C = A · B (SpGEMM in CSR format).
//
// Math:
//   dA[i,j] = dot(dC row i, B row j)  sampled at A's sparsity pattern
//           → csr_sddmm_sparse(target=A, L=dC[m,n], M=B[k,n], P=m, Q=k, R=n)
//
//   dB[j,l] = dot(Aᵀ row j, dCᵀ row l) sampled at B's sparsity pattern
//           → Aᵀ  from csr_to_csc(A,  m, k)
//             dCᵀ from csr_to_csc(dC, m, n)
//             csr_sddmm_sparse(target=B, L=Aᵀ[k,m], M=dCᵀ[n,m], P=k, Q=n, R=m)
//
// Inputs (9 arrays):
//   [0] aValues     [annz]  float  — non-zero values of A
//   [1] aColIdx     [annz]  int    — column indices of A
//   [2] aRowPtr     [m+1]   int    — row pointers of A
//   [3] bValues     [bnnz]  float  — non-zero values of B
//   [4] bColIdx     [bnnz]  int    — column indices of B
//   [5] bRowPtr     [k+1]   int    — row pointers of B
//   [6] cColIdx     [cnnz]  int    — column indices of C (forward output)
//   [7] cRowPtr     [m+1]   int    — row pointers of C (forward output)
//   [8] gradCValues [cnnz]  float  — upstream gradient w.r.t. C values
// IArgs:
//   [0] m  — rows of A (= rows of C)
//   [1] k  — cols of A (= rows of B)
//   [2] n  — cols of B (= cols of C)
// Outputs (2 arrays):
//   [0] dAValues    [annz]  float  — gradient w.r.t. aValues
//   [1] dBValues    [bnnz]  float  — gradient w.r.t. bValues
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_spgemm_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_spgemm_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_spgemm_bp, 9, 2, false, 0, 3) {
  auto aValues     = INPUT_VARIABLE(0);
  auto aColIdx     = INPUT_VARIABLE(1);
  auto aRowPtr     = INPUT_VARIABLE(2);
  auto bValues     = INPUT_VARIABLE(3);
  auto bColIdx     = INPUT_VARIABLE(4);
  auto bRowPtr     = INPUT_VARIABLE(5);
  auto cColIdx     = INPUT_VARIABLE(6);
  auto cRowPtr     = INPUT_VARIABLE(7);
  auto gradCValues = INPUT_VARIABLE(8);

  const LongType m = INT_ARG(0);
  const LongType k = INT_ARG(1);
  const LongType n = INT_ARG(2);

  // ── Validation ─────────────────────────────────────────────────────────────
  REQUIRE_TRUE(aValues->rankOf() == 1, 0,
               "csr_spgemm_bp: aValues must be 1D, got rank %d", aValues->rankOf());
  REQUIRE_TRUE(bValues->rankOf() == 1, 0,
               "csr_spgemm_bp: bValues must be 1D, got rank %d", bValues->rankOf());
  REQUIRE_TRUE(gradCValues->rankOf() == 1, 0,
               "csr_spgemm_bp: gradCValues must be 1D, got rank %d", gradCValues->rankOf());

  REQUIRE_TRUE(aValues->lengthOf() == aColIdx->lengthOf(), 0,
               "csr_spgemm_bp: aValues and aColIdx must have the same length");
  REQUIRE_TRUE(bValues->lengthOf() == bColIdx->lengthOf(), 0,
               "csr_spgemm_bp: bValues and bColIdx must have the same length");
  REQUIRE_TRUE(gradCValues->lengthOf() == cColIdx->lengthOf(), 0,
               "csr_spgemm_bp: gradCValues and cColIdx must have the same length");

  REQUIRE_TRUE(aRowPtr->lengthOf() == m + 1, 0,
               "csr_spgemm_bp: aRowPtr length must be m+1=%lld, got %lld",
               (long long)(m + 1), (long long)aRowPtr->lengthOf());
  REQUIRE_TRUE(bRowPtr->lengthOf() == k + 1, 0,
               "csr_spgemm_bp: bRowPtr length must be k+1=%lld, got %lld",
               (long long)(k + 1), (long long)bRowPtr->lengthOf());
  REQUIRE_TRUE(cRowPtr->lengthOf() == m + 1, 0,
               "csr_spgemm_bp: cRowPtr length must be m+1=%lld, got %lld",
               (long long)(m + 1), (long long)cRowPtr->lengthOf());

  REQUIRE_TRUE(aValues->dataType() == bValues->dataType(), 0,
               "csr_spgemm_bp: aValues and bValues must have the same float dtype");
  REQUIRE_TRUE(aValues->dataType() == gradCValues->dataType(), 0,
               "csr_spgemm_bp: aValues and gradCValues must have the same float dtype");
  REQUIRE_TRUE(aColIdx->dataType() == bColIdx->dataType(), 0,
               "csr_spgemm_bp: A and B index arrays must have the same integer dtype");
  REQUIRE_TRUE(aColIdx->dataType() == cColIdx->dataType(), 0,
               "csr_spgemm_bp: C index arrays must have the same integer dtype as A/B");

  // ── Execute ────────────────────────────────────────────────────────────────
  auto dAValues = OUTPUT_NULLIFIED(0);
  auto dBValues = OUTPUT_NULLIFIED(1);

  sd::ops::helpers::csr_spgemm_bp(*aValues,     *aColIdx,     *aRowPtr,
                                   *bValues,     *bColIdx,     *bRowPtr,
                                   *cColIdx,     *cRowPtr,
                                   *gradCValues,
                                   *dAValues,    *dBValues,
                                   m, k, n);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_spgemm_bp) {
  // dAValues shape == aValues shape  (same annz, same float dtype)
  // dBValues shape == bValues shape  (same bnnz, same float dtype)
  // Shapes are known statically from input shapes — no data scan needed.
  auto aValues = INPUT_VARIABLE(0);
  auto bValues = INPUT_VARIABLE(3);

  auto daShape = ConstantShapeHelper::getInstance().createShapeInfo(
      aValues->dataType(), 'c', {aValues->lengthOf()});
  auto dbShape = ConstantShapeHelper::getInstance().createShapeInfo(
      bValues->dataType(), 'c', {bValues->lengthOf()});

  return SHAPELIST(daShape, dbShape);
}

DECLARE_TYPES(csr_spgemm_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})   // aValues
      ->setAllowedInputTypes(1, {ALL_INTS})     // aColIdx
      ->setAllowedInputTypes(2, {ALL_INTS})     // aRowPtr
      ->setAllowedInputTypes(3, {ALL_FLOATS})   // bValues
      ->setAllowedInputTypes(4, {ALL_INTS})     // bColIdx
      ->setAllowedInputTypes(5, {ALL_INTS})     // bRowPtr
      ->setAllowedInputTypes(6, {ALL_INTS})     // cColIdx
      ->setAllowedInputTypes(7, {ALL_INTS})     // cRowPtr
      ->setAllowedInputTypes(8, {ALL_FLOATS})   // gradCValues
      ->setAllowedOutputTypes(0, {ALL_FLOATS})  // dAValues
      ->setAllowedOutputTypes(1, {ALL_FLOATS}); // dBValues
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_spgemm_bp)
