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
// csr_diag_mm_bp — backward pass for csr_diag_mm.
//
// Forward: out[e] = dl[i_e] * aValues[e] * dr[j_e]
//
// Backward (exact):
//   dAValues[e]  = gradOut[e] * dl[i_e] * dr[j_e]
//   ddl[i]       = sum_{e in row i}    gradOut[e] * aValues[e] * dr[j_e]
//   ddr[j]       = sum_{e: col j_e==j} gradOut[e] * aValues[e] * dl[i_e]
//
// Inputs (6 arrays):
//   [0] aValues   [nnz]     float  — original nonzero values of A
//   [1] aColIdx   [nnz]     int    — column indices of A
//   [2] aRowPtr   [rows+1]  int    — row pointers of A
//   [3] dl        [rows]    float  — left diagonal
//   [4] dr        [cols]    float  — right diagonal
//   [5] gradOut   [nnz]     float  — upstream gradient w.r.t. outValues
// IArgs:
//   [0] rows
//   [1] cols
// Outputs (3 arrays):
//   [0] dAValues  [nnz]     float  — gradient w.r.t. aValues
//   [1] ddl       [rows]    float  — gradient w.r.t. dl
//   [2] ddr       [cols]    float  — gradient w.r.t. dr
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_diag_mm_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_diag_mm_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_diag_mm_bp, 6, 3, false, 0, 2) {
  auto aValues = INPUT_VARIABLE(0);
  auto aColIdx = INPUT_VARIABLE(1);
  auto aRowPtr = INPUT_VARIABLE(2);
  auto dl      = INPUT_VARIABLE(3);
  auto dr      = INPUT_VARIABLE(4);
  auto gradOut = INPUT_VARIABLE(5);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  // ── Validation ─────────────────────────────────────────────────────────────
  REQUIRE_TRUE(aValues->rankOf() == 1, 0,
               "csr_diag_mm_bp: aValues must be 1D, got rank %d", aValues->rankOf());
  REQUIRE_TRUE(aColIdx->rankOf() == 1, 0,
               "csr_diag_mm_bp: aColIdx must be 1D, got rank %d", aColIdx->rankOf());
  REQUIRE_TRUE(aRowPtr->rankOf() == 1, 0,
               "csr_diag_mm_bp: aRowPtr must be 1D, got rank %d", aRowPtr->rankOf());
  REQUIRE_TRUE(dl->rankOf() == 1, 0,
               "csr_diag_mm_bp: dl must be 1D, got rank %d", dl->rankOf());
  REQUIRE_TRUE(dr->rankOf() == 1, 0,
               "csr_diag_mm_bp: dr must be 1D, got rank %d", dr->rankOf());
  REQUIRE_TRUE(gradOut->rankOf() == 1, 0,
               "csr_diag_mm_bp: gradOut must be 1D, got rank %d", gradOut->rankOf());

  REQUIRE_TRUE(aValues->lengthOf() == aColIdx->lengthOf(), 0,
               "csr_diag_mm_bp: aValues and aColIdx must have the same length");
  REQUIRE_TRUE(aValues->lengthOf() == gradOut->lengthOf(), 0,
               "csr_diag_mm_bp: aValues and gradOut must have the same length");
  REQUIRE_TRUE(aRowPtr->lengthOf() == rows + 1, 0,
               "csr_diag_mm_bp: aRowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)aRowPtr->lengthOf());
  REQUIRE_TRUE(dl->lengthOf() == rows, 0,
               "csr_diag_mm_bp: dl length must equal rows=%lld, got %lld",
               (long long)rows, (long long)dl->lengthOf());
  REQUIRE_TRUE(dr->lengthOf() == cols, 0,
               "csr_diag_mm_bp: dr length must equal cols=%lld, got %lld",
               (long long)cols, (long long)dr->lengthOf());

  REQUIRE_TRUE(aValues->dataType() == dl->dataType(), 0,
               "csr_diag_mm_bp: aValues and dl must have the same float dtype");
  REQUIRE_TRUE(aValues->dataType() == dr->dataType(), 0,
               "csr_diag_mm_bp: aValues and dr must have the same float dtype");
  REQUIRE_TRUE(aValues->dataType() == gradOut->dataType(), 0,
               "csr_diag_mm_bp: aValues and gradOut must have the same float dtype");
  REQUIRE_TRUE(aColIdx->dataType() == aRowPtr->dataType(), 0,
               "csr_diag_mm_bp: aColIdx and aRowPtr must have the same integer dtype");

  // ── Execute ────────────────────────────────────────────────────────────────
  auto dAValues = OUTPUT_NULLIFIED(0);
  auto ddl      = OUTPUT_NULLIFIED(1);
  auto ddr      = OUTPUT_NULLIFIED(2);

  sd::ops::helpers::csr_diag_mm_bp(*aValues, *aColIdx, *aRowPtr, *dl, *dr, *gradOut,
                                    *dAValues, *ddl, *ddr, rows, cols);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_diag_mm_bp) {
  // dAValues shape == aValues shape  (same nnz, same float dtype)
  // ddl      shape == dl      shape  (rows)
  // ddr      shape == dr      shape  (cols)
  auto aValues = INPUT_VARIABLE(0);
  auto dl      = INPUT_VARIABLE(3);
  auto dr      = INPUT_VARIABLE(4);

  auto daShape  = ConstantShapeHelper::getInstance().createShapeInfo(
      aValues->dataType(), 'c', {aValues->lengthOf()});
  auto ddlShape = ConstantShapeHelper::getInstance().createShapeInfo(
      dl->dataType(), 'c', {dl->lengthOf()});
  auto ddrShape = ConstantShapeHelper::getInstance().createShapeInfo(
      dr->dataType(), 'c', {dr->lengthOf()});

  return SHAPELIST(daShape, ddlShape, ddrShape);
}

DECLARE_TYPES(csr_diag_mm_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})   // aValues
      ->setAllowedInputTypes(1, {ALL_INTS})     // aColIdx
      ->setAllowedInputTypes(2, {ALL_INTS})     // aRowPtr
      ->setAllowedInputTypes(3, {ALL_FLOATS})   // dl
      ->setAllowedInputTypes(4, {ALL_FLOATS})   // dr
      ->setAllowedInputTypes(5, {ALL_FLOATS})   // gradOut
      ->setAllowedOutputTypes(0, {ALL_FLOATS})  // dAValues
      ->setAllowedOutputTypes(1, {ALL_FLOATS})  // ddl
      ->setAllowedOutputTypes(2, {ALL_FLOATS}); // ddr
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_diag_mm_bp)
