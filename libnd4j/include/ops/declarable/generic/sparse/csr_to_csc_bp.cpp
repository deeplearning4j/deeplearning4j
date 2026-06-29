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
// csr_to_csc_bp — backward pass for csr_to_csc.
//
// Forward:  (csrValues[nnz], csrColIdx[nnz], csrRowPtr[rows+1]), IArgs (rows, cols)
//              → cscValues[nnz], cscRowIdx[nnz], cscColPtr[cols+1]
//
// The forward permutation P: CSR(A) → CSC(A) is computed by a counting-sort
// transpose.  The backward pass applies P⁻¹ via a second csr_to_csc call:
// treat (gradCscValues, cscRowIdx, cscColPtr) as the CSR of A^T (shape [cols, rows])
// and transpose back with dimensions (cols, rows) → values land in CSR(A) row-major order.
//
// Inputs:
//   [0] csrColIdx     [nnz],     INT32/INT64 — CSR column indices (forward input[1])
//   [1] csrRowPtr     [rows+1],  same INT — CSR row pointers (forward input[2])
//   [2] cscRowIdx     [nnz],     INT32 — CSC row indices (forward output[1])
//   [3] cscColPtr     [cols+1],  INT32 — CSC column pointers (forward output[2])
//   [4] gradCscValues [nnz],     float — upstream gradient w.r.t. cscValues output
// IArgs:
//   [0] rows
//   [1] cols
// Output:
//   [0] dAValues [nnz], same float dtype as gradCscValues — gradient w.r.t. csrValues input
//
// csrColIdx and csrRowPtr are accepted for structural validation but are not
// used in the computation (the helper only needs the CSC arrays to invert P).
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_to_csc_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_convert_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_to_csc_bp, 5, 1, false, 0, 2) {
  auto csrColIdx     = INPUT_VARIABLE(0);
  auto csrRowPtr     = INPUT_VARIABLE(1);
  auto cscRowIdx     = INPUT_VARIABLE(2);
  auto cscColPtr     = INPUT_VARIABLE(3);
  auto gradCscValues = INPUT_VARIABLE(4);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);
  const LongType nnz  = gradCscValues->lengthOf();

  // ── Validation ─────────────────────────────────────────────────────────────
  REQUIRE_TRUE(csrColIdx->rankOf() == 1, 0,
               "csr_to_csc_bp: csrColIdx must be 1D, got rank %d", csrColIdx->rankOf());
  REQUIRE_TRUE(csrRowPtr->rankOf() == 1, 0,
               "csr_to_csc_bp: csrRowPtr must be 1D, got rank %d", csrRowPtr->rankOf());
  REQUIRE_TRUE(cscRowIdx->rankOf() == 1, 0,
               "csr_to_csc_bp: cscRowIdx must be 1D, got rank %d", cscRowIdx->rankOf());
  REQUIRE_TRUE(cscColPtr->rankOf() == 1, 0,
               "csr_to_csc_bp: cscColPtr must be 1D, got rank %d", cscColPtr->rankOf());
  REQUIRE_TRUE(gradCscValues->rankOf() == 1, 0,
               "csr_to_csc_bp: gradCscValues must be 1D, got rank %d", gradCscValues->rankOf());
  REQUIRE_TRUE(csrRowPtr->lengthOf() == rows + 1, 0,
               "csr_to_csc_bp: csrRowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)csrRowPtr->lengthOf());
  REQUIRE_TRUE(cscColPtr->lengthOf() == cols + 1, 0,
               "csr_to_csc_bp: cscColPtr length must be cols+1=%lld, got %lld",
               (long long)(cols + 1), (long long)cscColPtr->lengthOf());
  REQUIRE_TRUE(csrColIdx->lengthOf() == nnz, 0,
               "csr_to_csc_bp: csrColIdx length must match gradCscValues length (%lld), got %lld",
               (long long)nnz, (long long)csrColIdx->lengthOf());
  REQUIRE_TRUE(cscRowIdx->lengthOf() == nnz, 0,
               "csr_to_csc_bp: cscRowIdx length must match gradCscValues length (%lld), got %lld",
               (long long)nnz, (long long)cscRowIdx->lengthOf());

  // ── Execute ────────────────────────────────────────────────────────────────
  auto dAValues = OUTPUT_VARIABLE(0);

  if (nnz > 0) {
    sd::ops::helpers::csr_to_csc_bp(*cscRowIdx, *cscColPtr, *gradCscValues, *dAValues, rows, cols);
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_to_csc_bp) {
  // dAValues has same shape as csrColIdx (nnz elements) and dtype of gradCscValues.
  auto csrColIdx     = INPUT_VARIABLE(0);
  auto gradCscValues = INPUT_VARIABLE(4);

  const LongType nnz = csrColIdx->lengthOf();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      gradCscValues->dataType(), 'c', {nnz});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_to_csc_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})     // csrColIdx
      ->setAllowedInputTypes(1, {ALL_INTS})     // csrRowPtr
      ->setAllowedInputTypes(2, {ALL_INTS})     // cscRowIdx
      ->setAllowedInputTypes(3, {ALL_INTS})     // cscColPtr
      ->setAllowedInputTypes(4, {ALL_FLOATS})   // gradCscValues
      ->setAllowedOutputTypes(0, {ALL_FLOATS}); // dAValues
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_to_csc_bp)
