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
// csr_diag_mm — two-sided diagonal scaling of a CSR matrix.
//
// Computes out[i,j] = dl[i] * A[i,j] * dr[j] for each stored nonzero of A.
// Only the VALUE array changes — the sparsity structure (colIdx, rowPtr) is
// UNCHANGED and is NOT an output; the caller reuses A's structural arrays.
//
// This implements the GCN normalization D^{-1/2} A D^{-1/2} where dl=dr=D^{-1/2}.
//
// Inputs:
//   [0] aValues  [nnz]     float  — stored nonzero values of A
//   [1] aColIdx  [nnz]     int    — column indices of A (sorted per row)
//   [2] aRowPtr  [rows+1]  int    — row pointers of A
//   [3] dl       [rows]    float  — left diagonal (one scalar per row)
//   [4] dr       [cols]    float  — right diagonal (one scalar per column)
// IArgs:
//   [0] rows
//   [1] cols
// Output:
//   [0] outValues [nnz]  float  — same dtype as aValues; outValues[e] = dl[i]*aValues[e]*dr[j]
//                                 where i is the row containing entry e and j = aColIdx[e].
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_diag_mm)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_diag_mm.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_diag_mm, 5, 1, false, 0, 2) {
  auto aValues = INPUT_VARIABLE(0);
  auto aColIdx = INPUT_VARIABLE(1);
  auto aRowPtr = INPUT_VARIABLE(2);
  auto dl      = INPUT_VARIABLE(3);
  auto dr      = INPUT_VARIABLE(4);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(aValues->rankOf() == 1, 0,
               "csr_diag_mm: aValues must be 1D, got rank %d", aValues->rankOf());
  REQUIRE_TRUE(aColIdx->rankOf() == 1, 0,
               "csr_diag_mm: aColIdx must be 1D, got rank %d", aColIdx->rankOf());
  REQUIRE_TRUE(aRowPtr->rankOf() == 1, 0,
               "csr_diag_mm: aRowPtr must be 1D, got rank %d", aRowPtr->rankOf());
  REQUIRE_TRUE(dl->rankOf() == 1, 0,
               "csr_diag_mm: dl must be 1D, got rank %d", dl->rankOf());
  REQUIRE_TRUE(dr->rankOf() == 1, 0,
               "csr_diag_mm: dr must be 1D, got rank %d", dr->rankOf());

  REQUIRE_TRUE(aValues->lengthOf() == aColIdx->lengthOf(), 0,
               "csr_diag_mm: aValues and aColIdx must have the same length");
  REQUIRE_TRUE(aRowPtr->lengthOf() == rows + 1, 0,
               "csr_diag_mm: aRowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)aRowPtr->lengthOf());
  REQUIRE_TRUE(dl->lengthOf() == rows, 0,
               "csr_diag_mm: dl length must equal rows=%lld, got %lld",
               (long long)rows, (long long)dl->lengthOf());
  REQUIRE_TRUE(dr->lengthOf() == cols, 0,
               "csr_diag_mm: dr length must equal cols=%lld, got %lld",
               (long long)cols, (long long)dr->lengthOf());

  REQUIRE_TRUE(aValues->dataType() == dl->dataType(), 0,
               "csr_diag_mm: aValues and dl must have the same float dtype");
  REQUIRE_TRUE(aValues->dataType() == dr->dataType(), 0,
               "csr_diag_mm: aValues and dr must have the same float dtype");
  REQUIRE_TRUE(aColIdx->dataType() == aRowPtr->dataType(), 0,
               "csr_diag_mm: aColIdx and aRowPtr must have the same integer dtype");

  auto outValues = OUTPUT_NULLIFIED(0);

  sd::ops::helpers::csr_diag_mm(*aValues, *aColIdx, *aRowPtr, *dl, *dr, *outValues, rows, cols);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_diag_mm) {
  // Output shape is identical to aValues: same length and same float dtype.
  auto aValues = INPUT_VARIABLE(0);

  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      aValues->dataType(), 'c', {aValues->lengthOf()});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_diag_mm) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})  // aValues
      ->setAllowedInputTypes(1, {ALL_INTS})    // aColIdx
      ->setAllowedInputTypes(2, {ALL_INTS})    // aRowPtr
      ->setAllowedInputTypes(3, {ALL_FLOATS})  // dl
      ->setAllowedInputTypes(4, {ALL_FLOATS})  // dr
      ->setAllowedOutputTypes({ALL_FLOATS});   // outValues
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_diag_mm)
