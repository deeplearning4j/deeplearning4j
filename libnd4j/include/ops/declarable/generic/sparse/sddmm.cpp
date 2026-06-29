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
// SDDMM — sampled dense-dense matrix multiplication.
// Computes output values at the CSR sparsity pattern:
//
//   outValues[k] = sum_{l=0}^{p-1} D1[i,l] * D2[j,l]
//
// where i is the row of nonzero k (rowPtr[i] <= k < rowPtr[i+1])
// and j = colIdx[k].
// Equivalently: outValues = (D1 · D2^T) sampled at the (rowPtr, colIdx) pattern.
//
// Inputs:
//   [0] rowPtr    – 1D [rows+1], INT32/INT64  (row pointer array)
//   [1] colIdx    – 1D [nnz],   same INT       (column indices)
//   [2] D1        – dense 2D [rows, p]
//   [3] D2        – dense 2D [cols, p]
// IArgs:
//   [0] rows
//   [1] cols
// Output:
//   [0] outValues – 1D [nnz], same float dtype as D1
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_sddmm)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_blas.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(sddmm, 4, 1, false, 0, 2) {
  auto rowPtr    = INPUT_VARIABLE(0);
  auto colIdx    = INPUT_VARIABLE(1);
  auto D1        = INPUT_VARIABLE(2);
  auto D2        = INPUT_VARIABLE(3);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "sddmm: rowPtr must be 1D, got rank %d", rowPtr->rankOf());
  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "sddmm: colIdx must be 1D, got rank %d", colIdx->rankOf());
  REQUIRE_TRUE(D1->rankOf() == 2, 0,
               "sddmm: D1 must be 2D [rows, p], got rank %d", D1->rankOf());
  REQUIRE_TRUE(D2->rankOf() == 2, 0,
               "sddmm: D2 must be 2D [cols, p], got rank %d", D2->rankOf());
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "sddmm: rowPtr length must equal rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(rowPtr->dataType() == colIdx->dataType(), 0,
               "sddmm: rowPtr and colIdx must have the same integer dtype");
  REQUIRE_TRUE(D1->sizeAt(0) == rows, 0,
               "sddmm: D1 must have %lld rows, got %lld",
               (long long)rows, (long long)D1->sizeAt(0));
  REQUIRE_TRUE(D2->sizeAt(0) == cols, 0,
               "sddmm: D2 must have %lld rows (= cols of A), got %lld",
               (long long)cols, (long long)D2->sizeAt(0));
  REQUIRE_TRUE(D1->sizeAt(1) == D2->sizeAt(1), 0,
               "sddmm: D1 and D2 must share the inner dimension p: %lld vs %lld",
               (long long)D1->sizeAt(1), (long long)D2->sizeAt(1));

  auto outValues = OUTPUT_NULLIFIED(0);

  sd::ops::helpers::sddmm(*rowPtr, *colIdx, *D1, *D2, *outValues, rows, cols);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(sddmm) {
  auto colIdx = INPUT_VARIABLE(1);

  // nnz is the length of colIdx
  const LongType nnz = colIdx->lengthOf();

  auto D1 = INPUT_VARIABLE(2);
  auto dtype    = D1->dataType();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {nnz});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(sddmm) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})         // rowPtr
      ->setAllowedInputTypes(1, {ALL_INTS})         // colIdx
      ->setAllowedInputTypes(2, {ALL_FLOATS})   // D1
      ->setAllowedInputTypes(3, {ALL_FLOATS})   // D2
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_sddmm)
