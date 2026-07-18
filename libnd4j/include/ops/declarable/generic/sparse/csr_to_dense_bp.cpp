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
// csr_to_dense_bp — backward pass for csr_to_dense.
//
// Forward:  (values[nnz], colIdx[nnz], rowPtr[rows+1]), IArgs (rows, cols)
//              → dense[rows, cols]
//
// Backward: gather gradDense at the CSR sparsity pattern.
//
// Inputs:
//   [0] colIdx    [nnz],     INT32/INT64 — CSR column indices
//   [1] rowPtr    [rows+1],  same INT dtype — CSR row pointers
//   [2] gradDense [rows, cols], float dtype — upstream gradient w.r.t. dense output
// IArgs:
//   [0] rows
//   [1] cols
// Output:
//   [0] dValues   [nnz], same float dtype as gradDense — gradient w.r.t. CSR values
//
// Math:  dValues[e] = gradDense[i, colIdx[e]]
// where i is the row containing entry e (rowPtr[i] <= e < rowPtr[i+1]).
// This is a pure gather — no atomics.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_to_dense_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_convert_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_to_dense_bp, 3, 1, false, 0, 2) {
  auto colIdx    = INPUT_VARIABLE(0);
  auto rowPtr    = INPUT_VARIABLE(1);
  auto gradDense = INPUT_VARIABLE(2);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_to_dense_bp: colIdx must be 1D, got rank %d", colIdx->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_to_dense_bp: rowPtr must be 1D, got rank %d", rowPtr->rankOf());
  REQUIRE_TRUE(gradDense->rankOf() == 2, 0,
               "csr_to_dense_bp: gradDense must be 2D, got rank %d", gradDense->rankOf());
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_to_dense_bp: rowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(gradDense->sizeAt(0) == rows && gradDense->sizeAt(1) == cols, 0,
               "csr_to_dense_bp: gradDense shape must be [rows=%lld, cols=%lld], got [%lld, %lld]",
               (long long)rows, (long long)cols,
               (long long)gradDense->sizeAt(0), (long long)gradDense->sizeAt(1));
  REQUIRE_TRUE(colIdx->dataType() == rowPtr->dataType(), 0,
               "csr_to_dense_bp: colIdx and rowPtr must have the same integer dtype");

  auto dValues = OUTPUT_VARIABLE(0);

  const LongType nnz = colIdx->lengthOf();
  if (nnz > 0) {
    sd::ops::helpers::csr_to_dense_bp(*colIdx, *rowPtr, *gradDense, *dValues, rows, cols);
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_to_dense_bp) {
  auto colIdx    = INPUT_VARIABLE(0);
  auto gradDense = INPUT_VARIABLE(2);

  // dValues has same nnz as colIdx, same float dtype as gradDense
  const LongType nnz = colIdx->lengthOf();
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      gradDense->dataType(), 'c', {nnz});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_to_dense_bp) {

  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})     // colIdx
      ->setAllowedInputTypes(1, {ALL_INTS})     // rowPtr
      ->setAllowedInputTypes(2, {ALL_FLOATS})   // gradDense
      ->setAllowedOutputTypes(0, {ALL_FLOATS}); // dValues

  getOpDescriptor()->addTraits(OP_TRAIT_BACKWARD);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_to_dense_bp)
