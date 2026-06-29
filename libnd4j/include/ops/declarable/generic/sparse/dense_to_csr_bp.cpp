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
// dense_to_csr_bp — backward pass for dense_to_csr.
//
// Forward:  dense[rows, cols] (TArgs: threshold)
//              → values[nnz], colIdx[nnz], rowPtr[rows+1]
//
// Backward: scatter gradValues back into a zeroed dDense matrix.
//
// Inputs:
//   [0] colIdx     [nnz],     INT32 — CSR column indices (forward output[1])
//   [1] rowPtr     [rows+1],  INT32 — CSR row pointers  (forward output[2])
//   [2] gradValues [nnz],     float dtype — upstream gradient w.r.t. values output
// IArgs:
//   [0] rows
//   [1] cols
// Output:
//   [0] dDense [rows, cols], same float dtype as gradValues — gradient w.r.t. dense input
//       (OUTPUT_NULLIFIED — framework pre-zeros the output; only non-zero positions are written)
//
// Math:  dDense[i, colIdx[e]] = gradValues[e]  for each entry e.
// In a valid CSR pattern every (i, colIdx[e]) pair is unique → no atomics.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dense_to_csr_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_convert_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dense_to_csr_bp, 3, 1, false, 0, 2) {
  auto colIdx     = INPUT_VARIABLE(0);
  auto rowPtr     = INPUT_VARIABLE(1);
  auto gradValues = INPUT_VARIABLE(2);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "dense_to_csr_bp: colIdx must be 1D, got rank %d", colIdx->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "dense_to_csr_bp: rowPtr must be 1D, got rank %d", rowPtr->rankOf());
  REQUIRE_TRUE(gradValues->rankOf() == 1, 0,
               "dense_to_csr_bp: gradValues must be 1D, got rank %d", gradValues->rankOf());
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "dense_to_csr_bp: rowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(colIdx->lengthOf() == gradValues->lengthOf(), 0,
               "dense_to_csr_bp: colIdx and gradValues must have the same length");
  REQUIRE_TRUE(colIdx->dataType() == rowPtr->dataType(), 0,
               "dense_to_csr_bp: colIdx and rowPtr must have the same integer dtype");

  auto dDense = OUTPUT_NULLIFIED(0);

  const LongType nnz = colIdx->lengthOf();
  if (nnz > 0) {
    sd::ops::helpers::dense_to_csr_bp(*colIdx, *rowPtr, *gradValues, *dDense, rows, cols);
  }
  // If nnz == 0, dDense stays all-zero (already provided by OUTPUT_NULLIFIED).

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(dense_to_csr_bp) {
  auto gradValues = INPUT_VARIABLE(2);

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  // dDense has shape [rows, cols] with the same float dtype as gradValues
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      gradValues->dataType(), 'c', {rows, cols});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(dense_to_csr_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})     // colIdx
      ->setAllowedInputTypes(1, {ALL_INTS})     // rowPtr
      ->setAllowedInputTypes(2, {ALL_FLOATS})   // gradValues
      ->setAllowedOutputTypes(0, {ALL_FLOATS}); // dDense
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_dense_to_csr_bp)
