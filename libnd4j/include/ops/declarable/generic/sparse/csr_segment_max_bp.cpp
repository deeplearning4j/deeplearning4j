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
// csr_segment_max_bp — backward pass of csr_segment_max.
//
// For each (row i, feature feat):
//   Recompute k* = argmax_{k in row i} X[colIdx[k], feat].
//   dX[colIdx[k*], feat] += gradOut[i, feat].
//
// Multiple rows can map to the same (node, feat) → atomicAdd in CUDA.
// dX is initialised to zero (OUTPUT_NULLIFIED); CUDA helper also calls
// cudaMemsetAsync to guarantee the device buffer is zero before scatter.
//
// Inputs:
//   [0] colIdx   [nnz]     int    — CSR column indices
//   [1] rowPtr   [rows+1]  int    — CSR row pointer array
//   [2] X        [n, f]    float  — original node features
//   [3] gradOut  [rows, f] float  — upstream gradient
// IArgs:
//   [0] rows
// Outputs:
//   [0] dX  [n, f]  float  — gradient w.r.t. node features
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_segment_max_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_segment_max.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_segment_max_bp, 4, 1, false, 0, 1) {
  auto colIdx  = INPUT_VARIABLE(0);  // [nnz]     int
  auto rowPtr  = INPUT_VARIABLE(1);  // [rows+1]  int
  auto X       = INPUT_VARIABLE(2);  // [n, f]    float
  auto gradOut = INPUT_VARIABLE(3);  // [rows, f] float

  const sd::LongType rows = INT_ARG(0);
  const sd::LongType f    = X->sizeAt(1);

  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_segment_max_bp: colIdx must be rank-1, got %d", colIdx->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_segment_max_bp: rowPtr must be rank-1, got %d", rowPtr->rankOf());
  REQUIRE_TRUE(X->rankOf() == 2, 0,
               "csr_segment_max_bp: X must be rank-2 [n, f], got %d", X->rankOf());
  REQUIRE_TRUE(gradOut->rankOf() == 2, 0,
               "csr_segment_max_bp: gradOut must be rank-2 [rows, f], got %d",
               gradOut->rankOf());
  REQUIRE_TRUE(gradOut->sizeAt(0) == rows && gradOut->sizeAt(1) == f, 0,
               "csr_segment_max_bp: gradOut must be [%lld, %lld], got [%lld, %lld]",
               (long long)rows, (long long)f,
               (long long)gradOut->sizeAt(0), (long long)gradOut->sizeAt(1));
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_segment_max_bp: rowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());

  auto dX = OUTPUT_NULLIFIED(0);  // [n, f] — zeroed by nullification + helper

  sd::ops::helpers::csr_segment_max_bp(*colIdx, *rowPtr, *X, *gradOut, *dX, rows);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_segment_max_bp) {
  // dX has the same shape and dtype as X
  auto X = INPUT_VARIABLE(2);
  auto dxShape = ConstantShapeHelper::getInstance().createShapeInfo(
      X->dataType(), 'c', {X->sizeAt(0), X->sizeAt(1)});
  return SHAPELIST(dxShape);
}

DECLARE_TYPES(csr_segment_max_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})    // colIdx
      ->setAllowedInputTypes(1, {ALL_INTS})    // rowPtr
      ->setAllowedInputTypes(2, {ALL_FLOATS})  // X
      ->setAllowedInputTypes(3, {ALL_FLOATS})  // gradOut
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_segment_max_bp)
