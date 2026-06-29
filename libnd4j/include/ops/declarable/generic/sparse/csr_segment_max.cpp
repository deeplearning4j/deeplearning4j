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
// csr_segment_max — GraphSAGE-style max neighbor aggregation over CSR graph.
//
// For each row i and feature dimension feat:
//   out[i, feat] = max over k in [rowPtr[i], rowPtr[i+1]) of X[colIdx[k], feat]
// Empty row → out[i, :] = 0.
//
// Inputs:
//   [0] colIdx  [nnz]   int    — CSR column indices (neighbour node ids)
//   [1] rowPtr  [rows+1] int   — CSR row pointer array
//   [2] X       [n, f]  float  — dense node feature matrix
// IArgs:
//   [0] rows  — number of rows (output nodes)
// Outputs:
//   [0] out   [rows, f]  float  — aggregated features
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_segment_max)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_segment_max.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_segment_max, 3, 1, false, 0, 1) {
  auto colIdx = INPUT_VARIABLE(0);  // [nnz]    int
  auto rowPtr = INPUT_VARIABLE(1);  // [rows+1] int
  auto X      = INPUT_VARIABLE(2);  // [n, f]   float

  const sd::LongType rows = INT_ARG(0);

  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_segment_max: colIdx must be rank-1, got %d", colIdx->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_segment_max: rowPtr must be rank-1, got %d", rowPtr->rankOf());
  REQUIRE_TRUE(X->rankOf() == 2, 0,
               "csr_segment_max: X must be rank-2 [n, f], got %d", X->rankOf());
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_segment_max: rowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());

  auto out = OUTPUT_NULLIFIED(0);  // [rows, f]

  sd::ops::helpers::csr_segment_max(*colIdx, *rowPtr, *X, *out, rows);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_segment_max) {
  auto X      = INPUT_VARIABLE(2);
  const sd::LongType rows = INT_ARG(0);
  const sd::LongType f    = X->sizeAt(1);

  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      X->dataType(), 'c', {rows, f});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_segment_max) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})    // colIdx
      ->setAllowedInputTypes(1, {ALL_INTS})    // rowPtr
      ->setAllowedInputTypes(2, {ALL_FLOATS})  // X
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_segment_max)
