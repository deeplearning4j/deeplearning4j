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
// csr_subgraph_extract_bp — backward pass of csr_subgraph_extract.
//
// Gradient flows only through values (edge weights).
// For each kept edge e (original index) that maps to extracted position e':
//   dValues[e] = dNewValues[e']
// Dropped edges receive zero gradient.
// Index arrays (colIdx, rowPtr, nodeIdx) are structural — no gradient.
//
// Inputs:
//   [0] values      [nnz]   float  — forward input edge weights
//   [1] colIdx      [nnz]   int    — column indices
//   [2] rowPtr      [N+1]   int    — row pointers
//   [3] nodeIdx     [K]     int    — selected node ids (sorted ascending)
//   [4] dNewValues  [nnz']  float  — upstream gradient w.r.t. newValues
// IArgs:
//   [0] N
//   [1] K
// Output:
//   [0] dValues  [nnz]  float  — gradient w.r.t. values (zero for dropped edges)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_subgraph_extract_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_subgraph.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_subgraph_extract_bp, 5, 1, false, 0, 2) {
  auto values     = INPUT_VARIABLE(0);  // [nnz]  float
  auto colIdx     = INPUT_VARIABLE(1);  // [nnz]  int
  auto rowPtr     = INPUT_VARIABLE(2);  // [N+1]  int
  auto nodeIdx    = INPUT_VARIABLE(3);  // [K]    int
  auto dNewValues = INPUT_VARIABLE(4);  // [nnz'] float

  const sd::LongType N = INT_ARG(0);
  const sd::LongType K = INT_ARG(1);

  REQUIRE_TRUE(values->rankOf() == 1, 0,
               "csr_subgraph_extract_bp: values must be rank-1, got %d", values->rankOf());
  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_subgraph_extract_bp: colIdx must be rank-1, got %d", colIdx->rankOf());
  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_subgraph_extract_bp: rowPtr must be rank-1, got %d", rowPtr->rankOf());
  REQUIRE_TRUE(nodeIdx->rankOf() == 1, 0,
               "csr_subgraph_extract_bp: nodeIdx must be rank-1, got %d", nodeIdx->rankOf());
  REQUIRE_TRUE(dNewValues->rankOf() == 1, 0,
               "csr_subgraph_extract_bp: dNewValues must be rank-1, got %d",
               dNewValues->rankOf());
  REQUIRE_TRUE(rowPtr->lengthOf() == N + 1, 0,
               "csr_subgraph_extract_bp: rowPtr length must be N+1=%lld, got %lld",
               (long long)(N + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(nodeIdx->lengthOf() == K, 0,
               "csr_subgraph_extract_bp: nodeIdx length must be K=%lld, got %lld",
               (long long)K, (long long)nodeIdx->lengthOf());

  auto dValues = OUTPUT_NULLIFIED(0);  // [nnz] float — zeroed by nullification

  if (K == 0 || values->lengthOf() == 0 || dNewValues->lengthOf() == 0) {
    return sd::Status::OK;
  }

  sd::ops::helpers::csr_subgraph_extract_bp(
      block.launchContext(),
      *values, *colIdx, *rowPtr, *nodeIdx, *dNewValues,
      *dValues, N, K);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_subgraph_extract_bp) {
  // dValues has the same shape and dtype as values (input[0])
  auto values   = INPUT_VARIABLE(0);
  auto dValShape = ConstantShapeHelper::getInstance().createShapeInfo(
      values->dataType(), 'c', {values->lengthOf()});
  return SHAPELIST(dValShape);
}

DECLARE_TYPES(csr_subgraph_extract_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})  // values
      ->setAllowedInputTypes(1, {ALL_INTS})    // colIdx
      ->setAllowedInputTypes(2, {ALL_INTS})    // rowPtr
      ->setAllowedInputTypes(3, {ALL_INTS})    // nodeIdx
      ->setAllowedInputTypes(4, {ALL_FLOATS})  // dNewValues
      ->setAllowedOutputTypes({ALL_FLOATS});   // dValues
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_subgraph_extract_bp)
