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
// csr_edge_aggregate_bp — backward pass of csr_edge_aggregate.
//
// Routes upstream gradients from per-node outputs back to per-edge messages:
//
//   mode 0 (SUM):  dEdgeMsg[e, f] = gradOut[i, f]              for all e in row i
//   mode 1 (MEAN): dEdgeMsg[e, f] = gradOut[i, f] / degree(i)
//   mode 2 (MAX):  recompute argmax e* for each (i, f);
//                  dEdgeMsg[e*, f] += gradOut[i, f]
//                  (other edges in row → zero gradient, handled by pre-zero)
//
// Inputs:
//   [0] rowPtr   [rows+1]  int   — CSR row pointer array
//   [1] edgeMsg  [nnz, F]  float — forward edgeMsg (needed only for MAX argmax)
//   [2] gradOut  [rows, F] float — upstream gradient
// IArgs:
//   [0] rows  — number of nodes / output rows
//   [1] mode  — 0=SUM, 1=MEAN, 2=MAX (must match forward)
// Output:
//   [0] dEdgeMsg  [nnz, F]  float — gradient w.r.t. edgeMsg
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_edge_aggregate_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_edge_aggregate.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_edge_aggregate_bp, 3, 1, false, 0, 2) {
  auto rowPtr  = INPUT_VARIABLE(0);  // [rows+1]  int
  auto edgeMsg = INPUT_VARIABLE(1);  // [nnz, F]  float
  auto gradOut = INPUT_VARIABLE(2);  // [rows, F] float

  const sd::LongType rows = INT_ARG(0);
  const int          mode = static_cast<int>(INT_ARG(1));

  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_edge_aggregate_bp: rowPtr must be rank-1, got %d", rowPtr->rankOf());
  REQUIRE_TRUE(edgeMsg->rankOf() == 2, 0,
               "csr_edge_aggregate_bp: edgeMsg must be rank-2 [nnz, F], got %d",
               edgeMsg->rankOf());
  REQUIRE_TRUE(gradOut->rankOf() == 2, 0,
               "csr_edge_aggregate_bp: gradOut must be rank-2 [rows, F], got %d",
               gradOut->rankOf());
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_edge_aggregate_bp: rowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(gradOut->sizeAt(0) == rows, 0,
               "csr_edge_aggregate_bp: gradOut.shape[0]=%lld must equal rows=%lld",
               (long long)gradOut->sizeAt(0), (long long)rows);
  REQUIRE_TRUE(gradOut->sizeAt(1) == edgeMsg->sizeAt(1), 0,
               "csr_edge_aggregate_bp: gradOut.shape[1]=%lld must equal edgeMsg.shape[1]=%lld",
               (long long)gradOut->sizeAt(1), (long long)edgeMsg->sizeAt(1));
  REQUIRE_TRUE(mode >= 0 && mode <= 2, 0,
               "csr_edge_aggregate_bp: mode must be 0(SUM), 1(MEAN), or 2(MAX), got %d", mode);

  auto dEdgeMsg = OUTPUT_NULLIFIED(0);  // [nnz, F]

  sd::ops::helpers::csr_edge_aggregate_bp(*rowPtr, *edgeMsg, *gradOut, *dEdgeMsg, rows, mode);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_edge_aggregate_bp) {
  // dEdgeMsg has the same shape as edgeMsg
  auto edgeMsg = INPUT_VARIABLE(1);
  auto deShape = ConstantShapeHelper::getInstance().createShapeInfo(
      edgeMsg->dataType(), 'c', {edgeMsg->sizeAt(0), edgeMsg->sizeAt(1)});
  return SHAPELIST(deShape);
}

DECLARE_TYPES(csr_edge_aggregate_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})    // rowPtr
      ->setAllowedInputTypes(1, {ALL_FLOATS})  // edgeMsg
      ->setAllowedInputTypes(2, {ALL_FLOATS})  // gradOut
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_edge_aggregate_bp)
