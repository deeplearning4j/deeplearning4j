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
// csr_edge_aggregate — aggregate per-edge messages to per-node outputs.
//
// For each target node (row) i and feature dim f:
//   mode 0 (SUM):  out[i, f] = Σ_{e in row i} edgeMsg[e, f]
//   mode 1 (MEAN): out[i, f] = (Σ_{e in row i} edgeMsg[e, f]) / degree(i)
//                              (empty row → 0)
//   mode 2 (MAX):  out[i, f] = max_{e in row i} edgeMsg[e, f]
//                              (empty row → 0)
//
// This is the N-step of MPNN: reduce incoming edge messages into a per-node
// summary that feeds the next node-update MLP.
//
// Inputs:
//   [0] rowPtr   [rows+1] int   — CSR row pointer array
//   [1] edgeMsg  [nnz, F] float — per-edge message vectors
// IArgs:
//   [0] rows  — number of target nodes (output rows)
//   [1] mode  — 0=SUM, 1=MEAN, 2=MAX
// Output:
//   [0] out  [rows, F]  float — aggregated per-node messages
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_edge_aggregate)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_edge_aggregate.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_edge_aggregate, 2, 1, false, 0, 2) {
  auto rowPtr  = INPUT_VARIABLE(0);  // [rows+1] int
  auto edgeMsg = INPUT_VARIABLE(1);  // [nnz, F] float

  const sd::LongType rows = INT_ARG(0);
  const int          mode = static_cast<int>(INT_ARG(1));

  REQUIRE_TRUE(rowPtr->rankOf() == 1, 0,
               "csr_edge_aggregate: rowPtr must be rank-1, got %d", rowPtr->rankOf());
  REQUIRE_TRUE(edgeMsg->rankOf() == 2, 0,
               "csr_edge_aggregate: edgeMsg must be rank-2 [nnz, F], got %d",
               edgeMsg->rankOf());
  REQUIRE_TRUE(rowPtr->lengthOf() == rows + 1, 0,
               "csr_edge_aggregate: rowPtr length must be rows+1=%lld, got %lld",
               (long long)(rows + 1), (long long)rowPtr->lengthOf());
  REQUIRE_TRUE(mode >= 0 && mode <= 2, 0,
               "csr_edge_aggregate: mode must be 0(SUM), 1(MEAN), or 2(MAX), got %d", mode);

  auto out = OUTPUT_NULLIFIED(0);  // [rows, F]

  sd::ops::helpers::csr_edge_aggregate(*rowPtr, *edgeMsg, *out, rows, mode);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_edge_aggregate) {
  auto edgeMsg = INPUT_VARIABLE(1);
  const sd::LongType rows = INT_ARG(0);
  const sd::LongType F    = edgeMsg->sizeAt(1);

  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      edgeMsg->dataType(), 'c', {rows, F});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_edge_aggregate) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})    // rowPtr
      ->setAllowedInputTypes(1, {ALL_FLOATS})  // edgeMsg
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_edge_aggregate)
