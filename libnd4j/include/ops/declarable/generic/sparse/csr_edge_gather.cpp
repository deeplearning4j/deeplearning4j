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
// csr_edge_gather — gather node features to per-edge feature vectors.
//
// For each edge index e in [0, nnz) and feature dimension f in [0, F):
//   edgeFeat[e, f] = X[colIdx[e], f]
//
// colIdx[e] is the source-node (neighbour) id for edge e.
// This is the E-step of MPNN / edge-conditioned GNNs: pull node embeddings
// onto edges so that message functions can operate purely on edge tensors.
//
// Inputs:
//   [0] colIdx   [nnz]  int   — source-node id for each edge
//   [1] X        [n, F] float — dense node-feature matrix
// IArgs: none   (n and F are inferred from X)
// Output:
//   [0] edgeFeat [nnz, F] float — gathered per-edge node features
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_edge_gather)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_edge_gather.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_edge_gather, 2, 1, false, 0, 0) {
  auto colIdx = INPUT_VARIABLE(0);  // [nnz]   int
  auto X      = INPUT_VARIABLE(1);  // [n, F]  float

  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_edge_gather: colIdx must be rank-1, got %d", colIdx->rankOf());
  REQUIRE_TRUE(X->rankOf() == 2, 0,
               "csr_edge_gather: X must be rank-2 [n, F], got %d", X->rankOf());

  auto edgeFeat = OUTPUT_NULLIFIED(0);  // [nnz, F]

  sd::ops::helpers::csr_edge_gather(*colIdx, *X, *edgeFeat);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_edge_gather) {
  auto colIdx = INPUT_VARIABLE(0);
  auto X      = INPUT_VARIABLE(1);

  const sd::LongType nnz = colIdx->lengthOf();
  const sd::LongType F   = X->sizeAt(1);

  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      X->dataType(), 'c', {nnz, F});
  return SHAPELIST(outShape);
}

DECLARE_TYPES(csr_edge_gather) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})    // colIdx
      ->setAllowedInputTypes(1, {ALL_FLOATS})  // X
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_edge_gather)
