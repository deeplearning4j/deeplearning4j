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
// csr_edge_gather_bp — backward pass of csr_edge_gather.
//
// Scatter-add upstream gradients back to node features:
//   dX[colIdx[e], f] += gradEdge[e, f]   for all e, f
//
// Multiple edges may reference the same source node → accumulate.
// dX is pre-zeroed before accumulation (OUTPUT_NULLIFIED; helpers also clear).
// On CUDA the helper uses sd_atomicAdd for the scatter; on CPU it is sequential.
//
// Inputs:
//   [0] colIdx    [nnz]    int   — source-node ids (same as csr_edge_gather forward)
//   [1] X         [n, F]   float — original node features (used ONLY for shape; values ignored)
//   [2] gradEdge  [nnz, F] float — upstream gradient w.r.t. edgeFeat
// IArgs: none
// Output:
//   [0] dX  [n, F]  float — gradient w.r.t. node features X (same shape as X)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_csr_edge_gather_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_edge_gather.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(csr_edge_gather_bp, 3, 1, false, 0, 0) {
  auto colIdx   = INPUT_VARIABLE(0);  // [nnz]    int
  auto xArr     = INPUT_VARIABLE(1);  // [n, F]   float — shape source only, values unused
  auto gradEdge = INPUT_VARIABLE(2);  // [nnz, F] float

  REQUIRE_TRUE(colIdx->rankOf() == 1, 0,
               "csr_edge_gather_bp: colIdx must be rank-1, got %d", colIdx->rankOf());
  REQUIRE_TRUE(xArr->rankOf() == 2, 0,
               "csr_edge_gather_bp: X must be rank-2 [n, F], got %d", xArr->rankOf());
  REQUIRE_TRUE(gradEdge->rankOf() == 2, 0,
               "csr_edge_gather_bp: gradEdge must be rank-2 [nnz, F], got %d",
               gradEdge->rankOf());
  REQUIRE_TRUE(gradEdge->sizeAt(0) == colIdx->lengthOf(), 0,
               "csr_edge_gather_bp: gradEdge.shape[0] (%lld) must equal nnz (%lld)",
               (long long)gradEdge->sizeAt(0), (long long)colIdx->lengthOf());
  REQUIRE_TRUE(gradEdge->sizeAt(1) == xArr->sizeAt(1), 0,
               "csr_edge_gather_bp: gradEdge.shape[1] (%lld) must equal X.shape[1] (%lld)",
               (long long)gradEdge->sizeAt(1), (long long)xArr->sizeAt(1));

  auto dX = OUTPUT_NULLIFIED(0);  // [n, F] — same shape as X

  sd::ops::helpers::csr_edge_gather_bp(*colIdx, *xArr, *gradEdge, *dX);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(csr_edge_gather_bp) {
  // Input [1] is X[n, F]: use its shape for dX output shape.
  // Input [2] is gradEdge[nnz, F]: use its dtype for dX dtype.
  auto xShapeInfo        = inputShape->at(1);
  auto gradEdgeShapeInfo = inputShape->at(2);

  const sd::LongType n   = shape::sizeAt(xShapeInfo, 0);
  const sd::LongType F   = shape::sizeAt(xShapeInfo, 1);
  const sd::DataType dt  = ArrayOptions::dataType(gradEdgeShapeInfo);

  auto dxShape = ConstantShapeHelper::getInstance().createShapeInfo(dt, 'c', {n, F});
  return SHAPELIST(dxShape);
}

DECLARE_TYPES(csr_edge_gather_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})    // colIdx
      ->setAllowedInputTypes(1, {ALL_FLOATS})  // X (shape source)
      ->setAllowedInputTypes(2, {ALL_FLOATS})  // gradEdge
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_csr_edge_gather_bp)
