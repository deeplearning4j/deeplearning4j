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
// Helpers for csr_edge_gather / csr_edge_gather_bp.
//
// csr_edge_gather — gather node features to per-edge feature vectors.
//   edgeFeat[e, f] = X[colIdx[e], f]   for e in [0, nnz), f in [0, F)
//
// csr_edge_gather_bp — backward: scatter-add upstream gradients to node features.
//   dX[colIdx[e], f] += gradEdge[e, f]   (multiple edges → same node → atomicAdd/accumulate)
//   dX is pre-zeroed by the helper before accumulation.
//

#ifndef SAMEDIFF_SPARSE_EDGE_GATHER_H
#define SAMEDIFF_SPARSE_EDGE_GATHER_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

// Forward: edgeFeat[nnz, F] = xArr[colIdx, :]
SD_LIB_HIDDEN void csr_edge_gather(NDArray& colIdx,   // [nnz]   int
                                    NDArray& xArr,     // [n, F]  float
                                    NDArray& edgeFeat); // [nnz, F] float (output)

// Backward: dX[n, F] = scatter-sum of gradEdge over colIdx.
//   xArr is the original node-feature matrix X[n, F]; its shape determines dX's shape.
//   X values are NOT read — only shape information is used.
SD_LIB_HIDDEN void csr_edge_gather_bp(NDArray& colIdx,    // [nnz]    int
                                       NDArray& xArr,      // [n, F]   float (shape source)
                                       NDArray& gradEdge,  // [nnz, F] float
                                       NDArray& dX);       // [n, F]   float (output)

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_EDGE_GATHER_H
