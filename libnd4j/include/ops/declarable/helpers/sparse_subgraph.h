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
// Helpers for csr_subgraph_extract / csr_subgraph_extract_bp.
//
// csr_subgraph_extract:
//   Given a CSR graph (values, colIdx, rowPtr) of N nodes and a sorted list
//   of K selected node ids (nodeIdx), extract the induced subgraph CSR:
//   keep edge (i->j) iff BOTH i and j appear in nodeIdx; remap both endpoints
//   to their 0-based position in nodeIdx.
//   Outputs: newValues[nnz'], newColIdx[nnz'], newRowPtr[K+1].
//
// csr_subgraph_extract_bp:
//   Gradient flows only through values. For each kept edge e (original) -> e'
//   (extracted position): dValues[e] = dNewValues[e']. Dropped edges get 0.
//

#ifndef SAMEDIFF_SPARSE_SUBGRAPH_H
#define SAMEDIFF_SPARSE_SUBGRAPH_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

// Forward: extract induced-subgraph CSR for K selected nodes.
SD_LIB_HIDDEN void csr_subgraph_extract(LaunchContext* ctx,
                                         NDArray& values,     // [nnz]  float
                                         NDArray& colIdx,     // [nnz]  int
                                         NDArray& rowPtr,     // [N+1]  int
                                         NDArray& nodeIdx,    // [K]    int  (sorted)
                                         NDArray& newValues,  // [nnz'] float (output)
                                         NDArray& newColIdx,  // [nnz'] INT32 (output)
                                         NDArray& newRowPtr,  // [K+1]  INT32 (output)
                                         sd::LongType N,
                                         sd::LongType K);

// Backward: scatter dNewValues back to dValues at kept-edge positions.
SD_LIB_HIDDEN void csr_subgraph_extract_bp(LaunchContext* ctx,
                                             NDArray& values,      // [nnz]  float  (forward input)
                                             NDArray& colIdx,      // [nnz]  int
                                             NDArray& rowPtr,      // [N+1]  int
                                             NDArray& nodeIdx,     // [K]    int    (sorted)
                                             NDArray& dNewValues,  // [nnz'] float  (upstream grad)
                                             NDArray& dValues,     // [nnz]  float  (output grad)
                                             sd::LongType N,
                                             sd::LongType K);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_SUBGRAPH_H
