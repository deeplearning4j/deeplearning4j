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
// Helpers for csr_edge_aggregate / csr_edge_aggregate_bp.
//
// csr_edge_aggregate — reduce per-edge messages to per-node outputs along CSR rows.
//
//   mode 0 = SUM:  out[i, f] = Σ_{e in row i} edgeMsg[e, f]
//   mode 1 = MEAN: out[i, f] = (Σ_{e in row i} edgeMsg[e, f]) / degree(i)
//                              (empty row → 0)
//   mode 2 = MAX:  out[i, f] = max_{e in row i} edgeMsg[e, f]
//                              (empty row → 0)
//
// csr_edge_aggregate_bp — backward pass; all modes route gradients to edges.
//   SUM:  dEdgeMsg[e, f] = gradOut[i, f]          for all e in row i
//   MEAN: dEdgeMsg[e, f] = gradOut[i, f] / degree(i)
//   MAX:  dEdgeMsg[argmax_e, f] = gradOut[i, f]   (recompute argmax; pre-zeroed)
//

#ifndef SAMEDIFF_SPARSE_EDGE_AGGREGATE_H
#define SAMEDIFF_SPARSE_EDGE_AGGREGATE_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

// Forward: out[rows, F] = row-wise reduction of edgeMsg[nnz, F].
SD_LIB_HIDDEN void csr_edge_aggregate(NDArray& rowPtr,    // [rows+1] int
                                       NDArray& edgeMsg,  // [nnz, F] float
                                       NDArray& out,      // [rows, F] float (output)
                                       sd::LongType rows,
                                       int mode);

// Backward: dEdgeMsg[nnz, F] from gradOut[rows, F].
SD_LIB_HIDDEN void csr_edge_aggregate_bp(NDArray& rowPtr,    // [rows+1] int
                                          NDArray& edgeMsg,  // [nnz, F] float (fwd input)
                                          NDArray& gradOut,  // [rows, F] float
                                          NDArray& dEdgeMsg, // [nnz, F] float (output)
                                          sd::LongType rows,
                                          int mode);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_EDGE_AGGREGATE_H
