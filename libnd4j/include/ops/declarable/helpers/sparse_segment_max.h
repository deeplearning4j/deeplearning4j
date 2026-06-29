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
// Helpers for csr_segment_max / csr_segment_max_bp.
//
// csr_segment_max — GraphSAGE-style max neighbor aggregation.
//   For each row i and feature feat:
//     out[i, feat] = max over k in [rowPtr[i], rowPtr[i+1]) of X[colIdx[k], feat]
//   Empty row → 0.
//
// csr_segment_max_bp — backward pass.
//   For each (i, feat), locate argmax k*; then
//     dX[colIdx[k*], feat] += gradOut[i, feat]
//   Multiple rows may target the same (node, feat) → atomicAdd in CUDA.
//

#ifndef SAMEDIFF_SPARSE_SEGMENT_MAX_H
#define SAMEDIFF_SPARSE_SEGMENT_MAX_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

// Forward: out[rows, f] = row-wise max of neighbor features xArr[n, f].
SD_LIB_HIDDEN void csr_segment_max(NDArray& colIdx,   // [nnz]   int
                                     NDArray& rowPtr,   // [rows+1] int
                                     NDArray& xArr,     // [n, f]  float
                                     NDArray& out,      // [rows, f] float (output)
                                     sd::LongType rows);

// Backward: dX[n, f] accumulates gradients from the argmax positions.
SD_LIB_HIDDEN void csr_segment_max_bp(NDArray& colIdx,   // [nnz]     int
                                        NDArray& rowPtr,   // [rows+1]  int
                                        NDArray& xArr,     // [n, f]    float
                                        NDArray& gradOut,  // [rows, f] float
                                        NDArray& dX,       // [n, f]    float (output)
                                        sd::LongType rows);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_SEGMENT_MAX_H
