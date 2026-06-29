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
// Helpers for csr_row_softmax / csr_row_softmax_bp.
//
// csr_row_softmax  — numerically-stable per-row softmax over the stored
//                    non-zeros of a CSR matrix (GAT edge-softmax).
// csr_row_softmax_bp — backward pass; per-row Jacobian-vector product of
//                      the softmax.
//

#ifndef SAMEDIFF_SPARSE_ROW_SOFTMAX_H
#define SAMEDIFF_SPARSE_ROW_SOFTMAX_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

// Forward: alpha[k] = softmax over row-k's window, numerically stable.
SD_LIB_HIDDEN void csr_row_softmax(NDArray& values,   // [nnz]    float
                                    NDArray& rowPtr,   // [rows+1] int
                                    NDArray& alpha,    // [nnz]    float (output)
                                    sd::LongType rows);

// Backward: dValues[k] = alpha[k] * (gradOut[k] - dot_i)
//           where dot_i = sum_{k in row i} alpha[k]*gradOut[k]
SD_LIB_HIDDEN void csr_row_softmax_bp(NDArray& alpha,    // [nnz]    float (fwd output)
                                       NDArray& rowPtr,   // [rows+1] int
                                       NDArray& gradOut,  // [nnz]    float
                                       NDArray& dValues,  // [nnz]    float (output)
                                       sd::LongType rows);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_ROW_SOFTMAX_H
