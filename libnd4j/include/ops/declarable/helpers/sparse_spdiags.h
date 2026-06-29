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

#ifndef SAMEDIFF_SPARSE_SPDIAGS_H
#define SAMEDIFF_SPARSE_SPDIAGS_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Build the CSR representation of an n×n diagonal matrix from a float vector.
 *
 * The diagonal matrix has diag[i] at position (i, i) for i in [0, n).
 * nnz = n (never data-dependent for a diagonal).
 *
 * Outputs are sized by DECLARE_SHAPE_FN before this function is called:
 *   values [n]   — same float dtype as diag, values[i] = diag[i]
 *   colIdx [n]   — INT32, colIdx[i] = i
 *   rowPtr [n+1] — INT32, rowPtr[i] = i  (rowPtr[n] = n)
 *
 * @param context  Launch context (selects CUDA stream on GPU path)
 * @param diag     Input 1D vector [n], floating dtype
 * @param values   Output values array [n], same float dtype as diag
 * @param colIdx   Output column-index array [n], INT32
 * @param rowPtr   Output row-pointer array [n+1], INT32
 * @param n        Diagonal size (and matrix dimension)
 */
SD_LIB_HIDDEN void spdiags(sd::LaunchContext* context, NDArray& diag, NDArray& values,
                            NDArray& colIdx, NDArray& rowPtr, sd::LongType n);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_SPDIAGS_H
