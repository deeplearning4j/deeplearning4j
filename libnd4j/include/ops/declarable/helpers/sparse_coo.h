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

#ifndef SAMEDIFF_SPARSE_COO_H
#define SAMEDIFF_SPARSE_COO_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Converts a dense 2-D matrix to COO sparse representation.
 *
 * Scans the dense matrix in row-major order; collects every entry whose
 * absolute value exceeds threshold.
 *
 * @param context   Launch context (used by CUDA path for stream selection)
 * @param dense     Dense input array [rows, cols], floating dtype.
 *                  Host primary buffer must be valid before this call
 *                  (DECLARE_SHAPE_FN calls syncToHost() first).
 * @param indices   Output indices array [nnz, 2], INT64.
 *                  indices[k, 0] = row index of the k-th non-zero.
 *                  indices[k, 1] = col index of the k-th non-zero.
 * @param values    Output values array [nnz], same float dtype as dense.
 * @param threshold Keep entries where |x| > threshold.
 */
SD_LIB_HIDDEN void denseToCoo(sd::LaunchContext* context, NDArray* dense,
                               NDArray* indices, NDArray* values, double threshold);

/**
 * Converts COO sparse representation to CSR sparse representation.
 *
 * Entries are sorted by (row, then col) before building the CSR arrays so
 * that the resulting CSR matrix has its non-zeros in the canonical order
 * expected by BLAS/cuSPARSE routines.
 *
 * @param context      Launch context (used by CUDA path for stream selection)
 * @param cooIndices   COO indices array [nnz, 2], INT.
 *                     cooIndices[k, 0] = row index of the k-th entry.
 *                     cooIndices[k, 1] = col index of the k-th entry.
 * @param cooValues    COO values array [nnz], float.
 * @param csrValues    Output CSR values array [nnz], same float dtype as cooValues.
 * @param colIdx       Output column index array [nnz], INT32.
 * @param rowPtr       Output row pointer array [rows+1], INT32.
 * @param rows         Number of rows in the sparse matrix.
 */
SD_LIB_HIDDEN void cooToCsr(sd::LaunchContext* context, NDArray* cooIndices, NDArray* cooValues,
                             NDArray* csrValues, NDArray* colIdx, NDArray* rowPtr,
                             sd::LongType rows);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_COO_H
