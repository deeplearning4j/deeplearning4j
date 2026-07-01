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

#ifndef SAMEDIFF_SPARSE_CSC_H
#define SAMEDIFF_SPARSE_CSC_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Convert a CSR sparse matrix to CSC sparse format.
 *
 * CSC of A is equivalent to CSR of A^T with the roles of rows/cols swapped.
 *
 * @param csrValues   1D [nnz], non-zero values of A (float dtype)
 * @param csrColIdx   1D [nnz], column indices of CSR (INT32 or INT64)
 * @param csrRowPtr   1D [rows+1], row pointers of CSR (same INT dtype)
 * @param cscValues   1D [nnz], output non-zero values in CSC order
 * @param cscRowIdx   1D [nnz], INT32 — row indices in CSC order
 * @param cscColPtr   1D [cols+1], INT32 — column pointer array
 * @param rows        number of rows in the matrix
 * @param cols        number of columns in the matrix
 */
SD_LIB_HIDDEN void csr_to_csc(NDArray& csrValues, NDArray& csrColIdx, NDArray& csrRowPtr,
                               NDArray& cscValues, NDArray& cscRowIdx, NDArray& cscColPtr,
                               LongType rows, LongType cols);

/**
 * Scatter CSC values into a pre-zeroed dense output.
 *
 * @param cscValues   1D [nnz], non-zero values (float dtype)
 * @param cscRowIdx   1D [nnz], INT32 row indices
 * @param cscColPtr   1D [cols+1], INT32 column pointers
 * @param output      Dense output [rows, cols] — must be pre-zeroed (OUTPUT_NULLIFIED)
 */
SD_LIB_HIDDEN void csc_to_dense(NDArray& cscValues, NDArray& cscRowIdx, NDArray& cscColPtr,
                                 NDArray& output);

/**
 * Convert a dense matrix to CSC sparse representation (column-major scan order).
 *
 * DECLARE_SHAPE_FN calls dense->syncToHost() before this to make nnz known.
 *
 * @param input       Dense input [rows, cols]
 * @param cscValues   1D [nnz], output values (same dtype as input)
 * @param cscRowIdx   1D [nnz], INT32 output row indices
 * @param cscColPtr   1D [cols+1], INT32 output column pointers
 * @param threshold   Keep entries where |x| > threshold
 */
SD_LIB_HIDDEN void dense_to_csc(sd::LaunchContext* context, NDArray& input, NDArray& cscValues,
                                 NDArray& cscRowIdx, NDArray& cscColPtr, double threshold);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_CSC_H
