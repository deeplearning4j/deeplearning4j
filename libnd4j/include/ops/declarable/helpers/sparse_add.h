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

#ifndef SAMEDIFF_SPARSE_ADD_H
#define SAMEDIFF_SPARSE_ADD_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * CSR sparse-sparse elementwise addition: C = A + B
 *
 * A and B are [m, n] in CSR format with the same logical shape.
 * C is produced in CSR format [m, n] whose sparsity pattern is the
 * column-set union of A and B per row; overlapping entries are summed.
 *
 * The output arrays (cValues, cColIdx, cRowPtr) must be pre-allocated to the
 * exact sizes computed by DECLARE_SHAPE_FN (symbolic union pass).
 *
 * @param aValues  1D [annz]  float, non-zero values of A
 * @param aColIdx  1D [annz]  int,   column indices of A  (sorted per row)
 * @param aRowPtr  1D [m+1]   int,   row pointers of A
 * @param bValues  1D [bnnz]  float, non-zero values of B
 * @param bColIdx  1D [bnnz]  int,   column indices of B  (sorted per row)
 * @param bRowPtr  1D [m+1]   int,   row pointers of B
 * @param cValues  1D [cnnz]  float output, non-zero values of C
 * @param cColIdx  1D [cnnz]  INT32 output, column indices of C
 * @param cRowPtr  1D [m+1]   INT32 output, row pointers of C
 * @param m        Number of rows (shared by A, B, C)
 * @param n        Number of columns (shared by A, B, C)
 */
SD_LIB_HIDDEN void csr_add(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                            NDArray& bValues, NDArray& bColIdx, NDArray& bRowPtr,
                            NDArray& cValues, NDArray& cColIdx, NDArray& cRowPtr,
                            sd::LongType m, sd::LongType n);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_ADD_H
