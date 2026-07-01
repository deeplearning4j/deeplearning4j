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

#ifndef SAMEDIFF_SPARSE_CSC_BP_H
#define SAMEDIFF_SPARSE_CSC_BP_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Backward pass for csc_to_dense.
 *
 * The forward scatter places CSC entry k (row = cscRowIdx[k], column c found
 * via cscColPtr) into dense[row, col].  The gradient flows back as a gather:
 *   dCscValues[k] = gradDense[cscRowIdx[k], col]
 * where col is determined by binary search in cscColPtr.
 *
 * Pure gather — one dense element per sparse entry, no atomics.
 *
 * @param cscRowIdx  1D [nnz]      INT — CSC row indices
 * @param cscColPtr  1D [cols+1]   INT — CSC column pointers
 * @param gradDense  2D [rows,cols] float — upstream gradient w.r.t. dense output
 * @param dCscValues 1D [nnz]      float — OUTPUT gradient w.r.t. CSC values
 * @param rows       number of rows
 * @param cols       number of columns
 */
SD_LIB_HIDDEN void csc_to_dense_bp(
    NDArray& cscRowIdx, NDArray& cscColPtr,
    NDArray& gradDense, NDArray& dCscValues,
    LongType rows, LongType cols);

/**
 * Backward pass for dense_to_csc.
 *
 * dDense is pre-zeroed (OUTPUT_NULLIFIED). For each CSC entry k:
 *   dDense[cscRowIdx[k], col] = gradCscValues[k]
 * where col is the column containing entry k (binary search in cscColPtr).
 * In a valid CSC pattern every (row, col) pair is unique, so this scatter
 * requires no atomics.
 *
 * @param cscRowIdx      1D [nnz]      INT — CSC row indices (forward output[1])
 * @param cscColPtr      1D [cols+1]   INT — CSC column pointers (forward output[2])
 * @param gradCscValues  1D [nnz]      float — upstream gradient w.r.t. CSC values
 * @param dDense         2D [rows,cols] float — OUTPUT gradient w.r.t. dense input, pre-zeroed
 * @param rows           number of rows
 * @param cols           number of columns
 */
SD_LIB_HIDDEN void dense_to_csc_bp(
    NDArray& cscRowIdx, NDArray& cscColPtr,
    NDArray& gradCscValues, NDArray& dDense,
    LongType rows, LongType cols);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_CSC_BP_H
