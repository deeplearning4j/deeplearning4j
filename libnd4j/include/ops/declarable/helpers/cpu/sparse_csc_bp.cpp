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
// CPU backward-pass helpers for CSC sparse operations.
//
// csc_to_dense_bp : gather gradDense values at the CSC sparsity pattern → dCscValues.
// dense_to_csc_bp : scatter gradCscValues back into a zeroed dDense matrix.
//
// Both use column-major traversal matching the forward csc_to_dense / dense_to_csc
// ordering.  In a valid CSC format every (row, col) pair is unique, so neither
// function requires atomics.
//

#include <ops/declarable/helpers/sparse_csc_bp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ===========================================================================
// csc_to_dense_bp — CPU
//
// Forward csc_to_dense: for each column j, scatter cscValues[k] →
//   dense[cscRowIdx[k], j] for k in [cscColPtr[j], cscColPtr[j+1]).
//
// Backward is a pure gather from the same positions:
//   dCscValues[k] = gradDense[cscRowIdx[k], j]
// ===========================================================================

template <typename X, typename I>
static void cscToDenseBp_(
    NDArray& cscRowIdx, NDArray& cscColPtr,
    NDArray& gradDense, NDArray& dCscValues,
    LongType cols) {
  const auto* riBuf  = cscRowIdx.bufferAsT<I>();
  const auto* cpBuf  = cscColPtr.bufferAsT<I>();
  const auto* gBuf   = gradDense.bufferAsT<X>();
  auto*       dvBuf  = dCscValues.bufferAsT<X>();

  const LongType gStride0 = gradDense.stridesOf()[0];
  const LongType gStride1 = gradDense.stridesOf()[1];

  for (LongType j = 0; j < cols; ++j) {
    const I start = cpBuf[j];
    const I end   = cpBuf[j + 1];
    for (I k = start; k < end; ++k) {
      const LongType r = static_cast<LongType>(riBuf[k]);
      dvBuf[k] = gBuf[r * gStride0 + j * gStride1];
    }
  }
}

void csc_to_dense_bp(
    NDArray& cscRowIdx, NDArray& cscColPtr,
    NDArray& gradDense, NDArray& dCscValues,
    LongType /*rows*/, LongType cols) {
  NDArray::preparePrimaryUse({&dCscValues}, {&cscRowIdx, &cscColPtr, &gradDense});

  BUILD_DOUBLE_SELECTOR(gradDense.dataType(), cscRowIdx.dataType(), cscToDenseBp_,
      (cscRowIdx, cscColPtr, gradDense, dCscValues, cols),
      SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dCscValues}, {&cscRowIdx, &cscColPtr, &gradDense});
}

// ===========================================================================
// dense_to_csc_bp — CPU
//
// dDense is OUTPUT_NULLIFIED (pre-zeroed by the framework).
// Column-major scatter: for each column j, for each entry k in
//   [cscColPtr[j], cscColPtr[j+1]):
//     dDense[cscRowIdx[k], j] = gradCscValues[k]
//
// In a valid CSC format every (cscRowIdx[k], j) pair is unique → no atomics.
// ===========================================================================

template <typename X, typename I>
static void denseToCscBp_(
    NDArray& cscRowIdx, NDArray& cscColPtr,
    NDArray& gradCscValues, NDArray& dDense,
    LongType cols) {
  const auto* riBuf  = cscRowIdx.bufferAsT<I>();
  const auto* cpBuf  = cscColPtr.bufferAsT<I>();
  const auto* gvBuf  = gradCscValues.bufferAsT<X>();
  auto*       ddBuf  = dDense.bufferAsT<X>();

  const LongType ddStride0 = dDense.stridesOf()[0];
  const LongType ddStride1 = dDense.stridesOf()[1];

  for (LongType j = 0; j < cols; ++j) {
    const I start = cpBuf[j];
    const I end   = cpBuf[j + 1];
    for (I k = start; k < end; ++k) {
      const LongType r = static_cast<LongType>(riBuf[k]);
      ddBuf[r * ddStride0 + j * ddStride1] = gvBuf[k];
    }
  }
}

void dense_to_csc_bp(
    NDArray& cscRowIdx, NDArray& cscColPtr,
    NDArray& gradCscValues, NDArray& dDense,
    LongType /*rows*/, LongType cols) {
  NDArray::preparePrimaryUse({&dDense}, {&cscRowIdx, &cscColPtr, &gradCscValues});

  BUILD_DOUBLE_SELECTOR(gradCscValues.dataType(), cscRowIdx.dataType(), denseToCscBp_,
      (cscRowIdx, cscColPtr, gradCscValues, dDense, cols),
      SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dDense}, {&cscRowIdx, &cscColPtr, &gradCscValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
