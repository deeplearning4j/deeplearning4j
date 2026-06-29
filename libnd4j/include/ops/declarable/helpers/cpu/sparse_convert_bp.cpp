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
// CPU backward-pass helpers for csr_to_dense, dense_to_csr, and csr_to_csc.
//
// csr_to_dense_bp: gather gradDense values at the CSR sparsity pattern → dValues.
// dense_to_csr_bp: scatter gradValues back into a zeroed dDense matrix.
// csr_to_csc_bp:   inverse-permute gradCscValues via a second csr_to_csc call
//                  (treating the CSC arrays as the CSR of A^T, then transposing
//                  back to CSR(A) order).  Two temporary INT32 arrays are
//                  allocated on the host and discarded; no external memory
//                  management is required.
//

#include <array/NDArrayFactory.h>
#include <memory>
#include <ops/declarable/helpers/sparse_convert_bp.h>
#include <ops/declarable/helpers/sparse_csc.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------
// csr_to_dense_bp — CPU
//
// Row-major scan (same order as the forward dense_to_csr scatter):
//   for each row r, for each entry k in [rowPtr[r], rowPtr[r+1]):
//     dValues[k] = gradDense[r, colIdx[k]]   (stride-aware read)
// ---------------------------------------------------------------------------

template <typename X, typename I>
static void csrToDenseBp_(NDArray& colIdx, NDArray& rowPtr, NDArray& gradDense,
                           NDArray& dValues, LongType rows) {
  const auto* ciBuf  = colIdx.bufferAsT<I>();
  const auto* rpBuf  = rowPtr.bufferAsT<I>();
  const auto* gBuf   = gradDense.bufferAsT<X>();
  auto*       dvBuf  = dValues.bufferAsT<X>();

  const auto* gStride = gradDense.stridesOf();  // honour arbitrary strides

  for (LongType r = 0; r < rows; ++r) {
    const I start = rpBuf[r];
    const I end   = rpBuf[r + 1];
    for (I k = start; k < end; ++k) {
      const I c   = ciBuf[k];
      dvBuf[k] = gBuf[static_cast<LongType>(r) * gStride[0] +
                       static_cast<LongType>(c) * gStride[1]];
    }
  }
}

void csr_to_dense_bp(NDArray& colIdx, NDArray& rowPtr, NDArray& gradDense,
                     NDArray& dValues, LongType rows, LongType /*cols*/) {
  NDArray::preparePrimaryUse({&dValues}, {&colIdx, &rowPtr, &gradDense});

  BUILD_DOUBLE_SELECTOR(gradDense.dataType(), colIdx.dataType(), csrToDenseBp_,
                        (colIdx, rowPtr, gradDense, dValues, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dValues}, {&colIdx, &rowPtr, &gradDense});
}

// ---------------------------------------------------------------------------
// dense_to_csr_bp — CPU
//
// dDense is OUTPUT_NULLIFIED (pre-zeroed by the framework).
// Row-major scan of the CSR pattern:
//   for each row r, for each entry k in [rowPtr[r], rowPtr[r+1]):
//     dDense[r, colIdx[k]] = gradValues[k]   (stride-aware write)
//
// In a valid CSR matrix every (r, colIdx[k]) pair is unique, so no atomics.
// ---------------------------------------------------------------------------

template <typename X, typename I>
static void denseToCsrBp_(NDArray& colIdx, NDArray& rowPtr, NDArray& gradValues,
                           NDArray& dDense, LongType rows) {
  const auto* ciBuf  = colIdx.bufferAsT<I>();
  const auto* rpBuf  = rowPtr.bufferAsT<I>();
  const auto* gvBuf  = gradValues.bufferAsT<X>();
  auto*       ddBuf  = dDense.bufferAsT<X>();

  const auto* ddStride = dDense.stridesOf();

  for (LongType r = 0; r < rows; ++r) {
    const I start = rpBuf[r];
    const I end   = rpBuf[r + 1];
    for (I k = start; k < end; ++k) {
      const I c = ciBuf[k];
      ddBuf[static_cast<LongType>(r) * ddStride[0] +
            static_cast<LongType>(c) * ddStride[1]] = gvBuf[k];
    }
  }
}

void dense_to_csr_bp(NDArray& colIdx, NDArray& rowPtr, NDArray& gradValues,
                     NDArray& dDense, LongType rows, LongType /*cols*/) {
  NDArray::preparePrimaryUse({&dDense}, {&colIdx, &rowPtr, &gradValues});

  BUILD_DOUBLE_SELECTOR(gradValues.dataType(), colIdx.dataType(), denseToCsrBp_,
                        (colIdx, rowPtr, gradValues, dDense, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dDense}, {&colIdx, &rowPtr, &gradValues});
}

// ---------------------------------------------------------------------------
// csr_to_csc_bp — CPU
//
// The forward csr_to_csc applies a value-permutation P: CSR(A) → CSC(A).
// The backward pass applies P⁻¹.
//
// Key insight: CSC(A) viewed as a matrix of shape [cols, rows] is exactly the
// CSR of A^T.  Transposing that CSR back via a second csr_to_csc call with
// dimensions (cols, rows) yields values in the original CSR(A) order.
//
// Concretely:
//   csr_to_csc(gradCscValues, cscRowIdx, cscColPtr,
//              dAValues, tempRowIdx, tempColPtr,
//              rows_new=cols, cols_new=rows)
//
// The two temporary INT32 arrays (tempRowIdx[nnz], tempColPtr[rows+1]) mirror
// csrColIdx and csrRowPtr of the original A; they are allocated on the host
// stack/heap and discarded after the call.
// ---------------------------------------------------------------------------

void csr_to_csc_bp(NDArray& cscRowIdx, NDArray& cscColPtr,
                   NDArray& gradCscValues, NDArray& dAValues,
                   LongType rows, LongType cols) {
  const LongType nnz = gradCscValues.lengthOf();

  // Temporaries that receive the re-transposed index arrays (not needed by caller).
  // NDArrayFactory::create returns NDArray* — own it via unique_ptr (sparse-helper idiom).
  std::unique_ptr<NDArray> tempRowIdx(NDArrayFactory::create('c', {nnz},      sd::DataType::INT32));
  std::unique_ptr<NDArray> tempColPtr(NDArrayFactory::create('c', {rows + 1}, sd::DataType::INT32));

  // Re-use the forward helper: treat (gradCscValues, cscRowIdx, cscColPtr) as
  // the CSR representation of A^T (shape [cols, rows]), then transpose back.
  // The output dAValues receives values in the original CSR(A) row-major order.
  csr_to_csc(gradCscValues, cscRowIdx, cscColPtr,
              dAValues,     *tempRowIdx, *tempColPtr,
              cols, rows);
  // tempRowIdx and tempColPtr are discarded; dAValues is the result.
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
