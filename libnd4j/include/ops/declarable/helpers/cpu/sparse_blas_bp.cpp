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

#include <ops/declarable/helpers/sparse_blas.h>
#include <ops/declarable/helpers/sparse_blas_bp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ===========================================================================
// csr_spmv_bp — CPU
//
// dAValues[k at (i,j)]:
//   transposeA=0: gradY[i] * x[j]
//   transposeA=1: gradY[j] * x[i]
//
// dx:
//   transposeA=0: A^T * gradY  → csr_spmv(values, colIdx, rowPtr, gradY, dx, transposeA=1)
//   transposeA=1: A   * gradY  → csr_spmv(values, colIdx, rowPtr, gradY, dx, transposeA=0)
// ===========================================================================

template <typename X, typename I>
static void csrSpMVBpValues_(NDArray& colIdx, NDArray& rowPtr, NDArray& x, NDArray& gradY,
                              NDArray& dAValues, sd::LongType rows, int transposeA) {
  const I nrows = static_cast<I>(rows);

  const auto* ciBuf = colIdx.bufferAsT<I>();
  const auto* rpBuf = rowPtr.bufferAsT<I>();
  const auto* xBuf  = x.bufferAsT<X>();
  const auto* gyBuf = gradY.bufferAsT<X>();
  auto*       daBuf = dAValues.bufferAsT<X>();

  const LongType xStride  = x.stridesOf()[0];
  const LongType gyStride = gradY.stridesOf()[0];

  for (I i = 0; i < nrows; ++i) {
    const I start = rpBuf[i];
    const I end   = rpBuf[i + 1];
    if (transposeA == 0) {
      // forward: y[i] = sum_k A[i,k]*x[k]
      // dA[k at (i,j)] = gradY[i] * x[j]
      const X gi = gyBuf[static_cast<LongType>(i) * gyStride];
      for (I k = start; k < end; ++k)
        daBuf[k] = gi * xBuf[static_cast<LongType>(ciBuf[k]) * xStride];
    } else {
      // forward: y[j] = sum_i A[i,j]*x[i]
      // dA[k at (i,j)] = gradY[j] * x[i]
      const X xi = xBuf[static_cast<LongType>(i) * xStride];
      for (I k = start; k < end; ++k)
        daBuf[k] = gyBuf[static_cast<LongType>(ciBuf[k]) * gyStride] * xi;
    }
  }
}

void csr_spmv_bp(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                  NDArray& x, NDArray& gradY,
                  NDArray& dAValues, NDArray& dx,
                  sd::LongType rows, sd::LongType cols, int transposeA) {
  // Step 1: dAValues via sampled outer-product (values not needed for the kernel)
  NDArray::preparePrimaryUse({&dAValues}, {&colIdx, &rowPtr, &x, &gradY});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSpMVBpValues_,
                        (colIdx, rowPtr, x, gradY, dAValues, rows, transposeA),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dAValues}, {&colIdx, &rowPtr, &x, &gradY});

  // Step 2: dx = op_{!transposeA}(A) * gradY  — reuse forward helper
  csr_spmv(values, colIdx, rowPtr, gradY, dx, rows, cols, 1 - transposeA);
}

// ===========================================================================
// csr_spmm_bp — CPU
//
// dAValues:
//   transposeA=0: SDDMM(rowPtr, colIdx, gradC, B)
//   transposeA=1: SDDMM(rowPtr, colIdx, B,     gradC)
//
// dB:
//   transposeA=0: A^T * gradC  → csr_spmm(... transposeA=1)
//   transposeA=1: A   * gradC  → csr_spmm(... transposeA=0)
// ===========================================================================

void csr_spmm_bp(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                  NDArray& B, NDArray& gradC,
                  NDArray& dAValues, NDArray& dB,
                  sd::LongType rows, sd::LongType cols, int transposeA) {
  // dAValues via SDDMM (reuses existing helper with its own prepare/register)
  if (transposeA == 0) {
    // C = A*B, dA[k at (i,j)] = sum_n gradC[i,n]*B[j,n] = sddmm(rowPtr, colIdx, gradC, B)
    sddmm(rowPtr, colIdx, gradC, B, dAValues, rows, cols);
  } else {
    // C = A^T*B, dA[k at (i,j)] = sum_n B[i,n]*gradC[j,n] = sddmm(rowPtr, colIdx, B, gradC)
    sddmm(rowPtr, colIdx, B, gradC, dAValues, rows, cols);
  }

  // dB = op_{!transposeA}(A) * gradC
  csr_spmm(values, colIdx, rowPtr, gradC, dB, rows, cols, 1 - transposeA);
}

// ===========================================================================
// sddmm_bp — CPU
//
// Forward: out[k at (i,j)] = D1[i,:] · D2[j,:]
//
// dD1[i,:] += gradOut[k] * D2[j,:]   for each k with row i
// dD2[j,:] += gradOut[k] * D1[i,:]   for each k with col j
//
// dD1 and dD2 must be pre-zeroed (OUTPUT_NULLIFIED in the calling op).
// ===========================================================================

template <typename X, typename I>
static void sddmmBp_(NDArray& rowPtr, NDArray& colIdx,
                     NDArray& D1, NDArray& D2, NDArray& gradOut,
                     NDArray& dD1, NDArray& dD2,
                     sd::LongType rows) {
  const I nrows = static_cast<I>(rows);
  const LongType p = D1.sizeAt(1);

  const auto* rpBuf  = rowPtr.bufferAsT<I>();
  const auto* ciBuf  = colIdx.bufferAsT<I>();
  const auto* d1Buf  = D1.bufferAsT<X>();
  const auto* d2Buf  = D2.bufferAsT<X>();
  const auto* gBuf   = gradOut.bufferAsT<X>();
  auto*       dd1Buf = dD1.bufferAsT<X>();
  auto*       dd2Buf = dD2.bufferAsT<X>();

  const LongType d1Stride0  = D1.stridesOf()[0];
  const LongType d1Stride1  = D1.stridesOf()[1];
  const LongType d2Stride0  = D2.stridesOf()[0];
  const LongType d2Stride1  = D2.stridesOf()[1];
  const LongType dd1Stride0 = dD1.stridesOf()[0];
  const LongType dd1Stride1 = dD1.stridesOf()[1];
  const LongType dd2Stride0 = dD2.stridesOf()[0];
  const LongType dd2Stride1 = dD2.stridesOf()[1];

  for (I i = 0; i < nrows; ++i) {
    const I start = rpBuf[i];
    const I end   = rpBuf[i + 1];
    for (I k = start; k < end; ++k) {
      const LongType j = static_cast<LongType>(ciBuf[k]);
      const X g = gBuf[k];
      for (LongType l = 0; l < p; ++l) {
        dd1Buf[static_cast<LongType>(i) * dd1Stride0 + l * dd1Stride1]
            += g * d2Buf[j * d2Stride0 + l * d2Stride1];
        dd2Buf[j * dd2Stride0 + l * dd2Stride1]
            += g * d1Buf[static_cast<LongType>(i) * d1Stride0 + l * d1Stride1];
      }
    }
  }
}

void sddmm_bp(NDArray& rowPtr, NDArray& colIdx,
              NDArray& D1, NDArray& D2, NDArray& gradOut,
              NDArray& dD1, NDArray& dD2,
              sd::LongType rows, sd::LongType cols) {
  NDArray::preparePrimaryUse({&dD1, &dD2}, {&rowPtr, &colIdx, &D1, &D2, &gradOut});

  BUILD_DOUBLE_SELECTOR(D1.dataType(), colIdx.dataType(), sddmmBp_,
                        (rowPtr, colIdx, D1, D2, gradOut, dD1, dD2, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dD1, &dD2}, {&rowPtr, &colIdx, &D1, &D2, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
