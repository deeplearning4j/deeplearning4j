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

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_blas.h>
#include <ops/declarable/helpers/sparse_blas_bp.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

// ===========================================================================
// Utility: binary-search for the row owning CSR nonzero k.
// rowPtr has (rows+1) entries; returns the row index i s.t.
//   rowPtr[i] <= k < rowPtr[i+1].
// Called from device kernels — SD_DEVICE ensures inlining.
// ===========================================================================

template <typename I>
static SD_INLINE SD_DEVICE LongType findRow(const I* rowPtr, LongType rows, LongType k) {
  LongType lo = 0, hi = rows - 1, row = 0;
  while (lo <= hi) {
    LongType mid = (lo + hi) / 2;
    if (static_cast<LongType>(rowPtr[mid]) <= k &&
        k < static_cast<LongType>(rowPtr[mid + 1])) {
      row = mid;
      break;
    } else if (static_cast<LongType>(rowPtr[mid]) > k) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  return row;
}

// ===========================================================================
// csr_spmv_bp — CUDA kernels
// ===========================================================================

// dAValues[k at (i,j)]:
//   transposeA=0: gradY[i] * x[j]
//   transposeA=1: gradY[j] * x[i]
template <typename X, typename I>
static SD_KERNEL void csrSpMVBpValuesKernel(const I* rowPtr, const I* colIdx,
                                             const X* x, const X* gradY, X* dAValues,
                                             LongType rows, LongType nnz,
                                             LongType xStride0, LongType gyStride0,
                                             int transposeA) {
  const LongType k = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= nnz) return;

  const LongType row = findRow(rowPtr, rows, k);
  const LongType j   = static_cast<LongType>(colIdx[k]);

  if (transposeA == 0) {
    dAValues[k] = gradY[row * gyStride0] * x[j * xStride0];
  } else {
    dAValues[k] = gradY[j * gyStride0] * x[row * xStride0];
  }
}

template <typename X, typename I>
static void csrSpMVBpCuda_(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                            NDArray& x, NDArray& gradY,
                            NDArray& dAValues, NDArray& dx,
                            sd::LongType rows, sd::LongType cols, int transposeA) {
  auto stream = dAValues.getContext()->getCudaStream();

  const I* ciBuf  = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpBuf  = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const X* xBuf   = reinterpret_cast<const X*>(x.specialBuffer());
  const X* gyBuf  = reinterpret_cast<const X*>(gradY.specialBuffer());
  X*       daBuf  = reinterpret_cast<X*>(dAValues.specialBuffer());

  const LongType nnz      = colIdx.lengthOf();
  const LongType xStride0 = x.stridesOf()[0];
  const LongType gyStride0 = gradY.stridesOf()[0];

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);

  csrSpMVBpValuesKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      rpBuf, ciBuf, xBuf, gyBuf, daBuf,
      rows, nnz, xStride0, gyStride0, transposeA);
}

void csr_spmv_bp(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                  NDArray& x, NDArray& gradY,
                  NDArray& dAValues, NDArray& dx,
                  sd::LongType rows, sd::LongType cols, int transposeA) {
  // Step 1: dAValues — custom kernel
  NDArray::prepareSpecialUse({&dAValues}, {&colIdx, &rowPtr, &x, &gradY});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSpMVBpCuda_,
                        (values, colIdx, rowPtr, x, gradY, dAValues, dx, rows, cols, transposeA),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dAValues}, {&colIdx, &rowPtr, &x, &gradY});

  // Step 2: dx = op_{!transposeA}(A) * gradY — reuse forward helper
  csr_spmv(values, colIdx, rowPtr, gradY, dx, rows, cols, 1 - transposeA);
}

// ===========================================================================
// csr_spmm_bp — CUDA
//
// dAValues: computed by the SDDMM forward helper (runs on device)
// dB:       computed by the SpMM forward helper (runs on device)
// Both helpers manage their own prepareSpecialUse / registerSpecialUse.
// ===========================================================================

void csr_spmm_bp(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                  NDArray& B, NDArray& gradC,
                  NDArray& dAValues, NDArray& dB,
                  sd::LongType rows, sd::LongType cols, int transposeA) {
  // dAValues = SDDMM at A's pattern of (gradC * B^T) or (B * gradC^T)
  if (transposeA == 0) {
    sddmm(rowPtr, colIdx, gradC, B, dAValues, rows, cols);
  } else {
    sddmm(rowPtr, colIdx, B, gradC, dAValues, rows, cols);
  }

  // dB = op_{!transposeA}(A) * gradC
  csr_spmm(values, colIdx, rowPtr, gradC, dB, rows, cols, 1 - transposeA);
}

// ===========================================================================
// sddmm_bp — CUDA kernels
//
// Forward: out[k at (i,j)] = D1[i,:] · D2[j,:]
//
// dD1[i, l] += gradOut[k] * D2[j, l]   for each NNZ k with row i
// dD2[j, l] += gradOut[k] * D1[i, l]   for each NNZ k with col j
//
// Uses sd_atomicAdd because multiple NNZs can share the same row/col.
// dD1 and dD2 are pre-zeroed (OUTPUT_NULLIFIED) before the kernel runs.
// ===========================================================================

template <typename X, typename I>
static SD_KERNEL void sddmmBpD1Kernel(const I* rowPtr, const I* colIdx,
                                       const X* D2, const X* gradOut, X* dD1,
                                       LongType rows, LongType nnz, LongType p,
                                       LongType d2Stride0, LongType d2Stride1,
                                       LongType dd1Stride0, LongType dd1Stride1,
                                       LongType goStride0) {
  const LongType k = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= nnz) return;

  const LongType row = findRow(rowPtr, rows, k);
  const LongType j   = static_cast<LongType>(colIdx[k]);
  const X        g   = gradOut[k * goStride0];

  for (LongType l = 0; l < p; ++l) {
    sd::math::atomics::sd_atomicAdd(
        &dD1[row * dd1Stride0 + l * dd1Stride1],
        static_cast<X>(g * D2[j * d2Stride0 + l * d2Stride1]));
  }
}

template <typename X, typename I>
static SD_KERNEL void sddmmBpD2Kernel(const I* rowPtr, const I* colIdx,
                                       const X* D1, const X* gradOut, X* dD2,
                                       LongType rows, LongType nnz, LongType p,
                                       LongType d1Stride0, LongType d1Stride1,
                                       LongType dd2Stride0, LongType dd2Stride1,
                                       LongType goStride0) {
  const LongType k = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= nnz) return;

  const LongType row = findRow(rowPtr, rows, k);
  const LongType j   = static_cast<LongType>(colIdx[k]);
  const X        g   = gradOut[k * goStride0];

  for (LongType l = 0; l < p; ++l) {
    sd::math::atomics::sd_atomicAdd(
        &dD2[j * dd2Stride0 + l * dd2Stride1],
        static_cast<X>(g * D1[row * d1Stride0 + l * d1Stride1]));
  }
}

template <typename X, typename I>
static void sddmmBpCuda_(NDArray& rowPtr, NDArray& colIdx,
                          NDArray& D1, NDArray& D2, NDArray& gradOut,
                          NDArray& dD1, NDArray& dD2,
                          sd::LongType rows, sd::LongType cols) {
  auto stream = dD1.getContext()->getCudaStream();

  const I* rpBuf  = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const I* ciBuf  = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const X* d1Buf  = reinterpret_cast<const X*>(D1.specialBuffer());
  const X* d2Buf  = reinterpret_cast<const X*>(D2.specialBuffer());
  const X* gBuf   = reinterpret_cast<const X*>(gradOut.specialBuffer());
  X*       dd1Buf = reinterpret_cast<X*>(dD1.specialBuffer());
  X*       dd2Buf = reinterpret_cast<X*>(dD2.specialBuffer());

  const LongType nnz = colIdx.lengthOf();
  const LongType p   = D1.sizeAt(1);

  const LongType d1Stride0  = D1.stridesOf()[0];
  const LongType d1Stride1  = D1.stridesOf()[1];
  const LongType d2Stride0  = D2.stridesOf()[0];
  const LongType d2Stride1  = D2.stridesOf()[1];
  const LongType dd1Stride0 = dD1.stridesOf()[0];
  const LongType dd1Stride1 = dD1.stridesOf()[1];
  const LongType dd2Stride0 = dD2.stridesOf()[0];
  const LongType dd2Stride1 = dD2.stridesOf()[1];
  // gradOut is 1D; stride may be non-unit for non-contiguous views
  const LongType goStride0  = gradOut.stridesOf()[0];

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);

  // Two separate kernel launches: dD1 and dD2
  sddmmBpD1Kernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      rpBuf, ciBuf, d2Buf, gBuf, dd1Buf,
      rows, nnz, p,
      d2Stride0, d2Stride1, dd1Stride0, dd1Stride1, goStride0);

  sddmmBpD2Kernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      rpBuf, ciBuf, d1Buf, gBuf, dd2Buf,
      rows, nnz, p,
      d1Stride0, d1Stride1, dd2Stride0, dd2Stride1, goStride0);
}

void sddmm_bp(NDArray& rowPtr, NDArray& colIdx,
              NDArray& D1, NDArray& D2, NDArray& gradOut,
              NDArray& dD1, NDArray& dD2,
              sd::LongType rows, sd::LongType cols) {
  NDArray::prepareSpecialUse({&dD1, &dD2}, {&rowPtr, &colIdx, &D1, &D2, &gradOut});

  BUILD_DOUBLE_SELECTOR(D1.dataType(), colIdx.dataType(), sddmmBpCuda_,
                        (rowPtr, colIdx, D1, D2, gradOut, dD1, dD2, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dD1, &dD2}, {&rowPtr, &colIdx, &D1, &D2, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
