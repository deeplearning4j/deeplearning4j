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
// GraphBLAS semiring SpMV and SpMM on CUDA using custom kernels.
//
// cuSPARSE is intentionally NOT used: its generic SpMV/SpMM API only supports
// the PLUS_TIMES semiring (standard linear algebra).  All five semirings here
// (PLUS_TIMES, MIN_PLUS, MAX_PLUS, OR_AND, MIN_TIMES) are implemented with
// hand-written kernels so the semiring abstraction is exact.
//
// Supported semiring codes (int semiring parameter):
//   0 = PLUS_TIMES  (add=+,  mul=*,   identity=0)
//   1 = MIN_PLUS    (add=min, mul=+,   identity=+INF)
//   2 = MAX_PLUS    (add=max, mul=+,   identity=-INF)
//   3 = OR_AND      (add=||,  mul=&&,  identity=0)
//   4 = MIN_TIMES   (add=min, mul=*,   identity=+INF)
//

#include <cuda_runtime.h>
#include <ops/declarable/helpers/sparse_blas_semiring.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>
#include <math/templatemath.h>

namespace sd {
namespace ops {
namespace helpers {

// ===========================================================================
// Semiring functor structs
//
// Each struct exposes three SD_DEVICE static methods:
//   identity() — the additive identity (fill value before accumulation)
//   add(a, b)  — the semiring addition operator
//   mul(a, b)  — the semiring multiplication operator
// ===========================================================================

template <typename X>
struct PlusTimesSR {
  SD_DEVICE static X identity() { return static_cast<X>(0); }
  SD_DEVICE static X add(X a, X b) { return a + b; }
  SD_DEVICE static X mul(X a, X b) { return a * b; }
};

template <typename X>
struct MinPlusSR {
  // Additive identity for min is +INF
  SD_DEVICE static X identity() { return static_cast<X>(1.0f / 0.0f); }
  SD_DEVICE static X add(X a, X b) { return a < b ? a : b; }
  // Multiplication in the tropical (min,+) semiring is plain addition
  SD_DEVICE static X mul(X a, X b) { return a + b; }
};

template <typename X>
struct MaxPlusSR {
  // Additive identity for max is -INF
  SD_DEVICE static X identity() { return -static_cast<X>(1.0f / 0.0f); }
  SD_DEVICE static X add(X a, X b) { return a > b ? a : b; }
  // Multiplication in the (max,+) semiring is plain addition
  SD_DEVICE static X mul(X a, X b) { return a + b; }
};

template <typename X>
struct OrAndSR {
  // Additive identity for logical OR is false (0)
  SD_DEVICE static X identity() { return static_cast<X>(0); }
  SD_DEVICE static X add(X a, X b) {
    return ((a != static_cast<X>(0)) || (b != static_cast<X>(0)))
               ? static_cast<X>(1)
               : static_cast<X>(0);
  }
  SD_DEVICE static X mul(X a, X b) {
    return ((a != static_cast<X>(0)) && (b != static_cast<X>(0)))
               ? static_cast<X>(1)
               : static_cast<X>(0);
  }
};

template <typename X>
struct MinTimesSR {
  // Additive identity for min is +INF
  SD_DEVICE static X identity() { return static_cast<X>(1.0f / 0.0f); }
  SD_DEVICE static X add(X a, X b) { return a < b ? a : b; }
  SD_DEVICE static X mul(X a, X b) { return a * b; }
};

// ===========================================================================
// SpMV kernel
//
// One thread per output row.  Thread r computes:
//   y[r] = SR::add over k in [rowPtr[r], rowPtr[r+1]) of SR::mul(values[k], x[colIdx[k]])
//
// The accumulator is initialised to SR::identity() so empty rows produce the
// correct result without a separate zeroing pass.
// ===========================================================================

template <typename X, typename I, template <typename> class SR>
static SD_KERNEL void csrSpMVSemiringKernel(
    const X* values, const I* colIdx, const I* rowPtr,
    const X* x, X* y,
    LongType rows, LongType xStride0, LongType yStride0) {
  const LongType r = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;

  const I start = rowPtr[r];
  const I end   = rowPtr[r + 1];

  X acc = SR<X>::identity();
  for (I k = start; k < end; ++k) {
    const X v = values[k];
    const X xi = x[static_cast<LongType>(colIdx[k]) * xStride0];
    acc = SR<X>::add(acc, SR<X>::mul(v, xi));
  }

  y[r * yStride0] = acc;
}

// ===========================================================================
// SpMM kernel
//
// 2D grid: (row, column-of-B).  Thread (r, j) computes:
//   C[r,j] = SR::add over k in row r of SR::mul(values[k], B[colIdx[k], j])
// ===========================================================================

template <typename X, typename I, template <typename> class SR>
static SD_KERNEL void csrSpMMSemiringKernel(
    const X* values, const I* colIdx, const I* rowPtr,
    const X* B, X* C,
    LongType rows, LongType n,
    LongType bStride0, LongType bStride1,
    LongType cStride0, LongType cStride1) {
  const LongType r = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType j = static_cast<LongType>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (r >= rows || j >= n) return;

  const I start = rowPtr[r];
  const I end   = rowPtr[r + 1];

  X acc = SR<X>::identity();
  for (I k = start; k < end; ++k) {
    const X v  = values[k];
    const X bv = B[static_cast<LongType>(colIdx[k]) * bStride0 + j * bStride1];
    acc = SR<X>::add(acc, SR<X>::mul(v, bv));
  }

  C[r * cStride0 + j * cStride1] = acc;
}

// ===========================================================================
// Type-dispatched SpMV — one instantiation per (X, I) pair
// ===========================================================================

template <typename X, typename I>
static void csrSpMVSemiringCuda_(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                                  NDArray& x, NDArray& y,
                                  sd::LongType rows, sd::LongType /*cols*/, int semiring) {
  auto stream = y.getContext()->getCudaStream();

  const X* vBuf  = reinterpret_cast<const X*>(values.specialBuffer());
  const I* ciBuf = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpBuf = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const X* xBuf  = reinterpret_cast<const X*>(x.specialBuffer());
  X*       yBuf  = reinterpret_cast<X*>(y.specialBuffer());

  const LongType xStride0 = x.stridesOf()[0];
  const LongType yStride0 = y.stridesOf()[0];

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((rows + blockSize - 1) / blockSize);

  switch (semiring) {
    case 0:
      csrSpMVSemiringKernel<X, I, PlusTimesSR><<<gridSize, blockSize, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, xBuf, yBuf, rows, xStride0, yStride0);
      break;
    case 1:
      csrSpMVSemiringKernel<X, I, MinPlusSR><<<gridSize, blockSize, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, xBuf, yBuf, rows, xStride0, yStride0);
      break;
    case 2:
      csrSpMVSemiringKernel<X, I, MaxPlusSR><<<gridSize, blockSize, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, xBuf, yBuf, rows, xStride0, yStride0);
      break;
    case 3:
      csrSpMVSemiringKernel<X, I, OrAndSR><<<gridSize, blockSize, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, xBuf, yBuf, rows, xStride0, yStride0);
      break;
    case 4:
      csrSpMVSemiringKernel<X, I, MinTimesSR><<<gridSize, blockSize, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, xBuf, yBuf, rows, xStride0, yStride0);
      break;
    default:
      THROW_EXCEPTION("csr_spmv_semiring: unknown semiring code");
  }
}

// ===========================================================================
// Type-dispatched SpMM — one instantiation per (X, I) pair
// ===========================================================================

template <typename X, typename I>
static void csrSpMMSemiringCuda_(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                                  NDArray& B, NDArray& C,
                                  sd::LongType rows, sd::LongType /*cols*/, int semiring) {
  auto stream = C.getContext()->getCudaStream();

  const X* vBuf  = reinterpret_cast<const X*>(values.specialBuffer());
  const I* ciBuf = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpBuf = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const X* bBuf  = reinterpret_cast<const X*>(B.specialBuffer());
  X*       cBuf  = reinterpret_cast<X*>(C.specialBuffer());

  const LongType n       = B.sizeAt(1);
  const LongType bStride0 = B.stridesOf()[0];
  const LongType bStride1 = B.stridesOf()[1];
  const LongType cStride0 = C.stridesOf()[0];
  const LongType cStride1 = C.stridesOf()[1];

  const dim3 block(16, 16);
  const dim3 grid(static_cast<unsigned>((rows + block.x - 1) / block.x),
                  static_cast<unsigned>((n    + block.y - 1) / block.y));

  switch (semiring) {
    case 0:
      csrSpMMSemiringKernel<X, I, PlusTimesSR><<<grid, block, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, bBuf, cBuf,
          rows, n, bStride0, bStride1, cStride0, cStride1);
      break;
    case 1:
      csrSpMMSemiringKernel<X, I, MinPlusSR><<<grid, block, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, bBuf, cBuf,
          rows, n, bStride0, bStride1, cStride0, cStride1);
      break;
    case 2:
      csrSpMMSemiringKernel<X, I, MaxPlusSR><<<grid, block, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, bBuf, cBuf,
          rows, n, bStride0, bStride1, cStride0, cStride1);
      break;
    case 3:
      csrSpMMSemiringKernel<X, I, OrAndSR><<<grid, block, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, bBuf, cBuf,
          rows, n, bStride0, bStride1, cStride0, cStride1);
      break;
    case 4:
      csrSpMMSemiringKernel<X, I, MinTimesSR><<<grid, block, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, bBuf, cBuf,
          rows, n, bStride0, bStride1, cStride0, cStride1);
      break;
    default:
      THROW_EXCEPTION("csr_spmm_semiring: unknown semiring code");
  }
}

// ===========================================================================
// Public API
// ===========================================================================

void csr_spmv_semiring(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                       NDArray& x, NDArray& y,
                       sd::LongType rows, sd::LongType cols, int semiring) {
  NDArray::prepareSpecialUse({&y}, {&values, &colIdx, &rowPtr, &x});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSpMVSemiringCuda_,
                        (values, colIdx, rowPtr, x, y, rows, cols, semiring),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&y}, {&values, &colIdx, &rowPtr, &x});
}

void csr_spmm_semiring(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                       NDArray& B, NDArray& C,
                       sd::LongType rows, sd::LongType cols, int semiring) {
  NDArray::prepareSpecialUse({&C}, {&values, &colIdx, &rowPtr, &B});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSpMMSemiringCuda_,
                        (values, colIdx, rowPtr, B, C, rows, cols, semiring),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&C}, {&values, &colIdx, &rowPtr, &B});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
