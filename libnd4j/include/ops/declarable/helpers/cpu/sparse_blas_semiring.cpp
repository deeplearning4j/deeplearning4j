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

#include <ops/declarable/helpers/sparse_blas_semiring.h>
#include <system/op_boilerplate.h>
#include <array/DataTypeUtils.h>

namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------
// Semiring functor structs
// Each exposes:   identity()  — neutral element for the add operation
//                 add(a, b)   — semiring addition
//                 mul(a, b)   — semiring multiplication
// ---------------------------------------------------------------------------

template <typename X>
struct PlusTimesSR {
  // Standard arithmetic ring: (add=+, mul=*, identity=0)
  static X identity() { return static_cast<X>(0); }
  static X add(X a, X b) { return a + b; }
  static X mul(X a, X b) { return a * b; }
};

template <typename X>
struct MinPlusSR {
  // Tropical min-plus: (add=min, mul=+, identity=+inf)
  static X identity() { return DataTypeUtils::infOrMax<X>(); }
  static X add(X a, X b) { return a < b ? a : b; }
  static X mul(X a, X b) { return a + b; }
};

template <typename X>
struct MaxPlusSR {
  // Max-plus: (add=max, mul=+, identity=-inf)
  static X identity() { return -DataTypeUtils::infOrMax<X>(); }
  static X add(X a, X b) { return a > b ? a : b; }
  static X mul(X a, X b) { return a + b; }
};

template <typename X>
struct OrAndSR {
  // Boolean OR/AND semiring: (add=||, mul=&&, identity=0)
  // Values are treated as booleans: non-zero == true, zero == false.
  static X identity() { return static_cast<X>(0); }
  static X add(X a, X b) {
    return ((a != static_cast<X>(0)) || (b != static_cast<X>(0)))
               ? static_cast<X>(1)
               : static_cast<X>(0);
  }
  static X mul(X a, X b) {
    return ((a != static_cast<X>(0)) && (b != static_cast<X>(0)))
               ? static_cast<X>(1)
               : static_cast<X>(0);
  }
};

template <typename X>
struct MinTimesSR {
  // Min-times: (add=min, mul=*, identity=+inf)
  static X identity() { return DataTypeUtils::infOrMax<X>(); }
  static X add(X a, X b) { return a < b ? a : b; }
  static X mul(X a, X b) { return a * b; }
};

// ---------------------------------------------------------------------------
// csr_spmv_semiring — CPU
// y[i] = SR.reduce_{k in [rowPtr[i], rowPtr[i+1])}( SR.mul(values[k], x[colIdx[k]]) )
// ---------------------------------------------------------------------------

// Inner kernel: one semiring type baked in as template parameter SR.
template <typename X, typename I, typename SR>
static void csrSpMVSemiringKernel_(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                                    NDArray& x, NDArray& y,
                                    sd::LongType rows) {
  const I nrows = static_cast<I>(rows);

  const auto* vBuf  = values.bufferAsT<X>();
  const auto* ciBuf = colIdx.bufferAsT<I>();
  const auto* rpBuf = rowPtr.bufferAsT<I>();
  const auto* xBuf  = x.bufferAsT<X>();
  auto*       yBuf  = y.bufferAsT<X>();

  const LongType xStride0 = x.stridesOf()[0];
  const LongType yStride0 = y.stridesOf()[0];

  for (I i = 0; i < nrows; ++i) {
    X acc = SR::identity();
    const I start = rpBuf[i];
    const I end   = rpBuf[i + 1];
    for (I k = start; k < end; ++k) {
      acc = SR::add(acc, SR::mul(vBuf[k], xBuf[static_cast<LongType>(ciBuf[k]) * xStride0]));
    }
    yBuf[static_cast<LongType>(i) * yStride0] = acc;
  }
}

// Dispatch function: switches on the runtime semiring code, then calls the
// templated kernel with the appropriate functor type.
template <typename X, typename I>
static void csrSpMVSemiring_(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                              NDArray& x, NDArray& y,
                              sd::LongType rows, sd::LongType cols, int semiring) {
  switch (semiring) {
    case 0:
      csrSpMVSemiringKernel_<X, I, PlusTimesSR<X>>(values, colIdx, rowPtr, x, y, rows);
      break;
    case 1:
      csrSpMVSemiringKernel_<X, I, MinPlusSR<X>>(values, colIdx, rowPtr, x, y, rows);
      break;
    case 2:
      csrSpMVSemiringKernel_<X, I, MaxPlusSR<X>>(values, colIdx, rowPtr, x, y, rows);
      break;
    case 3:
      csrSpMVSemiringKernel_<X, I, OrAndSR<X>>(values, colIdx, rowPtr, x, y, rows);
      break;
    case 4:
      csrSpMVSemiringKernel_<X, I, MinTimesSR<X>>(values, colIdx, rowPtr, x, y, rows);
      break;
    default: {
      std::string msg = "csr_spmv_semiring: unknown semiring code " + std::to_string(semiring);
      THROW_EXCEPTION(msg.c_str());
    }
  }
}

void csr_spmv_semiring(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                        NDArray& x, NDArray& y,
                        sd::LongType rows, sd::LongType cols, int semiring) {
  NDArray::preparePrimaryUse({&y}, {&values, &colIdx, &rowPtr, &x});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSpMVSemiring_,
                        (values, colIdx, rowPtr, x, y, rows, cols, semiring),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&y}, {&values, &colIdx, &rowPtr, &x});
}

// ---------------------------------------------------------------------------
// csr_spmm_semiring — CPU
// C[i,j] = SR.reduce_{k in [rowPtr[i], rowPtr[i+1])}( SR.mul(values[k], B[colIdx[k],j]) )
// ---------------------------------------------------------------------------

// Inner kernel: one semiring type baked in as template parameter SR.
template <typename X, typename I, typename SR>
static void csrSpMMSemiringKernel_(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                                    NDArray& B, NDArray& C,
                                    sd::LongType rows) {
  const I nrows = static_cast<I>(rows);
  const LongType n = B.sizeAt(1);

  const auto* vBuf  = values.bufferAsT<X>();
  const auto* ciBuf = colIdx.bufferAsT<I>();
  const auto* rpBuf = rowPtr.bufferAsT<I>();
  const auto* bBuf  = B.bufferAsT<X>();
  auto*       cBuf  = C.bufferAsT<X>();

  const LongType bStride0 = B.stridesOf()[0];
  const LongType bStride1 = B.stridesOf()[1];
  const LongType cStride0 = C.stridesOf()[0];
  const LongType cStride1 = C.stridesOf()[1];

  for (I i = 0; i < nrows; ++i) {
    const I start = rpBuf[i];
    const I end   = rpBuf[i + 1];
    for (LongType j = 0; j < n; ++j) {
      X acc = SR::identity();
      for (I k = start; k < end; ++k) {
        acc = SR::add(acc,
                      SR::mul(vBuf[k],
                              bBuf[static_cast<LongType>(ciBuf[k]) * bStride0 + j * bStride1]));
      }
      cBuf[static_cast<LongType>(i) * cStride0 + j * cStride1] = acc;
    }
  }
}

// Dispatch function: switches on the runtime semiring code.
template <typename X, typename I>
static void csrSpMMSemiring_(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                              NDArray& B, NDArray& C,
                              sd::LongType rows, sd::LongType cols, int semiring) {
  switch (semiring) {
    case 0:
      csrSpMMSemiringKernel_<X, I, PlusTimesSR<X>>(values, colIdx, rowPtr, B, C, rows);
      break;
    case 1:
      csrSpMMSemiringKernel_<X, I, MinPlusSR<X>>(values, colIdx, rowPtr, B, C, rows);
      break;
    case 2:
      csrSpMMSemiringKernel_<X, I, MaxPlusSR<X>>(values, colIdx, rowPtr, B, C, rows);
      break;
    case 3:
      csrSpMMSemiringKernel_<X, I, OrAndSR<X>>(values, colIdx, rowPtr, B, C, rows);
      break;
    case 4:
      csrSpMMSemiringKernel_<X, I, MinTimesSR<X>>(values, colIdx, rowPtr, B, C, rows);
      break;
    default: {
      std::string msg = "csr_spmm_semiring: unknown semiring code " + std::to_string(semiring);
      THROW_EXCEPTION(msg.c_str());
    }
  }
}

void csr_spmm_semiring(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                        NDArray& B, NDArray& C,
                        sd::LongType rows, sd::LongType cols, int semiring) {
  NDArray::preparePrimaryUse({&C}, {&values, &colIdx, &rowPtr, &B});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSpMMSemiring_,
                        (values, colIdx, rowPtr, B, C, rows, cols, semiring),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&C}, {&values, &colIdx, &rowPtr, &B});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
