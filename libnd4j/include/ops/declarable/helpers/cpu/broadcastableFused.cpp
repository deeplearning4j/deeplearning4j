/* ******************************************************************************
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

#include <execution/Threads.h>
#include <ops/declarable/helpers/broadcastableFused.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// Template implementations for same-shape contiguous operations

template <typename T>
static void addContiguous_(const void* vx, const void* vy, void* vz, sd::LongType len) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto func = PRAGMA_THREADS_FOR {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; ++i) {
      z[i] = x[i] + y[i];
    }
  };
  samediff::Threads::parallel_for(func, 0, len);
}

template <typename T>
static void subtractContiguous_(const void* vx, const void* vy, void* vz, sd::LongType len) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto func = PRAGMA_THREADS_FOR {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; ++i) {
      z[i] = x[i] - y[i];
    }
  };
  samediff::Threads::parallel_for(func, 0, len);
}

template <typename T>
static void multiplyContiguous_(const void* vx, const void* vy, void* vz, sd::LongType len) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto func = PRAGMA_THREADS_FOR {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; ++i) {
      z[i] = x[i] * y[i];
    }
  };
  samediff::Threads::parallel_for(func, 0, len);
}

template <typename T>
static void divideContiguous_(const void* vx, const void* vy, void* vz, sd::LongType len) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto func = PRAGMA_THREADS_FOR {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; ++i) {
      z[i] = x[i] / y[i];
    }
  };
  samediff::Threads::parallel_for(func, 0, len);
}

//////////////////////////////////////////////////////////////////////////
// Template implementations for 1D broadcast along last dimension

template <typename T>
static void add1DLast_(const void* vx, const void* vy, void* vz, sd::LongType numRows, sd::LongType lastDim) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto func = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; ++row) {
      const T* xRow = x + row * lastDim;
      T* zRow = z + row * lastDim;
      PRAGMA_OMP_SIMD
      for (sd::LongType i = 0; i < lastDim; ++i) {
        zRow[i] = xRow[i] + y[i];
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, numRows);
}

template <typename T>
static void multiply1DLast_(const void* vx, const void* vy, void* vz, sd::LongType numRows, sd::LongType lastDim) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  auto func = PRAGMA_THREADS_FOR {
    for (auto row = start; row < stop; ++row) {
      const T* xRow = x + row * lastDim;
      T* zRow = z + row * lastDim;
      PRAGMA_OMP_SIMD
      for (sd::LongType i = 0; i < lastDim; ++i) {
        zRow[i] = xRow[i] * y[i];
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, numRows);
}

//////////////////////////////////////////////////////////////////////////
// Public API implementations

void fusedAddContiguous(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType len = x.lengthOf();
  // Guard: empty arrays have null data buffers. Mirrors the pairwise null-buffer guard
  // in NativeOpExecutioner_pairwise_*.cpp. Without this, parallel_for invokes the lambda
  // even when len==0 (delta==0 fast-path calls function(0,0,0,1)), and SIMD preamble code
  // may load from address 0 before the loop-bound check, causing SIGSEGV.
  void* xBuf = x.buffer();
  void* yBuf = y.buffer();
  void* zBuf = z.buffer();
  if (xBuf == nullptr || yBuf == nullptr || zBuf == nullptr || len == 0) return;
  BUILD_SINGLE_SELECTOR(x.dataType(), addContiguous_, (xBuf, yBuf, zBuf, len), SD_COMMON_TYPES);
}

void fusedSubtractContiguous(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType len = x.lengthOf();
  void* xBuf = x.buffer();
  void* yBuf = y.buffer();
  void* zBuf = z.buffer();
  if (xBuf == nullptr || yBuf == nullptr || zBuf == nullptr || len == 0) return;
  BUILD_SINGLE_SELECTOR(x.dataType(), subtractContiguous_, (xBuf, yBuf, zBuf, len), SD_COMMON_TYPES);
}

void fusedMultiplyContiguous(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType len = x.lengthOf();
  void* xBuf = x.buffer();
  void* yBuf = y.buffer();
  void* zBuf = z.buffer();
  if (xBuf == nullptr || yBuf == nullptr || zBuf == nullptr || len == 0) return;
  BUILD_SINGLE_SELECTOR(x.dataType(), multiplyContiguous_, (xBuf, yBuf, zBuf, len), SD_COMMON_TYPES);
}

void fusedDivideContiguous(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType len = x.lengthOf();
  void* xBuf = x.buffer();
  void* yBuf = y.buffer();
  void* zBuf = z.buffer();
  if (xBuf == nullptr || yBuf == nullptr || zBuf == nullptr || len == 0) return;
  BUILD_SINGLE_SELECTOR(x.dataType(), divideContiguous_, (xBuf, yBuf, zBuf, len), SD_COMMON_TYPES);
}

void fusedAdd1DLast(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType lastDim = x.sizeAt(-1);
  // Guard lastDim before the division to avoid UB (divide-by-zero) on empty arrays.
  if (lastDim == 0) return;
  const sd::LongType numRows = x.lengthOf() / lastDim;
  void* xBuf = x.buffer();
  void* yBuf = y.buffer();
  void* zBuf = z.buffer();
  if (xBuf == nullptr || yBuf == nullptr || zBuf == nullptr || numRows == 0) return;
  BUILD_SINGLE_SELECTOR(x.dataType(), add1DLast_, (xBuf, yBuf, zBuf, numRows, lastDim), SD_COMMON_TYPES);
}

void fusedMultiply1DLast(NDArray& x, NDArray& y, NDArray& z) {
  const sd::LongType lastDim = x.sizeAt(-1);
  // Guard lastDim before the division to avoid UB (divide-by-zero) on empty arrays.
  if (lastDim == 0) return;
  const sd::LongType numRows = x.lengthOf() / lastDim;
  void* xBuf = x.buffer();
  void* yBuf = y.buffer();
  void* zBuf = z.buffer();
  if (xBuf == nullptr || yBuf == nullptr || zBuf == nullptr || numRows == 0) return;
  BUILD_SINGLE_SELECTOR(x.dataType(), multiply1DLast_, (xBuf, yBuf, zBuf, numRows, lastDim), SD_COMMON_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
