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
// Created by raver119 on 20.12.17.
//

#ifndef LIBND4J_HELPERS_MATMUL_H
#define LIBND4J_HELPERS_MATMUL_H
#include <array/NDArray.h>
#include <ops/op_types.h>
#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

inline bool matmulSerialFloatStorage(DataType dtype) {
  return dtype == DataType::HALF || dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32;
}

inline bool matmulSerialStorageSupported(DataType x, DataType y, DataType z) {
  return (z == DataType::FLOAT32 && matmulSerialFloatStorage(x) && matmulSerialFloatStorage(y)) ||
      (x == y && y == z && (matmulSerialFloatStorage(x) || x == DataType::DOUBLE));
}

// SERIAL_FMA arithmetic is deliberately independent of M, N, tiling and launch
// geometry. Explicit FMA starts at +0; alpha multiplication is rounded before
// the optional beta FMA. No output read when beta == 0.
template <typename T>
SD_HOST_DEVICE SD_INLINE T matmulFma(T a, T b, T c) {
#if defined(__CUDA_ARCH__)
  if constexpr (sizeof(T) == sizeof(double)) return __fma_rn(a, b, c);
  else return __fmaf_ieee_rn(a, b, c);  // preserve subnormals even in an FTZ build
#else
  return std::fma(a, b, c);
#endif
}

template <typename T>
SD_HOST_DEVICE SD_INLINE T matmulMultiply(T a, T b) {
#if defined(__CUDA_ARCH__)
  // Adding -0 preserves the sign of an exact zero product in round-to-nearest.
  return matmulFma(a, b, static_cast<T>(-0.0));
#else
  volatile T rounded = a * b;  // prohibit contraction with the following beta FMA
  return rounded;
#endif
}

// Project one logical output coordinate through each operand's own strides.
// Rank-one operands remove their M/N output axes; leading axes broadcast from
// the right. Buffers are already shifted to the view base.
SD_HOST_DEVICE SD_INLINE void matmulSerialOffsets(
    LongType linearIndex, const LongType* xs, const LongType* ys, const LongType* zs,
    bool tx, bool ty, LongType& xo, LongType& yo, LongType& zo,
    LongType& kSize, LongType& xkStride, LongType& ykStride) {
  const int xr = shape::rank(xs), yr = shape::rank(ys), zr = shape::rank(zs);
  const int batchRank = zr - (xr > 1 ? 1 : 0) - (yr > 1 ? 1 : 0);
  LongType coords[SD_MAX_RANK] = {};
  INDEX2COORDS(linearIndex, zr, shape::shapeOf(zs), coords);
  COORDS2INDEX(zr, shape::stride(zs), coords, zo);
  xo = 0;
  yo = 0;
  for (int d = 0; d < xr - 2; ++d)
    if (shape::shapeOf(xs)[d] != 1)
      xo += coords[batchRank - (xr - 2) + d] * shape::stride(xs)[d];
  for (int d = 0; d < yr - 2; ++d)
    if (shape::shapeOf(ys)[d] != 1)
      yo += coords[batchRank - (yr - 2) + d] * shape::stride(ys)[d];
  const int xk = xr == 1 ? 0 : xr - (tx ? 2 : 1);
  const int yk = yr == 1 ? 0 : yr - (ty ? 1 : 2);
  if (xr > 1) xo += coords[batchRank] * shape::stride(xs)[xr - (tx ? 1 : 2)];
  if (yr > 1) yo += coords[batchRank + (xr > 1 ? 1 : 0)] * shape::stride(ys)[yr - (ty ? 2 : 1)];
  kSize = shape::shapeOf(xs)[xk];
  xkStride = shape::stride(xs)[xk];
  ykStride = shape::stride(ys)[yk];
}

template <typename X, typename Y = X, typename Z = X>
SD_HOST_DEVICE SD_INLINE void matmulSerialElement(
    LongType linearIndex, const X* x, const Y* y, Z* z,
    const LongType* xs, const LongType* ys, const LongType* zs,
    bool tx, bool ty, double alpha, double beta) {
  using AccT = typename simdOps::AggregateType<Z>::type;
  LongType xo, yo, zo, kSize, xkStride, ykStride;
  matmulSerialOffsets(linearIndex, xs, ys, zs, tx, ty, xo, yo, zo, kSize, xkStride, ykStride);
  AccT sum = static_cast<AccT>(0);
  for (LongType k = 0; k < kSize; ++k, xo += xkStride, yo += ykStride)
    sum = matmulFma(static_cast<AccT>(x[xo]), static_cast<AccT>(y[yo]), sum);
  AccT result = matmulMultiply(static_cast<AccT>(alpha), sum);
  if (beta != 0.0)
    result = matmulFma(static_cast<AccT>(beta), static_cast<AccT>(z[zo]), result);
  z[zo] = static_cast<Z>(result);
}

SD_LIB_HIDDEN void _matmul(LaunchContext *context, NDArray *A, NDArray *B, NDArray *C, int transA, int transB,
                           double alpha = 1., double beta = 0.);
}
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_MATMUL_H
