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

// The type of a matmul's output: the wider input type, the first on a tie (HALF and BFLOAT16 <
// FLOAT32 < DOUBLE), and a float input over an integer one. Java's Mmul.promoteMatmulOutputDataType
// is the same rule.
inline DataType matmulOutputType(DataType x, DataType y) {
  const bool xFloat = DataTypeUtils::isR(x);
  const bool yFloat = DataTypeUtils::isR(y);
  if (xFloat != yFloat) return xFloat ? x : y;
  return DataTypeUtils::sizeOf(x) >= DataTypeUtils::sizeOf(y) ? x : y;
}

// The one type a GEMM over mixed storage computes in when no typed BLAS path covers it: FLOAT32
// for floating storage (DOUBLE when any operand is DOUBLE), so no floating operand is narrowed,
// and INT64 when every operand is an integer, which holds every signed and unsigned operand value.
inline DataType mixedGemmComputeType(DataType a, DataType b, DataType c) {
  if (a == DataType::DOUBLE || b == DataType::DOUBLE || c == DataType::DOUBLE) return DataType::DOUBLE;
  if (DataTypeUtils::isR(a) || DataTypeUtils::isR(b) || DataTypeUtils::isR(c)) return DataType::FLOAT32;
  return DataType::INT64;
}

// A matrix-vector product whose matrix and vector storage types differ, as when a decode step
// multiplies an FP32 activation by HALF or BFLOAT16 weights:
//   y[r * yStride] = alpha * sum_k w[r * rowStride + k * depthStride] * x[k * xStride]
//                    + beta * y[r * yStride]
// The mixed GEMV reads every operand in place in its storage type and sums in
// ProductAccumulator. BLAS takes no such pair, so the matrix used to be widened first: a copy
// of every weight on every call.
struct MixedGemvLayout {
  LongType rows, depth;
  LongType rowStride, depthStride;
  LongType xStride, yStride;

  // Whether one row's products are the adjacent ones, so a row is a contiguous dot product;
  // otherwise neighbouring rows are adjacent at each depth.
  bool depthMajor() const { return rows == 1 || (depth > 1 && depthStride <= rowStride); }
};

// The type products of X and Y storage accumulate in on their way to Z storage: FLOAT for HALF,
// BFLOAT16 and FLOAT operands, DOUBLE when any is DOUBLE, an integer type for integers.
template <typename X, typename Y, typename Z>
using ProductAccumulator = typename simdOps::AggregateType<typename math::promote_type3<X, Y, Z>::type>::type;

// The storage types the mixed GEMV reads: the float types the selectors dispatch.
inline bool mixedGemvStorage(DataType type) {
  for (const DataType floatType : {ALL_FLOATS})
    if (type == floatType) return true;
  return false;
}

// Float storage throughout with the matrix's type not the vector's: no BLAS reads such a pair.
// (cuBLAS reads a matrix and a vector of one type into an output of another.)
inline bool mixedGemvApplies(DataType matrix, DataType vector, DataType output) {
  return matrix != vector && mixedGemvStorage(matrix) && mixedGemvStorage(vector) && mixedGemvStorage(output);
}

// The product [M,K] x [K,N] -> [M,N] with one output row (M == 1: the matrix is b) or one
// output column (N == 1: the matrix is a) as a GEMV.
inline MixedGemvLayout mixedGemvLayoutOfGemm(NDArray* a, NDArray* b, NDArray* c) {
  if (a->sizeAt(0) == 1)
    return {b->sizeAt(1), b->sizeAt(0), b->strideAt(1), b->strideAt(0), a->strideAt(1), c->strideAt(1)};
  return {a->sizeAt(0), a->sizeAt(1), a->strideAt(0), a->strideAt(1), b->strideAt(0), c->strideAt(0)};
}

SD_LIB_HIDDEN void _matmul(LaunchContext *context, NDArray *A, NDArray *B, NDArray *C, int transA, int transB,
                           double alpha = 1., double beta = 0.);
}
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_MATMUL_H
