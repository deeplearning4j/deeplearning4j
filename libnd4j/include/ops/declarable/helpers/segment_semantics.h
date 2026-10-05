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
//  The semantics of segment_{sum,mean,max,min,prod}, unsorted_segment_{sum,mean,max,min,prod,sqrt_n} and their
//  backprops, written once for both backends. The CPU helpers (helpers/cpu/segment.cpp) call the host range
//  functions below; the CUDA kernels (helpers/cuda/segment_ops.cuh) call the same policy functions per element, so
//  the two backends cannot drift apart on
//    * the accumulator type (AccOf): HALF/BFLOAT16 sum in FLOAT; integers sum and multiply modulo 2^32 / 2^64, which
//      is exactly what accumulating in the narrow type with wrap-around gives; max/min compare in a type that keeps
//      the order of the input type; the mean and sqrt_n of integers sum in DOUBLE;
//    * the identity of a reduction and the value of a segment no row maps to:
//        sorted   : sum, mean, max, min -> 0; prod -> 1                    (TensorFlow's segment_*)
//        unsorted : sum, mean, sqrt_n -> 0; prod -> 1; max -> the lowest and min -> the highest value of the type
//                   (numeric_limits<T>::lowest() / max(), so -FLT_MAX and FLT_MAX for floats)   (TensorFlow's
//                   unsorted_segment_*)
//      and a segment with rows reduces exactly those rows: a segment holding only -inf has the max -inf;
//    * NaN: a NaN input makes a max or min NaN, whatever its position;
//    * a mean is sum / count and a sqrt_n is sum / sqrt(count), both applied once to the finished accumulator;
//    * the gradient of a product is the product of the other elements of the segment, also when some are zero
//      (prod / x only without zeros; one zero: it alone has the product of the nonzero elements; more: none has a
//      gradient); the gradients of a maximum and a minimum compare exactly.
//  Rows are the entries of dimension 0, a "column" is one element of a row (the trailing dimensions in C order).
//
#ifndef LIBND4J_SEGMENT_SEMANTICS_H
#define LIBND4J_SEGMENT_SEMANTICS_H

#include <array/DataTypeUtils.h>
#include <helpers/shape.h>
#include <system/common.h>
#include <types/bfloat16.h>
#include <types/float16.h>

#include <math.h>

#include <cstdint>
#include <type_traits>

namespace sd {
namespace ops {
namespace helpers {
namespace segment_sem {

// ============================================================================ //
// Accumulator types
// ============================================================================ //

// sum / prod: HALF, BFLOAT16 and FLOAT in FLOAT, DOUBLE in DOUBLE; integers modulo 2^32 (up to 32 bits) or 2^64.
template <typename T>
struct AccArith;
template <> struct AccArith<float16> { using type = float; };
template <> struct AccArith<bfloat16> { using type = float; };
template <> struct AccArith<float> { using type = float; };
template <> struct AccArith<double> { using type = double; };
template <> struct AccArith<int8_t> { using type = uint32_t; };
template <> struct AccArith<int16_t> { using type = uint32_t; };
template <> struct AccArith<int32_t> { using type = uint32_t; };
template <> struct AccArith<uint8_t> { using type = uint32_t; };
template <> struct AccArith<uint16_t> { using type = uint32_t; };
template <> struct AccArith<uint32_t> { using type = uint32_t; };
template <> struct AccArith<int64_t> { using type = uint64_t; };
template <> struct AccArith<uint64_t> { using type = uint64_t; };

// max / min: a type that keeps the order of the input type (and holds every value of it exactly).
template <typename T>
struct AccOrd;
template <> struct AccOrd<float16> { using type = float; };
template <> struct AccOrd<bfloat16> { using type = float; };
template <> struct AccOrd<float> { using type = float; };
template <> struct AccOrd<double> { using type = double; };
template <> struct AccOrd<int8_t> { using type = int32_t; };
template <> struct AccOrd<int16_t> { using type = int32_t; };
template <> struct AccOrd<int32_t> { using type = int32_t; };
template <> struct AccOrd<uint8_t> { using type = uint32_t; };
template <> struct AccOrd<uint16_t> { using type = uint32_t; };
template <> struct AccOrd<uint32_t> { using type = uint32_t; };
template <> struct AccOrd<int64_t> { using type = int64_t; };
template <> struct AccOrd<uint64_t> { using type = uint64_t; };

// mean / sqrt_n / gradients: DOUBLE when either side is DOUBLE or the input is an integer, FLOAT otherwise.
template <typename X, typename Z>
struct AccFloat {
  using type = typename std::conditional<
      (std::is_same<X, double>::value || std::is_same<Z, double>::value || !std::is_floating_point<X>::value), double,
      float>::type;
};

// ============================================================================ //
// Limits and small scalar helpers
// ============================================================================ //

// numeric_limits<Z>::lowest(): -max for floating types (HALF -65504, BFLOAT16 -3.39e38, FLOAT -FLT_MAX), the minimum
// of an integer type (0 for the unsigned ones).
template <typename Z>
SD_HOST_DEVICE SD_INLINE Z lowestOf() {
  if constexpr (std::is_same<Z, float>::value || std::is_same<Z, double>::value) {
    return -DataTypeUtils::max<Z>();
  } else if constexpr (std::is_floating_point<Z>::value) {
    return static_cast<Z>(-static_cast<float>(DataTypeUtils::max<Z>()));
  } else {
    return DataTypeUtils::min<Z>();
  }
}

// numeric_limits<Z>::max()
template <typename Z>
SD_HOST_DEVICE SD_INLINE Z highestOf() {
  return DataTypeUtils::max<Z>();
}

SD_HOST_DEVICE SD_INLINE float segSqrt(float v) { return ::sqrtf(v); }
SD_HOST_DEVICE SD_INLINE double segSqrt(double v) { return ::sqrt(v); }

// A segment id of any integer type as a signed 64 bit value; an unsigned 64 bit id above the signed range becomes the
// largest signed value (it is out of every range of classes).
template <typename I>
SD_HOST_DEVICE SD_INLINE LongType idToLong(I v) {
  if constexpr (std::is_unsigned<I>::value && sizeof(I) >= 8) {
    return v > static_cast<I>(DataTypeUtils::max<LongType>()) ? DataTypeUtils::max<LongType>()
                                                              : static_cast<LongType>(v);
  } else {
    return static_cast<LongType>(v);
  }
}

// Offset of the element with the given logical (C order) linear index in an array of the given shape and strides.
SD_HOST_DEVICE SD_INLINE LongType logicalOffset(LongType index, LongType rank, const LongType* shape,
                                                const LongType* stride) {
  if (rank == 0) return 0;
  if (rank == 1) return index * stride[0];
  LongType offset = 0;
  for (LongType d = rank - 1; d >= 0; --d) {
    const LongType size = shape[d];
    const LongType c = index % size;
    index /= size;
    offset += c * stride[d];
  }
  return offset;
}

// ============================================================================ //
// Layout of an array as rows x columns
// ============================================================================ //

// An array seen as [rows, inner]: row r starts rowStride elements per row after the base, column e of a row is
// e * innerStep further when the trailing dimensions are one run of equal steps (any C-contiguous or strided vector,
// the common case) and is found through the trailing dimensions of the shape info otherwise (F order, permuted and
// stepped views). Offsets are relative to the array's own (already view-shifted) buffer.
struct SegMat {
  LongType rowStride;
  LongType inner;
  LongType innerStep;          // >= 0: column e at e * innerStep; < 0: use shapeInfo
  const LongType* shapeInfo;   // read only when innerStep < 0; must be readable by the code that calls colOffset

  SD_HOST_DEVICE SD_INLINE LongType colOffset(LongType e) const {
    if (innerStep >= 0) return e * innerStep;
    const LongType rank = shape::rank(shapeInfo);
    const LongType* shp = shape::shapeOf(shapeInfo);
    const LongType* str = shape::stride(shapeInfo);
    LongType offset = 0;
    for (LongType d = rank - 1; d >= 1; --d) {
      const LongType size = shp[d];
      const LongType c = e % size;
      e /= size;
      offset += c * str[d];
    }
    return offset;
  }

  SD_HOST_DEVICE SD_INLINE LongType offset(LongType row, LongType col) const {
    return row * rowStride + colOffset(col);
  }
};

// hostShapeInfo is read here; executingShapeInfo is what colOffset reads later (the device copy of the shape info
// for a CUDA kernel, the same pointer on the CPU).
SD_HOST SD_INLINE SegMat makeSegMat(const LongType* hostShapeInfo, const LongType* executingShapeInfo) {
  SegMat m;
  m.shapeInfo = executingShapeInfo;
  const LongType rank = shape::rank(hostShapeInfo);
  if (rank == 0) {
    m.rowStride = 0;
    m.inner = 1;
    m.innerStep = 1;
    return m;
  }
  const LongType* shp = shape::shapeOf(hostShapeInfo);
  const LongType* str = shape::stride(hostShapeInfo);
  m.rowStride = str[0];
  m.inner = 1;
  for (LongType d = 1; d < rank; ++d) m.inner *= shp[d];

  // the trailing dimensions are "one run" when walking them in C order always adds the same step
  bool run = true;
  LongType step = 1;
  LongType expected = -1;
  for (LongType d = rank - 1; d >= 1; --d) {
    if (shp[d] == 1) continue;
    if (expected < 0) {
      step = str[d];
    } else if (str[d] != expected) {
      run = false;
      break;
    }
    expected = str[d] * shp[d];
  }
  m.innerStep = run ? step : -1;
  return m;
}

// A dense C-order [rows, inner] buffer.
SD_HOST_DEVICE SD_INLINE SegMat denseSegMat(LongType inner) {
  SegMat m;
  m.rowStride = inner;
  m.inner = inner;
  m.innerStep = 1;
  m.shapeInfo = nullptr;
  return m;
}

// ============================================================================ //
// Forward policies
// ============================================================================ //
//  AccOf<X, Z>::type          accumulator for input type X and output type Z
//  kNeedsCount                the result of a segment depends on the number of its rows (unsorted ops: needed to
//                             tell an empty segment from one whose reduction happens to equal the identity)
//  kIsAdd                     combine is an addition (a native atomic add exists)
//  identity<A>()              start value of an accumulator
//  combine<A>(acc, v)         acc = acc (+) v
//  finalize<A>(acc, count)    value of a segment with count >= 1 rows
//  emptySorted<Z>() / emptyUnsorted<Z>()   value of a segment no row maps to

struct SegSum {
  static constexpr bool kNeedsCount = false;
  static constexpr bool kIsAdd = true;
  template <typename X, typename Z> struct AccOf { using type = typename AccArith<X>::type; };
  template <typename A> static SD_HOST_DEVICE SD_INLINE A identity() { return static_cast<A>(0); }
  template <typename A> static SD_HOST_DEVICE SD_INLINE void combine(A& acc, A v) { acc += v; }
  template <typename A> static SD_HOST_DEVICE SD_INLINE A finalize(A acc, LongType) { return acc; }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptySorted() { return static_cast<Z>(0); }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptyUnsorted() { return static_cast<Z>(0); }
};

struct SegProd {
  static constexpr bool kNeedsCount = false;
  static constexpr bool kIsAdd = false;
  template <typename X, typename Z> struct AccOf { using type = typename AccArith<X>::type; };
  template <typename A> static SD_HOST_DEVICE SD_INLINE A identity() { return static_cast<A>(1); }
  template <typename A> static SD_HOST_DEVICE SD_INLINE void combine(A& acc, A v) { acc *= v; }
  template <typename A> static SD_HOST_DEVICE SD_INLINE A finalize(A acc, LongType) { return acc; }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptySorted() { return static_cast<Z>(1); }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptyUnsorted() { return static_cast<Z>(1); }
};

// The two reductions the gradient of a product reads (GradProd): the product of the NONZERO elements of a segment, and
// the number of its zeros. A product cannot be divided by a zero element, so the gradient is built from these.
struct SegProdNonZero {
  static constexpr bool kNeedsCount = false;
  static constexpr bool kIsAdd = false;
  template <typename X, typename Z> struct AccOf { using type = typename AccArith<X>::type; };
  template <typename A> static SD_HOST_DEVICE SD_INLINE A identity() { return static_cast<A>(1); }
  template <typename A> static SD_HOST_DEVICE SD_INLINE void combine(A& acc, A v) {
    if (v != static_cast<A>(0)) acc *= v;
  }
  template <typename A> static SD_HOST_DEVICE SD_INLINE A finalize(A acc, LongType) { return acc; }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptySorted() { return static_cast<Z>(1); }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptyUnsorted() { return static_cast<Z>(1); }
};

struct SegZeroCount {
  static constexpr bool kNeedsCount = false;
  static constexpr bool kIsAdd = false;  // a zero adds one and any other value nothing: not the addition of the value
  template <typename X, typename Z> struct AccOf { using type = typename AccArith<X>::type; };
  template <typename A> static SD_HOST_DEVICE SD_INLINE A identity() { return static_cast<A>(0); }
  template <typename A> static SD_HOST_DEVICE SD_INLINE void combine(A& acc, A v) {
    if (v == static_cast<A>(0)) acc += static_cast<A>(1);
  }
  template <typename A> static SD_HOST_DEVICE SD_INLINE A finalize(A acc, LongType) { return acc; }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptySorted() { return static_cast<Z>(0); }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptyUnsorted() { return static_cast<Z>(0); }
};

struct SegMax {
  static constexpr bool kNeedsCount = true;
  static constexpr bool kIsAdd = false;
  template <typename X, typename Z> struct AccOf { using type = typename AccOrd<X>::type; };
  // -inf for floating accumulators, the lowest value of an integer accumulator
  template <typename A> static SD_HOST_DEVICE SD_INLINE A identity() {
    if constexpr (std::is_floating_point<A>::value) {
      return -DataTypeUtils::infOrMax<A>();
    } else {
      return lowestOf<A>();
    }
  }
  // a NaN (as the accumulator or as the value) makes the maximum NaN
  template <typename A> static SD_HOST_DEVICE SD_INLINE void combine(A& acc, A v) {
    if constexpr (std::is_floating_point<A>::value) {
      if (v > acc || v != v) acc = v;
    } else {
      if (v > acc) acc = v;
    }
  }
  template <typename A> static SD_HOST_DEVICE SD_INLINE A finalize(A acc, LongType) { return acc; }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptySorted() { return static_cast<Z>(0); }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptyUnsorted() { return lowestOf<Z>(); }
};

struct SegMin {
  static constexpr bool kNeedsCount = true;
  static constexpr bool kIsAdd = false;
  template <typename X, typename Z> struct AccOf { using type = typename AccOrd<X>::type; };
  // +inf for floating accumulators, the highest value of an integer accumulator
  template <typename A> static SD_HOST_DEVICE SD_INLINE A identity() {
    if constexpr (std::is_floating_point<A>::value) {
      return DataTypeUtils::infOrMax<A>();
    } else {
      return highestOf<A>();
    }
  }
  template <typename A> static SD_HOST_DEVICE SD_INLINE void combine(A& acc, A v) {
    if constexpr (std::is_floating_point<A>::value) {
      if (v < acc || v != v) acc = v;
    } else {
      if (v < acc) acc = v;
    }
  }
  template <typename A> static SD_HOST_DEVICE SD_INLINE A finalize(A acc, LongType) { return acc; }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptySorted() { return static_cast<Z>(0); }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptyUnsorted() { return highestOf<Z>(); }
};

struct SegMean {
  static constexpr bool kNeedsCount = true;
  static constexpr bool kIsAdd = true;
  template <typename X, typename Z> struct AccOf { using type = typename AccFloat<X, Z>::type; };
  template <typename A> static SD_HOST_DEVICE SD_INLINE A identity() { return static_cast<A>(0); }
  template <typename A> static SD_HOST_DEVICE SD_INLINE void combine(A& acc, A v) { acc += v; }
  template <typename A> static SD_HOST_DEVICE SD_INLINE A finalize(A acc, LongType count) {
    return acc / static_cast<A>(count);
  }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptySorted() { return static_cast<Z>(0); }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptyUnsorted() { return static_cast<Z>(0); }
};

struct SegSqrtN {
  static constexpr bool kNeedsCount = true;
  static constexpr bool kIsAdd = true;
  template <typename X, typename Z> struct AccOf { using type = typename AccFloat<X, Z>::type; };
  template <typename A> static SD_HOST_DEVICE SD_INLINE A identity() { return static_cast<A>(0); }
  template <typename A> static SD_HOST_DEVICE SD_INLINE void combine(A& acc, A v) { acc += v; }
  template <typename A> static SD_HOST_DEVICE SD_INLINE A finalize(A acc, LongType count) {
    return acc / segSqrt(static_cast<A>(count));
  }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptySorted() { return static_cast<Z>(0); }
  template <typename Z> static SD_HOST_DEVICE SD_INLINE Z emptyUnsorted() { return static_cast<Z>(0); }
};

// ============================================================================ //
// Backprop policies
// ============================================================================ //
//  kNeedsForward   the input element and the forward result of its segment are read (max, min, prod)
//  kNeedsCount     the number of rows of the segment is read (mean, sqrt_n)
//  kNeedsZeros     the number of zeros in the segment's column is read (prod)
//  apply(go, x, fwd, count, zeros)  the gradient of one element: go = the gradient of its segment, x the input element,
//                  fwd the forward result of its segment (for prod: the product of its nonzero elements), count the
//                  rows of its segment, zeros the zeros among the segment's elements of this column

// Elements equal to the segment's extremum get the whole gradient (an exact comparison: the extremum IS one of the
// elements; two NaNs are equal here, a NaN extremum is the NaN elements').
struct GradCompare {
  static constexpr bool kNeedsForward = true;
  static constexpr bool kNeedsCount = false;
  static constexpr bool kNeedsZeros = false;
  template <typename T>
  static SD_HOST_DEVICE SD_INLINE T apply(T go, T x, T fwd, LongType, LongType) {
    using A = typename AccOrd<T>::type;
    const A ax = static_cast<A>(x);
    const A af = static_cast<A>(fwd);
    const bool tie = (ax == af) || (ax != ax && af != af);
    return tie ? go : static_cast<T>(0);
  }
};

// d prod / d x_i is the product of the other elements of the segment (TensorFlow's three cases). Without a zero in the
// segment that is prod / x_i; with exactly one, only that zero has a gradient (the product of the nonzero elements);
// with more, no element has one. fwd is the product of the NONZERO elements (SegProdNonZero) and zeros the number of
// zeros of the segment's column (SegZeroCount): dividing the whole product by a zero element gave NaN.
struct GradProd {
  static constexpr bool kNeedsForward = true;
  static constexpr bool kNeedsCount = false;
  static constexpr bool kNeedsZeros = true;
  template <typename T>
  static SD_HOST_DEVICE SD_INLINE T apply(T go, T x, T fwd, LongType, LongType zeros) {
    using A = typename AccFloat<T, T>::type;
    const A ax = static_cast<A>(x);
    const A scaled = static_cast<A>(go) * static_cast<A>(fwd);
    if (ax == static_cast<A>(0)) return zeros == 1 ? static_cast<T>(scaled) : static_cast<T>(0);
    return zeros == 0 ? static_cast<T>(scaled / ax) : static_cast<T>(0);
  }
};

struct GradSum {
  static constexpr bool kNeedsForward = false;
  static constexpr bool kNeedsCount = false;
  static constexpr bool kNeedsZeros = false;
  template <typename T>
  static SD_HOST_DEVICE SD_INLINE T apply(T go, T, T, LongType, LongType) {
    return go;
  }
};

struct GradMean {
  static constexpr bool kNeedsForward = false;
  static constexpr bool kNeedsCount = true;
  static constexpr bool kNeedsZeros = false;
  template <typename T>
  static SD_HOST_DEVICE SD_INLINE T apply(T go, T, T, LongType count, LongType) {
    using A = typename AccFloat<T, T>::type;
    return static_cast<T>(static_cast<A>(go) / static_cast<A>(count));
  }
};

struct GradSqrtN {
  static constexpr bool kNeedsForward = false;
  static constexpr bool kNeedsCount = true;
  static constexpr bool kNeedsZeros = false;
  template <typename T>
  static SD_HOST_DEVICE SD_INLINE T apply(T go, T, T, LongType count, LongType) {
    using A = typename AccFloat<T, T>::type;
    return static_cast<T>(static_cast<A>(go) / segSqrt(static_cast<A>(count)));
  }
};

// ============================================================================ //
// Host (CPU) range functions: plain loops over a range of output / input elements. helpers/cpu/segment.cpp splits
// the ranges over threads; the standalone checks of these semantics call them directly.
// ============================================================================ //
namespace host {

// begin / end row of every segment of SORTED ids; a class no id names has begin == end == 0. Ids outside [0, C) are
// ignored (they were rejected by the validation; this keeps a caller that skipped it inside its buffers).
inline void buildRanges(const LongType* ids, LongType n, LongType C, LongType* starts, LongType* ends) {
  for (LongType s = 0; s < C; ++s) {
    starts[s] = 0;
    ends[s] = 0;
  }
  for (LongType i = 0; i < n; ++i) {
    const LongType s = ids[i];
    if (s < 0 || s >= C) continue;
    if (i == 0 || ids[i - 1] != s) starts[s] = i;
    ends[s] = i + 1;
  }
}

// rows per class
inline void countIds(const LongType* ids, LongType n, LongType C, LongType* counts) {
  for (LongType s = 0; s < C; ++s) counts[s] = 0;
  for (LongType i = 0; i < n; ++i) {
    const LongType s = ids[i];
    if (s >= 0 && s < C) counts[s]++;
  }
}

// Output elements [gBegin, gEnd) of a sorted segment op: element g is column g % K of segment g / K.
template <typename Op, typename X, typename Z>
inline void sortedForwardRange(const X* x, const SegMat& xm, const LongType* starts, const LongType* ends, LongType K,
                               Z* z, const SegMat& zm, LongType gBegin, LongType gEnd) {
  using A = typename Op::template AccOf<X, Z>::type;
  for (LongType g = gBegin; g < gEnd; ++g) {
    const LongType s = g / K;
    const LongType e = g - s * K;
    const LongType begin = starts[s];
    const LongType rows = ends[s] - begin;
    const LongType xc = xm.colOffset(e);
    A acc = Op::template identity<A>();
    for (LongType r = begin; r < begin + rows; ++r) Op::template combine<A>(acc, static_cast<A>(x[r * xm.rowStride + xc]));
    z[s * zm.rowStride + zm.colOffset(e)] =
        rows > 0 ? static_cast<Z>(Op::template finalize<A>(acc, rows)) : Op::template emptySorted<Z>();
  }
}

// Accumulates columns [eBegin, eEnd) of every row of x into acc[C, K] (acc holds the identity); the rows are visited
// in order, so the result does not depend on how the columns are split over threads.
template <typename Op, typename X, typename A>
inline void unsortedScatterColumns(const X* x, const SegMat& xm, const LongType* ids, LongType n, LongType C,
                                   LongType K, A* acc, LongType eBegin, LongType eEnd) {
  for (LongType r = 0; r < n; ++r) {
    const LongType s = ids[r];
    if (s < 0 || s >= C) continue;
    A* a = acc + s * K;
    const X* row = x + r * xm.rowStride;
    for (LongType e = eBegin; e < eEnd; ++e) Op::template combine<A>(a[e], static_cast<A>(row[xm.colOffset(e)]));
  }
}

// Output elements [gBegin, gEnd) of an unsorted op from the finished accumulator acc[C, K].
template <typename Op, typename A, typename Z>
inline void unsortedFinalizeRange(const A* acc, const LongType* counts, LongType K, Z* z, const SegMat& zm,
                                  LongType gBegin, LongType gEnd) {
  for (LongType g = gBegin; g < gEnd; ++g) {
    const LongType s = g / K;
    const LongType e = g - s * K;
    Z out;
    if constexpr (Op::kNeedsCount) {
      const LongType count = counts[s];
      out = count > 0 ? static_cast<Z>(Op::template finalize<A>(acc[g], count)) : Op::template emptyUnsorted<Z>();
    } else {
      out = static_cast<Z>(Op::template finalize<A>(acc[g], 0));
    }
    z[s * zm.rowStride + zm.colOffset(e)] = out;
  }
}

// Elements [gBegin, gEnd) (row g / K, column g % K) of the gradient w.r.t. the rows of the input. x may be null when
// the policy does not read it; fwd (dense [C, K]) when it does not read the forward result; counts (one per class) and
// zeros (dense [C, K]) likewise.
template <typename Grad, typename T>
inline void backpropRange(const T* x, const SegMat& xm, const T* gradOut, const SegMat& gm, const T* fwd,
                          const LongType* counts, const LongType* zeros, const LongType* ids, LongType C, LongType K,
                          T* z, const SegMat& zm, LongType gBegin, LongType gEnd) {
  for (LongType g = gBegin; g < gEnd; ++g) {
    const LongType r = g / K;
    const LongType e = g - r * K;
    const LongType s = ids[r];
    T out = static_cast<T>(0);
    if (s >= 0 && s < C) {
      const T go = gradOut[s * gm.rowStride + gm.colOffset(e)];
      T xv = static_cast<T>(0);
      T fv = static_cast<T>(0);
      LongType count = 0;
      LongType zeroCount = 0;
      if constexpr (Grad::kNeedsForward) {
        xv = x[r * xm.rowStride + xm.colOffset(e)];
        fv = fwd[s * K + e];
      }
      if constexpr (Grad::kNeedsCount) count = counts[s];
      if constexpr (Grad::kNeedsZeros) zeroCount = zeros[s * K + e];
      out = Grad::template apply<T>(go, xv, fv, count, zeroCount);
    }
    z[r * zm.rowStride + zm.colOffset(e)] = out;
  }
}

}  // namespace host

}  // namespace segment_sem
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_SEGMENT_SEMANTICS_H
