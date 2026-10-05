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
//  Shared implementation of the CUDA segment ops (segment_{sum,mean,max,min,prod},
//  unsorted_segment_{sum,mean,max,min,prod,sqrt_n} and their backprops). The arithmetic is the one of
//  segment_semantics.h (the CPU helpers run the same policies); this file adds the kernels:
//
//    sorted ids    gather. The ids are one run per segment (validated), so every output element is reduced by one
//                  thread in row order (deterministic, bit-identical to the CPU for rows x columns inputs), and a
//                  vector input is reduced by one block per segment with a fixed shared-memory tree (deterministic
//                  for a given launch).
//    unsorted ids  scatter into a [classes, columns] accumulator of the policy's accumulator type with atomics that
//                  do not go through templatemath.h (native atomicAdd where it exists, a compare-and-swap loop on
//                  the bit pattern otherwise, for HALF/BFLOAT16 in FLOAT), then one kernel finishes every segment
//                  (mean: one division; empty segments get their value).
//    backprop      one kernel over the elements of the input.
//
//  Every kernel strides over its work with 64 bit indices, so the launch dimensions of LaunchDims.cu
//  (segmentDims, segmentTad, segmentBpDims, segmentBpTad, getFillUpSegmentsDims) only cap the grid. All arrays are read
//  through their own shapes and strides (SegMat), so any rank, order, offset or stepped view is correct; the ids of any
//  integer dtype / rank are first read into one dense int64 sequence (segmentReadIds). A segment id outside the range of
//  classes is dropped by every kernel (the ops reject such ids up front; during a graph capture nothing can be read
//  back, and nothing indexes with them).
//
#ifndef LIBND4J_SEGMENT_OPS_CUH
#define LIBND4J_SEGMENT_OPS_CUH

#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/segment.h>
#include <ops/declarable/helpers/segment_common.h>
#include <ops/declarable/helpers/segment_semantics.h>
#include <system/selective_rendering.h>

#include <string.h>

#include <string>
#include <type_traits>

namespace sd {
namespace ops {
namespace helpers {
namespace segment_ops {

using segment_sem::SegMat;

// the sorted linear kernel reduces in a static shared array of this many partials: its launches use at most this
// many threads, a power of two (LaunchDims.cu: segmentDims)
static constexpr int kSegmentTreeThreads = 256;

static SD_INLINE int clampInt(LongType value) {
  if (value < 0) return 0;
  if (value > static_cast<LongType>(2147483647)) return 2147483647;
  return static_cast<int>(value);
}

static SD_INLINE void checkLaunch(cudaStream_t* stream, const char* what) {
  if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode(what);
}

static SD_INLINE SegMat segMat(NDArray* array) {
  return segment_sem::makeSegMat(array->shapeInfo(), array->specialShapeInfo());
}

// ============================================================================ //
// Atomics (device). Native where CUDA has them, a compare-and-swap loop on the bit pattern otherwise; the operand is
// always the accumulator type, which is at least 32 bits wide, so no sub-word atomics are involved.
// ============================================================================ //
template <typename To, typename From>
static SD_DEVICE SD_INLINE To bitCast(const From& from) {
  static_assert(sizeof(To) == sizeof(From), "bitCast needs equally sized types");
  To to;
  memcpy(&to, &from, sizeof(To));
  return to;
}

template <typename Op, typename A, typename U>
static SD_DEVICE SD_INLINE void atomicCasCombine(A* address, A value) {
  U* word = reinterpret_cast<U*>(address);
  U old = *word;
  U assumed;
  do {
    assumed = old;
    A updated = bitCast<A>(assumed);
    Op::template combine<A>(updated, value);
    const U wanted = bitCast<U>(updated);
    if (wanted == assumed) break;  // the value is already in (a max of a smaller value, a product by one)
    old = atomicCAS(word, assumed, wanted);
  } while (assumed != old);
}

template <typename Op, typename A>
static SD_DEVICE SD_INLINE void atomicCombine(A* address, A value) {
  if constexpr (Op::kIsAdd) {
    if constexpr (std::is_same<A, float>::value) {
      atomicAdd(address, value);
    } else if constexpr (std::is_same<A, double>::value) {
      atomicAdd(address, value);
    } else if constexpr (sizeof(A) == 4) {
      atomicAdd(reinterpret_cast<unsigned int*>(address), static_cast<unsigned int>(value));
    } else {
      atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value));
    }
  } else if constexpr (sizeof(A) == 4) {
    atomicCasCombine<Op, A, unsigned int>(address, value);
  } else {
    atomicCasCombine<Op, A, unsigned long long>(address, value);
  }
}

// ============================================================================ //
// Forward kernels
// ============================================================================ //

// Sorted ids, vector (one value per row): one block per segment (grid-stride over the segments), a fixed
// shared-memory tree. blockDim.x is a power of two <= kSegmentTreeThreads.
template <typename Op, typename X, typename Z>
static SD_KERNEL void sortedLinearKernel(const X* x, LongType xStride, const LongType* begin, const LongType* end,
                                         LongType numClasses, Z* z, LongType zStride) {
  using A = typename Op::template AccOf<X, Z>::type;
  __shared__ A partial[kSegmentTreeThreads];
  for (LongType s = blockIdx.x; s < numClasses; s += gridDim.x) {
    const LongType first = begin[s];
    const LongType last = end[s];
    A acc = Op::template identity<A>();
    for (LongType r = first + threadIdx.x; r < last; r += blockDim.x)
      Op::template combine<A>(acc, static_cast<A>(x[r * xStride]));
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) Op::template combine<A>(partial[threadIdx.x], partial[threadIdx.x + offset]);
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const LongType rows = last - first;
      z[s * zStride] =
          rows > 0 ? static_cast<Z>(Op::template finalize<A>(partial[0], rows)) : Op::template emptySorted<Z>();
    }
    __syncthreads();  // the partials are reused by the next segment of this block
  }
}

// Sorted ids, rows x columns: one thread per output element reduces the rows of its segment in order.
template <typename Op, typename X, typename Z>
static SD_KERNEL void sortedMatrixKernel(const X* x, SegMat xm, const LongType* begin, const LongType* end,
                                         LongType numClasses, LongType K, Z* z, SegMat zm) {
  using A = typename Op::template AccOf<X, Z>::type;
  const LongType total = numClasses * K;
  for (LongType g = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; g < total;
       g += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType s = g / K;
    const LongType e = g - s * K;
    const LongType first = begin[s];
    const LongType rows = end[s] - first;
    const LongType xc = xm.colOffset(e);
    A acc = Op::template identity<A>();
    for (LongType r = first; r < first + rows; ++r) Op::template combine<A>(acc, static_cast<A>(x[r * xm.rowStride + xc]));
    z[s * zm.rowStride + zm.colOffset(e)] =
        rows > 0 ? static_cast<Z>(Op::template finalize<A>(acc, rows)) : Op::template emptySorted<Z>();
  }
}

template <typename Op, typename A>
static SD_KERNEL void fillIdentityKernel(A* data, LongType length) {
  const A value = Op::template identity<A>();
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < length;
       i += static_cast<LongType>(gridDim.x) * blockDim.x)
    data[i] = value;
}

// Unsorted ids: every element of the first n rows is combined into its segment's accumulator.
template <typename Op, typename X, typename A>
static SD_KERNEL void unsortedScatterKernel(const X* x, SegMat xm, const LongType* ids, LongType n,
                                            LongType numClasses, LongType K, A* acc) {
  const LongType total = n * K;
  for (LongType g = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; g < total;
       g += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType r = g / K;
    const LongType e = g - r * K;
    const LongType s = ids[r];
    if (s < 0 || s >= numClasses) continue;
    atomicCombine<Op, A>(acc + s * K + e, static_cast<A>(x[r * xm.rowStride + xm.colOffset(e)]));
  }
}

// Unsorted ids: one thread per output element finishes its segment (an empty one gets the op's value for it).
template <typename Op, typename A, typename Z>
static SD_KERNEL void unsortedFinalizeKernel(const A* acc, const unsigned long long* counts, LongType numClasses,
                                             LongType K, Z* z, SegMat zm) {
  const LongType total = numClasses * K;
  for (LongType g = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; g < total;
       g += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType s = g / K;
    const LongType e = g - s * K;
    Z out;
    if constexpr (Op::kNeedsCount) {
      const LongType count = static_cast<LongType>(counts[s]);
      out = count > 0 ? static_cast<Z>(Op::template finalize<A>(acc[g], count)) : Op::template emptyUnsorted<Z>();
    } else {
      out = static_cast<Z>(Op::template finalize<A>(acc[g], 0));
    }
    z[s * zm.rowStride + zm.colOffset(e)] = out;
  }
}

// ============================================================================ //
// Backprop kernel: element g (row g / K, column g % K) of the gradient w.r.t. the input
// ============================================================================ //
template <typename Grad, typename T>
static SD_KERNEL void backpropKernel(const T* x, SegMat xm, const T* gradOut, SegMat gm, const T* fwd,
                                     const unsigned long long* counts, const LongType* zeros, const LongType* ids,
                                     LongType n, LongType numClasses, LongType K, T* z, SegMat zm) {
  const LongType total = n * K;
  for (LongType g = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; g < total;
       g += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType r = g / K;
    const LongType e = g - r * K;
    const LongType s = ids[r];
    T out = static_cast<T>(0);
    if (s >= 0 && s < numClasses) {
      const T go = gradOut[s * gm.rowStride + gm.colOffset(e)];
      T xv = static_cast<T>(0);
      T fv = static_cast<T>(0);
      LongType count = 0;
      LongType zeroCount = 0;
      if constexpr (Grad::kNeedsForward) {
        xv = x[r * xm.rowStride + xm.colOffset(e)];
        fv = fwd[s * K + e];
      }
      if constexpr (Grad::kNeedsCount) count = static_cast<LongType>(counts[s]);
      if constexpr (Grad::kNeedsZeros) zeroCount = zeros[s * K + e];
      out = Grad::template apply<T>(go, xv, fv, count, zeroCount);
    }
    z[r * zm.rowStride + zm.colOffset(e)] = out;
  }
}

// ============================================================================ //
// Host
// ============================================================================ //

// sorted ids: output [C, ...] from input [n, ...]
template <typename Op, typename X, typename Z>
static void sortedForward_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  const LongType C = output->sizeAt(0);
  const SegMat xm = segMat(input);
  const SegMat zm = segMat(output);
  const LongType K = zm.inner;
  if (C == 0 || K == 0) return;
  auto stream = context->getCudaStream();
  const LongType n = indices->lengthOf();

  PointersManager manager(context, "segmentSortedForward");
  auto* ids = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(n) * sizeof(LongType)));
  auto* begin = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(C) * sizeof(LongType)));
  auto* end = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(C) * sizeof(LongType)));
  segmentReadIds(context, indices, ids);
  segmentBuildRanges(context, ids, n, C, begin, end);

  const X* x = reinterpret_cast<const X*>(input->specialBuffer());
  Z* z = reinterpret_cast<Z*>(output->specialBuffer());
  if (K == 1) {
    dim3 dims = segmentDims(clampInt(C), clampInt(n));
    sortedLinearKernel<Op, X, Z><<<dims.x, dims.y, dims.z, *stream>>>(x, xm.rowStride, begin, end, C, z, zm.rowStride);
    checkLaunch(stream, "sortedLinearKernel failed");
  } else {
    dim3 dims = segmentTad(clampInt(C * K));
    sortedMatrixKernel<Op, X, Z><<<dims.x, dims.y, dims.z, *stream>>>(x, xm, begin, end, C, K, z, zm);
    checkLaunch(stream, "sortedMatrixKernel failed");
  }
}

// unsorted ids: accumulates the first n rows of x into [C, K] and writes the finished segments to z
template <typename Op, typename X, typename Z>
static void unsortedForwardInto(LaunchContext* context, PointersManager& manager, const X* x, const SegMat& xm,
                                const LongType* ids, LongType n, LongType C, LongType K, Z* z, const SegMat& zm) {
  using A = typename Op::template AccOf<X, Z>::type;
  if (C == 0 || K == 0) return;
  auto stream = context->getCudaStream();
  auto* acc = reinterpret_cast<A*>(manager.allocateDevMem(static_cast<size_t>(C * K) * sizeof(A)));
  unsigned long long* counts = nullptr;
  if constexpr (Op::kNeedsCount) {
    counts = reinterpret_cast<unsigned long long*>(manager.allocateDevMem(static_cast<size_t>(C) * sizeof(unsigned long long)));
    segmentCountIds(context, ids, n, C, counts);
  }
  dim3 flat = segmentTad(clampInt(C * K));
  fillIdentityKernel<Op, A><<<flat.x, flat.y, flat.z, *stream>>>(acc, C * K);
  checkLaunch(stream, "fillIdentityKernel failed");
  if (n > 0) {
    dim3 dims = segmentBpTad(clampInt(n), clampInt(n * K));
    unsortedScatterKernel<Op, X, A><<<dims.x, dims.y, dims.z, *stream>>>(x, xm, ids, n, C, K, acc);
    checkLaunch(stream, "unsortedScatterKernel failed");
  }
  unsortedFinalizeKernel<Op, A, Z><<<flat.x, flat.y, flat.z, *stream>>>(acc, counts, C, K, z, zm);
  checkLaunch(stream, "unsortedFinalizeKernel failed");
}

template <typename Op, typename X, typename Z>
static void unsortedForward_(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                             NDArray* output) {
  const SegMat xm = segMat(input);
  const SegMat zm = segMat(output);
  const LongType K = zm.inner;
  // the classes are the rows of the output (the op's shape function made them numOfClasses)
  const LongType C = output->sizeAt(0);
  if (C == 0 || K == 0) return;
  // an id list longer than the rows of x has no row to read (the op validates the lengths)
  LongType n = indices->lengthOf();
  if (n > input->sizeAt(0)) n = input->sizeAt(0);

  PointersManager manager(context, "segmentUnsortedForward");
  auto* ids = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(indices->lengthOf()) * sizeof(LongType)));
  segmentReadIds(context, indices, ids);
  unsortedForwardInto<Op, X, Z>(context, manager, reinterpret_cast<const X*>(input->specialBuffer()), xm, ids, n, C, K,
                                reinterpret_cast<Z*>(output->specialBuffer()), zm);
}

// backprop: output (gradient w.r.t. the rows of the input) from gradOut ([C, ...], the gradient of the segments)
template <typename Op, typename Grad, typename T>
static Status backprop_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
  const LongType C = gradOut->sizeAt(0);
  const SegMat xm = segMat(input);
  const SegMat gm = segMat(gradOut);
  const SegMat zm = segMat(output);
  const LongType K = zm.inner;
  const LongType n = indices->lengthOf();
  if (n == 0 || K == 0) return Status::OK;
  auto stream = context->getCudaStream();

  PointersManager manager(context, "segmentBackprop");
  auto* ids = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(n) * sizeof(LongType)));
  segmentReadIds(context, indices, ids);
  unsigned long long* counts = nullptr;
  if constexpr (Grad::kNeedsCount) {
    if (C > 0) {
      counts = reinterpret_cast<unsigned long long*>(manager.allocateDevMem(static_cast<size_t>(C) * sizeof(unsigned long long)));
      segmentCountIds(context, ids, n, C, counts);
    }
  }
  const T* x = nullptr;
  T* fwd = nullptr;
  LongType* zeros = nullptr;
  if constexpr (Grad::kNeedsForward) {
    // the forward result of every segment, recomputed from the input (dense [C, K])
    x = reinterpret_cast<const T*>(input->specialBuffer());
    if (C > 0) {
      fwd = reinterpret_cast<T*>(manager.allocateDevMem(static_cast<size_t>(C * K) * sizeof(T)));
      unsortedForwardInto<Op, T, T>(context, manager, x, xm, ids, n, C, K, fwd, segment_sem::denseSegMat(K));
      if constexpr (Grad::kNeedsZeros) {
        // the zeros of every segment's column (dense [C, K]): a product's gradient cannot divide by a zero element
        zeros = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(C * K) * sizeof(LongType)));
        unsortedForwardInto<segment_sem::SegZeroCount, T, LongType>(context, manager, x, xm, ids, n, C, K, zeros,
                                                                    segment_sem::denseSegMat(K));
      }
    }
  }
  dim3 dims = segmentBpDims(clampInt(C * K), clampInt(n * K));
  backpropKernel<Grad, T><<<dims.x, dims.y, dims.z, *stream>>>(
      x, xm, reinterpret_cast<const T*>(gradOut->specialBuffer()), gm, fwd, counts, zeros, ids, n, C, K,
      reinterpret_cast<T*>(output->specialBuffer()), zm);
  checkLaunch(stream, "backpropKernel failed");
  return Status::OK;
}

}  // namespace segment_ops
}  // namespace helpers
}  // namespace ops
}  // namespace sd

// ============================================================================ //
// Instantiation macros: a segment_<op>.cu is a thin TU using them.
//   SEGMENT_OP_SAME_TYPE(NAME, OP)        segment<NAME>Functor + unsortedSegment<NAME>Functor, output type == input
//                                         type (sum, prod, max, min)
//   SEGMENT_OP_FLOAT_OUT(NAME, OP)        segment<NAME>Functor + unsortedSegment<NAME>Functor, floating output of any
//                                         numeric input (mean)
//   SEGMENT_OP_BACKPROP_NUMERIC / _FLOAT  segment<NAME>FunctorBP + unsortedSegment<NAME>FunctorBP
// (the type lists are spelled out in each macro: a list passed as a macro argument is expanded before the selector
// sees it)
// ============================================================================ //

#define SEGMENT_OP_SAME_TYPE(NAME, OP)                                                                                \
  namespace sd {                                                                                                      \
  namespace ops {                                                                                                     \
  namespace helpers {                                                                                                 \
  namespace segment_ops {                                                                                             \
  template <typename T>                                                                                               \
  static void sorted##NAME##_(LaunchContext* c, NDArray* in, NDArray* idx, NDArray* out) {                            \
    sortedForward_<segment_sem::OP, T, T>(c, in, idx, out);                                                           \
  }                                                                                                                   \
  template <typename T>                                                                                               \
  static void unsorted##NAME##_(LaunchContext* c, NDArray* in, NDArray* idx, LongType n, NDArray* out) {              \
    unsortedForward_<segment_sem::OP, T, T>(c, in, idx, n, out);                                                      \
  }                                                                                                                   \
  }                                                                                                                   \
  void segment##NAME##Functor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {          \
    NDArray::prepareSpecialUse({output}, {input, indices});                                                          \
    BUILD_SINGLE_SELECTOR(input->dataType(), segment_ops::sorted##NAME##_, (context, input, indices, output),        \
                          SD_NUMERIC_TYPES);                                                                          \
    NDArray::registerSpecialUse({output}, {input, indices});                                                         \
  }                                                                                                                   \
  void unsortedSegment##NAME##Functor(LaunchContext* context, NDArray* input, NDArray* indices,                      \
                                      LongType numOfClasses, NDArray* output) {                                      \
    NDArray::prepareSpecialUse({output}, {input, indices});                                                          \
    BUILD_SINGLE_SELECTOR(input->dataType(), segment_ops::unsorted##NAME##_,                                         \
                          (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES);                         \
    NDArray::registerSpecialUse({output}, {input, indices});                                                         \
  }                                                                                                                   \
  }                                                                                                                   \
  }                                                                                                                   \
  }

#define SEGMENT_OP_FLOAT_OUT(NAME, OP)                                                                                \
  namespace sd {                                                                                                      \
  namespace ops {                                                                                                     \
  namespace helpers {                                                                                                 \
  namespace segment_ops {                                                                                             \
  template <typename X, typename Z>                                                                                   \
  static void sorted##NAME##_(LaunchContext* c, NDArray* in, NDArray* idx, NDArray* out) {                            \
    sortedForward_<segment_sem::OP, X, Z>(c, in, idx, out);                                                           \
  }                                                                                                                   \
  template <typename X, typename Z>                                                                                   \
  static void unsorted##NAME##_(LaunchContext* c, NDArray* in, NDArray* idx, LongType n, NDArray* out) {              \
    unsortedForward_<segment_sem::OP, X, Z>(c, in, idx, n, out);                                                      \
  }                                                                                                                   \
  }                                                                                                                   \
  void segment##NAME##Functor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {          \
    NDArray::prepareSpecialUse({output}, {input, indices});                                                          \
    BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), segment_ops::sorted##NAME##_,                       \
                          (context, input, indices, output), SD_NUMERIC_TYPES, SD_FLOAT_TYPES);                       \
    NDArray::registerSpecialUse({output}, {input, indices});                                                         \
  }                                                                                                                   \
  void unsortedSegment##NAME##Functor(LaunchContext* context, NDArray* input, NDArray* indices,                      \
                                      LongType numOfClasses, NDArray* output) {                                      \
    NDArray::prepareSpecialUse({output}, {input, indices});                                                          \
    BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), segment_ops::unsorted##NAME##_,                     \
                          (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_FLOAT_TYPES);         \
    NDArray::registerSpecialUse({output}, {input, indices});                                                         \
  }                                                                                                                   \
  }                                                                                                                   \
  }                                                                                                                   \
  }

#define SEGMENT_OP_BACKPROP_NUMERIC(NAME, OP, GRAD)                                                                   \
  namespace sd {                                                                                                      \
  namespace ops {                                                                                                     \
  namespace helpers {                                                                                                 \
  namespace segment_ops {                                                                                             \
  template <typename T>                                                                                               \
  static Status bp##NAME##_(LaunchContext* c, NDArray* in, NDArray* idx, NDArray* go, NDArray* out) {                 \
    return backprop_<segment_sem::OP, segment_sem::GRAD, T>(c, in, idx, go, out);                                     \
  }                                                                                                                   \
  }                                                                                                                   \
  Status segment##NAME##FunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,        \
                                  NDArray* output) {                                                                  \
    NDArray::prepareSpecialUse({output}, {input, indices, gradOut});                                                 \
    BUILD_SINGLE_SELECTOR(output->dataType(), segment_ops::bp##NAME##_, (context, input, indices, gradOut, output),  \
                          SD_NUMERIC_TYPES);                                                                          \
    NDArray::registerSpecialUse({output}, {input, indices, gradOut});                                                \
    return Status::OK;                                                                                                \
  }                                                                                                                   \
  Status unsortedSegment##NAME##FunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,                  \
                                          NDArray* gradOut, LongType numOfClasses, NDArray* output) {                \
    NDArray::prepareSpecialUse({output}, {input, indices, gradOut});                                                 \
    BUILD_SINGLE_SELECTOR(output->dataType(), segment_ops::bp##NAME##_, (context, input, indices, gradOut, output),  \
                          SD_NUMERIC_TYPES);                                                                          \
    NDArray::registerSpecialUse({output}, {input, indices, gradOut});                                                \
    return Status::OK;                                                                                                \
  }                                                                                                                   \
  }                                                                                                                   \
  }                                                                                                                   \
  }

#define SEGMENT_OP_BACKPROP_FLOAT(NAME, OP, GRAD)                                                                     \
  namespace sd {                                                                                                      \
  namespace ops {                                                                                                     \
  namespace helpers {                                                                                                 \
  namespace segment_ops {                                                                                             \
  template <typename T>                                                                                               \
  static Status bp##NAME##_(LaunchContext* c, NDArray* in, NDArray* idx, NDArray* go, NDArray* out) {                 \
    return backprop_<segment_sem::OP, segment_sem::GRAD, T>(c, in, idx, go, out);                                     \
  }                                                                                                                   \
  }                                                                                                                   \
  Status segment##NAME##FunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,        \
                                  NDArray* output) {                                                                  \
    NDArray::prepareSpecialUse({output}, {input, indices, gradOut});                                                 \
    BUILD_SINGLE_SELECTOR(output->dataType(), segment_ops::bp##NAME##_, (context, input, indices, gradOut, output),  \
                          SD_FLOAT_TYPES);                                                                            \
    NDArray::registerSpecialUse({output}, {input, indices, gradOut});                                                \
    return Status::OK;                                                                                                \
  }                                                                                                                   \
  Status unsortedSegment##NAME##FunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,                  \
                                          NDArray* gradOut, LongType numOfClasses, NDArray* output) {                \
    NDArray::prepareSpecialUse({output}, {input, indices, gradOut});                                                 \
    BUILD_SINGLE_SELECTOR(output->dataType(), segment_ops::bp##NAME##_, (context, input, indices, gradOut, output),  \
                          SD_FLOAT_TYPES);                                                                            \
    NDArray::registerSpecialUse({output}, {input, indices, gradOut});                                                \
    return Status::OK;                                                                                                \
  }                                                                                                                   \
  }                                                                                                                   \
  }                                                                                                                   \
  }

#endif  // LIBND4J_SEGMENT_OPS_CUH
