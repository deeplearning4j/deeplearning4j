/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
//  @author GS <sgazeos@gmail.com>
//
//  CPU implementation of segment_{sum,mean,max,min,prod}, unsorted_segment_{sum,mean,max,min,prod,sqrt_n} and their
//  backprops. The arithmetic (accumulator types, identities, the value of an empty segment, NaN, one division for a
//  mean) lives in segment_semantics.h and is shared with the CUDA kernels; this file reads the arrays through their
//  own shapes and strides (any rank, order, offset or stepped view), validates and splits the ranges over threads.
//
#include <execution/Threads.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/segment.h>
#include <ops/declarable/helpers/segment_semantics.h>

#include <vector>

#if NOT_EXCLUDED(OP_segment)
namespace sd {
namespace ops {
namespace helpers {

namespace segment_cpu {

using namespace segment_sem;
namespace shost = segment_sem::host;

// The segment ids of any integer type, rank and strides as one dense sequence of signed 64 bit values.
template <typename I>
static void readIds_(NDArray* indices, LongType* dst) {
  const I* ids = indices->bufferAsT<I>();
  const LongType n = indices->lengthOf();
  const LongType rank = indices->rankOf();
  const LongType* shp = shape::shapeOf(indices->shapeInfo());
  const LongType* str = shape::stride(indices->shapeInfo());
  for (LongType i = 0; i < n; ++i) dst[i] = idToLong<I>(ids[logicalOffset(i, rank, shp, str)]);
}

static void readIds(NDArray* indices, std::vector<LongType>& out) {
  out.resize(static_cast<size_t>(indices->lengthOf()));
  if (out.empty()) return;
  BUILD_SINGLE_SELECTOR(indices->dataType(), readIds_, (indices, out.data()), SD_INTEGER_TYPES);
}

static SegMat matOf(NDArray* array) { return makeSegMat(array->shapeInfo(), array->shapeInfo()); }

// ---------------------------------------------------------------------------------------------------------- //
// forward
// ---------------------------------------------------------------------------------------------------------- //

// sorted ids: every segment is one run of rows, reduced in row order
template <typename Op, typename X, typename Z>
static void sortedForward_(NDArray* input, NDArray* indices, NDArray* output) {
  const LongType C = output->sizeAt(0);
  const SegMat xm = matOf(input);
  const SegMat zm = matOf(output);
  const LongType K = zm.inner;
  if (C == 0 || K == 0) return;

  std::vector<LongType> ids;
  readIds(indices, ids);
  const LongType n = static_cast<LongType>(ids.size());
  std::vector<LongType> starts(static_cast<size_t>(C)), ends(static_cast<size_t>(C));
  shost::buildRanges(ids.data(), n, C, starts.data(), ends.data());

  const X* x = input->bufferAsT<X>();
  Z* z = output->bufferAsT<Z>();
  auto func = PRAGMA_THREADS_FOR {
    shost::sortedForwardRange<Op, X, Z>(x, xm, starts.data(), ends.data(), K, z, zm, start, stop);
  };
  samediff::Threads::parallel_for(func, 0, C * K);
}

// any ids: rows are accumulated in row order into a [C, K] accumulator, then every segment is finished once
template <typename Op, typename X, typename Z>
static void unsortedForwardInto(const X* x, const SegMat& xm, const std::vector<LongType>& ids, LongType C, Z* z,
                                const SegMat& zm) {
  using A = typename Op::template AccOf<X, Z>::type;
  const LongType K = zm.inner;
  const LongType n = static_cast<LongType>(ids.size());
  std::vector<A> acc(static_cast<size_t>(C * K), Op::template identity<A>());
  std::vector<LongType> counts;
  if constexpr (Op::kNeedsCount) {
    counts.resize(static_cast<size_t>(C));
    shost::countIds(ids.data(), n, C, counts.data());
  }
  // columns are split over threads, every thread visits all rows: the sum of a segment never depends on the split
  auto scatter = PRAGMA_THREADS_FOR {
    shost::unsortedScatterColumns<Op, X, A>(x, xm, ids.data(), n, C, K, acc.data(), start, stop);
  };
  samediff::Threads::parallel_for(scatter, 0, K);
  auto finish = PRAGMA_THREADS_FOR {
    shost::unsortedFinalizeRange<Op, A, Z>(acc.data(), counts.data(), K, z, zm, start, stop);
  };
  samediff::Threads::parallel_for(finish, 0, C * K);
}

template <typename Op, typename X, typename Z>
static void unsortedForward_(NDArray* input, NDArray* indices, LongType numOfClasses, NDArray* output) {
  const SegMat xm = matOf(input);
  const SegMat zm = matOf(output);
  // the classes are the rows of the output (the op's shape function made them numOfClasses)
  const LongType C = output->sizeAt(0);
  if (C == 0 || zm.inner == 0) return;
  std::vector<LongType> ids;
  readIds(indices, ids);
  // an id list longer than the rows of x has no row to read (the op validates the lengths)
  if (static_cast<LongType>(ids.size()) > input->sizeAt(0)) ids.resize(static_cast<size_t>(input->sizeAt(0)));
  unsortedForwardInto<Op, X, Z>(input->bufferAsT<X>(), xm, ids, C, output->bufferAsT<Z>(), zm);
}

// ---------------------------------------------------------------------------------------------------------- //
// backprop: output = gradient w.r.t. the rows of the input, gradOut = gradient of the segments ([C, ...])
// ---------------------------------------------------------------------------------------------------------- //
template <typename Op, typename Grad, typename T>
static Status backprop_(NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
  const LongType C = gradOut->sizeAt(0);
  const SegMat xm = matOf(input);
  const SegMat gm = matOf(gradOut);
  const SegMat zm = matOf(output);
  const LongType K = zm.inner;
  std::vector<LongType> ids;
  readIds(indices, ids);
  const LongType n = static_cast<LongType>(ids.size());
  if (n == 0 || K == 0) return Status::OK;

  std::vector<LongType> counts;
  if constexpr (Grad::kNeedsCount) {
    counts.resize(static_cast<size_t>(C));
    shost::countIds(ids.data(), n, C, counts.data());
  }
  std::vector<T> forward;
  std::vector<LongType> zeros;
  const T* x = nullptr;
  if constexpr (Grad::kNeedsForward) {
    // the forward result of every segment, recomputed from the input (dense [C, K])
    forward.assign(static_cast<size_t>(C * K), static_cast<T>(0));
    x = input->bufferAsT<T>();
    unsortedForwardInto<Op, T, T>(x, xm, ids, C, forward.data(), denseSegMat(K));
    if constexpr (Grad::kNeedsZeros) {
      // the zeros of every segment's column (dense [C, K]): a product's gradient cannot divide by a zero element
      zeros.assign(static_cast<size_t>(C * K), 0);
      unsortedForwardInto<SegZeroCount, T, LongType>(x, xm, ids, C, zeros.data(), denseSegMat(K));
    }
  }
  const T* go = gradOut->bufferAsT<T>();
  T* z = output->bufferAsT<T>();
  auto func = PRAGMA_THREADS_FOR {
    shost::backpropRange<Grad, T>(x, xm, go, gm, forward.data(), counts.data(), zeros.data(), ids.data(), C, K, z, zm,
                                  start, stop);
  };
  samediff::Threads::parallel_for(func, 0, n * K);
  return Status::OK;
}

}  // namespace segment_cpu

// ---------------------------------------------------------------------------------------------------------- //
// Index validation
// ---------------------------------------------------------------------------------------------------------- //

// Sorted ids: no negative id and no id smaller than the one before it. On failure previous holds the id before the
// offending one (the offending one itself for a negative first id) and offending that id.
bool segmentIndicesValidate(LaunchContext* context, NDArray* indices, LongType& previous, LongType& offending) {
  NDArray::preparePrimaryUse({}, {indices});
  std::vector<LongType> ids;
  segment_cpu::readIds(indices, ids);
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0 || (i > 0 && ids[i] < ids[i - 1])) {
      previous = i > 0 ? ids[i - 1] : ids[i];
      offending = ids[i];
      return false;
    }
  }
  return true;
}

// Unsorted ids: every id in [0, numOfClasses). On failure output holds the first offending id, on success
// numOfClasses.
bool unsortedSegmentIndicesValidate(LaunchContext* context, NDArray* indices, LongType expected, LongType& output) {
  NDArray::preparePrimaryUse({}, {indices});
  std::vector<LongType> ids;
  segment_cpu::readIds(indices, ids);
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0 || ids[i] >= expected) {
      output = ids[i];
      return false;
    }
  }
  output = expected;
  return true;
}

// ---------------------------------------------------------------------------------------------------------- //
// Functors. sum / prod / max / min keep the dtype of the input; mean and sqrt_n write floating point (an integer input
// is summed in DOUBLE and divided once).
// ---------------------------------------------------------------------------------------------------------- //

#define SEGMENT_CPU_SAME_TYPE_FORWARD(NAME, OP)                                                                      \
  namespace segment_cpu {                                                                                             \
  template <typename T>                                                                                               \
  static void sorted##NAME##_(NDArray* in, NDArray* idx, NDArray* out) {                                              \
    sortedForward_<segment_sem::OP, T, T>(in, idx, out);                                                              \
  }                                                                                                                   \
  template <typename T>                                                                                               \
  static void unsorted##NAME##_(NDArray* in, NDArray* idx, LongType n, NDArray* out) {                                \
    unsortedForward_<segment_sem::OP, T, T>(in, idx, n, out);                                                         \
  }                                                                                                                   \
  }                                                                                                                   \
  void segment##NAME##Functor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {          \
    NDArray::preparePrimaryUse({output}, {input, indices});                                                          \
    BUILD_SINGLE_SELECTOR(input->dataType(), segment_cpu::sorted##NAME##_, (input, indices, output),                 \
                          SD_NUMERIC_TYPES);                                                                          \
    NDArray::registerPrimaryUse({output}, {input, indices});                                                         \
  }                                                                                                                   \
  void unsortedSegment##NAME##Functor(LaunchContext* context, NDArray* input, NDArray* indices,                      \
                                      LongType numOfClasses, NDArray* output) {                                      \
    NDArray::preparePrimaryUse({output}, {input, indices});                                                          \
    BUILD_SINGLE_SELECTOR(input->dataType(), segment_cpu::unsorted##NAME##_, (input, indices, numOfClasses, output), \
                          SD_NUMERIC_TYPES);                                                                          \
    NDArray::registerPrimaryUse({output}, {input, indices});                                                         \
  }

SEGMENT_CPU_SAME_TYPE_FORWARD(Sum, SegSum)
SEGMENT_CPU_SAME_TYPE_FORWARD(Prod, SegProd)
SEGMENT_CPU_SAME_TYPE_FORWARD(Max, SegMax)
SEGMENT_CPU_SAME_TYPE_FORWARD(Min, SegMin)

namespace segment_cpu {
template <typename X, typename Z>
static void sortedMean_(NDArray* in, NDArray* idx, NDArray* out) {
  sortedForward_<SegMean, X, Z>(in, idx, out);
}
template <typename X, typename Z>
static void unsortedMean_(NDArray* in, NDArray* idx, LongType n, NDArray* out) {
  unsortedForward_<SegMean, X, Z>(in, idx, n, out);
}
template <typename X, typename Z>
static void unsortedSqrtN_(NDArray* in, NDArray* idx, LongType n, NDArray* out) {
  unsortedForward_<SegSqrtN, X, Z>(in, idx, n, out);
}
}  // namespace segment_cpu

void segmentMeanFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  NDArray::preparePrimaryUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), segment_cpu::sortedMean_, (input, indices, output),
                        SD_NUMERIC_TYPES, SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({output}, {input, indices});
}

void unsortedSegmentMeanFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                                NDArray* output) {
  NDArray::preparePrimaryUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), segment_cpu::unsortedMean_,
                        (input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({output}, {input, indices});
}

void unsortedSegmentSqrtNFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                                 NDArray* output) {
  NDArray::preparePrimaryUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), segment_cpu::unsortedSqrtN_,
                        (input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({output}, {input, indices});
}

// ---------------------------------------------------------------------------------------------------------- //
// Backprop functors. Sorted and unsorted share one implementation: the gradient of a row only depends on the
// segment its id names. max / min / sum take every numeric type, prod / mean / sqrt_n the floating ones.
// ---------------------------------------------------------------------------------------------------------- //

#define SEGMENT_CPU_BACKPROP_NUMERIC(NAME, OP, GRAD)                                                                \
  namespace segment_cpu {                                                                                           \
  template <typename T>                                                                                             \
  static Status bp##NAME##_(NDArray* in, NDArray* idx, NDArray* go, NDArray* out) {                                 \
    return backprop_<segment_sem::OP, segment_sem::GRAD, T>(in, idx, go, out);                                      \
  }                                                                                                                 \
  }                                                                                                                 \
  Status segment##NAME##FunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,      \
                                  NDArray* output) {                                                                \
    NDArray::preparePrimaryUse({output}, {input, indices, gradOut});                                               \
    BUILD_SINGLE_SELECTOR(output->dataType(), segment_cpu::bp##NAME##_, (input, indices, gradOut, output),         \
                          SD_NUMERIC_TYPES);                                                                        \
    NDArray::registerPrimaryUse({output}, {input, indices, gradOut});                                              \
    return Status::OK;                                                                                              \
  }                                                                                                                 \
  Status unsortedSegment##NAME##FunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,                \
                                          NDArray* gradOut, LongType numOfClasses, NDArray* output) {              \
    NDArray::preparePrimaryUse({output}, {input, indices, gradOut});                                               \
    BUILD_SINGLE_SELECTOR(output->dataType(), segment_cpu::bp##NAME##_, (input, indices, gradOut, output),         \
                          SD_NUMERIC_TYPES);                                                                        \
    NDArray::registerPrimaryUse({output}, {input, indices, gradOut});                                              \
    return Status::OK;                                                                                              \
  }

#define SEGMENT_CPU_BACKPROP_FLOAT(NAME, OP, GRAD)                                                                  \
  namespace segment_cpu {                                                                                           \
  template <typename T>                                                                                             \
  static Status bp##NAME##_(NDArray* in, NDArray* idx, NDArray* go, NDArray* out) {                                 \
    return backprop_<segment_sem::OP, segment_sem::GRAD, T>(in, idx, go, out);                                      \
  }                                                                                                                 \
  }                                                                                                                 \
  Status segment##NAME##FunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,      \
                                  NDArray* output) {                                                                \
    NDArray::preparePrimaryUse({output}, {input, indices, gradOut});                                               \
    BUILD_SINGLE_SELECTOR(output->dataType(), segment_cpu::bp##NAME##_, (input, indices, gradOut, output),         \
                          SD_FLOAT_TYPES);                                                                          \
    NDArray::registerPrimaryUse({output}, {input, indices, gradOut});                                              \
    return Status::OK;                                                                                              \
  }                                                                                                                 \
  Status unsortedSegment##NAME##FunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,                \
                                          NDArray* gradOut, LongType numOfClasses, NDArray* output) {              \
    NDArray::preparePrimaryUse({output}, {input, indices, gradOut});                                               \
    BUILD_SINGLE_SELECTOR(output->dataType(), segment_cpu::bp##NAME##_, (input, indices, gradOut, output),         \
                          SD_FLOAT_TYPES);                                                                          \
    NDArray::registerPrimaryUse({output}, {input, indices, gradOut});                                              \
    return Status::OK;                                                                                              \
  }

SEGMENT_CPU_BACKPROP_NUMERIC(Sum, SegSum, GradSum)
SEGMENT_CPU_BACKPROP_NUMERIC(Max, SegMax, GradCompare)
SEGMENT_CPU_BACKPROP_NUMERIC(Min, SegMin, GradCompare)
SEGMENT_CPU_BACKPROP_FLOAT(Prod, SegProdNonZero, GradProd)
SEGMENT_CPU_BACKPROP_FLOAT(Mean, SegMean, GradMean)

namespace segment_cpu {
template <typename T>
static Status bpSqrtN_(NDArray* in, NDArray* idx, NDArray* go, NDArray* out) {
  return backprop_<SegSqrtN, GradSqrtN, T>(in, idx, go, out);
}
}  // namespace segment_cpu

Status unsortedSegmentSqrtNFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                     LongType numOfClasses, NDArray* output) {
  NDArray::preparePrimaryUse({output}, {input, indices, gradOut});
  BUILD_SINGLE_SELECTOR(output->dataType(), segment_cpu::bpSqrtN_, (input, indices, gradOut, output), SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({output}, {input, indices, gradOut});
  return Status::OK;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
