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
// @author AbdelRauf
//
#include <execution/ThreadPool.h>
#include <execution/Threads.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/reductions.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#if 1
#define LOG_CALLS(X)
#else

#define LOG_CALLS(X) sd_printf("___%s_________%d+\n", __PRETTY_FUNCTION__, X);
#endif
namespace sd {
namespace ops {
namespace helpers {
constexpr int threadingThreshold = 4096;
template <typename X, typename Z, typename ReductionOp>
static SD_INLINE void indexInnerReductionRank1(const X* buffer, X& current, Z& argCurrent,
                                               const sd::LongType& loopCount) {
  argCurrent = 0;
  current = buffer[0];
  LOG_CALLS(0)
  sd::LongType j_offset = 0;
  for (Z j = 0; j < loopCount; j++) {
    ReductionOp::update(current, argCurrent, buffer[j], j);
  }
}

template <typename X, typename Z, typename ReductionOp>
static SD_INLINE void indexInnerReductionRank1(const X* buffer, X& current, Z& argCurrent,
                                               const sd::LongType& loopCount, const sd::LongType& inner_stride) {
  argCurrent = 0;
  current = buffer[0];
  LOG_CALLS(0)
  sd::LongType j_offset = 0;
  for (Z j = 0; j < loopCount; j++) {
    ReductionOp::update(current, argCurrent, buffer[j_offset], j);
    j_offset += inner_stride;
  }
}

template <typename X, typename Z, typename ReductionOp, size_t constRank, bool LastIndexFaster = true>
static SD_INLINE void indexInnerReductionConstRank(const X* buffer, X& current, Z& argCurrent,
                                                   const sd::LongType* bases, const sd::LongType* strides,
                                                   const sd::LongType outerLoopCount,
                                                   const sd::LongType& innerLoopCount) {
  // skip 1 from the beginning or end depending the Order
  constexpr size_t updated_index = LastIndexFaster ? 0 : 1;
  constexpr size_t updated_rank = constRank - 1;
  sd::CoordsState<updated_rank - 1> cst;
  // we skip 1
  size_t offset =
      sd::init_coords<updated_rank, 0, LastIndexFaster>(cst, 0, bases + updated_index, strides + updated_index);
  Z startIndex = 0;
  argCurrent = 0;
  current = buffer[offset];
  LOG_CALLS(0)
  for (Z i = 0; i < outerLoopCount; i++) {
    const X* inner_buffer = &(buffer[offset]);
    // typename std::make_signed<Z>::type iArgMax = -1;
    for (Z j = 0; j < innerLoopCount; j++) {
      ReductionOp::update(current, argCurrent, inner_buffer[j], j + startIndex);
    }
    // we skip 1
    offset = sd::inc_coords<updated_rank, 0, LastIndexFaster>(cst, offset);
    startIndex += innerLoopCount;
  }
}

template <typename X, typename Z, typename ReductionOp, size_t constRank, bool LastIndexFaster = true>
static SD_INLINE void indexInnerReductionConstRank(const X* buffer, X& current, Z& argCurrent,
                                                   const sd::LongType* bases, const sd::LongType* strides,
                                                   const sd::LongType outerLoopCount,
                                                   const sd::LongType& innerLoopCount,
                                                   const sd::LongType& inner_stride) {
  // skip 1 from the beginning or end depending the Order
  constexpr size_t updated_index = LastIndexFaster ? 0 : 1;
  constexpr size_t updated_rank = constRank - 1;
  sd::CoordsState<updated_rank - 1> cst;
  // we skip 1
  size_t offset =
      sd::init_coords<updated_rank, 0, LastIndexFaster>(cst, 0, bases + updated_index, strides + updated_index);
  Z startIndex = 0;
  argCurrent = 0;
  current = buffer[offset];
  LOG_CALLS(0)
  for (Z i = 0; i < outerLoopCount; i++) {
    const X* inner_buffer = &(buffer[offset]);
    for (Z j = 0; j < innerLoopCount; j++) {
      ReductionOp::update(current, argCurrent, *inner_buffer, j + startIndex);
      inner_buffer += inner_stride;
    }
    // we alreaddy skiped
    offset = sd::inc_coords<updated_rank, 0, LastIndexFaster>(cst, offset);
    startIndex += innerLoopCount;
  }
}

template <typename X, typename Z, typename ReductionOp, bool LastIndexFaster = true>
static SD_INLINE void indexInnerReduction(const LongType& rank, const X* buffer, X& current, Z& argCurrent,
                                          const sd::LongType* bases, const sd::LongType* strides,
                                          const sd::LongType& outerLoopStart, const sd::LongType& outerLoopStop,
                                          const sd::LongType& innerLoopCount) {
  size_t offset = 0;
  sd::LongType outerLoopCount = outerLoopStop - outerLoopStart;
  sd::LongType coords[SD_MAX_RANK] = {};
  sd::LongType* ptr_coords = (sd::LongType*)&coords;
  if (outerLoopStart > 0) {
    sd::index2coords_C(outerLoopStart, rank - 1, bases, ptr_coords);
    offset = sd::offset_from_coords(strides, ptr_coords, rank);
  }
  Z startIndex = outerLoopStart * innerLoopCount;
  argCurrent = startIndex;
  current = buffer[offset];
  LOG_CALLS(0)
  for (Z i = 0; i < outerLoopCount; i++) {
    const X* inner_buffer = &(buffer[offset]);
    // typename std::make_signed<Z>::type iArgMax = -1;
    for (Z j = 0; j < innerLoopCount; j++) {
      ReductionOp::update(current, argCurrent, inner_buffer[j], j + startIndex);
    }
    offset = inc_coords<true>(bases, strides, ptr_coords, offset, rank, 1);
    startIndex += innerLoopCount;
  }
}

template <typename X, typename Z, typename ReductionOp, bool LastIndexFaster = true>
static SD_INLINE void indexInnerReduction(const int& rank, const X* buffer, X& current, Z& argCurrent,
                                          const sd::LongType* bases, const sd::LongType* strides,
                                          const sd::LongType& outerLoopStart, const sd::LongType& outerLoopStop,
                                          const sd::LongType& innerLoopCount, const sd::LongType& inner_stride) {
  size_t offset = 0;
  sd::LongType outerLoopCount = outerLoopStop - outerLoopStart;
  sd::LongType coords[SD_MAX_RANK] = {};
  sd::LongType* ptr_coords = (sd::LongType*)&coords;
  if (outerLoopStart > 0) {
    sd::index2coords_C(outerLoopStart, rank - 1, bases, ptr_coords);
    offset = sd::offset_from_coords(strides, ptr_coords, rank);
  }
  Z startIndex = outerLoopStart * innerLoopCount;
  argCurrent = startIndex;
  current = buffer[offset];
  LOG_CALLS(0)
  for (Z i = 0; i < outerLoopCount; i++) {
    const X* inner_buffer = &(buffer[offset]);
    // typename std::make_signed<Z>::type iArgMax = -1;
    for (Z j = 0; j < innerLoopCount; j++) {
      ReductionOp::update(current, argCurrent, inner_buffer[j * inner_stride], startIndex + j);
    }
    offset = inc_coords<true>(bases, strides, ptr_coords, offset, rank, 1);
    // offset = inc_coords<LastIndexFaster>(bases, strides, ptr_coords, offset, rank, 1);
    // if (iArgMax >= 0) argCurrent = startIndex + iArgMax;
    startIndex += innerLoopCount;
  }
}

template <typename X, typename Z, typename ReductionOp>
static SD_INLINE void indexInnerReductionRank1Block4WithMerge(const X* buffer, X& current, Z& argCurrent,
                                                              const sd::LongType& loopCount) {
  argCurrent = 0;
  current = buffer[0];
  LOG_CALLS(0)
  sd::LongType loopCount4 = loopCount / 4;
  sd::LongType loopCountEnd = loopCount4 + (loopCount & 3);
  const X* buffer1 = buffer + 1 * loopCount4;
  const X* buffer2 = buffer1 + 1 * loopCount4;
  const X* buffer3 = buffer2 + 1 * loopCount4;
  X current1 = *buffer1;
  X current2 = *buffer2;
  X current3 = *buffer3;
  Z argCurrent1 = 0;
  Z argCurrent2 = 0;
  Z argCurrent3 = 0;
  for (Z j = 0; j < loopCount4; j++) {
    ReductionOp::update(current, argCurrent, buffer[j], j);
    ReductionOp::update(current1, argCurrent1, buffer1[j], j);
    ReductionOp::update(current2, argCurrent2, buffer2[j], j);
    ReductionOp::update(current3, argCurrent3, buffer3[j], j);
  }
  // tail
  for (Z j = loopCount4; j < loopCountEnd; j++) {
    ReductionOp::update(current3, argCurrent3, buffer3[j], j);
  }
  // merge
  argCurrent1 += loopCount4;
  argCurrent2 += 2 * loopCount4;
  argCurrent3 += 3 * loopCount4;
  ReductionOp::update(current, argCurrent, current1, argCurrent1);
  ReductionOp::update(current, argCurrent, current2, argCurrent2);
  ReductionOp::update(current, argCurrent, current3, argCurrent3);
}

template <typename X, typename Z, typename ReductionOp>
static SD_INLINE void indexInnerReductionRank1Block4WithMerge(const X* buffer, X& current, Z& argCurrent,
                                                              const sd::LongType& loopCount,
                                                              const sd::LongType& inner_stride) {
  argCurrent = 0;
  current = buffer[0];
  LOG_CALLS(0)
  sd::LongType loopCount4 = loopCount / 4;
  sd::LongType loopCountEnd = loopCount4 + (loopCount & 3);
  const X* buffer1 = buffer + inner_stride * loopCount4;
  const X* buffer2 = buffer1 + inner_stride * loopCount4;
  const X* buffer3 = buffer2 + inner_stride * loopCount4;
  X current1 = *buffer1;
  X current2 = *buffer2;
  X current3 = *buffer3;
  Z argCurrent1 = 0;
  Z argCurrent2 = 0;
  Z argCurrent3 = 0;
  sd::LongType j_offset = 0;
  for (Z j = 0; j < loopCount4; j++) {
    ReductionOp::update(current, argCurrent, buffer[j_offset], j);
    ReductionOp::update(current1, argCurrent1, buffer1[j_offset], j);
    ReductionOp::update(current2, argCurrent2, buffer2[j_offset], j);
    ReductionOp::update(current3, argCurrent3, buffer3[j_offset], j);
    j_offset += inner_stride;
  }
  // tail
  for (Z j = loopCount4; j < loopCountEnd; j++) {
    ReductionOp::update(current3, argCurrent3, buffer3[j_offset], j);
    j_offset += inner_stride;
  }
  // merge
  argCurrent1 += loopCount4;
  argCurrent2 += 2 * loopCount4;
  argCurrent3 += 3 * loopCount4;
  ReductionOp::update(current, argCurrent, current1, argCurrent1);
  ReductionOp::update(current, argCurrent, current2, argCurrent2);
  ReductionOp::update(current, argCurrent, current3, argCurrent3);
}

template <typename X, typename Z, typename ReductionOp>
static SD_INLINE void indexInnerReductionRank1Block4(const X* buffer, const X* buffer1, const X* buffer2,
                                                     const X* buffer3, Z* output, Z* output1, Z* output2, Z* output3,
                                                     const sd::LongType& loopCount) {
  LOG_CALLS(0)
  Z argCurrent = 0;
  Z argCurrent1 = 0;
  Z argCurrent2 = 0;
  Z argCurrent3 = 0;
  X current = buffer[0];
  X current1 = buffer1[0];
  X current2 = buffer2[0];
  X current3 = buffer3[0];
  for (Z j = 0; j < loopCount; j++) {
    ReductionOp::update(current, argCurrent, buffer[j], j);
    ReductionOp::update(current1, argCurrent1, buffer1[j], j);
    ReductionOp::update(current2, argCurrent2, buffer2[j], j);
    ReductionOp::update(current3, argCurrent3, buffer3[j], j);
  }
  *output = argCurrent;
  *output1 = argCurrent1;
  *output2 = argCurrent2;
  *output3 = argCurrent3;
  return;
}

template <typename X, typename Z, typename ReductionOp>
static SD_INLINE void indexInnerReductionRank1Block4(const X* buffer, const X* buffer1, const X* buffer2,
                                                     const X* buffer3, Z* output, Z* output1, Z* output2, Z* output3,
                                                     const sd::LongType& loopCount, const sd::LongType& inner_stride) {
  LOG_CALLS(0)
  Z argCurrent = 0;
  Z argCurrent1 = 0;
  Z argCurrent2 = 0;
  Z argCurrent3 = 0;
  X current = buffer[0];
  X current1 = buffer1[0];
  X current2 = buffer2[0];
  X current3 = buffer3[0];
  sd::LongType j_offset = 0;
  for (Z j = 0; j < loopCount; j++) {
    ReductionOp::update(current, argCurrent, buffer[j_offset], j);
    ReductionOp::update(current1, argCurrent1, buffer1[j_offset], j);
    ReductionOp::update(current2, argCurrent2, buffer2[j_offset], j);
    ReductionOp::update(current3, argCurrent3, buffer3[j_offset], j);
    j_offset += inner_stride;
  }
  *output = argCurrent;
  *output1 = argCurrent1;
  *output2 = argCurrent2;
  *output3 = argCurrent3;
  return;
}

template <typename X, typename Z, typename ReductionOp, size_t constRank, bool LastIndexFaster = true>
static SD_INLINE void indexInnerReductionConstRankBlock4(const X* buffer, const X* buffer1, const X* buffer2,
                                                         const X* buffer3, Z* output, Z* output1, Z* output2,
                                                         Z* output3, const sd::LongType* bases,
                                                         const sd::LongType* strides,
                                                         const sd::LongType& outerLoopCount,
                                                         const sd::LongType& innerLoopCount) {
  LOG_CALLS(0)
  // skip 1 from the beginning or end depending the Order
  constexpr size_t updated_index = LastIndexFaster ? 0 : 1;
  constexpr size_t updated_rank = constRank - 1;
  sd::CoordsState<updated_rank - 1> cst;
  // we skip 1
  size_t offset =
      sd::init_coords<updated_rank, 0, LastIndexFaster>(cst, 0, bases + updated_index, strides + updated_index);
  Z startIndex = 0;
  Z argCurrent = 0;
  Z argCurrent1 = 0;
  Z argCurrent2 = 0;
  Z argCurrent3 = 0;
  X current = buffer[0];
  X current1 = buffer1[0];
  X current2 = buffer2[0];
  X current3 = buffer3[0];
  // LOG_CALLS(0)
  for (Z i = 0; i < outerLoopCount; i++) {
    const X* inner_buffer = &(buffer[offset]);
    const X* inner_buffer1 = &(buffer1[offset]);
    const X* inner_buffer2 = &(buffer2[offset]);
    const X* inner_buffer3 = &(buffer3[offset]);
    // typename std::make_signed<Z>::type iArgMax = -1;
    for (Z j = 0; j < innerLoopCount; j++) {
      ReductionOp::update(current, argCurrent, inner_buffer[j], j + startIndex);
      ReductionOp::update(current1, argCurrent1, inner_buffer1[j], j + startIndex);
      ReductionOp::update(current2, argCurrent2, inner_buffer2[j], j + startIndex);
      ReductionOp::update(current3, argCurrent3, inner_buffer3[j], j + startIndex);
    }
    // we skip 1
    offset = sd::inc_coords<updated_rank, 0, LastIndexFaster>(cst, offset);
    startIndex += innerLoopCount;
  }
  *output = argCurrent;
  *output1 = argCurrent1;
  *output2 = argCurrent2;
  *output3 = argCurrent3;
  return;
}

template <typename X, typename Z, typename ReductionOp, size_t constRank, bool LastIndexFaster = true>
static SD_INLINE void indexInnerReductionConstRankBlock4(
    const X* buffer, const X* buffer1, const X* buffer2, const X* buffer3, Z* output, Z* output1, Z* output2,
    Z* output3, const sd::LongType* bases, const sd::LongType* strides, const sd::LongType& outerLoopCount,
    const sd::LongType& innerLoopCount, const sd::LongType& inner_stride) {
  LOG_CALLS(0)
  // skip 1 from the beginning or end depending the Order
  constexpr size_t updated_index = LastIndexFaster ? 0 : 1;
  constexpr size_t updated_rank = constRank - 1;
  sd::CoordsState<updated_rank - 1> cst;
  // we skip 1
  size_t offset =
      sd::init_coords<updated_rank, 0, LastIndexFaster>(cst, 0, bases + updated_index, strides + updated_index);
  Z startIndex = 0;
  Z argCurrent = 0;
  Z argCurrent1 = 0;
  Z argCurrent2 = 0;
  Z argCurrent3 = 0;
  X current = buffer[0];
  X current1 = buffer1[0];
  X current2 = buffer2[0];
  X current3 = buffer3[0];
  // LOG_CALLS(0)
  for (Z i = 0; i < outerLoopCount; i++) {
    const X* inner_buffer = &(buffer[offset]);
    const X* inner_buffer1 = &(buffer1[offset]);
    const X* inner_buffer2 = &(buffer2[offset]);
    const X* inner_buffer3 = &(buffer3[offset]);
    // typename std::make_signed<Z>::type iArgMax = -1;
    sd::LongType inner_offset = 0;
    for (Z j = 0; j < innerLoopCount; j++) {
      ReductionOp::update(current, argCurrent, inner_buffer[inner_offset], j + startIndex);
      ReductionOp::update(current1, argCurrent1, inner_buffer1[inner_offset], j + startIndex);
      ReductionOp::update(current2, argCurrent2, inner_buffer2[inner_offset], j + startIndex);
      ReductionOp::update(current3, argCurrent3, inner_buffer3[inner_offset], j + startIndex);
      inner_offset += inner_stride;
    }
    // we skip 1
    offset = sd::inc_coords<updated_rank, 0, LastIndexFaster>(cst, offset);
    startIndex += innerLoopCount;
  }
  *output = argCurrent;
  *output1 = argCurrent1;
  *output2 = argCurrent2;
  *output3 = argCurrent3;
  return;
}

template <typename X, typename Z, typename ReductionOp, bool LastIndexFaster = true>
static void argIndexCase1Scalar(const int& second_rank, const sd::LongType* inner_bases,
                                const sd::LongType* inner_strides, const X* bufferX, Z* outputZ) {
  sd::LongType inner_total;
  sd::LongType inner_last = 0;
  int maxThreads = sd::Environment::getInstance().maxMasterThreads();
  if (second_rank == 1) {
    inner_total = inner_bases[0];
    if (inner_total < threadingThreshold) {
      maxThreads = 1;
    }
  } else {
    inner_total = getLength<LastIndexFaster>(inner_bases, second_rank, 1, inner_last);
    if (inner_total * inner_last < threadingThreshold) {
      maxThreads = 1;
    }
  }

  std::unique_ptr<X[]> maxValues(new X[maxThreads]);
  std::unique_ptr<Z[]> maxIndices(new Z[maxThreads]);
  X* ptrMaxValues = maxValues.get();
  Z* ptrMaxIndices = maxIndices.get();
  auto func = [ptrMaxValues, ptrMaxIndices, inner_last, second_rank, inner_bases, inner_strides, bufferX](
                  uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
    // LOG_CALLS(0)
    const sd::LongType inner_stride = LastIndexFaster ? inner_strides[second_rank - 1] : inner_strides[0];
    Z argCurrent;
    X current;
    if (second_rank == 1) {
      const sd::LongType loopTotal = stop - start;
      if (inner_stride == 1) {
        indexInnerReductionRank1Block4WithMerge<X, Z, ReductionOp>(&(bufferX[start]), current, argCurrent, loopTotal);
      } else {
        indexInnerReductionRank1Block4WithMerge<X, Z, ReductionOp>(&(bufferX[start * inner_stride]), current,
                                                                   argCurrent, loopTotal, inner_stride);
      }
      ptrMaxIndices[thread_id] = argCurrent + start;
    } else {
      if (inner_stride == 1) {
        indexInnerReduction<X, Z, ReductionOp, LastIndexFaster>(second_rank, bufferX, current, argCurrent, inner_bases,
                                                                inner_strides, start, stop, inner_last, inner_stride);
      } else {
        indexInnerReduction<X, Z, ReductionOp, LastIndexFaster>(second_rank, bufferX, current, argCurrent, inner_bases,
                                                                inner_strides, start, stop, inner_last, inner_stride);
      }
      ptrMaxIndices[thread_id] = argCurrent;
    }
    ptrMaxValues[thread_id] = current;
  };
#if 0
                int Count = 0;
                func(0, 0, inner_total, 1);
#else
  int Count = samediff::Threads::parallel_tad(func, 0, inner_total, 1, maxThreads);
#endif
  Z arg = 0;
  X current = ptrMaxValues[0];

  for (Z i = 1; i < Count; i++) {
    ReductionOp::update(current, arg, ptrMaxValues[i], i);
  }

  *outputZ = ptrMaxIndices[arg];
}

template <typename X, typename Z, typename ReductionOp, typename Movement, bool LastIndexFaster = true>
static void argReductionInnerCases(Movement& movement, sd::LongType loopTotal, const int& second_rank,
                                   const sd::LongType* inner_bases, const sd::LongType* inner_strides, const X* bufferX,
                                   Z* outputZ) {
  sd::LongType inner_stride = true /*LastIndexFaster*/ ? inner_strides[second_rank - 1] : inner_strides[0];

  sd::LongType loopTotal_K = loopTotal / 4;
  sd::LongType loopTotal_Tail = loopTotal & 3;
  if (inner_stride == 1) {
    if (second_rank == 1) {
      LOG_CALLS(0)
      sd::LongType inner_total = getLength<true>(inner_bases, second_rank);
      for (sd::LongType i = 0; i < loopTotal_K; i++) {
        const X* buffer0 = &(bufferX[movement.First()]);
        Z* output0 = &(outputZ[movement.Second()]);
        movement.increment();
        const X* buffer1 = &(bufferX[movement.First()]);
        Z* output1 = &(outputZ[movement.Second()]);
        movement.increment();
        const X* buffer2 = &(bufferX[movement.First()]);
        Z* output2 = &(outputZ[movement.Second()]);
        movement.increment();
        const X* buffer3 = &(bufferX[movement.First()]);
        Z* output3 = &(outputZ[movement.Second()]);
        movement.increment();
        indexInnerReductionRank1Block4<X, Z, ReductionOp>(buffer0, buffer1, buffer2, buffer3, output0, output1, output2,
                                                          output3, inner_total);
      }
      if (inner_total >= 2048) {
        for (sd::LongType i = 0; i < loopTotal_Tail; i++) {
          X current;
          const X* buffer0 = &(bufferX[movement.First()]);
          indexInnerReductionRank1Block4WithMerge<X, Z, ReductionOp>(buffer0, current, outputZ[movement.Second()],
                                                                     inner_total);
          movement.increment();
        }
      } else {
        for (sd::LongType i = 0; i < loopTotal_Tail; i++) {
          X current;
          const X* buffer0 = &(bufferX[movement.First()]);
          indexInnerReductionRank1<X, Z, ReductionOp>(buffer0, current, outputZ[movement.Second()], inner_total);
          movement.increment();
        }
      }

    } else {
      sd::LongType inner_last;
      sd::LongType inner_loop = getLength<true>(inner_bases, second_rank, 1, inner_last);
      if (second_rank == 2) {
        LOG_CALLS(1)
        for (sd::LongType i = 0; i < loopTotal_K; i++) {
          const X* buffer0 = &(bufferX[movement.First()]);
          Z* output0 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer1 = &(bufferX[movement.First()]);
          Z* output1 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer2 = &(bufferX[movement.First()]);
          Z* output2 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer3 = &(bufferX[movement.First()]);
          Z* output3 = &(outputZ[movement.Second()]);
          movement.increment();
          indexInnerReductionConstRankBlock4<X, Z, ReductionOp, 2>(buffer0, buffer1, buffer2, buffer3, output0, output1,
                                                                   output2, output3, inner_bases, inner_strides,
                                                                   inner_loop, inner_last);
        }
        for (sd::LongType i = 0; i < loopTotal_Tail; i++) {
          X current;
          const X* buffer0 = &(bufferX[movement.First()]);
          indexInnerReductionConstRank<X, Z, ReductionOp, 2>(buffer0, current, outputZ[movement.Second()], inner_bases,
                                                             inner_strides, inner_loop, inner_last);
          movement.increment();
        }

      } else if (second_rank == 3) {
        LOG_CALLS(2)
        for (sd::LongType i = 0; i < loopTotal_K; i++) {
          const X* buffer0 = &(bufferX[movement.First()]);
          Z* output0 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer1 = &(bufferX[movement.First()]);
          Z* output1 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer2 = &(bufferX[movement.First()]);
          Z* output2 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer3 = &(bufferX[movement.First()]);
          Z* output3 = &(outputZ[movement.Second()]);
          movement.increment();
          indexInnerReductionConstRankBlock4<X, Z, ReductionOp, 3>(buffer0, buffer1, buffer2, buffer3, output0, output1,
                                                                   output2, output3, inner_bases, inner_strides,
                                                                   inner_loop, inner_last);
        }
        for (sd::LongType i = 0; i < loopTotal_Tail; i++) {
          X current;
          const X* buffer0 = &(bufferX[movement.First()]);
          indexInnerReductionConstRank<X, Z, ReductionOp, 3>(buffer0, current, outputZ[movement.Second()], inner_bases,
                                                             inner_strides, inner_loop, inner_last);
          movement.increment();
        }

      } else {
        LOG_CALLS(3)
        for (sd::LongType i = 0; i < loopTotal; i++) {
          X current;
          const X* buffer0 = &(bufferX[movement.First()]);
          indexInnerReduction<X, Z, ReductionOp>(second_rank, buffer0, current, outputZ[movement.Second()], inner_bases,
                                                 inner_strides, 0, inner_loop, inner_last);
          movement.increment();
        }
      }
    }

  } else {
    if (second_rank == 1) {
      LOG_CALLS(10)
      sd::LongType inner_total = getLength<true>(inner_bases, second_rank);
      for (sd::LongType i = 0; i < loopTotal_K; i++) {
        const X* buffer0 = &(bufferX[movement.First()]);
        Z* output0 = &(outputZ[movement.Second()]);
        movement.increment();
        const X* buffer1 = &(bufferX[movement.First()]);
        Z* output1 = &(outputZ[movement.Second()]);
        movement.increment();
        const X* buffer2 = &(bufferX[movement.First()]);
        Z* output2 = &(outputZ[movement.Second()]);
        movement.increment();
        const X* buffer3 = &(bufferX[movement.First()]);
        Z* output3 = &(outputZ[movement.Second()]);
        movement.increment();
        indexInnerReductionRank1Block4<X, Z, ReductionOp>(buffer0, buffer1, buffer2, buffer3, output0, output1, output2,
                                                          output3, inner_total, inner_stride);
      }
      if (inner_total >= 2048) {
        for (sd::LongType i = 0; i < loopTotal_Tail; i++) {
          X current;
          const X* buffer0 = &(bufferX[movement.First()]);
          indexInnerReductionRank1Block4WithMerge<X, Z, ReductionOp>(buffer0, current, outputZ[movement.Second()],
                                                                     inner_total, inner_stride);
          movement.increment();
        }
      } else {
        for (sd::LongType i = 0; i < loopTotal_Tail; i++) {
          X current;
          const X* buffer0 = &(bufferX[movement.First()]);
          indexInnerReductionRank1<X, Z, ReductionOp>(buffer0, current, outputZ[movement.Second()], inner_total,
                                                      inner_stride);
          movement.increment();
        }
      }

    } else {
      sd::LongType inner_last;
      sd::LongType inner_loop = getLength<true>(inner_bases, second_rank, 1, inner_last);
      if (second_rank == 2) {
        LOG_CALLS(11)
        for (sd::LongType i = 0; i < loopTotal_K; i++) {
          const X* buffer0 = &(bufferX[movement.First()]);
          Z* output0 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer1 = &(bufferX[movement.First()]);
          Z* output1 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer2 = &(bufferX[movement.First()]);
          Z* output2 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer3 = &(bufferX[movement.First()]);
          Z* output3 = &(outputZ[movement.Second()]);
          movement.increment();
          indexInnerReductionConstRankBlock4<X, Z, ReductionOp, 2>(buffer0, buffer1, buffer2, buffer3, output0, output1,
                                                                   output2, output3, inner_bases, inner_strides,
                                                                   inner_loop, inner_last, inner_stride);
        }
        for (sd::LongType i = 0; i < loopTotal_Tail; i++) {
          X current;
          const X* buffer0 = &(bufferX[movement.First()]);
          indexInnerReductionConstRank<X, Z, ReductionOp, 2>(buffer0, current, outputZ[movement.Second()], inner_bases,
                                                             inner_strides, inner_loop, inner_last, inner_stride);
          movement.increment();
        }

      } else if (second_rank == 3) {
        LOG_CALLS(12)
        for (sd::LongType i = 0; i < loopTotal_K; i++) {
          const X* buffer0 = &(bufferX[movement.First()]);
          Z* output0 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer1 = &(bufferX[movement.First()]);
          Z* output1 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer2 = &(bufferX[movement.First()]);
          Z* output2 = &(outputZ[movement.Second()]);
          movement.increment();
          const X* buffer3 = &(bufferX[movement.First()]);
          Z* output3 = &(outputZ[movement.Second()]);
          movement.increment();
          indexInnerReductionConstRankBlock4<X, Z, ReductionOp, 3>(buffer0, buffer1, buffer2, buffer3, output0, output1,
                                                                   output2, output3, inner_bases, inner_strides,
                                                                   inner_loop, inner_last, inner_stride);
        }
        for (sd::LongType i = 0; i < loopTotal_Tail; i++) {
          X current;
          const X* buffer0 = &(bufferX[movement.First()]);
          indexInnerReductionConstRank<X, Z, ReductionOp, 3>(buffer0, current, outputZ[movement.Second()], inner_bases,
                                                             inner_strides, inner_loop, inner_last, inner_stride);
          movement.increment();
        }

      } else {
        LOG_CALLS(13)
        for (sd::LongType i = 0; i < loopTotal; i++) {
          X current;
          const X* buffer0 = &(bufferX[movement.First()]);
          indexInnerReduction<X, Z, ReductionOp>(second_rank, buffer0, current, outputZ[movement.Second()], inner_bases,
                                                 inner_strides, 0, inner_loop, inner_last, inner_stride);
          movement.increment();
        }
      }
    }
  }
}

template <typename X, typename Z, typename ReductionOp, bool LastIndexFaster = true>
static void argIndexCaseNonScalar(const int& first_rank, const int& output_rank, bool squashed, const int& second_rank,
                                  const sd::LongType*& outer_bases, const sd::LongType* outer_strides,
                                  const sd::LongType* output_strides, const sd::LongType& output_stride,
                                  const sd::LongType*& inner_bases, const sd::LongType* inner_strides, const X* bufferX,
                                  Z* outputZ) {
  sd::LongType total = getLength<LastIndexFaster>(outer_bases, first_rank);
  sd::LongType inner_stride = true /*LastIndexFaster*/ ? inner_strides[second_rank - 1] : inner_strides[0];
  sd::LongType outer_stride = LastIndexFaster ? outer_strides[second_rank - 1] : outer_strides[0];
  auto func = [first_rank, output_rank, squashed, outer_bases, outer_strides, output_strides, output_stride,
               second_rank, inner_bases, inner_strides, bufferX,
               outputZ](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
    sd::LongType loopTotal = stop - start;
    sd::LongType stride = LastIndexFaster ? outer_strides[first_rank - 1] : outer_strides[0];
    if (first_rank == 1) {
      if (stride == 1) {
        ZipGenericCoordsRank1Stride1 movement;
        movement.init(nullptr, nullptr, nullptr, 0, start);
        argReductionInnerCases<X, Z, ReductionOp>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX,
                                                  outputZ);
      } else {
        ZipGenericCoordsRank1BothStrideN movement;
        movement.init(nullptr, &stride, &output_stride, 0, start);
        argReductionInnerCases<X, Z, ReductionOp>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX,
                                                  outputZ);
      }

    } else if (squashed && first_rank <= output_rank) {
      if (first_rank == 2) {
        if (output_stride == 1) {
          ZipGenericCoordsConstMovementSecondStride1<2, LastIndexFaster> movement;
          movement.init(outer_bases, outer_strides, nullptr, first_rank, start);
          argReductionInnerCases<X, Z, ReductionOp>(movement, loopTotal, second_rank, inner_bases, inner_strides,
                                                    bufferX, outputZ);

        } else {
          ZipGenericCoordsConstMovementSecondStrideN<2, LastIndexFaster> movement;
          movement.init(outer_bases, outer_strides, &output_stride, first_rank, start);
          argReductionInnerCases<X, Z, ReductionOp>(movement, loopTotal, second_rank, inner_bases, inner_strides,
                                                    bufferX, outputZ);
        }
      } else if (first_rank == 3) {
        if (output_stride == 1) {
          ZipGenericCoordsConstMovementSecondStride1<3, LastIndexFaster> movement;
          movement.init(outer_bases, outer_strides, nullptr, first_rank, start);
          argReductionInnerCases<X, Z, ReductionOp>(movement, loopTotal, second_rank, inner_bases, inner_strides,
                                                    bufferX, outputZ);

        } else {
          ZipGenericCoordsConstMovementSecondStrideN<3, LastIndexFaster> movement;
          movement.init(outer_bases, outer_strides, &output_stride, first_rank, start);
          argReductionInnerCases<X, Z, ReductionOp>(movement, loopTotal, second_rank, inner_bases, inner_strides,
                                                    bufferX, outputZ);
        }
      } else {
        ZipGenericCoordsMovementSecondStrideN<LastIndexFaster> movement;
        movement.init(outer_bases, outer_strides, &output_stride, first_rank, start);

        argReductionInnerCases<X, Z, ReductionOp>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX,
                                                  outputZ);
      }

    } else {
      ZipGenericCoordsMovement<LastIndexFaster> movement;
      movement.init(outer_bases, outer_strides, output_strides, first_rank, start);

      argReductionInnerCases<X, Z, ReductionOp>(movement, loopTotal, second_rank, inner_bases, inner_strides, bufferX,
                                                outputZ);
    }
  };
#if 0
                func(0, 0, total, 1);
#else
  //
  uint32_t numThreads = sd::Environment::getInstance().maxMasterThreads();
  sd::LongType inner_total = getLength<true>(inner_bases, second_rank);
  if (total * inner_total <= threadingThreshold) {
    numThreads = 1;
  } else {
    if (inner_stride > outer_stride && total <= 256) {
      auto desired = total > 4 ? (total / 4) : 1;
      numThreads = numThreads > desired ? desired : numThreads;
    }
  }

  samediff::Threads::parallel_tad(func, 0, total, 1, numThreads);
#endif
}

template <typename X, typename Z, typename ReductionOp>
SD_LIB_HIDDEN void argIndex_(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  char input_order = input.ordering();
  bool try_squash_outer = (input_order == output.ordering()) && output.ews() != 0;
  const sd::LongType* input_shapeInfo = input.shapeInfo();
  const sd::LongType* output_shapeInfo = output.shapeInfo();
  const sd::LongType rank = input_shapeInfo[0];
  const sd::LongType* input_bases = &(input_shapeInfo[1]);
  const sd::LongType* input_strides = &(input_shapeInfo[rank + 1]);
  const sd::LongType output_rank = output_shapeInfo[0];
  const sd::LongType* output_strides = &(output_shapeInfo[output_rank + 1]);
  sd::LongType new_bases[SD_MAX_RANK];
  sd::LongType new_strides[SD_MAX_RANK];
  sd::LongType first_begin, first_end, second_begin, second_end;
  // rePartition into two parts based on the selection
  rePartition(input_order, dimensions, rank, input_bases, input_strides, new_bases, new_strides, first_begin, first_end,
              second_begin, second_end, try_squash_outer, input_order == 'c');
  int first_rank = first_end - first_begin;  // the first rank can be 0 for scalar cases
  int second_rank = second_end - second_begin;
  auto bufferX = input.bufferAsT<X>();
  auto outputZ = output.bufferAsT<Z>();
  const sd::LongType* outer_bases = &(new_bases[first_begin]);
  const sd::LongType* outer_strides = &(new_strides[first_begin]);
  const sd::LongType* inner_bases = &(new_bases[second_begin]);
  const sd::LongType* inner_strides = &(new_strides[second_begin]);
  const sd::LongType output_stride = output.ordering() == 'c' ? output_strides[output_rank - 1] : output_strides[0];
  if (input_order == 'c') {
    if (first_rank == 0) {
      argIndexCase1Scalar<X, Z, ReductionOp>(second_rank, inner_bases, inner_strides, bufferX, outputZ);
    } else {
      argIndexCaseNonScalar<X, Z, ReductionOp>(first_rank, output_rank, try_squash_outer, second_rank, outer_bases,
                                               outer_strides, output_strides, output_stride, inner_bases, inner_strides,
                                               bufferX, outputZ);
    }
  } else {
    if (first_rank == 0) {
      LOG_CALLS(0);
      if (second_rank == 1) {
        argIndexCase1Scalar<X, Z, ReductionOp, false>(second_rank, inner_bases, inner_strides, bufferX, outputZ);
      } else {
        argIndexCase1Scalar<X, Z, ReductionOp, true>(second_rank, inner_bases, inner_strides, bufferX, outputZ);
      }
    } else {
      LOG_CALLS(1);
      argIndexCaseNonScalar<X, Z, ReductionOp, false>(first_rank, output_rank, try_squash_outer, second_rank,
                                                      outer_bases, outer_strides, output_strides, output_stride,
                                                      inner_bases, inner_strides, bufferX, outputZ);
    }
  }
}

template <typename X, typename Z>
struct IndexMax {
  static SD_INLINE void update(X& current, Z& currentIndex, const X& candidate, const Z& candidateIndex) {
    if (candidate > current) {
      current = candidate;
      currentIndex = candidateIndex;
    }
  }
};

template <typename X, typename Z>
struct IndexMin {
  static SD_INLINE void update(X& current, Z& currentIndex, const X& candidate, const Z& candidateIndex) {
    if (candidate < current) {
      current = candidate;
      currentIndex = candidateIndex;
    }
  }
};

template <typename X, typename Z>
struct IndexAbsMax {
  static SD_INLINE void update(X& current, Z& currentIndex, const X& candidate, const Z& candidateIndex) {
    auto absCandidate = sd::math::sd_abs<X,X>(candidate);
    if (absCandidate > current) {
      current = absCandidate;
      currentIndex = candidateIndex;
    }
  }
};

template <typename X, typename Z>
struct IndexAbsMin {
  static SD_INLINE void update(X& current, Z& currentIndex, const X& candidate, const Z& candidateIndex) {
    auto absCandidate = sd::math::sd_abs<X,X>(candidate);
    if (absCandidate < current) {
      current = absCandidate;
      currentIndex = candidateIndex;
    }
  }
};

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_LIB_HIDDEN void argMax_(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  return argIndex_<X, Z, IndexMax<X, Z>>(input, output, dimensions);
}

template <typename X, typename Z>
SD_LIB_HIDDEN void argMin_(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  return argIndex_<X, Z, IndexMin<X, Z>>(input, output, dimensions);
}

template <typename X, typename Z>
SD_LIB_HIDDEN void argAbsMax_(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  return argIndex_<X, Z, IndexAbsMax<X, Z>>(input, output, dimensions);
}

template <typename X, typename Z>
SD_LIB_HIDDEN void argAbsMin_(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  return argIndex_<X, Z, IndexAbsMin<X, Z>>(input, output, dimensions);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
