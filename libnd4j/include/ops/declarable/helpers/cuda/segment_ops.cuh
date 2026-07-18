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
//  Shared, policy-templated implementation of the segment reduction ops
//  (segment_max / segment_min / segment_sum / segment_prod / segment_mean).
//
//  Historically each op carried its own near-duplicate copy of five kernels +
//  four host functors that had drifted apart (two forward strategies, four BP
//  formulas, divergent signatures). This header unifies them:
//    * forward LINEAR  -> deterministic shared-memory tree reduction (all ops)
//    * forward TAD      -> global atomics (all ops, already uniform)
//    * backprop         -> one scaffold + a per-op GradOp policy
//  Behaviour is selected by two tiny policy structs per op (ReduceOp + GradOp);
//  each segment_<op>.cu is reduced to a thin TU that instantiates them.
//
#ifndef LIBND4J_SEGMENT_OPS_CUH
#define LIBND4J_SEGMENT_OPS_CUH

#include <array/NDArrayFactory.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/segment.h>
#include <ops/declarable/helpers/segment_common.h>
#include <system/selective_rendering.h>

namespace sd {
namespace ops {
namespace helpers {
namespace segment_ops {

// ============================================================================ //
// ReduceOp policies (forward). Each provides:
//   neutral<T>()          - identity element for the reduction
//   combine(acc, v)       - fold v into the running accumulator (device)
//   atomic(addr, v)       - atomic fold into global memory (device, TAD path)
//   finalize(acc, len)    - post-reduction transform for the linear write
//   tadValue(v, len)      - per-element transform before the TAD atomic
// ============================================================================ //

struct MaxReduce {
  template <typename T> static SD_HOST_DEVICE SD_INLINE T neutral() { return -DataTypeUtils::max<T>(); }
  template <typename T> static SD_DEVICE SD_INLINE void combine(T& acc, T v) { if (v > acc) acc = v; }
  template <typename T> static SD_DEVICE SD_INLINE void atomic(T* addr, T v) { math::atomics::sd_atomicMax<T>(addr, v); }
  template <typename T> static SD_DEVICE SD_INLINE T finalize(T acc, LongType) { return acc; }
  template <typename T> static SD_DEVICE SD_INLINE T tadValue(T v, LongType) { return v; }
};

struct MinReduce {
  template <typename T> static SD_HOST_DEVICE SD_INLINE T neutral() { return DataTypeUtils::infOrMax<T>(); }
  template <typename T> static SD_DEVICE SD_INLINE void combine(T& acc, T v) { if (v < acc) acc = v; }
  template <typename T> static SD_DEVICE SD_INLINE void atomic(T* addr, T v) { math::atomics::sd_atomicMin<T>(addr, v); }
  template <typename T> static SD_DEVICE SD_INLINE T finalize(T acc, LongType) { return acc; }
  template <typename T> static SD_DEVICE SD_INLINE T tadValue(T v, LongType) { return v; }
};

struct SumReduce {
  template <typename T> static SD_HOST_DEVICE SD_INLINE T neutral() { return static_cast<T>(0); }
  template <typename T> static SD_DEVICE SD_INLINE void combine(T& acc, T v) { acc += v; }
  template <typename T> static SD_DEVICE SD_INLINE void atomic(T* addr, T v) { math::atomics::sd_atomicAdd<T>(addr, v); }
  template <typename T> static SD_DEVICE SD_INLINE T finalize(T acc, LongType) { return acc; }
  template <typename T> static SD_DEVICE SD_INLINE T tadValue(T v, LongType) { return v; }
};

struct ProdReduce {
  template <typename T> static SD_HOST_DEVICE SD_INLINE T neutral() { return static_cast<T>(1); }
  template <typename T> static SD_DEVICE SD_INLINE void combine(T& acc, T v) { acc *= v; }
  template <typename T> static SD_DEVICE SD_INLINE void atomic(T* addr, T v) { math::atomics::sd_atomicMul<T>(addr, v); }
  template <typename T> static SD_DEVICE SD_INLINE T finalize(T acc, LongType) { return acc; }
  template <typename T> static SD_DEVICE SD_INLINE T tadValue(T v, LongType) { return v; }
};

struct MeanReduce {
  template <typename T> static SD_HOST_DEVICE SD_INLINE T neutral() { return static_cast<T>(0); }
  template <typename T> static SD_DEVICE SD_INLINE void combine(T& acc, T v) { acc += v; }
  template <typename T> static SD_DEVICE SD_INLINE void atomic(T* addr, T v) { math::atomics::sd_atomicAdd<T>(addr, v); }
  template <typename T> static SD_DEVICE SD_INLINE T finalize(T acc, LongType len) {
    return len > 0 ? acc / static_cast<T>(len) : acc;
  }
  template <typename T> static SD_DEVICE SD_INLINE T tadValue(T v, LongType len) {
    return len > 0 ? v / static_cast<T>(len) : v;
  }
};

// ============================================================================ //
// GradOp policies (backprop). Each provides:
//   static constexpr bool needsForward  - re-run forward into tempRes?
//   static constexpr bool needsLengths  - materialize segment lengths?
//   grad(gradOut, x, fwd, len, out)     - returns true + fills out when the
//                                         element receives gradient (device)
// ============================================================================ //

// max/min: gradient flows only to the element(s) that produced the extremum.
struct CompareGrad {
  static constexpr bool needsForward = true;
  static constexpr bool needsLengths = false;
  template <typename T>
  static SD_DEVICE SD_INLINE bool grad(T gradOut, T x, T fwd, LongType, T& out) {
    if (math::sd_abs<T, T>(fwd - x) <= T(1.e-6)) { out = gradOut; return true; }
    return false;
  }
};

// prod: d/dx_i (prod) = prod / x_i, so grad = gradOut * fwdProduct / x_i.
struct ProdGrad {
  static constexpr bool needsForward = true;
  static constexpr bool needsLengths = false;
  template <typename T>
  static SD_DEVICE SD_INLINE bool grad(T gradOut, T x, T fwd, LongType, T& out) {
    out = gradOut * fwd / x;
    return true;
  }
};

// sum: gradient is broadcast unchanged to every segment member.
struct SumGrad {
  static constexpr bool needsForward = false;
  static constexpr bool needsLengths = false;
  template <typename T>
  static SD_DEVICE SD_INLINE bool grad(T gradOut, T, T, LongType, T& out) {
    out = gradOut;
    return true;
  }
};

// mean: gradient is broadcast and scaled by 1/segmentLength.
struct MeanGrad {
  static constexpr bool needsForward = false;
  static constexpr bool needsLengths = true;
  template <typename T>
  static SD_DEVICE SD_INLINE bool grad(T gradOut, T, T, LongType len, T& out) {
    out = len > 0 ? gradOut / static_cast<T>(len) : gradOut;
    return true;
  }
};

// ============================================================================ //
// Forward kernels
// ============================================================================ //

// Linear (vector input) forward: one block per segment, shared-memory tree reduce.
template <typename Op, typename T, typename I>
static SD_KERNEL void segmentLinearKernel(void* input, LongType const* inputShape, LongType* starts, LongType* lengths,
                                          LongType numOfClasses, void* output, LongType const* outputShape) {
  extern __shared__ char shmem[];
  T* sdata = reinterpret_cast<T*>(shmem);

  auto segment = blockIdx.x;
  if (segment >= numOfClasses) return;

  const T* x = reinterpret_cast<const T*>(input);
  T* z = reinterpret_cast<T*>(output);

  auto start = starts[segment];
  auto len = lengths[segment];
  if (len == 0) return;
  auto finish = start + len;

  const LongType* inputStridePtr = shape::stride(inputShape);
  const LongType* outputStridePtr = shape::stride(outputShape);

  T threadAcc = Op::template neutral<T>();
  for (auto e = start + threadIdx.x; e < finish; e += blockDim.x) {
    LongType eCoords[] = {e};
    LongType xIndex;
    COORDS2INDEX(1, inputStridePtr, eCoords, xIndex);
    Op::template combine<T>(threadAcc, x[xIndex]);
  }

  sdata[threadIdx.x] = threadAcc;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x) {
      Op::template combine<T>(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    LongType segmentCoords[] = {segment};
    LongType zIndex;
    COORDS2INDEX(1, outputStridePtr, segmentCoords, zIndex);
    z[zIndex] = Op::template finalize<T>(sdata[0], len);
  }
}

// Unsorted linear forward: one block per segment, scans all elements gated by index.
template <typename Op, typename T, typename I>
static SD_KERNEL void unsortedSegmentLinearKernel(void* input, LongType const* inputShape, void* indices,
                                                  LongType const* indicesShape, LongType* starts, LongType* lengths,
                                                  LongType numOfClasses, void* output, LongType const* outputShape) {
  extern __shared__ char shmem[];
  T* sdata = reinterpret_cast<T*>(shmem);

  auto segment = blockIdx.x;
  if (segment >= numOfClasses) return;
  if (lengths[segment] == 0) return;

  const T* x = reinterpret_cast<const T*>(input);
  T* z = reinterpret_cast<T*>(output);
  const I* y = reinterpret_cast<const I*>(indices);

  LongType xLen = shape::length(inputShape);
  const LongType* inputStridePtr = shape::stride(inputShape);
  const LongType* indicesStridePtr = shape::stride(indicesShape);
  const LongType* outputStridePtr = shape::stride(outputShape);

  T threadAcc = Op::template neutral<T>();
  for (auto e = threadIdx.x; e < xLen; e += blockDim.x) {
    LongType eCoords[] = {e};
    LongType xIndex, yIndex;
    COORDS2INDEX(1, inputStridePtr, eCoords, xIndex);
    COORDS2INDEX(1, indicesStridePtr, eCoords, yIndex);
    if (y[yIndex] == static_cast<I>(segment)) {
      Op::template combine<T>(threadAcc, x[xIndex]);
    }
  }

  sdata[threadIdx.x] = threadAcc;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x) {
      Op::template combine<T>(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    LongType segmentCoords[] = {segment};
    LongType zIndex;
    COORDS2INDEX(1, outputStridePtr, segmentCoords, zIndex);
    z[zIndex] = Op::template finalize<T>(sdata[0], lengths[segment]);
  }
}

// TAD (multi-dim) forward: global atomics into the pre-initialized output.
template <typename Op, typename T, typename I>
static SD_KERNEL void segmentTadKernel(void* inputBuf, LongType const* inputShape, LongType const* inputTads,
                                       LongType const* inputTadOffsets, I* indices, LongType* starts, LongType* lengths,
                                       LongType numOfClasses, void* outputBuf, LongType const* outputShape,
                                       LongType const* outputTads, LongType const* outputTadOffsets,
                                       LongType indicesLength, LongType numInputTads) {
  __shared__ LongType len, start, finish;
  __shared__ I segment;
  __shared__ sd::LongType inputTadRank, outputTadRank;
  __shared__ const sd::LongType* inputTadShapePtr;
  __shared__ const sd::LongType* outputTadShapePtr;
  __shared__ const sd::LongType* inputTadStridePtr;
  __shared__ const sd::LongType* outputTadStridePtr;

  if (threadIdx.x == 0 && blockIdx.x < indicesLength) {
    segment = indices[blockIdx.x];
    len = shape::length(inputTads);
    inputTadRank = shape::rank(inputTads);
    outputTadRank = shape::rank(outputTads);
    inputTadShapePtr = shape::shapeOf(inputTads);
    outputTadShapePtr = shape::shapeOf(outputTads);
    inputTadStridePtr = shape::stride(inputTads);
    outputTadStridePtr = shape::stride(outputTads);
    start = starts[segment];
    finish = start + lengths[segment];
  }
  __syncthreads();

  auto idx = blockIdx.x;
  if (idx < numInputTads) {
    auto x = reinterpret_cast<T*>(inputBuf) + inputTadOffsets[idx];
    auto z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
    if (blockIdx.x == start || lengths[segment]) {
      for (auto e = threadIdx.x; e < len; e += blockDim.x) {
        LongType xCoords[SD_MAX_RANK];
        LongType zCoords[SD_MAX_RANK];
        LongType xIndex;
        LongType zIndex;
        INDEX2COORDS(e, inputTadRank, inputTadShapePtr, xCoords);
        COORDS2INDEX(inputTadRank, inputTadStridePtr, xCoords, xIndex);
        INDEX2COORDS(e, outputTadRank, outputTadShapePtr, zCoords);
        COORDS2INDEX(outputTadRank, outputTadStridePtr, zCoords, zIndex);
        Op::template atomic<T>(&z[zIndex], Op::template tadValue<T>(x[xIndex], lengths[segment]));
      }
    }
  }
}

// ============================================================================ //
// Backprop kernels (one scaffold, GradOp-parameterized)
// ============================================================================ //

template <typename Grad, typename T, typename I>
static SD_KERNEL void segmentBPLinearKernel(void* inputBuf, LongType const* inputShape, void* forwardOutput,
                                            LongType const* forwardShape, void* eps, LongType const* epsShape,
                                            void* indicesBuf, LongType const* indicesShape, LongType* lengths,
                                            void* outputBuf, LongType const* outputShape, LongType indicesLen) {
  auto x = reinterpret_cast<T*>(inputBuf);
  auto y = reinterpret_cast<I*>(indicesBuf);
  auto z = reinterpret_cast<T*>(outputBuf);
  auto gradIn = reinterpret_cast<T*>(forwardOutput);
  auto gradOut = reinterpret_cast<T*>(eps);

  const LongType xRank = shape::rank(inputShape), yRank = shape::rank(indicesShape), zRank = shape::rank(outputShape);
  const LongType* xStridePtr = shape::stride(inputShape);
  const LongType* yStridePtr = shape::stride(indicesShape);
  const LongType* zStridePtr = shape::stride(outputShape);
  const LongType* xShapePtr = shape::shapeOf(inputShape);
  const LongType* yShapePtr = shape::shapeOf(indicesShape);
  const LongType* zShapePtr = shape::shapeOf(outputShape);
  const LongType gradInRank = Grad::needsForward ? shape::rank(forwardShape) : 0;
  const LongType* gradInStridePtr = Grad::needsForward ? shape::stride(forwardShape) : nullptr;
  const LongType* gradInShapePtr = Grad::needsForward ? shape::shapeOf(forwardShape) : nullptr;
  const LongType gradOutRank = shape::rank(epsShape);
  const LongType* gradOutStridePtr = shape::stride(epsShape);
  const LongType* gradOutShapePtr = shape::shapeOf(epsShape);

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  for (auto e = tid; e < indicesLen; e += step) {
    LongType zCoords[SD_MAX_RANK], xCoords[SD_MAX_RANK], yCoords[SD_MAX_RANK];
    LongType zOffset, xOffset, yOffset;
    INDEX2COORDS(e, zRank, zShapePtr, zCoords);
    COORDS2INDEX(zRank, zStridePtr, zCoords, zOffset);
    INDEX2COORDS(e, xRank, xShapePtr, xCoords);
    COORDS2INDEX(xRank, xStridePtr, xCoords, xOffset);
    INDEX2COORDS(e, yRank, yShapePtr, yCoords);
    COORDS2INDEX(yRank, yStridePtr, yCoords, yOffset);

    auto classIndex = y[yOffset];
    LongType gradOffsetO;
    LongType gradOCoords[SD_MAX_RANK];
    INDEX2COORDS(classIndex, gradOutRank, gradOutShapePtr, gradOCoords);
    COORDS2INDEX(gradOutRank, gradOutStridePtr, gradOCoords, gradOffsetO);

    T fwd = T(0);
    if (Grad::needsForward) {
      LongType gradICoords[SD_MAX_RANK];
      LongType gradOffsetI;
      INDEX2COORDS(classIndex, gradInRank, gradInShapePtr, gradICoords);
      COORDS2INDEX(gradInRank, gradInStridePtr, gradICoords, gradOffsetI);
      fwd = gradIn[gradOffsetI];
    }
    LongType len = Grad::needsLengths ? lengths[classIndex] : 0;

    T out;
    if (Grad::template grad<T>(gradOut[gradOffsetO], x[xOffset], fwd, len, out)) z[zOffset] = out;
  }
}

template <typename Grad, typename T, typename I>
static SD_KERNEL void segmentBPTadKernel(void* inputBuf, LongType const* inputShape, void* forwardOutput,
                                         LongType const* forwardShape, void* eps, LongType const* epsShape,
                                         void* indicesBuf, LongType const* indicesShape, LongType* lengths,
                                         void* outputBuf, LongType const* outputShape, LongType const* inputTad,
                                         LongType const* inputOffsets, LongType const* gradInTad,
                                         LongType const* gradInOffsets, LongType const* gradOutTad,
                                         LongType const* gradOutOffsets, LongType const* outTad,
                                         LongType const* outOffsets, LongType indicesLen) {
  auto x = reinterpret_cast<T*>(inputBuf);
  auto indices = reinterpret_cast<I*>(indicesBuf);
  auto z = reinterpret_cast<T*>(outputBuf);
  auto gradIn = reinterpret_cast<T*>(forwardOutput);
  auto gradOut = reinterpret_cast<T*>(eps);

  const LongType currentLen = shape::length(inputTad);
  const LongType inputTadRank = shape::rank(inputTad);
  const LongType* inputTadShapePtr = shape::shapeOf(inputTad);
  const LongType* inputTadStridePtr = shape::stride(inputTad);
  const LongType gradOutTadRank = shape::rank(gradOutTad);
  const LongType* gradOutTadShapePtr = shape::shapeOf(gradOutTad);
  const LongType* gradOutTadStridePtr = shape::stride(gradOutTad);
  const LongType outTadRank = shape::rank(outTad);
  const LongType* outTadShapePtr = shape::shapeOf(outTad);
  const LongType* outTadStridePtr = shape::stride(outTad);
  const LongType gradInTadRank = Grad::needsForward ? shape::rank(gradInTad) : 0;
  const LongType* gradInTadShapePtr = Grad::needsForward ? shape::shapeOf(gradInTad) : nullptr;
  const LongType* gradInTadStridePtr = Grad::needsForward ? shape::stride(gradInTad) : nullptr;

  for (auto i = blockIdx.x; i < indicesLen; i += gridDim.x) {
    I segment = indices[i];
    auto current2 = x + inputOffsets[i];
    auto currentOut2 = z + outOffsets[i];
    auto currentGradOut2 = gradOut + gradOutOffsets[segment];
    auto gradIn2 = Grad::needsForward ? gradIn + gradInOffsets[segment] : nullptr;
    LongType len = Grad::needsLengths ? lengths[segment] : 0;

    for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
      LongType xCoords[SD_MAX_RANK], gradOutCoords[SD_MAX_RANK], outCoords[SD_MAX_RANK];
      LongType xIndex, gradOutIndex, outIndex;
      INDEX2COORDS(e, inputTadRank, inputTadShapePtr, xCoords);
      COORDS2INDEX(inputTadRank, inputTadStridePtr, xCoords, xIndex);
      INDEX2COORDS(e, gradOutTadRank, gradOutTadShapePtr, gradOutCoords);
      COORDS2INDEX(gradOutTadRank, gradOutTadStridePtr, gradOutCoords, gradOutIndex);
      INDEX2COORDS(e, outTadRank, outTadShapePtr, outCoords);
      COORDS2INDEX(outTadRank, outTadStridePtr, outCoords, outIndex);

      T fwd = T(0);
      if (Grad::needsForward) {
        LongType gradInCoords[SD_MAX_RANK];
        LongType gradInIndex;
        INDEX2COORDS(e, gradInTadRank, gradInTadShapePtr, gradInCoords);
        COORDS2INDEX(gradInTadRank, gradInTadStridePtr, gradInCoords, gradInIndex);
        fwd = gradIn2[gradInIndex];
      }

      T out;
      if (Grad::template grad<T>(currentGradOut2[gradOutIndex], current2[xIndex], fwd, len, out))
        currentOut2[outIndex] = out;
    }
  }
}

// ============================================================================ //
// Host: segment length materialization (shared by every functor)
// ============================================================================ //

struct SegmentRanges {
  NDArray* begs;
  NDArray* lens;
  LongType* begins;
  LongType* lengths;
};

template <typename I>
static SegmentRanges buildSegmentRanges(LaunchContext* context, NDArray* indices, LongType numOfClasses) {
  auto begs = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
  auto lens = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
  LongType len = indices->lengthOf();
  int zero = 0;
  begs->assign(len);
  lens->assign(zero);
  fillUpSegments(indices, numOfClasses, *begs, *lens);
  return {begs, lens, reinterpret_cast<LongType*>(begs->specialBuffer()),
          reinterpret_cast<LongType*>(lens->specialBuffer())};
}

static SD_INLINE unsigned int roundUpPow2(unsigned int threads) {
  threads--;
  threads |= threads >> 1;
  threads |= threads >> 2;
  threads |= threads >> 4;
  threads |= threads >> 8;
  threads |= threads >> 16;
  threads++;
  return threads < 1 ? 1 : threads;
}

// ============================================================================ //
// Host: forward
// ============================================================================ //

template <typename Op, typename T, typename I>
static void segmentForward_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  T neutralVal = Op::template neutral<T>();
  output->assign(neutralVal);
  auto stream = context->getCudaStream();
  indices->syncToHost();
  LongType numOfClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
  auto ranges = buildSegmentRanges<I>(context, indices, numOfClasses);
  NDArray* rBegs = ranges.begs;
  NDArray* rLens = ranges.lens;

  NDArray::prepareSpecialUse({output}, {input, indices, rBegs, rLens});
  if (input->isVector() || input->isScalar()) {
    dim3 launchDims = segmentDims(numOfClasses, input->lengthOf());
    unsigned int threads = roundUpPow2(launchDims.x);
    LongType shmemSize = threads * sizeof(T);
    if (shmemSize < static_cast<LongType>(launchDims.z)) shmemSize = launchDims.z;
    segmentLinearKernel<Op, T, I><<<launchDims.y, threads, shmemSize, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), ranges.begins, ranges.lengths, numOfClasses,
        output->specialBuffer(), output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentLinearKernel failed");
  } else {
    LongType zero = 0;
    std::vector<LongType>* dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1, &zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    dim3 launchDims = segmentTad(packX->numberOfTads());
    segmentTadKernel<Op, T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), packX->specialShapeInfo(), packX->specialOffsets(),
        reinterpret_cast<I*>(indices->specialBuffer()), ranges.begins, ranges.lengths, numOfClasses,
        output->specialBuffer(), output->specialShapeInfo(), packZ->specialShapeInfo(), packZ->specialOffsets(),
        indices->lengthOf(), packX->numberOfTads());
    sd::DebugHelper::checkErrorCode(stream, "segmentTadKernel failed");
    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, rBegs, rLens});
  delete ranges.begs;
  delete ranges.lens;
}

template <typename Op, typename T, typename I>
static void unsortedSegmentForward_(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                                    NDArray* output) {
  T neutralVal = Op::template neutral<T>();
  output->assign(neutralVal);
  auto stream = context->getCudaStream();
  auto ranges = buildSegmentRanges<I>(context, indices, numOfClasses);
  NDArray* rBegs = ranges.begs;
  NDArray* rLens = ranges.lens;

  NDArray::prepareSpecialUse({output}, {input, indices, rBegs, rLens});
  if (input->isVector() || input->isScalar()) {
    dim3 dims = getFillUpSegmentsDims(numOfClasses, indices->lengthOf());
    unsigned int threads = roundUpPow2(dims.y);
    LongType shmemSize = threads * sizeof(T);
    if (shmemSize < static_cast<LongType>(dims.z)) shmemSize = dims.z;
    unsortedSegmentLinearKernel<Op, T, I><<<dims.x, threads, shmemSize, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        ranges.begins, ranges.lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentLinearKernel failed");
  } else {
    LongType zero = 0;
    std::vector<LongType>* dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1, &zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    dim3 launchDims = segmentTad(packX->numberOfTads());
    segmentTadKernel<Op, T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), packX->specialShapeInfo(), packX->specialOffsets(),
        reinterpret_cast<I*>(indices->specialBuffer()), ranges.begins, ranges.lengths, numOfClasses,
        output->specialBuffer(), output->specialShapeInfo(), packZ->specialShapeInfo(), packZ->specialOffsets(),
        indices->lengthOf(), packX->numberOfTads());
    sd::DebugHelper::checkErrorCode(stream, "segmentTadKernel(unsorted) failed");
    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, rBegs, rLens});
  delete ranges.begs;
  delete ranges.lens;
}

// ============================================================================ //
// Host: backprop
// ============================================================================ //

template <typename Op, typename Grad, typename T, typename I>
static Status segmentBackprop_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                               LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();

  NDArray* tempRes = nullptr;
  if (Grad::needsForward) {
    tempRes = gradOut->ulike();
    unsortedSegmentForward_<Op, T, I>(context, input, indices, numOfClasses, tempRes);
  }
  SegmentRanges ranges{nullptr, nullptr, nullptr, nullptr};
  if (Grad::needsLengths) ranges = buildSegmentRanges<I>(context, indices, numOfClasses);

  auto fwdBuf = Grad::needsForward ? tempRes->specialBuffer() : nullptr;
  auto fwdShape = Grad::needsForward ? tempRes->specialShapeInfo() : nullptr;

  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  if (input->isVector() || input->isScalar()) {
    dim3 dims = segmentBpDims(1 + gradOut->lengthOf(), input->lengthOf());
    segmentBPLinearKernel<Grad, T, I><<<dims.y, dims.x, dims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), fwdBuf, fwdShape, gradOut->specialBuffer(),
        gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), ranges.lengths,
        output->specialBuffer(), output->specialShapeInfo(), indices->lengthOf());
    sd::DebugHelper::checkErrorCode(stream, "segmentBPLinearKernel failed");
  } else {
    LongType zero = 0;
    std::vector<LongType>* dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1, &zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    if (Grad::needsForward) NDArray::preparePrimaryUse({tempRes}, {tempRes});
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
    LongType const* gradInTad = nullptr;
    LongType const* gradInOffsets = nullptr;
    if (Grad::needsForward) {
      auto packGradIn = ConstantTadHelper::getInstance().tadForDimensions(tempRes->shapeInfo(), dimensions);
      gradInTad = packGradIn->specialShapeInfo();
      gradInOffsets = packGradIn->specialOffsets();
    }
    dim3 dims = segmentBpTad(gradOut->lengthOf(), input->lengthOf());
    segmentBPTadKernel<Grad, T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), fwdBuf, fwdShape, gradOut->specialBuffer(),
        gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), ranges.lengths,
        output->specialBuffer(), output->specialShapeInfo(), packX->specialShapeInfo(), packX->specialOffsets(),
        gradInTad, gradInOffsets, packGradOut->specialShapeInfo(), packGradOut->specialOffsets(),
        packZ->specialShapeInfo(), packZ->specialOffsets(), indices->lengthOf());
    sd::DebugHelper::checkErrorCode(stream, "segmentBPTadKernel failed");
    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
  if (tempRes) delete tempRes;
  if (ranges.begs) {
    delete ranges.begs;
    delete ranges.lens;
  }
  return Status::OK;
}

}  // namespace segment_ops
}  // namespace helpers
}  // namespace ops
}  // namespace sd

// ============================================================================ //
// Instantiation macros — a segment_<op>.cu becomes a thin TU using these.
//   SEGMENT_OP_SORTED   : segment<Op>Functor            (sorted forward)
//   SEGMENT_OP_UNSORTED : unsortedSegment<Op>Functor    (unsorted forward)
//   SEGMENT_OP_BP       : segment<Op>FunctorBP + unsortedSegment<Op>FunctorBP
// FWD_TYPES is SD_NUMERIC_TYPES (forward) / SD_FLOAT_TYPES (bp) per the originals.
// ============================================================================ //

#define SEGMENT_OP_INSTANTIATE(NAME, REDUCE, GRAD)                                                                 \
  namespace sd {                                                                                                   \
  namespace ops {                                                                                                  \
  namespace helpers {                                                                                             \
  namespace segment_ops {                                                                                          \
  template <typename T, typename I>                                                                                \
  static void segFwd_##NAME(LaunchContext* c, NDArray* in, NDArray* idx, NDArray* out) {                          \
    segmentForward_<REDUCE, T, I>(c, in, idx, out);                                                               \
  }                                                                                                                \
  template <typename T, typename I>                                                                                \
  static void segUFwd_##NAME(LaunchContext* c, NDArray* in, NDArray* idx, LongType n, NDArray* out) {             \
    unsortedSegmentForward_<REDUCE, T, I>(c, in, idx, n, out);                                                    \
  }                                                                                                                \
  template <typename T, typename I>                                                                                \
  static Status segBp_##NAME(LaunchContext* c, NDArray* in, NDArray* idx, NDArray* g, LongType n, NDArray* out) { \
    return segmentBackprop_<REDUCE, GRAD, T, I>(c, in, idx, g, n, out);                                           \
  }                                                                                                                \
  }  /* namespace segment_ops */                                                                                   \
  void segment##NAME##Functor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {        \
    NDArray::prepareSpecialUse({output}, {input, indices});                                                       \
    BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segment_ops::segFwd_##NAME,                     \
                          (context, input, indices, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);                \
    NDArray::registerSpecialUse({output}, {input, indices});                                                      \
  }                                                                                                                \
  void unsortedSegment##NAME##Functor(LaunchContext* context, NDArray* input, NDArray* indices,                   \
                                      LongType numOfClasses, NDArray* output) {                                   \
    NDArray::prepareSpecialUse({output}, {input, indices});                                                       \
    output->nullify();                                                                                            \
    BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segment_ops::segUFwd_##NAME,                    \
                          (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);  \
    NDArray::registerSpecialUse({output}, {input, indices});                                                      \
  }                                                                                                                \
  Status segment##NAME##FunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,     \
                                  NDArray* output) {                                                              \
    NDArray::prepareSpecialUse({output}, {input, indices, gradOut});                                              \
    BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), segment_ops::segBp_##NAME,                     \
                          (context, input, indices, gradOut, gradOut->sizeAt(0), output), SD_FLOAT_TYPES,         \
                          SD_INDEXING_TYPES);                                                                     \
    NDArray::registerSpecialUse({output}, {input, indices, gradOut});                                            \
    return Status::OK;                                                                                            \
  }                                                                                                                \
  Status unsortedSegment##NAME##FunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,               \
                                          NDArray* gradOut, LongType numOfClasses, NDArray* output) {             \
    NDArray::prepareSpecialUse({output}, {input, indices, gradOut});                                              \
    BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), segment_ops::segBp_##NAME,                     \
                          (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES,               \
                          SD_INDEXING_TYPES);                                                                     \
    NDArray::registerSpecialUse({output}, {input, indices, gradOut});                                            \
    return Status::OK;                                                                                            \
  }                                                                                                                \
  }                                                                                                                \
  }                                                                                                                \
  }

#endif  // LIBND4J_SEGMENT_OPS_CUH
