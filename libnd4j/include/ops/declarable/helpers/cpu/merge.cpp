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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//
#include <helpers/Loops.h>
#include <ops/declarable/helpers/transforms.h>
#include <ops/op_types.h>
#if NOT_EXCLUDED(OP_merge)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void mergeMaxIndex_(const std::vector<NDArray*>& inArrs, NDArray& output) {
  const sd::LongType numArgs = inArrs.size();
  const sd::LongType length = output.lengthOf();
  const int rank = output.rankOf();

  // Logical coordinates through each array's own strides, as mergeMax_ and mergeAvg_ use: t<X>(e)
  // and r<Z>(e) walk each array in its own order, so an 'f' output paired a 'c' input's element e
  // with a different element of its own (DL4J's ElementWiseVertex(Max) backward).
  auto outputShape = output.shapeInfo();
  std::vector<bool> vbSameShapeAndStrides(numArgs);
  std::vector<sd::LongType*> vStridePtrs(numArgs);
  std::vector<sd::LongType> vRanks(numArgs);
  std::vector<const X*> vBuffers(numArgs);
  for (int i = 0; i < numArgs; ++i) {
    vbSameShapeAndStrides[i] = shape::haveSameShapeAndStrides(outputShape, inArrs[i]->shapeInfo());
    vStridePtrs[i] = shape::stride(inArrs[i]->shapeInfo());
    vRanks[i] = shape::rank(inArrs[i]->shapeInfo());
    vBuffers[i] = inArrs[i]->bufferAsT<X>();
  }

  sd::LongType *outputShapeOf = shape::shapeOf(outputShape);
  sd::LongType *outputStride = shape::stride(outputShape);
  Z* outBuffer = output.bufferAsT<Z>();

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (auto e = start; e < stop; e++) {
      INDEX2COORDS(e, rank, outputShapeOf, coords);
      sd::LongType outOffset;
      COORDS2INDEX(rank, outputStride, coords, outOffset);

      X max = -DataTypeUtils::max<X>();
      Z idx = static_cast<Z>(0);
      for (sd::LongType i = 0; i < numArgs; i++) {
        sd::LongType xOffset;
        if (vbSameShapeAndStrides[i]) {
          xOffset = outOffset;
        } else {
          COORDS2INDEX(vRanks[i], vStridePtrs[i], coords, xOffset);
        }
        const X v = vBuffers[i][xOffset];
        if (v > max) {
          max = v;
          idx = static_cast<Z>(i);
        }
      }
      outBuffer[outOffset] = idx;
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}

void mergeMaxIndex(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  BUILD_DOUBLE_SELECTOR(inArrs[0]->dataType(), output.dataType(), mergeMaxIndex_, (inArrs, output), SD_NUMERIC_TYPES,
                        SD_INDEXING_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeMax_(const std::vector<NDArray*>& inArrs, NDArray& output) {
  const sd::LongType numArgs = inArrs.size();
  const sd::LongType length = output.lengthOf();
  const int rank = output.rankOf();
  
  // Check if all inputs have same shape and strides as output
  auto outputShape = output.shapeInfo();
  std::vector<bool> vbSameShapeAndStrides(numArgs);
  std::vector<sd::LongType*> vShapePtrs(numArgs);
  std::vector<sd::LongType*> vStridePtrs(numArgs);
  std::vector<sd::LongType> vRanks(numArgs);
  std::vector<const T*> vBuffers(numArgs);
  
  for (int i = 0; i < numArgs; ++i) {
    vbSameShapeAndStrides[i] = shape::haveSameShapeAndStrides(outputShape, inArrs[i]->shapeInfo());
    vShapePtrs[i] = shape::shapeOf(inArrs[i]->shapeInfo());
    vStridePtrs[i] = shape::stride(inArrs[i]->shapeInfo());
    vRanks[i] = shape::rank(inArrs[i]->shapeInfo());
    vBuffers[i] = inArrs[i]->bufferAsT<T>();
  }
  
  sd::LongType *outputShapeOf = shape::shapeOf(outputShape);
  sd::LongType *outputStride = shape::stride(outputShape);
  T* outBuffer = output.bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (auto e = start; e < stop; e++) {
      INDEX2COORDS(e, rank, outputShapeOf, coords);
      
      sd::LongType outOffset;
      COORDS2INDEX(rank, outputStride, coords, outOffset);
      
      T max = -DataTypeUtils::max<T>();
      for (sd::LongType i = 0; i < numArgs; i++) {
        sd::LongType xOffset;
        if (vbSameShapeAndStrides[i]) {
          xOffset = outOffset;
        } else {
          COORDS2INDEX(vRanks[i], vStridePtrs[i], coords, xOffset);
        }
        T v = vBuffers[i][xOffset];
        if (v > max) max = v;
      }
      outBuffer[outOffset] = max;
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}

void mergeMax(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  BUILD_SINGLE_SELECTOR(output.dataType(), mergeMax_, (inArrs, output), SD_NUMERIC_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeMaxBp_(const std::vector<NDArray*>& inArrs, std::vector<NDArray*>& outArrs) {
  // outArrs.size() == inArrs.size() - 1
  const sd::LongType numArgs = outArrs.size();
  // last array is gradient
  const auto gradient = inArrs[numArgs]->bufferAsT<T>();
  auto length = inArrs[numArgs]->lengthOf();

  auto gradShape = inArrs[numArgs]->shapeInfo();
  std::vector<bool> vbSameShaepeAndStrides(numArgs);
  std::vector<bool> vbOutSameShapeAndStrides(numArgs);
  std::vector<sd::LongType*> vShapePtrs(numArgs);
  std::vector<sd::LongType*> vStridePtrs(numArgs);
  std::vector<sd::LongType> vRanks(numArgs);
  for (int i = 0; i < numArgs; ++i) {
    vbSameShaepeAndStrides[i] = shape::haveSameShapeAndStrides(gradShape, inArrs[i]->shapeInfo());
    // Output i has its own strides: it took input i's flag, so an output laid out unlike its input
    // was written at the gradient's offset.
    vbOutSameShapeAndStrides[i] = shape::haveSameShapeAndStrides(gradShape, outArrs[i]->shapeInfo());
    vShapePtrs[i] = shape::shapeOf(inArrs[i]->shapeInfo());
    vStridePtrs[i] = shape::stride(inArrs[i]->shapeInfo());
    vRanks[i] = shape::rank(inArrs[i]->shapeInfo());
  }


  std::vector<sd::LongType *> outShapePtrs(numArgs);
  std::vector<sd::LongType *> outStridePtrs(numArgs);
  std::vector<sd::LongType> outRanks(numArgs);
  for (int i = 0; i < numArgs; ++i) {
    outShapePtrs[i] = shape::shapeOf(outArrs[i]->shapeInfo());
    outStridePtrs[i] = shape::stride(outArrs[i]->shapeInfo());
    outRanks[i] = shape::rank(outArrs[i]->shapeInfo());
  }

  sd::LongType gradRank = shape::rank(gradShape);
  sd::LongType *gradShapeOf = shape::shapeOf(gradShape);
  sd::LongType *gradStride = shape::stride(gradShape);
  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (auto e = start; e < stop; e++) {
      INDEX2COORDS(e, gradRank, gradShapeOf, coords);

      sd::LongType gradOffset;
      COORDS2INDEX(gradRank,gradStride, coords, gradOffset);

      T max = -DataTypeUtils::max<T>();
      sd::LongType nMaxIndex = 0;

      for (sd::LongType i = 0; i < numArgs; i++) {
        sd::LongType xOffset;
        if (vbSameShaepeAndStrides[i]) {
          xOffset = gradOffset;
        } else {
          COORDS2INDEX(vRanks[i],vStridePtrs[i], coords, xOffset);
        }
        const T* v = inArrs[i]->bufferAsT<T>();
        if (v[xOffset] > max) {
          max = v[xOffset];
          nMaxIndex = i;
        }
      }

      sd::LongType zOffset;
      if (vbOutSameShapeAndStrides[nMaxIndex]) {
        zOffset = gradOffset;
      } else {
        COORDS2INDEX(outRanks[nMaxIndex],outStridePtrs[nMaxIndex], coords, zOffset);
      }

      T* z = outArrs[nMaxIndex]->bufferAsT<T>();
      z[zOffset] = gradient[gradOffset];
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
  return;
}

void mergeMaxBp(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, std::vector<NDArray*>& outArrs) {
  BUILD_SINGLE_SELECTOR(outArrs[0]->dataType(), mergeMaxBp_, (inArrs, outArrs), SD_NUMERIC_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeAvg_(const std::vector<NDArray*>& inArrs, NDArray& output) {
  // the sum and its division by the number of arrays are in the aggregation type: a reciprocal taken in float scaled a
  // DOUBLE average by a float's worth of error
  using AccT = typename simdOps::AggregateType<T>::type;
  const sd::LongType numArgs = inArrs.size();
  const sd::LongType length = output.lengthOf();
  const int rank = output.rankOf();

  // Use coordinate-based access to correctly handle mixed F/C orderings.
  // mergeMax_ already uses this pattern. The old e<T>(e)/p<T>(e,...) approach
  // mis-pairs elements when inputs are F-order and output is C-order.
  auto outputShape = output.shapeInfo();
  std::vector<bool> vbSameShapeAndStrides(numArgs);
  std::vector<sd::LongType*> vStridePtrs(numArgs);
  std::vector<sd::LongType> vRanks(numArgs);
  std::vector<const T*> vBuffers(numArgs);

  for (int i = 0; i < numArgs; ++i) {
    vbSameShapeAndStrides[i] = shape::haveSameShapeAndStrides(outputShape, inArrs[i]->shapeInfo());
    vStridePtrs[i] = shape::stride(inArrs[i]->shapeInfo());
    vRanks[i] = shape::rank(inArrs[i]->shapeInfo());
    vBuffers[i] = inArrs[i]->bufferAsT<T>();
  }

  sd::LongType *outputShapeOf = shape::shapeOf(outputShape);
  sd::LongType *outputStride = shape::stride(outputShape);
  T* outBuffer = output.bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (auto e = start; e < stop; e++) {
      INDEX2COORDS(e, rank, outputShapeOf, coords);
      sd::LongType outOffset;
      COORDS2INDEX(rank, outputStride, coords, outOffset);

      AccT sum = static_cast<AccT>(0);
      for (sd::LongType i = 0; i < numArgs; i++) {
        sd::LongType xOffset;
        if (vbSameShapeAndStrides[i]) {
          xOffset = outOffset;
        } else {
          COORDS2INDEX(vRanks[i], vStridePtrs[i], coords, xOffset);
        }
        sum += static_cast<AccT>(vBuffers[i][xOffset]);
      }
      outBuffer[outOffset] = static_cast<T>(sum / static_cast<AccT>(numArgs));
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}

void mergeAvg(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  BUILD_SINGLE_SELECTOR(output.dataType(), mergeAvg_, (inArrs, output), SD_NUMERIC_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeAvgBp_(NDArray& gradient, std::vector<NDArray*>& outArrs) {
  const sd::LongType numArgs = outArrs.size();
  const sd::LongType length = gradient.lengthOf();
  const int gradRank = gradient.rankOf();

  // Use coordinate-based access to correctly handle mixed F/C orderings.
  // The old e<T>(e)/p<T>(e,...) approach mis-pairs elements when gradient is
  // C-order and outputs are F-order (matching the forward-pass input ordering).
  auto gradShape = gradient.shapeInfo();
  std::vector<bool> vbSameShapeAndStrides(numArgs);
  std::vector<sd::LongType*> vStridePtrs(numArgs);
  std::vector<sd::LongType> vRanks(numArgs);
  std::vector<T*> vOutBuffers(numArgs);

  for (int i = 0; i < numArgs; ++i) {
    vbSameShapeAndStrides[i] = shape::haveSameShapeAndStrides(gradShape, outArrs[i]->shapeInfo());
    vStridePtrs[i] = shape::stride(outArrs[i]->shapeInfo());
    vRanks[i] = shape::rank(outArrs[i]->shapeInfo());
    vOutBuffers[i] = outArrs[i]->bufferAsT<T>();
  }

  sd::LongType *gradShapeOf = shape::shapeOf(gradShape);
  sd::LongType *gradStride = shape::stride(gradShape);
  const T* gradBuffer = gradient.bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (auto e = start; e < stop; e++) {
      INDEX2COORDS(e, gradRank, gradShapeOf, coords);
      sd::LongType gradOffset;
      COORDS2INDEX(gradRank, gradStride, coords, gradOffset);

      T v = gradBuffer[gradOffset] / static_cast<T>(numArgs);

      for (sd::LongType i = 0; i < numArgs; i++) {
        sd::LongType outOffset;
        if (vbSameShapeAndStrides[i]) {
          outOffset = gradOffset;
        } else {
          COORDS2INDEX(vRanks[i], vStridePtrs[i], coords, outOffset);
        }
        vOutBuffers[i][outOffset] = v;
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}

void mergeAvgBp(sd::LaunchContext* context, NDArray& gradient, std::vector<NDArray*>& outArrs) {
  BUILD_SINGLE_SELECTOR(gradient.dataType(), mergeAvgBp_, (gradient, outArrs), SD_NUMERIC_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeAdd_(const std::vector<NDArray*>& inArrs, NDArray& output) {
  const sd::LongType numArgs = inArrs.size();
  const sd::LongType length = output.lengthOf();
  const int rank = output.rankOf();

  // Logical coordinates through each array's own strides, as mergeAvg_: e<T>(e) and p(e, ...) walk
  // each array in its own order and mispaired elements across 'c' and 'f' arrays.
  auto outputShape = output.shapeInfo();
  std::vector<bool> vbSameShapeAndStrides(numArgs);
  std::vector<sd::LongType*> vStridePtrs(numArgs);
  std::vector<sd::LongType> vRanks(numArgs);
  std::vector<const T*> vBuffers(numArgs);
  for (int i = 0; i < numArgs; ++i) {
    vbSameShapeAndStrides[i] = shape::haveSameShapeAndStrides(outputShape, inArrs[i]->shapeInfo());
    vStridePtrs[i] = shape::stride(inArrs[i]->shapeInfo());
    vRanks[i] = shape::rank(inArrs[i]->shapeInfo());
    vBuffers[i] = inArrs[i]->bufferAsT<T>();
  }

  sd::LongType *outputShapeOf = shape::shapeOf(outputShape);
  sd::LongType *outputStride = shape::stride(outputShape);
  T* outBuffer = output.bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (auto e = start; e < stop; e++) {
      INDEX2COORDS(e, rank, outputShapeOf, coords);
      sd::LongType outOffset;
      COORDS2INDEX(rank, outputStride, coords, outOffset);

      T sum = static_cast<T>(0);
      for (sd::LongType i = 0; i < numArgs; i++) {
        sd::LongType xOffset;
        if (vbSameShapeAndStrides[i]) {
          xOffset = outOffset;
        } else {
          COORDS2INDEX(vRanks[i], vStridePtrs[i], coords, xOffset);
        }
        sum += vBuffers[i][xOffset];
      }
      outBuffer[outOffset] = sum;
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}
void mergeAdd(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (inArrs, output), SD_NUMERIC_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeAddBp_(NDArray& gradient, std::vector<NDArray*>& outArrs) {
  const sd::LongType numArgs = outArrs.size();
  const sd::LongType length = gradient.lengthOf();
  const int gradRank = gradient.rankOf();

  // Logical coordinates through each array's own strides, as mergeAvgBp_.
  auto gradShape = gradient.shapeInfo();
  std::vector<bool> vbSameShapeAndStrides(numArgs);
  std::vector<sd::LongType*> vStridePtrs(numArgs);
  std::vector<sd::LongType> vRanks(numArgs);
  std::vector<T*> vOutBuffers(numArgs);
  for (int i = 0; i < numArgs; ++i) {
    vbSameShapeAndStrides[i] = shape::haveSameShapeAndStrides(gradShape, outArrs[i]->shapeInfo());
    vStridePtrs[i] = shape::stride(outArrs[i]->shapeInfo());
    vRanks[i] = shape::rank(outArrs[i]->shapeInfo());
    vOutBuffers[i] = outArrs[i]->bufferAsT<T>();
  }

  sd::LongType *gradShapeOf = shape::shapeOf(gradShape);
  sd::LongType *gradStride = shape::stride(gradShape);
  const T* gradBuffer = gradient.bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (auto e = start; e < stop; e++) {
      INDEX2COORDS(e, gradRank, gradShapeOf, coords);
      sd::LongType gradOffset;
      COORDS2INDEX(gradRank, gradStride, coords, gradOffset);

      const T v = gradBuffer[gradOffset];
      for (sd::LongType i = 0; i < numArgs; i++) {
        sd::LongType outOffset;
        if (vbSameShapeAndStrides[i]) {
          outOffset = gradOffset;
        } else {
          COORDS2INDEX(vRanks[i], vStridePtrs[i], coords, outOffset);
        }
        vOutBuffers[i][outOffset] = v;
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}

void mergeAddBp(sd::LaunchContext* context, NDArray& gradient, std::vector<NDArray*>& outArrs) {
  BUILD_SINGLE_SELECTOR(gradient.dataType(), mergeAddBp_, (gradient, outArrs), SD_NUMERIC_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif