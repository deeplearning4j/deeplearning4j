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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#include <array/NDArrayFactory.h>
#include <array/ResultSet.h>
#include <exceptions/cuda_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/helpers/transforms.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {
//////////////////////////////////////////////////////////////////////////
template <typename T, typename Z>
static SD_KERNEL void mergeMaxIndexCudaLauncher(void** inArrs, void** inShapes, const int numArrays, void* voutput,
                                                const LongType* outputShape, LongType length) {
  auto output = reinterpret_cast<Z*>(voutput);

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  __shared__ int rankOutput;
  __shared__ const LongType *shapeOutput, *strideOutput;

  if (threadIdx.x == 0) {
    rankOutput = shape::rank(outputShape);
    shapeOutput = shape::shapeOf(outputShape);
    strideOutput = shape::stride(outputShape);
  }
  __syncthreads();

  LongType outputCoords[SD_MAX_RANK];

  for (LongType e = tid; e < length; e += step) {
    T mVal = -DataTypeUtils::max<T>();
    Z mIdx(0);

    // Iterate through all input arrays to find the maximum value and its index
    for (int i = 0; i < numArrays; ++i) {
      auto x = reinterpret_cast<const T*>(inArrs[i]);
      auto xShape = reinterpret_cast<const LongType*>(inShapes[i]);

      __shared__ int rankInput;
      __shared__ const LongType *shapeInput, *strideInput;

      if (threadIdx.x == 0) {
        rankInput = shape::rank(xShape);
        shapeInput = shape::shapeOf(xShape);
        strideInput = shape::stride(xShape);
      }
      __syncthreads();

      LongType xCoords[SD_MAX_RANK];
      LongType xOffset;

      // Compute input coordinates and offset
      INDEX2COORDS(e, rankInput, shapeInput, xCoords);
      COORDS2INDEX(rankInput, strideInput, xCoords, xOffset);

      // Update maximum value and index
      const auto val = x[xOffset];
      if (mVal < val) {
        mIdx = static_cast<Z>(i);
        mVal = val;
      }
    }

    // Compute output coordinates and offset
    LongType outputOffset;
    INDEX2COORDS(e, rankOutput, shapeOutput, outputCoords);
    COORDS2INDEX(rankOutput, strideOutput, outputCoords, outputOffset);

    // Store the index of the maximum value in the output
    output[outputOffset] = mIdx;
  }
}


template <typename T, typename Z>
static void mergeMaxIndex_(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  int nArrSize = static_cast<int>(inArrs.size());
  std::vector<const void*> inBuffers(nArrSize), inShapes(nArrSize);

  for (int e = 0; e < nArrSize; e++) {
    inBuffers[e] = inArrs[e]->specialBuffer();
    inShapes[e] = inArrs[e]->specialShapeInfo();
  }

  PointersManager manager(context, "mergeMaxIndex");

  auto pInBuffers =
      reinterpret_cast<void**>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void*)));
  auto pInShapes = reinterpret_cast<void**>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void*)));
  auto length = output.lengthOf();

  dim3 mergeLaunchDims = mergeDims(length);
  mergeMaxIndexCudaLauncher<T, Z><<<mergeLaunchDims.y, mergeLaunchDims.x, mergeLaunchDims.z, *context->getCudaStream()>>>(
      pInBuffers, pInShapes, nArrSize, output.specialBuffer(), output.specialShapeInfo(), length);
  sd::DebugHelper::checkErrorCode(context->getCudaStream(), "mergeMaxIndexCudaLauncher failed");

  manager.synchronize();
}

void mergeMaxIndex(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  NDArray::prepareSpecialUse({&output}, inArrs);

  BUILD_DOUBLE_SELECTOR(inArrs[0]->dataType(), output.dataType(), mergeMaxIndex_, (context, inArrs, output),
                        SD_COMMON_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&output}, inArrs);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static SD_KERNEL void mergeMaxCudaLauncher(void** inArrs, void** inShapes, const int numArrays, void* voutput,
                                           const LongType* outputShape, LongType length) {
  auto output = reinterpret_cast<T*>(voutput);

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  __shared__ int rankOutput;
  __shared__ const LongType *shapeOutput, *strideOutput;

  if (threadIdx.x == 0) {
    rankOutput = shape::rank(outputShape);
    shapeOutput = shape::shapeOf(outputShape);
    strideOutput = shape::stride(outputShape);
  }
  __syncthreads();

  LongType outputCoords[SD_MAX_RANK];

  for (LongType e = tid; e < length; e += step) {
    T mVal = -DataTypeUtils::max<T>();

    // Iterate through all input arrays to find the maximum value
    for (int i = 0; i < numArrays; ++i) {
      auto x = reinterpret_cast<const T*>(inArrs[i]);
      auto xShape = reinterpret_cast<const LongType*>(inShapes[i]);

      __shared__ int rankInput;
      __shared__ const LongType *shapeInput, *strideInput;

      if (threadIdx.x == 0) {
        rankInput = shape::rank(xShape);
        shapeInput = shape::shapeOf(xShape);
        strideInput = shape::stride(xShape);
      }
      __syncthreads();

      LongType xCoords[SD_MAX_RANK];
      LongType xOffset;

      // Compute input coordinates and offset
      INDEX2COORDS(e, rankInput, shapeInput, xCoords);
      COORDS2INDEX(rankInput, strideInput, xCoords, xOffset);

      // Update maximum value
      const auto val = x[xOffset];
      if (mVal < val) {
        mVal = val;
      }
    }

    // Compute output coordinates and offset
    LongType outputOffset;
    INDEX2COORDS(e, rankOutput, shapeOutput, outputCoords);
    COORDS2INDEX(rankOutput, strideOutput, outputCoords, outputOffset);

    // Store the maximum value in the output
    output[outputOffset] = mVal;
  }
}

template <typename T>
static void mergeMax_(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  int nArrsSize = static_cast<int>(inArrs.size());

  std::vector<const void*> inBuffers(nArrsSize), inShapes(nArrsSize);

  for (int e = 0; e < nArrsSize; e++) {
    inBuffers[e] = inArrs[e]->specialBuffer();
    inShapes[e] = inArrs[e]->specialShapeInfo();
  }

  PointersManager manager(context, "mergeMax");

  auto pInBuffers =
      reinterpret_cast<void**>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void*)));
  auto pInShapes = reinterpret_cast<void**>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void*)));
  auto length = output.lengthOf();

  dim3 mergeLaunchDims = mergeDims(length);
  mergeMaxCudaLauncher<T><<<mergeLaunchDims.y, mergeLaunchDims.x, mergeLaunchDims.z, *context->getCudaStream()>>>(
      pInBuffers, pInShapes, nArrsSize, output.specialBuffer(), output.specialShapeInfo(), length);
  sd::DebugHelper::checkErrorCode(context->getCudaStream(), "mergeMaxCudaLauncher failed");
  manager.synchronize();
}

void mergeMax(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  NDArray::prepareSpecialUse({&output}, inArrs);

  BUILD_SINGLE_SELECTOR(output.dataType(), mergeMax_, (context, inArrs, output), SD_COMMON_TYPES);

  NDArray::registerSpecialUse({&output}, inArrs);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static SD_KERNEL void mergeMaxBpCudaLauncher(void** inArrs, void** inShapes, const void* vgradient,
                                             const LongType* gradientShape, const int numArrays, void** outArrs,
                                             void** outShapes, LongType length, bool bSameOrderAndEws1) {
  const auto grad = reinterpret_cast<const T*>(vgradient);

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  __shared__ int gradRank;
  __shared__ const LongType *gradShape, *gradStride;

  if (threadIdx.x == 0) {
    gradRank = shape::rank(gradientShape);
    gradShape = shape::shapeOf(gradientShape);
    gradStride = shape::stride(gradientShape);
  }
  __syncthreads();

  LongType coords[SD_MAX_RANK];

  for (LongType e = tid; e < length; e += step) {
    T mVal = -DataTypeUtils::max<T>();
    int nMaxIndex = 0;
    LongType gradOffset = bSameOrderAndEws1 ? e : 0;

    // Compute gradient offset if not same order and EWS=1
    if (!bSameOrderAndEws1) {
      INDEX2COORDS(e, gradRank, gradShape, coords);
      COORDS2INDEX(gradRank, gradStride, coords, gradOffset);
    }

    // Find the maximum value and its index across all input arrays
    for (int i = 0; i < numArrays; ++i) {
      auto x = reinterpret_cast<T*>(inArrs[i]);
      LongType xOffset = bSameOrderAndEws1 ? e : 0;

      if (!bSameOrderAndEws1) {
        auto xShape = reinterpret_cast<const LongType*>(inShapes[i]);
        COORDS2INDEX(shape::rank(xShape), shape::stride(xShape), coords, xOffset);
      }

      const auto val = x[xOffset];
      if (mVal < val) {
        mVal = val;
        nMaxIndex = i;
      }
    }

    // Assign gradient to the corresponding output array at the max index
    auto output = reinterpret_cast<T*>(outArrs[nMaxIndex]);
    LongType zOffset = bSameOrderAndEws1 ? e : 0;

    if (!bSameOrderAndEws1) {
      auto outShape = reinterpret_cast<const LongType*>(outShapes[nMaxIndex]);
      COORDS2INDEX(shape::rank(outShape), shape::stride(outShape), coords, zOffset);
    }

    output[zOffset] = grad[gradOffset];
  }
}


template <typename T>
static void mergeMaxBp_(LaunchContext* context, const std::vector<NDArray*>& inArrs,
                        std::vector<NDArray*>& outArrs, int nArrSize, bool bSameOrderAndEws1) {
  std::vector<const void*> inBuffers(nArrSize), inShapes(nArrSize), outBuffers(nArrSize), outShapes(nArrSize);

  for (int e = 0; e < nArrSize; e++) {
    inBuffers[e] = inArrs[e]->specialBuffer();
    inShapes[e] = inArrs[e]->specialShapeInfo();
    outBuffers[e] = outArrs[e]->specialBuffer();
    outShapes[e] = outArrs[e]->specialShapeInfo();
  }

  PointersManager manager(context, "mergeMaxBp");

  auto pInBuffers =
      reinterpret_cast<void**>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void*)));
  auto pInShapes = reinterpret_cast<void**>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void*)));

  auto pOutBuffers =
      reinterpret_cast<void**>(manager.replicatePointer(outBuffers.data(), outBuffers.size() * sizeof(void*)));
  auto pOutShapes =
      reinterpret_cast<void**>(manager.replicatePointer(outShapes.data(), outShapes.size() * sizeof(void*)));

  auto length = inArrs[nArrSize]->lengthOf();

  dim3 mergeLaunchDims = mergeDims(length);

  mergeMaxBpCudaLauncher<T><<<mergeLaunchDims.y, mergeLaunchDims.x, mergeLaunchDims.z, *context->getCudaStream()>>>(
      pInBuffers, pInShapes, inArrs[nArrSize]->specialBuffer(), inArrs[nArrSize]->specialShapeInfo(), nArrSize,
      pOutBuffers, pOutShapes, length, bSameOrderAndEws1);
  sd::DebugHelper::checkErrorCode(context->getCudaStream(), "mergeMaxBpCudaLauncher failed");

  manager.synchronize();
}

void mergeMaxBp(LaunchContext* context, const std::vector<NDArray*>& inArrs, std::vector<NDArray*>& outArrs) {
  // not use gradient
  int nArrSize = static_cast<int>(inArrs.size() - 1);

  const std::vector<NDArray*>& out = reinterpret_cast<const std::vector<NDArray*>&>(outArrs);

  NDArray::prepareSpecialUse(out, inArrs);

  bool bSameOrderAndEws1 = false;
  auto ordering = inArrs[nArrSize]->ordering();


  BUILD_SINGLE_SELECTOR(inArrs[nArrSize]->dataType(), mergeMaxBp_,
                        (context, inArrs, outArrs, nArrSize, bSameOrderAndEws1), SD_COMMON_TYPES);

  NDArray::registerSpecialUse(out, inArrs);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static SD_KERNEL void mergeAvgCudaLauncher(void** inArrs, void** inShapes, const int numArrays, void* voutput,
                                           const LongType* outputShape, LongType length) {
  auto output = reinterpret_cast<T*>(voutput);

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  __shared__ int rankOutput;
  __shared__ const LongType *shapeOutput, *strideOutput;

  if (threadIdx.x == 0) {
    rankOutput = shape::rank(outputShape);
    shapeOutput = shape::shapeOf(outputShape);
    strideOutput = shape::stride(outputShape);
  }
  __syncthreads();

  LongType outputCoords[SD_MAX_RANK];

  for (LongType e = tid; e < length; e += step) {
    T sum = static_cast<T>(0.0);

    // Sum values from all input arrays
    for (int i = 0; i < numArrays; ++i) {
      auto x = reinterpret_cast<T*>(inArrs[i]);
      auto xShape = reinterpret_cast<const LongType*>(inShapes[i]);

      __shared__ int rankInput;
      __shared__ const LongType *shapeInput, *strideInput;

      if (threadIdx.x == 0) {
        rankInput = shape::rank(xShape);
        shapeInput = shape::shapeOf(xShape);
        strideInput = shape::stride(xShape);
      }
      __syncthreads();

      LongType xCoords[SD_MAX_RANK];
      LongType xOffset;

      // Compute input coordinates and offset
      INDEX2COORDS(e, rankInput, shapeInput, xCoords);
      COORDS2INDEX(rankInput, strideInput, xCoords, xOffset);

      sum += x[xOffset];
    }

    // Compute output coordinates and offset
    LongType outputOffset;
    INDEX2COORDS(e, rankOutput, shapeOutput, outputCoords);
    COORDS2INDEX(rankOutput, strideOutput, outputCoords, outputOffset);

    // Store the averaged value in the output
    output[outputOffset] = sum / static_cast<T>(numArrays);
  }
}

template <typename T>
static void mergeAvg_(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  std::vector<const void*> inBuffers(inArrs.size()), inShapes(inArrs.size());

  for (int e = 0; e < inArrs.size(); e++) {
    inBuffers[e] = inArrs[e]->specialBuffer();
    inShapes[e] = inArrs[e]->specialShapeInfo();
  }

  PointersManager manager(context, "mergeAvg");

  auto pInBuffers =
      reinterpret_cast<void**>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void*)));
  auto pInShapes = reinterpret_cast<void**>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void*)));
  auto length = output.lengthOf();

  dim3 mergeLaunchDims = mergeDims(length);

  mergeAvgCudaLauncher<T><<<mergeLaunchDims.y, mergeLaunchDims.x, mergeLaunchDims.z, *context->getCudaStream()>>>(
      pInBuffers, pInShapes, (int)inArrs.size(), output.specialBuffer(), output.specialShapeInfo(), length);
  sd::DebugHelper::checkErrorCode(context->getCudaStream(), "mergeAvgCudaLauncher failed");

  manager.synchronize();
}

void mergeAvg(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  NDArray::prepareSpecialUse({&output}, inArrs);

  BUILD_SINGLE_SELECTOR(output.dataType(), mergeAvg_, (context, inArrs, output), SD_FLOAT_TYPES);

  NDArray::registerSpecialUse({&output}, inArrs);
}
//////////////////////////////////////////////////////////////////////////
template <typename T>
static SD_KERNEL void mergeAvgBpCudaLauncher(const void* vgradient, const LongType* gradientShape, void** outArrs,
                                             void** outShapes, const int numArrays, LongType length,
                                             bool bSameOrderAndEws1) {
  const auto grad = reinterpret_cast<const T*>(vgradient);

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  __shared__ int gradRank;
  __shared__ const LongType *gradShape, *gradStride;

  if (threadIdx.x == 0) {
    gradRank = shape::rank(gradientShape);
    gradShape = shape::shapeOf(gradientShape);
    gradStride = shape::stride(gradientShape);
  }
  __syncthreads();

  LongType coords[SD_MAX_RANK];

  for (LongType e = tid; e < length; e += step) {
    LongType gradOffset = bSameOrderAndEws1 ? e : 0;

    // Compute gradient offset if not using the same order and EWS=1
    if (!bSameOrderAndEws1) {
      INDEX2COORDS(e, gradRank, gradShape, coords);
      COORDS2INDEX(gradRank, gradStride, coords, gradOffset);
    }

    // Iterate through each output array and compute the average gradient
    for (int i = 0; i < numArrays; ++i) {
      auto output = reinterpret_cast<T*>(outArrs[i]);
      LongType zOffset = bSameOrderAndEws1 ? e : 0;

      if (!bSameOrderAndEws1) {
        auto outShape = reinterpret_cast<const LongType*>(outShapes[i]);
        COORDS2INDEX(shape::rank(outShape), shape::stride(outShape), coords, zOffset);
      }

      // Assign averaged gradient value to output
      output[zOffset] = grad[gradOffset] / static_cast<T>(numArrays);
    }
  }
}


template <typename T>
static void mergeAvgBp_(LaunchContext* context, NDArray& gradient, std::vector<NDArray*>& outArrs,
                        bool bSameOrderAndEws1) {
  int nArrSize = static_cast<int>(outArrs.size());

  std::vector<const void*> outBuffers(nArrSize), outShapes(nArrSize);

  for (int e = 0; e < nArrSize; e++) {
    outBuffers[e] = outArrs[e]->specialBuffer();
    outShapes[e] = outArrs[e]->specialShapeInfo();
  }

  PointersManager manager(context, "mergeAvgBp");

  auto pOutBuffers =
      reinterpret_cast<void**>(manager.replicatePointer(outBuffers.data(), outBuffers.size() * sizeof(void*)));
  auto pOutShapes =
      reinterpret_cast<void**>(manager.replicatePointer(outShapes.data(), outShapes.size() * sizeof(void*)));

  auto length = gradient.lengthOf();

  dim3 mergeLaunchDims = mergeDims(length);

  mergeAvgBpCudaLauncher<T><<<mergeLaunchDims.y, mergeLaunchDims.x,mergeLaunchDims.z, *context->getCudaStream()>>>(
      gradient.specialBuffer(), gradient.specialShapeInfo(), pOutBuffers, pOutShapes, nArrSize, length,
      bSameOrderAndEws1);
  sd::DebugHelper::checkErrorCode(context->getCudaStream(), "mergeAvgBpCudaLauncher failed");

  manager.synchronize();
}

void mergeAvgBp(LaunchContext* context, NDArray& gradient, std::vector<NDArray*>& outArrs) {
  const std::vector<NDArray*>& out = reinterpret_cast<const std::vector<NDArray*>&>(outArrs);

  NDArray::prepareSpecialUse(out, {&gradient});

  bool bSameOrderAndEws1 = false;
  auto ordering = gradient.ordering();

  for (const auto& v : outArrs) {
    bSameOrderAndEws1 &= (ordering == v->ordering());
  }

  BUILD_SINGLE_SELECTOR(gradient.dataType(), mergeAvgBp_, (context, gradient, outArrs, bSameOrderAndEws1),
                        SD_COMMON_TYPES);

  NDArray::prepareSpecialUse(out, {&gradient});
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static SD_KERNEL void mergeAddCudaLauncher(void** inArrs, void** inShapes, const int numArrays, void* voutput,
                                           const LongType* outputShape, LongType length) {
  auto output = reinterpret_cast<T*>(voutput);

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  __shared__ int rankOutput;
  __shared__ const LongType *shapeOutput, *strideOutput;

  if (threadIdx.x == 0) {
    rankOutput = shape::rank(outputShape);
    shapeOutput = shape::shapeOf(outputShape);
    strideOutput = shape::stride(outputShape);
  }
  __syncthreads();

  LongType outputCoords[SD_MAX_RANK];

  for (LongType e = tid; e < length; e += step) {
    T sum(0.0f);

    // Compute the sum across all input arrays
    for (int i = 0; i < numArrays; ++i) {
      auto x = reinterpret_cast<T*>(inArrs[i]);
      auto xShape = reinterpret_cast<const LongType*>(inShapes[i]);

      __shared__ int rankInput;
      __shared__ const LongType *shapeInput, *strideInput;

      if (threadIdx.x == 0) {
        rankInput = shape::rank(xShape);
        shapeInput = shape::shapeOf(xShape);
        strideInput = shape::stride(xShape);
      }
      __syncthreads();

      LongType xCoords[SD_MAX_RANK];
      LongType xOffset;

      // Compute input coordinates and offset
      INDEX2COORDS(e, rankInput, shapeInput, xCoords);
      COORDS2INDEX(rankInput, strideInput, xCoords, xOffset);

      sum += x[xOffset];
    }

    // Compute output coordinates and offset
    LongType outputOffset;
    INDEX2COORDS(e, rankOutput, shapeOutput, outputCoords);
    COORDS2INDEX(rankOutput, strideOutput, outputCoords, outputOffset);

    // Store the computed sum in the output
    output[outputOffset] = sum;
  }
}

template <typename T>
static void mergeAdd_(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  int nArrSize = static_cast<int>(inArrs.size());
  std::vector<const void*> inBuffers(nArrSize), inShapes(nArrSize);

  for (int e = 0; e < nArrSize; e++) {
    inBuffers[e] = inArrs[e]->specialBuffer();
    inShapes[e] = inArrs[e]->specialShapeInfo();
  }

  PointersManager manager(context, "mergeAdd");

  auto pInBuffers =
      reinterpret_cast<void**>(manager.replicatePointer(inBuffers.data(), inBuffers.size() * sizeof(void*)));
  auto pInShapes = reinterpret_cast<void**>(manager.replicatePointer(inShapes.data(), inShapes.size() * sizeof(void*)));
  auto length = output.lengthOf();

  dim3 mergeLaunchDims = mergeDims(length);

  mergeAddCudaLauncher<T><<<mergeLaunchDims.x, mergeLaunchDims.y, mergeLaunchDims.z, *context->getCudaStream()>>>(
      pInBuffers, pInShapes, nArrSize, output.specialBuffer(), output.specialShapeInfo(), length);
  sd::DebugHelper::checkErrorCode(context->getCudaStream(), "mergeAddCudaLauncher failed");

  manager.synchronize();
}
BUILD_SINGLE_TEMPLATE(template void mergeAdd_,
                      (sd::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output),
                      SD_NUMERIC_TYPES);

void mergeAdd(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output) {
  NDArray::prepareSpecialUse({&output}, inArrs);

  BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (context, inArrs, output), SD_NUMERIC_TYPES);

  NDArray::registerSpecialUse({&output}, inArrs);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static SD_KERNEL void mergeAddBpCudaLauncher(const void* vgradient, const LongType* gradientShape, void** outArrs,
                                             void** outShapes, const int numArrays, LongType length,
                                             bool bSameOrderAndEws1) {
  const auto grad = reinterpret_cast<const T*>(vgradient);
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  __shared__ int gradRank;
  __shared__ const LongType *gradShape, *gradStride;

  if (threadIdx.x == 0) {
    gradRank = shape::rank(gradientShape);
    gradShape = shape::shapeOf(gradientShape);
    gradStride = shape::stride(gradientShape);
  }
  __syncthreads();

  LongType coords[SD_MAX_RANK];

  for (LongType e = tid; e < length; e += step) {
    LongType gradOffset = bSameOrderAndEws1 ? e : 0;

    // Compute gradient offset if not using same order and EWS=1
    if (!bSameOrderAndEws1) {
      INDEX2COORDS(e, gradRank, gradShape, coords);
      COORDS2INDEX(gradRank, gradStride, coords, gradOffset);
    }

    for (int i = 0; i < numArrays; ++i) {
      auto output = reinterpret_cast<T*>(outArrs[i]);
      LongType zOffset = bSameOrderAndEws1 ? e : 0;

      // Compute output offset if not using same order and EWS=1
      if (!bSameOrderAndEws1) {
        auto outShape = reinterpret_cast<const LongType*>(outShapes[i]);
        COORDS2INDEX(shape::rank(outShape), shape::stride(outShape), coords, zOffset);
      }

      // Assign gradient value to output
      output[zOffset] = grad[gradOffset];
    }
  }
}


template <typename T>
static void mergeAddBp_(LaunchContext* context, NDArray& gradient, std::vector<NDArray*>& outArrs,
                        bool bSameOrderAndEws1) {
  int nArrSize = static_cast<int>(outArrs.size());

  std::vector<const void*> outBuffers(nArrSize), outShapes(nArrSize);

  for (int e = 0; e < nArrSize; e++) {
    outBuffers[e] = outArrs[e]->specialBuffer();
    outShapes[e] = outArrs[e]->specialShapeInfo();
  }

  PointersManager manager(context, "mergeAddBp");

  auto pOutBuffers =
      reinterpret_cast<void**>(manager.replicatePointer(outBuffers.data(), outBuffers.size() * sizeof(void*)));
  auto pOutShapes =
      reinterpret_cast<void**>(manager.replicatePointer(outShapes.data(), outShapes.size() * sizeof(void*)));

  auto length = gradient.lengthOf();

  const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
  const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

  mergeAddBpCudaLauncher<T><<<blocksPerGrid, threadsPerBlock, 512, *context->getCudaStream()>>>(
      gradient.specialBuffer(), gradient.specialShapeInfo(), pOutBuffers, pOutShapes, nArrSize, length,
      bSameOrderAndEws1);
  sd::DebugHelper::checkErrorCode(context->getCudaStream(), "mergeAddBpCudaLauncher failed");

  manager.synchronize();
}

void mergeAddBp(LaunchContext* context, NDArray& gradient, std::vector<NDArray*>& outArrs) {
  const std::vector<NDArray*>& out = reinterpret_cast<const std::vector<NDArray*>&>(outArrs);
  NDArray::prepareSpecialUse(out, {&gradient});

  bool bSameOrderAndEws1 = false;
  auto ordering = gradient.ordering();

  for (const auto& v : outArrs) {
    bSameOrderAndEws1 &= (ordering == v->ordering());
  }

  BUILD_SINGLE_SELECTOR(gradient.dataType(), mergeAddBp_, (context, gradient, outArrs, bSameOrderAndEws1),
                        SD_COMMON_TYPES);

  NDArray::prepareSpecialUse(out, {&gradient});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
