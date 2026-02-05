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

//
// Split from NativeOps.cu to reduce object file size for SD_GCC_FUNCTRACE builds
// Contains: sort, sortByKey, sortByValue, sortTad, sortTadByKey, sortTadByValue
//

#include <cuda.h>
#include <exceptions/cuda_exception.h>
#include <execution/LaunchContext.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>
#include <legacy/NativeOps.h>
#include <loops/special_kernels.h>
#include <ops/specials_cuda.h>
#include <system/common.h>
#include <array/ArrayOptions.h>
#include <helpers/shape.h>
#include <execution/cuda/LaunchDims.h>

void sort(sd::Pointer *extraPointers, OpaqueNDArray x, bool descending) {
  try {
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      stream = sd::LaunchContext::defaultContext()->getCudaStream();
    }

    const sd::LongType *xShapeInfo = x->shapeInfo();
    const sd::LongType *dXShapeInfo = x->specialShapeInfo();
    auto xLength = shape::length(xShapeInfo);
    auto xEWS = shape::elementWiseStride(xShapeInfo);
    auto xType = sd::ArrayOptions::dataType(xShapeInfo);

    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
      dim3 launchDims = getSortFullDims(xLength);
      for (int k = 2; k <= xLength; k *= 2) {
        for (int j = k >> 1; j > 0; j >>= 1) {
          BUILD_SINGLE_SELECTOR(xType, bitonicSortStepGeneric,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, j, k, xLength, descending),
                                SD_NUMERIC_TYPES);
        }
      }
    } else {
      dim3 launchDims = getSortFullDims(xLength);
      int max = 2, dg = 0;
      while (max < xLength) {
        max <<= 1;
        dg++;
      }
      max <<= 1;

      for (int window = 2; window < max; window <<= 1) {
        int n = window;
        int rev = 0;
        do {
          int half = n >> 1;
          BUILD_SINGLE_SELECTOR(xType, bitonicArbitraryStepGeneric,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, n, xLength, rev, descending),
                                SD_NUMERIC_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }

    sd::DebugHelper::checkErrorCode(stream, "sort(...) failed");
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortByKey(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y, bool descending) {
  try {
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      stream = sd::LaunchContext::defaultContext()->getCudaStream();
    }

    const sd::LongType *xShapeInfo = x->shapeInfo();
    const sd::LongType *dXShapeInfo = x->specialShapeInfo();
    const sd::LongType *yShapeInfo = y->shapeInfo();
    const sd::LongType *dyShapeInfo = y->specialShapeInfo();

    auto xLength = shape::length(xShapeInfo);
    auto yLength = shape::length(yShapeInfo);
    auto xEWS = shape::elementWiseStride(xShapeInfo);
    auto xType = sd::ArrayOptions::dataType(xShapeInfo);
    auto yType = sd::ArrayOptions::dataType(yShapeInfo);

    if (shape::isEmptyConst(xShapeInfo) || shape::isEmptyConst(yShapeInfo)) return;
    if (xLength != yLength) THROW_EXCEPTION("sortByKey: keys and values must have the same size");

    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
      dim3 launchDims = getSortFullDims(xLength);
      for (int k = 2; k <= xLength; k *= 2) {
        for (int j = k >> 1; j > 0; j >>= 1) {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicSortStepGenericKey,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, y->specialBuffer(), dyShapeInfo, j, k, xLength, descending),
                                SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
        }
      }
    } else {
      int numThreads = sd::math::sd_min<int>(512, xLength);
      int numBlocks = xLength / numThreads;
      if (xLength % numThreads > 0 || numBlocks == 0) numBlocks++;
      numBlocks = sd::math::sd_min<int>(512, numBlocks);
      dim3 launchDims(numBlocks, numThreads, 32768);

      int max = 2;
      while (max < xLength) {
        max <<= 1;
      }
      max <<= 1;

      for (int window = 2; window < max; window <<= 1) {
        int n = window;
        int rev = 0;
        do {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicArbitraryStepGenericKey,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, y->specialBuffer(), dyShapeInfo, n, xLength, rev, descending),
                                SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }

    sd::DebugHelper::checkErrorCode(stream, "sortByKey(...) failed");
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortByValue(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y, bool descending) {
  try {
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      stream = sd::LaunchContext::defaultContext()->getCudaStream();
    }

    const sd::LongType *xShapeInfo = x->shapeInfo();
    const sd::LongType *dXShapeInfo = x->specialShapeInfo();
    const sd::LongType *yShapeInfo = y->shapeInfo();
    const sd::LongType *dyShapeInfo = y->specialShapeInfo();

    auto xLength = shape::length(xShapeInfo);
    auto yLength = shape::length(yShapeInfo);
    auto xEWS = shape::elementWiseStride(xShapeInfo);
    auto xType = sd::ArrayOptions::dataType(yShapeInfo);
    auto yType = sd::ArrayOptions::dataType(xShapeInfo);

    if (shape::isEmptyConst(xShapeInfo) || shape::isEmptyConst(yShapeInfo)) return;
    if (xLength != yLength) THROW_EXCEPTION("sortByValue: keys and values must have the same size");

    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
      dim3 launchDims = getSortFullDims(xLength);
      for (int k = 2; k <= xLength; k *= 2) {
        for (int j = k >> 1; j > 0; j >>= 1) {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicSortStepGenericValue,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, y->specialBuffer(), dyShapeInfo, j, k, xLength, descending),
                                SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
        }
      }
    } else {
      int numThreads = sd::math::sd_min<int>(512, xLength);
      int numBlocks = xLength / numThreads;
      if (xLength % numThreads > 0 || numBlocks == 0) numBlocks++;
      numBlocks = sd::math::sd_min<int>(512, numBlocks);
      dim3 launchDims(numBlocks, numThreads, 32768);

      int max = 2;
      while (max < xLength) {
        max <<= 1;
      }
      max <<= 1;

      for (int window = 2; window < max; window <<= 1) {
        int n = window;
        int rev = 0;
        do {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicArbitraryStepGenericValue,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, y->specialBuffer(), dyShapeInfo, n, xLength, rev, descending),
                                SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }

    sd::DebugHelper::checkErrorCode(stream, "sortByValue(...) failed");
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTadByKey(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray dimension, bool descending) {
  try {
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      stream = sd::LaunchContext::defaultContext()->getCudaStream();
    }

    sd::LongType *xShapeInfo = x->shapeInfo();
    const sd::LongType *dXShapeInfo = x->specialShapeInfo();
    const sd::LongType *yShapeInfo = y->shapeInfo();
    const sd::LongType *dyShapeInfo = y->specialShapeInfo();

    auto xType = sd::ArrayOptions::dataType(xShapeInfo);
    auto yType = sd::ArrayOptions::dataType(yShapeInfo);

    auto dimensionPtr = reinterpret_cast<sd::LongType *>(dimension->buffer());
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimensionPtr, dimensionLength);
    auto numTads = tadPack->numberOfTads();

    dim3 launchDims = getSortTadDims(numTads);
    BUILD_DOUBLE_SELECTOR(xType, yType, oesTadGenericKey,
                          (launchDims, stream, x->specialBuffer(), dXShapeInfo, y->specialBuffer(), dyShapeInfo,
                           dimensionPtr, dimensionLength, tadPack->platformShapeInfo(), tadPack->platformOffsets(), descending),
                          SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);

    sd::DebugHelper::checkErrorCode(stream, "sortTadByKey(...) failed");
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTadByValue(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray dimension, bool descending) {
  try {
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      stream = sd::LaunchContext::defaultContext()->getCudaStream();
    }

    sd::LongType *xShapeInfo = x->shapeInfo();
    const sd::LongType *dXShapeInfo = x->specialShapeInfo();
    const sd::LongType *yShapeInfo = y->shapeInfo();
    const sd::LongType *dyShapeInfo = y->specialShapeInfo();

    auto xType = sd::ArrayOptions::dataType(yShapeInfo);
    auto yType = sd::ArrayOptions::dataType(xShapeInfo);

    auto dimensionPtr = reinterpret_cast<sd::LongType *>(dimension->buffer());
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimensionPtr, dimension->lengthOf());
    auto numTads = tadPack->numberOfTads();

    dim3 launchDims = getSortTadDims(numTads);
    BUILD_DOUBLE_SELECTOR(xType, yType, oesTadGenericKey,
                          (launchDims, stream, y->specialBuffer(), dyShapeInfo, x->specialBuffer(), dXShapeInfo,
                           dimensionPtr, dimensionLength, tadPack->platformShapeInfo(), tadPack->platformOffsets(), descending),
                          SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);

    sd::DebugHelper::checkErrorCode(stream, "sortTadByValue(...) failed");
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTad(sd::Pointer *extraPointers, OpaqueNDArray x,
             sd::LongType *dimension, sd::LongType dimensionLength,
             sd::LongType *tadShapeInfo, sd::LongType *tadOffsets, bool descending) {
  try {
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      stream = sd::LaunchContext::defaultContext()->getCudaStream();
    }

    sd::LongType *xShapeInfo = x->shapeInfo();
    sd::LongType *dXShapeInfo = x->specialShapeInfo();
    auto xType = sd::ArrayOptions::dataType(xShapeInfo);

    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);
    auto numTads = tadPack->numberOfTads();

    dim3 launchDims = getSortTadLarge(numTads);
    BUILD_SINGLE_SELECTOR(
        xType, oesTadGeneric,
        (launchDims, stream, x->specialBuffer(), dXShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending),
        SD_NUMERIC_TYPES);

    sd::DebugHelper::checkErrorCode(stream, "sortTad(...) failed");
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
