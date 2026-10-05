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
#include <array/NDArray.h>

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

#include <string>

namespace {

// The step launchers (ops/specials_cuda.h) take the window, the block size k and the length as int: the windows of an
// array longer than 2^30 elements reach 2^31, which none of them can hold.
constexpr sd::LongType SORT_MAX_LENGTH = static_cast<sd::LongType>(1) << 30;

// Power-of-two arrays up to this length run the plain bitonic network, any other length the arbitrary-length one.
constexpr sd::LongType SORT_BITONIC_MAX_LENGTH = static_cast<sd::LongType>(1024) * 1024 * 10;

// the stream a sort runs on: the caller's, else the default context's
cudaStream_t *sortStream(sd::Pointer *extraPointers) {
  if (extraPointers != nullptr && extraPointers[1] != nullptr) {
    return reinterpret_cast<cudaStream_t *>(extraPointers[1]);
  }
  return sd::LaunchContext::defaultContext()->getCudaStream();
}

void requireSortableLength(sd::LongType length, const char *operation) {
  if (length > SORT_MAX_LENGTH) {
    std::string message = std::string(operation) + ": " + std::to_string(length) +
                          " elements exceed the longest array the CUDA sort handles (" +
                          std::to_string(SORT_MAX_LENGTH) + ")";
    THROW_EXCEPTION(message.c_str());
  }
}

bool isPowerOfTwo(sd::LongType length) { return length > 0 && (length & (length - 1)) == 0; }

// the next power of two at or above length (length > 1), doubled: the bound of the window loop of the arbitrary-length
// network, whose last window is the next power of two itself
sd::LongType windowBound(sd::LongType length) {
  sd::LongType bound = 2;
  while (bound < length) bound <<= 1;
  return bound << 1;
}

}  // namespace

void sort(sd::Pointer *extraPointers, OpaqueNDArray x, bool descending) {
  try {
    cudaStream_t *stream = sortStream(extraPointers);

    const sd::LongType *xShapeInfo = x->shapeInfo();
    auto xLength = shape::length(xShapeInfo);
    auto xType = sd::ArrayOptions::dataType(xShapeInfo);

    // nothing to order: no step kernel could be launched for it (and getSortFullDims divides by the length)
    if (xLength <= 1) return;
    requireSortableLength(xLength, "sort");

    sd::NDArray::prepareSpecialUse({x}, {x});
    const sd::LongType *dXShapeInfo = x->specialShapeInfo();
    dim3 launchDims = getSortFullDims(static_cast<int>(xLength));

    if (isPowerOfTwo(xLength) && xLength <= SORT_BITONIC_MAX_LENGTH) {
      for (sd::LongType k = 2; k <= xLength; k <<= 1) {
        for (sd::LongType j = k >> 1; j > 0; j >>= 1) {
          BUILD_SINGLE_SELECTOR(xType, bitonicSortStepGeneric,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, static_cast<int>(j),
                                 static_cast<int>(k), static_cast<int>(xLength), descending),
                                SD_NUMERIC_TYPES);
        }
      }
    } else {
      const sd::LongType max = windowBound(xLength);

      for (sd::LongType window = 2; window < max; window <<= 1) {
        sd::LongType n = window;
        int rev = 0;
        do {
          BUILD_SINGLE_SELECTOR(xType, bitonicArbitraryStepGeneric,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, static_cast<int>(n),
                                 static_cast<int>(xLength), rev, descending),
                                SD_NUMERIC_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }

    sd::DebugHelper::checkErrorCode(stream, "sort(...) failed");
    sd::NDArray::registerSpecialUse({x}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortByKey(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y, bool descending) {
  try {
    cudaStream_t *stream = sortStream(extraPointers);

    const sd::LongType *xShapeInfo = x->shapeInfo();
    const sd::LongType *yShapeInfo = y->shapeInfo();

    auto xLength = shape::length(xShapeInfo);
    auto yLength = shape::length(yShapeInfo);
    auto xType = sd::ArrayOptions::dataType(xShapeInfo);
    auto yType = sd::ArrayOptions::dataType(yShapeInfo);

    if (shape::isEmptyConst(xShapeInfo) || shape::isEmptyConst(yShapeInfo)) return;
    if (xLength != yLength) THROW_EXCEPTION("sortByKey: keys and values must have the same size");
    if (xLength <= 1) return;
    requireSortableLength(xLength, "sortByKey");

    sd::NDArray::prepareSpecialUse({x, y}, {x, y});
    const sd::LongType *dXShapeInfo = x->specialShapeInfo();
    const sd::LongType *dyShapeInfo = y->specialShapeInfo();
    dim3 launchDims = getSortFullDims(static_cast<int>(xLength));

    if (isPowerOfTwo(xLength) && xLength <= SORT_BITONIC_MAX_LENGTH) {
      for (sd::LongType k = 2; k <= xLength; k <<= 1) {
        for (sd::LongType j = k >> 1; j > 0; j >>= 1) {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicSortStepGenericKey,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, y->specialBuffer(), dyShapeInfo,
                                 static_cast<int>(j), static_cast<int>(k), static_cast<int>(xLength), descending),
                                SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
        }
      }
    } else {
      const sd::LongType max = windowBound(xLength);

      for (sd::LongType window = 2; window < max; window <<= 1) {
        sd::LongType n = window;
        int rev = 0;
        do {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicArbitraryStepGenericKey,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, y->specialBuffer(), dyShapeInfo,
                                 static_cast<int>(n), static_cast<int>(xLength), rev, descending),
                                SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }

    sd::DebugHelper::checkErrorCode(stream, "sortByKey(...) failed");
    sd::NDArray::registerSpecialUse({x, y}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortByValue(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y, bool descending) {
  try {
    cudaStream_t *stream = sortStream(extraPointers);

    const sd::LongType *xShapeInfo = x->shapeInfo();
    const sd::LongType *yShapeInfo = y->shapeInfo();

    auto xLength = shape::length(xShapeInfo);
    auto yLength = shape::length(yShapeInfo);
    // the step kernels read the first buffer as X and the second as Y: the types follow the buffers
    auto xType = sd::ArrayOptions::dataType(xShapeInfo);
    auto yType = sd::ArrayOptions::dataType(yShapeInfo);

    if (shape::isEmptyConst(xShapeInfo) || shape::isEmptyConst(yShapeInfo)) return;
    if (xLength != yLength) THROW_EXCEPTION("sortByValue: keys and values must have the same size");
    if (xLength <= 1) return;
    requireSortableLength(xLength, "sortByValue");

    sd::NDArray::prepareSpecialUse({x, y}, {x, y});
    const sd::LongType *dXShapeInfo = x->specialShapeInfo();
    const sd::LongType *dyShapeInfo = y->specialShapeInfo();
    dim3 launchDims = getSortFullDims(static_cast<int>(xLength));

    if (isPowerOfTwo(xLength) && xLength <= SORT_BITONIC_MAX_LENGTH) {
      for (sd::LongType k = 2; k <= xLength; k <<= 1) {
        for (sd::LongType j = k >> 1; j > 0; j >>= 1) {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicSortStepGenericValue,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, y->specialBuffer(), dyShapeInfo,
                                 static_cast<int>(j), static_cast<int>(k), static_cast<int>(xLength), descending),
                                SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
        }
      }
    } else {
      const sd::LongType max = windowBound(xLength);

      for (sd::LongType window = 2; window < max; window <<= 1) {
        sd::LongType n = window;
        int rev = 0;
        do {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicArbitraryStepGenericValue,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, y->specialBuffer(), dyShapeInfo,
                                 static_cast<int>(n), static_cast<int>(xLength), rev, descending),
                                SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }

    sd::DebugHelper::checkErrorCode(stream, "sortByValue(...) failed");
    sd::NDArray::registerSpecialUse({x, y}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTadByKey(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray dimension, bool descending) {
  try {
    cudaStream_t *stream = sortStream(extraPointers);

    sd::LongType *xShapeInfo = x->shapeInfo();
    const sd::LongType *yShapeInfo = y->shapeInfo();

    auto xType = sd::ArrayOptions::dataType(xShapeInfo);
    auto yType = sd::ArrayOptions::dataType(yShapeInfo);

    // no TAD to sort, and no launch of no blocks
    if (shape::isEmptyConst(xShapeInfo) || shape::length(xShapeInfo) == 0) return;
    // the kernel addresses both arrays through the keys' TAD shape and offsets
    if (!shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo))
      THROW_EXCEPTION("sortTadByKey: keys and values must have the same shape and strides");

    auto dimensionPtr = reinterpret_cast<sd::LongType *>(dimension->buffer());
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimensionPtr, dimensionLength);
    auto numTads = tadPack->numberOfTads();
    if (numTads == 0) return;

    sd::NDArray::prepareSpecialUse({x, y}, {x, y});
    const sd::LongType *dXShapeInfo = x->specialShapeInfo();
    const sd::LongType *dyShapeInfo = y->specialShapeInfo();
    dim3 launchDims = getSortTadDims(static_cast<int>(numTads));
    BUILD_DOUBLE_SELECTOR(xType, yType, oesTadGenericKey,
                          (launchDims, stream, x->specialBuffer(), dXShapeInfo, y->specialBuffer(), dyShapeInfo,
                           dimensionPtr, dimensionLength, tadPack->platformShapeInfo(), tadPack->platformOffsets(), descending),
                          SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);

    sd::DebugHelper::checkErrorCode(stream, "sortTadByKey(...) failed");
    sd::NDArray::registerSpecialUse({x, y}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTadByValue(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray dimension, bool descending) {
  try {
    cudaStream_t *stream = sortStream(extraPointers);

    sd::LongType *xShapeInfo = x->shapeInfo();
    const sd::LongType *yShapeInfo = y->shapeInfo();

    // the values are the keys of the kernel: it reads the second array first
    auto xType = sd::ArrayOptions::dataType(yShapeInfo);
    auto yType = sd::ArrayOptions::dataType(xShapeInfo);

    // no TAD to sort, and no launch of no blocks
    if (shape::isEmptyConst(xShapeInfo) || shape::length(xShapeInfo) == 0) return;
    // the kernel addresses both arrays through one TAD shape and offsets (x's)
    if (!shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo))
      THROW_EXCEPTION("sortTadByValue: keys and values must have the same shape and strides");

    auto dimensionPtr = reinterpret_cast<sd::LongType *>(dimension->buffer());
    sd::LongType dimensionLength = static_cast<sd::LongType>(shape::length(dimension->shapeInfo()));

    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimensionPtr, dimension->lengthOf());
    auto numTads = tadPack->numberOfTads();
    if (numTads == 0) return;

    sd::NDArray::prepareSpecialUse({x, y}, {x, y});
    const sd::LongType *dXShapeInfo = x->specialShapeInfo();
    const sd::LongType *dyShapeInfo = y->specialShapeInfo();
    dim3 launchDims = getSortTadDims(static_cast<int>(numTads));
    BUILD_DOUBLE_SELECTOR(xType, yType, oesTadGenericKey,
                          (launchDims, stream, y->specialBuffer(), dyShapeInfo, x->specialBuffer(), dXShapeInfo,
                           dimensionPtr, dimensionLength, tadPack->platformShapeInfo(), tadPack->platformOffsets(), descending),
                          SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);

    sd::DebugHelper::checkErrorCode(stream, "sortTadByValue(...) failed");
    sd::NDArray::registerSpecialUse({x, y}, {x, y});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTad(sd::Pointer *extraPointers, OpaqueNDArray x,
             sd::LongType *dimension, sd::LongType dimensionLength,
             sd::LongType *tadShapeInfo, sd::LongType *tadOffsets, bool descending) {
  try {
    cudaStream_t *stream = sortStream(extraPointers);

    sd::LongType *xShapeInfo = x->shapeInfo();
    auto xType = sd::ArrayOptions::dataType(xShapeInfo);

    // no TAD to sort, and no launch of no blocks
    if (shape::isEmptyConst(xShapeInfo) || shape::length(xShapeInfo) == 0) return;

    // Ensure data is synced to device before sorting
    x->prepareSpecialUse({x}, {x});
    sd::LongType *dXShapeInfo = x->specialShapeInfo();

    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);
    auto numTads = tadPack->numberOfTads();
    if (numTads == 0) return;

    dim3 launchDims = getSortTadLarge(static_cast<int>(numTads));
    BUILD_SINGLE_SELECTOR(
        xType, oesTadGeneric,
        (launchDims, stream, x->specialBuffer(), dXShapeInfo, dimension, dimensionLength,
         tadPack->specialShapeInfo(), tadPack->specialOffsets(), descending),
        SD_NUMERIC_TYPES);

    sd::DebugHelper::checkErrorCode(stream, "sortTad(...) failed");

    x->registerSpecialUse({x}, {x});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
