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
// Created by agibsonccc on 2/21/16.
//

#define __STDC_CONSTANT_MACROS

#include <exceptions/allocation_exception.h>
#include <fcntl.h>
#include <array/DataTypeUtils.h>
#include <graph/GraphExecutioner.h>
#include <graph/GraphHolder.h>
#include <helpers/BlasHelper.h>
#include <helpers/helper_ptrmap.h>
#include <helpers/logger.h>
#include <legacy/NativeOps.h>
#include <loops/type_conversions.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/transforms.h>
#include <stdio.h>
#include <stdlib.h>

#include <types/float8.h>
#include <types/types.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>

#else
#include <helpers/mman.h>
#include <io.h>
#endif
#include <errno.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpExecutionLogger.h>
#include <graph/OpContextLifecycleTracker.h>
#include <array/NDArrayLifecycleTracker.h>
#include <array/DataBufferLifecycleTracker.h>
#include <sys/types.h>

#include <execution/Threads.h>
#include <graph/Context.h>
#include <graph/ResultWrapper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/DebugHelper.h>

#include <ops/declarable/OpRegistrator.h>
#include <ops/specials.h>
#include <system/Environment.h>
#ifdef CPU_FEATURES
#include <cpuinfo_x86.h>
#endif

#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>

#include <system/selective_rendering.h>



//these are mainly for cuda
sd::Pointer lcScalarPointer(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcReductionPointer(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcAllocationPointer(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcExecutionStream(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcCopyStream(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcBlasHandle(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcSolverHandle(OpaqueLaunchContext lc) { return nullptr; }


void execBroadcastBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,OpaqueNDArray  y,
                       OpaqueNDArray z,void *extraParams,  OpaqueNDArray dimension) {
    auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                      dimension->bufferAsT<sd::LongType>(),
                                                                      dimension->lengthOf());
    auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(),
                                                                      dimension->bufferAsT<sd::LongType>(),
                                                                      dimension->lengthOf());

    auto hTADShapeInfo = tadPackX->primaryShapeInfo();
    auto hTADOffsets = tadPackX->primaryOffsets();
    auto hTADShapeInfoZ = tadPackZ->primaryShapeInfo();
    auto hTADOffsetsZ = tadPackZ->primaryOffsets();

    sd::NDArray::prepareSpecialUse({z}, {x, y});

    NativeOpExecutioner::execBroadcastBool(nullptr, opNum,
                                           x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                           y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                           z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                           extraParams,
                                           dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                           hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);

    sd::NDArray::registerSpecialUse({z}, {x, y});

}




/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParamsVals
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 */
void execReduce3(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams,OpaqueNDArray y, OpaqueNDArray z) {
  #ifdef __cpp_exceptions
  try {
    auto dbX = x->dataBuffer();
    auto dbY = y->dataBuffer();
    auto dbZ = z->dataBuffer();

    x->preparePrimaryUse({z}, {x,y});
    NativeOpExecutioner::execReduce3(nullptr, opNum, dbX != nullptr ? x->buffer() : nullptr,
                                     x->shapeInfo(), dbX != nullptr ? dbX->special() : nullptr,
                                     x->specialShapeInfo(),
                                     extraParams, y->buffer(),
                                     y->shapeInfo(), y->specialBuffer(),
                                     y->specialShapeInfo(),
                                     dbZ != nullptr ? dbZ->primary() : nullptr, z->shapeInfo(),
                                     dbZ != nullptr ? z->specialBuffer() : nullptr,
                                     z->specialShapeInfo());
    x->registerPrimaryUse({z}, {x,y});
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto dbX = x->dataBuffer();
    auto dbY = y->dataBuffer();
    auto dbZ = z->dataBuffer();

    x->preparePrimaryUse({z}, {x,y});
    NativeOpExecutioner::execReduce3(nullptr, opNum, dbX != nullptr ? x->buffer() : nullptr,
                                     x->shapeInfo(), dbX != nullptr ? dbX->special() : nullptr,
                                     x->specialShapeInfo(),
                                     extraParams, y->buffer(),
                                     y->shapeInfo(), y->specialBuffer(),
                                     y->specialShapeInfo(),
                                     dbZ != nullptr ? dbZ->primary() : nullptr, z->shapeInfo(),
                                     dbZ != nullptr ? z->specialBuffer() : nullptr,
                                     z->specialShapeInfo());
    x->registerPrimaryUse({z}, {x,y});
  #endif
}
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParamsVals
 * @param hY
 * @param hYShapeInfo
 */
void execReduce3Scalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,  void *extraParams ,OpaqueNDArray y, OpaqueNDArray z) {
  #ifdef __cpp_exceptions
  try {
    auto dbX = x->dataBuffer();
    auto dbY = y->dataBuffer();
    auto dbZ = z->dataBuffer();

    x->preparePrimaryUse({z}, {x, y});
    NativeOpExecutioner::execReduce3Scalar(nullptr, opNum, dbX != nullptr ? x->buffer() : nullptr,
                                           x->shapeInfo(),
                                           dbX != nullptr ? x->specialBuffer() : nullptr, x->specialShapeInfo(),
                                           extraParams, y->buffer(), y->shapeInfo(),
                                           dbY->special(), y->specialShapeInfo(),
                                           dbZ != nullptr ? z->buffer() : nullptr,
                                           z->shapeInfo(), dbZ != nullptr ? dbZ->special() : nullptr,
                                           z->specialShapeInfo());
    x->registerPrimaryUse({z}, {x, y});
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto dbX = x->dataBuffer();
    auto dbY = y->dataBuffer();
    auto dbZ = z->dataBuffer();

    x->preparePrimaryUse({z}, {x, y});
    NativeOpExecutioner::execReduce3Scalar(nullptr, opNum, dbX != nullptr ? x->buffer() : nullptr,
                                           x->shapeInfo(),
                                           dbX != nullptr ? x->specialBuffer() : nullptr, x->specialShapeInfo(),
                                           extraParams, y->buffer(), y->shapeInfo(),
                                           dbY->special(), y->specialShapeInfo(),
                                           dbZ != nullptr ? z->buffer() : nullptr,
                                           z->shapeInfo(), dbZ != nullptr ? dbZ->special() : nullptr,
                                           z->specialShapeInfo());
    x->registerPrimaryUse({z}, {x, y});
  #endif
}


bool isBlasVersionMatches(int major, int minor, int build) { return true; }




/**
 * This is dummy method for JNI compatibility
 * Since we'll use this from java, jni compiler would like to have method no matter what.
 */
void initializeDevicesAndFunctions() {}

/**
 * Initialize the shape cache early to prevent race conditions during static initialization.
 * This ensures ConstantShapeHelper and its internal DirectShapeTrie are fully initialized
 * before any multi-threaded access occurs.
 *
 * Safe to call multiple times - subsequent calls are no-ops.
 */
void initializeShapeCache() {
  sd::ConstantShapeHelper::initializeEarly();
}

/**
 * Initialize the TAD (Tensor-Along-Dimension) cache early to prevent race conditions.
 * This ensures ConstantTadHelper and its internal DirectTadTrie are fully initialized
 * before any multi-threaded access occurs.
 *
 * Safe to call multiple times - subsequent calls are no-ops.
 */
void initializeTadCache() {
  sd::ConstantTadHelper::getInstance();
}

void initializeFunctions(sd::Pointer *functions) { sd::BlasHelper::getInstance().initializeFunctions(functions); }

/**
 * This method acquires memory chunk of requested size on host side
 *
 * @param sd::Pointer sd::Pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param flags optional parameter
 */
sd::Pointer mallocHost(sd::LongType memorySize, int flags) {
#if defined(SD_ALIGNED_ALLOC)
  return static_cast<sd::Pointer *>(
      aligned_alloc(SD_DESIRED_ALIGNMENT, (memorySize + SD_DESIRED_ALIGNMENT - 1) & (-SD_DESIRED_ALIGNMENT)));
#else
  return reinterpret_cast<sd::Pointer>(new int8_t[memorySize]);
#endif
}

/**
 * This method acquires memory chunk of requested size on specified device
 *
 * PLEASE NOTE: This method is NOT supported and has NO effect in CPU-based backend.
 *
 * @param sd::Pointer sd::Pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param ptrToDeviceId sd::Pointer to deviceId. For cuda that's just and int, for OpenCL that's sd::Pointer to device_id, etc
 * @param flags optional parameter
 */
sd::Pointer mallocDevice(sd::LongType memorySize, int deviceId, int flags) {
  // not supported
  return 0L;
}

/**
 * This method releases previously allocated host memory space
 *
 * @param sd::Pointer sd::Pointer that'll be freed
 */
int freeHost(sd::Pointer pointer) {
#if defined(SD_ALIGNED_ALLOC)
  free(pointer);
#else
  delete[] reinterpret_cast<int8_t *>(pointer);
#endif
  return 1L;
}

/**
 * This method releases previously allocated memory space on device
 *
 * PLEASE NOTE: This method is NOT supported and has NO effect in CPU-based backend.
 *
 * @param sd::Pointer sd::Pointer that'll be freed
 * @param ptrToDeviceId sd::Pointer to deviceId.
 */
int freeDevice(sd::Pointer pointer, int deviceId) {
  // not supported
  return 0L;
}

/**
 * Returns the maximum number open mp threads
 */
int ompGetMaxThreads() { return omp_get_max_threads(); }

/**
 * Returns the number open mp threads
 */
int ompGetNumThreads() { return omp_get_num_threads(); }

/**
 * Sets the number of openmp threads
 */
void setOmpNumThreads(int threads) { omp_set_num_threads(threads); }

sd::Pointer createContext() { return 0L; }

sd::Pointer createStream() { return 0L; }

sd::Pointer createEvent() { return 0L; }
int getDeviceBlockThreshold(int deviceId) { return 0; }
int getDeviceMajor(int deviceId) { return 0; }
int getDeviceSharedThreshold(int deviceId) {return 0; }
int getDeviceMinor(int deviceId) { return 0; }
int getDeviceId(void* deviceId) { return 0; }

int registerEvent(sd::Pointer event, sd::Pointer stream) { return 0L; }

int setDevice(int deviceId) { return 0L; }

sd::LongType getDeviceFreeMemory(int deviceId) { return 0L; }

sd::LongType getDeviceFreeMemoryDefault() { return 0L; }

sd::LongType getDeviceTotalMemory(int deviceId) { return 0L; }

int memcpySync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) { return 0L; }

int memcpyAsync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) { return 0L; }

int memsetSync(sd::Pointer dst, int value, sd::LongType size, int flags, sd::Pointer reserved) { return 0L; }

int memsetAsync(sd::Pointer dst, int value, sd::LongType size, int flags, sd::Pointer reserved) { return 0L; }

int destroyEvent(sd::Pointer event) { return 0L; }

int streamSynchronize(sd::Pointer stream) { return 0L; }

int eventSynchronize(sd::Pointer event) { return 0L; }

int getAvailableDevices() { return 0L; }

void enableDebugMode(bool reallyEnable) { sd::Environment::getInstance().setDebug(reallyEnable); }

void enableVerboseMode(bool reallyEnable) { sd::Environment::getInstance().setVerbose(reallyEnable); }

void setGridLimit(int gridSize) {
  // no-op
}

void prescanArrayRecursive(sd::Pointer *extras, int *dZ, int *dX, int numElements, int level) {
  THROW_EXCEPTION("prescanArrayRecursive Not implemented");
}



int memcpyConstantAsync(sd::LongType dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) {
  // no-op
  return 0L;
}

sd::Pointer getConstantSpace() {
  // no-op
  return 0L;
}
template <typename T>
void pullRowsGeneric(OpaqueNDArray vx, OpaqueNDArray vz, const int n, OpaqueNDArray indexes, sd::LongType dimension) {
  auto hX = vx->bufferAsT<T>();
  auto hZ = vz->bufferAsT<T>();

  auto hXShapeInfo = vx->shapeInfo();
  auto hZShapeInfo = vz->shapeInfo();

  auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, &dimension, 1);
  auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(hZShapeInfo, &dimension, 1);

  auto tadShapeInfo = tadPackX->primaryShapeInfo();
  auto tadOffsets = tadPackX->primaryOffsets();
  auto zTadShapeInfo = tadPackZ->primaryShapeInfo();
  auto zTadOffsets = tadPackZ->primaryOffsets();

  const auto tadLength = shape::length(tadShapeInfo);

  int elementsPerThread = n / TAD_THRESHOLD;
  int _threads = sd::math::sd_max<int>(1, elementsPerThread);
  _threads = sd::math::sd_min<int>(_threads, sd::Environment::getInstance().maxThreads());

  sd::LongType tadRank = shape::rank(tadShapeInfo);
  sd::LongType *tadShape = shape::shapeOf(tadShapeInfo);
  sd::LongType *tadStride = shape::stride(tadShapeInfo);

  sd::LongType zTadRank = shape::rank(zTadShapeInfo);
  sd::LongType *zTadShape = shape::shapeOf(zTadShapeInfo);
  sd::LongType *zTadStride = shape::stride(zTadShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    for (auto idx2 = start; idx2 < stop; idx2++) {
      auto xTadOffsetForBlock = tadOffsets[reinterpret_cast<sd::LongType *>(indexes->buffer())[idx2]];
      auto zTadOffsetForBlock = zTadOffsets[idx2];

      auto rX = hX + xTadOffsetForBlock;
      auto rZ = hZ + zTadOffsetForBlock;

      sd::LongType xCoords[SD_MAX_RANK];
      sd::LongType zCoords[SD_MAX_RANK];
      sd::LongType xOffset;
      sd::LongType zOffset;

      INDEX2COORDS(idx2, tadRank, tadShape, xCoords);
      COORDS2INDEX(tadRank, tadStride, xCoords, xOffset);
      INDEX2COORDS(idx2, zTadRank,zTadShape, zCoords);
      COORDS2INDEX(zTadRank, zTadStride, zCoords, zOffset);

      for (sd::LongType i = 0; i < tadLength; i++) {
        hZ[zOffset + i] = hX[xOffset + i];
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, n, 1, _threads);
}
void tryPointer(sd::Pointer extra, sd::Pointer p, int len) {
  #ifdef __cpp_exceptions
  try {
    auto buf = reinterpret_cast<int8_t *>(p);
    int cnt = 0;
    for (int i = 0; i < len; i++) cnt += buf[cnt];
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto buf = reinterpret_cast<int8_t *>(p);
    int cnt = 0;
    for (int i = 0; i < len; i++) cnt += buf[cnt];
  #endif
}

void pullRows(sd::Pointer *extraPointers,
              OpaqueNDArray x,
              OpaqueNDArray z,
              sd::LongType n,
              OpaqueNDArray indexes,
              sd::LongType dimension) {
  #ifdef __cpp_exceptions
  try {
    auto xType = sd::ArrayOptions::dataType(x->shapeInfo());

    BUILD_SINGLE_SELECTOR(xType, pullRowsGeneric, (x, z, n, indexes, dimension), SD_COMMON_TYPES);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto xType = sd::ArrayOptions::dataType(x->shapeInfo());

    BUILD_SINGLE_SELECTOR(xType, pullRowsGeneric, (x, z, n, indexes, dimension), SD_COMMON_TYPES);
  #endif
}
template <typename T>
void tearGeneric(void *vx, sd::LongType const *hXShapeInfo, sd::Pointer *targets, sd::LongType const *hZShapeInfo,
                 sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets) {
  auto hX = reinterpret_cast<T *>(vx);

  const auto tadLength = shape::length(tadShapeInfo);
  auto numTads = shape::length(hXShapeInfo) / tadLength;

  sd::LongType tadRank = shape::rank(tadShapeInfo);
  sd::LongType *tadShape = shape::shapeOf(tadShapeInfo);
  sd::LongType *tadStride = shape::stride(tadShapeInfo);

  sd::LongType zTadRank = shape::rank(hZShapeInfo);
  sd::LongType *zTadShape = shape::shapeOf(hZShapeInfo);
  sd::LongType *zTadStride = shape::stride(hZShapeInfo);


  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      auto hZ = reinterpret_cast<T *>(targets[i]);
      auto s = hX + tadOffsets[i];

      for (sd::LongType j = 0; j < tadLength; j++) {
        sd::LongType xCoords[SD_MAX_RANK];
        sd::LongType zCoords[SD_MAX_RANK];
        sd::LongType xOffset;
        sd::LongType zOffset;

        INDEX2COORDS(j, tadRank, tadShape, xCoords);
        COORDS2INDEX(tadRank, zTadStride, xCoords, xOffset);
        INDEX2COORDS(j, zTadRank, zTadStride, zCoords);
        COORDS2INDEX(zTadRank, zTadStride, zCoords, zOffset);

        hZ[zOffset] = s[xOffset];
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, numTads);
}

void tear(sd::Pointer *extraPointers, OpaqueDataBuffer *dbX, sd::LongType const *hXShapeInfo,
          sd::LongType const *dXShapeInfo, sd::Pointer *targets, sd::LongType const *hZShapeInfo,
          sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets) {
  #ifdef __cpp_exceptions
  try {
    auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, tearGeneric,
                          (dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, targets, hZShapeInfo, tadShapeInfo, tadOffsets),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, tearGeneric,
                          (dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, targets, hZShapeInfo, tadShapeInfo, tadOffsets),
                          SD_COMMON_TYPES);
  #endif
}



void enableP2P(bool enable) {
  // no-op
}



bool isP2PAvailable() {
  // always TRUE for cpu backend
  return true;
}

void checkP2P() {
  // no-op
}


template <typename T>
void shuffleGeneric(OpaqueNDArrayArr hX, OpaqueNDArrayArr hZ, int N, int *shuffleMap, sd::LongType *dimension, sd::LongType dimensionLength) {
  auto func = PRAGMA_THREADS_FOR {
    for (auto f = start; f < stop; f++) {
      T *hX2 = hX[f]->bufferAsT<T>();
      T *hZ2 = hZ[f]->bufferAsT<T>();

      auto xShapeInfo = hX[f]->shapeInfo();
      auto zShapeInfo = hZ[f]->shapeInfo();
      auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);
      auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(zShapeInfo, dimension, dimensionLength);
      auto tadOnlyShapeInfoX = tadPackX->primaryShapeInfo();
      auto tadOffsetsX = tadPackX->primaryOffsets();
      auto tadOnlyShapeInfoZ = tadPackZ->primaryShapeInfo();
      auto tadOffsetsZ = tadPackZ->primaryOffsets();

      const auto tadLength = shape::length(tadOnlyShapeInfoX);
      auto numTads = shape::length(xShapeInfo) / tadLength;

      sd::LongType xRank = shape::rank(xShapeInfo);
      sd::LongType zRank = shape::rank(zShapeInfo);
      sd::LongType *xShape = shape::shapeOf(xShapeInfo);
      sd::LongType *xStride = shape::stride(xShapeInfo);
      sd::LongType *zShape = shape::shapeOf(zShapeInfo);
      sd::LongType *zStride = shape::stride(zShapeInfo);

      if (shape::rank(xShapeInfo) == 1) {
        auto xLength = shape::length(xShapeInfo);
        for (sd::LongType r = 0; r < xLength; r++) {
          auto swapIdx = shuffleMap[r];
          if (swapIdx < 0) continue;

          sd::LongType xCoords[SD_MAX_RANK];
          sd::LongType zCoords[SD_MAX_RANK];
          sd::LongType xOffset;
          sd::LongType zOffset;

          INDEX2COORDS(r, xRank, xShape, xCoords);
          COORDS2INDEX(xRank, xStride, xCoords, xOffset);
          INDEX2COORDS(swapIdx,zRank, zShape, zCoords);
          COORDS2INDEX(zRank, zStride, zCoords, zOffset);

          sd::math::sd_swap<T>(hX2[xOffset], hZ2[zOffset]);
        }
      } else {


        sd::LongType tadShapeInfoRank = shape::rank(tadOnlyShapeInfoX);
        sd::LongType *tadShapeInfoShape = shape::shapeOf(tadOnlyShapeInfoX);
        sd::LongType *tadShapeInfoStride = shape::stride(tadOnlyShapeInfoX);

        sd::LongType zTadShapeInfoRank = shape::rank(tadOnlyShapeInfoZ);
        sd::LongType *zTadShapeInfoShape = shape::shapeOf(tadOnlyShapeInfoZ);
        sd::LongType *zTadShapeInfoStride = shape::stride(tadOnlyShapeInfoZ);
        for (sd::LongType r = 0; r < numTads; r++) {
          if (shuffleMap[r] < 0) continue;

          auto oldOffsetX = tadOffsetsX[r];
          auto newOffsetZ = tadOffsetsZ[shuffleMap[r]];

          auto rX = hX2 + oldOffsetX;
          auto rZ = hZ2 + newOffsetZ;

          for (sd::LongType i = 0; i < tadLength; i++) {
            sd::LongType xCoords[SD_MAX_RANK];
            sd::LongType zCoords[SD_MAX_RANK];
            sd::LongType xOffset;
            sd::LongType zOffset;

            INDEX2COORDS(i, tadShapeInfoRank, tadShapeInfoShape, xCoords);
            COORDS2INDEX(tadShapeInfoRank,tadShapeInfoStride, xCoords, xOffset);
            INDEX2COORDS(i, zTadShapeInfoRank,zTadShapeInfoShape, zCoords);
            COORDS2INDEX(zTadShapeInfoRank, zTadShapeInfoStride, zCoords, zOffset);

            sd::math::sd_swap<T>(rX[xOffset], rZ[zOffset]);
          }
        }
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, N);
}

void shuffle(sd::Pointer *extras,
             OpaqueNDArrayArr x,
             OpaqueNDArrayArr z,
             int N,
             OpaqueNDArray dimension,
             OpaqueNDArray shuffleMap) {
  #ifdef __cpp_exceptions
  try {
    auto dimensionData = reinterpret_cast<sd::LongType *>(dimension->buffer());
    auto dimensionLength = shape::length(dimension->shapeInfo());

    auto xType = sd::ArrayOptions::dataType(x[0]->shapeInfo());

    BUILD_SINGLE_SELECTOR(xType, shuffleGeneric, (x, z, N, reinterpret_cast<int *>(shuffleMap->buffer()), dimensionData, dimensionLength), SD_COMMON_TYPES);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto dimensionData = reinterpret_cast<sd::LongType *>(dimension->buffer());
    auto dimensionLength = shape::length(dimension->shapeInfo());

    auto xType = sd::ArrayOptions::dataType(x[0]->shapeInfo());

    BUILD_SINGLE_SELECTOR(xType, shuffleGeneric, (x, z, N, reinterpret_cast<int *>(shuffleMap->buffer()), dimensionData, dimensionLength), SD_COMMON_TYPES);
  #endif
}

bool isExperimentalEnabled() { return sd::Environment::getInstance().isExperimentalBuild(); }

void setOmpMinThreads(int threads) {
  // TODO: to be implemented
}

int getDevice() { return 0; }



char *name;
bool nameSet = false;

const char *getDeviceName(int deviceId) {
  #ifdef __cpp_exceptions
  try {
    if (!nameSet) {
      name = reinterpret_cast<char *>(malloc(256 * sizeof(char)));

      CHECK_ALLOC(name, "Failed to allocate new string buffer", 256);

      std::memset(name, 0, 256 * sizeof(char));
      nameSet = true;

      // TODO: provide proper CPU model name here
      sprintf(name, "x86-compatible CPU");
    }
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    if (!nameSet) {
      name = reinterpret_cast<char *>(malloc(256 * sizeof(char)));

      CHECK_ALLOC(name, "Failed to allocate new string buffer", 256);

      std::memset(name, 0, 256 * sizeof(char));
      nameSet = true;

      // TODO: provide proper CPU model name here
      sprintf(name, "x86-compatible CPU");
    }
  #endif

  return name;
}



void execRandom(sd::Pointer *extraPointers, int opNum, sd::Pointer state, OpaqueDataBuffer *dbZ,
                const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, void *extraArguments) {
  #ifdef __cpp_exceptions
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo,
                                    extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {});
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo,
                                    extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {});
  #endif
}

void execRandom3(sd::Pointer *extraPointers, int opNum, sd::Pointer state, OpaqueDataBuffer *dbX,
                 const sd::LongType *hXShapeInfo, const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbY,
                 const sd::LongType *hYShapeInfo, const sd::LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                 const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, void *extraArguments) {
  #ifdef __cpp_exceptions
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                    dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ != nullptr ? dbZ->primary() : nullptr,
                                    hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                    dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ != nullptr ? dbZ->primary() : nullptr,
                                    hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  #endif
}

void execRandom2(sd::Pointer *extraPointers, int opNum, sd::Pointer state, OpaqueDataBuffer *dbX,
                 const sd::LongType *hXShapeInfo, const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbZ,
                 const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, void *extraArguments) {
  #ifdef __cpp_exceptions
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                    dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                    dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  #endif
}

sd::Pointer initRandom(sd::Pointer *extraPointers, long seed, long bufferSize, sd::Pointer ptrToBuffer) {
  #ifdef __cpp_exceptions
  try {
    auto generator = new sd::graph::RandomGenerator(seed, seed);

    return (sd::Pointer)generator;
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }
  #else
    auto generator = new sd::graph::RandomGenerator(seed, seed);

    return (sd::Pointer)generator;
  #endif
}

void refreshBuffer(sd::Pointer *extraPointers, long seed, sd::Pointer ptrRandom) {
  auto generator = reinterpret_cast<sd::graph::RandomGenerator *>(ptrRandom);

  generator->setStates(seed);
}

void reSeedBuffer(sd::Pointer *extraPointers, long seed, sd::Pointer ptrRandom) {
  auto generator = reinterpret_cast<sd::graph::RandomGenerator *>(ptrRandom);

  generator->setStates(seed);
}

void destroyRandom(sd::Pointer ptrBuffer) {
  auto buffer = reinterpret_cast<sd::graph::RandomGenerator *>(ptrBuffer);
  delete buffer;
}

/**
 * Return the length of a shape buffer
 * based on the pointer
 * @param buffer  the buffer sd::Pointer to check
 * @return
 */
int lengthForShapeBufferPointer(sd::Pointer buffer) {
  auto shapeBuffer = reinterpret_cast<sd::LongType *>(buffer);
  return shape::shapeInfoLength(shape::rank(shapeBuffer));
}

/**
 * The sd::Pointer to get the address for
 *
 * @param address the address to get the pointer
 * @return the sd::Pointer for the given address
 */

sd::Pointer pointerForAddress(sd::LongType address) { return reinterpret_cast<sd::Pointer>(address); }

void sort(sd::Pointer *extraPointers, OpaqueNDArray x, bool descending) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execSort(x, descending);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execSort(x, descending);
  #endif
}

void sortTad(sd::Pointer *extraPointers, OpaqueNDArray  x,
             sd::LongType *dimension, sd::LongType dimensionLength,
             sd::LongType *tadShapeInfo,  sd::LongType *tadOffsets, bool descending) {
    NativeOpExecutioner::execSort(x, dimension, dimensionLength, descending);

}



sd::Status execCustomOp2(sd::Pointer *extraPointers, sd::LongType hash, OpaqueContext *context) {
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);

    // Set op name BEFORE execute() so allocations during execution are tagged
    // This is done unconditionally so per-op tracking works even without SD_GCC_FUNCTRACE
    if (op->getOpName() != nullptr) {
        const std::string& opName = *op->getOpName();

        // Set the op context in ALL lifecycle trackers so allocations are tagged
        sd::array::NDArrayLifecycleTracker::setCurrentOpContext(opName);
        sd::array::DataBufferLifecycleTracker::setCurrentOpContext(opName);
        sd::graph::OpContextLifecycleTracker::setCurrentOpContext(opName);

        // Also update the already-tracked context with the op name
        sd::graph::OpContextLifecycleTracker::getInstance().updateContextOpName(context, opName);

#if defined(SD_GCC_FUNCTRACE)
        // Also set for OpExecutionLogger when functrace is enabled
        sd::ops::OpExecutionLogger::setCurrentOpName(opName);
#endif
    }

    auto result = op->execute(context);

    // Clear op context after execution
    sd::array::NDArrayLifecycleTracker::clearCurrentOpContext();
    sd::array::DataBufferLifecycleTracker::clearCurrentOpContext();
    sd::graph::OpContextLifecycleTracker::clearCurrentOpContext();

#if defined(SD_GCC_FUNCTRACE)
    sd::ops::OpExecutionLogger::clearCurrentOpName();
#endif

    checkAndCleanupCaches();

    return result;
}



void setShapeBuffer(sd::LongType *inputShapeData,sd::DataType dt,sd::LongType *bufferToSet,char order,int elementWiseStride,bool isEmpty,bool isView) {
  if(inputShapeData == nullptr)
    THROW_EXCEPTION("setShapeBuffer: inputShapeData is null");

  if(bufferToSet == nullptr)
    THROW_EXCEPTION("setShapeBuffer: bufferToSet is null");
  sd::LongType  rank = inputShapeData[0];
  if(rank > SD_MAX_RANK || rank < 0)
    THROW_EXCEPTION("Invalid rank for shape buffer.");
  std::vector<sd::LongType> shape;
  std::vector<sd::LongType> strides;
  //shape, stride, data type
  for(sd::LongType i = 1; i < rank * 2 + 1; i++) {
    if(i <= rank) {
      shape.push_back(inputShapeData[i]);
    } else if(shape.size() == static_cast<size_t>(rank)) {
      strides.push_back(inputShapeData[i]);
    }
  }

  bufferToSet[0] = rank;

  shape::setOrder(bufferToSet,order);

  auto len = shape::shapeInfoLength(rank);

  auto origShape = shape::shapeOf(inputShapeData);
  auto origStride = shape::stride(inputShapeData);
  shape::setShape(bufferToSet,origShape);
  shape::setStride(bufferToSet,origStride);

  sd::ArrayOptions::setDataType(bufferToSet,dt);
  if(isView) {
    sd::ArrayOptions::toggleIsView(bufferToSet);
  }
  if(!sd::ArrayOptions::isEmpty(inputShapeData) && isEmpty) {
    sd::ArrayOptions::toggleIsEmpty(bufferToSet);
  }


  if(rank == 0) {
    //detect when the shape buffer values are unset.
    auto len2 = shape::shapeInfoLength(rank);
    //min number of values in a shape info buffer
    bool allZero = true;
    for(int i = 0; i < len2; i++) {
      if(bufferToSet[i] != 0) {
        allZero = false;
        break;
      }
    }

    if(allZero) {
      THROW_EXCEPTION("Found shape buffer with all zero values. Values likely unset.");
    }
  }
}



////////////////////////////////////////////////////////////////////////


void setGraphContextCudaContext(sd::graph::Context *ptr, void *stream, void *reductionPointer,
                                void *allocationPointer) {}


void saveNpy(std::string fname, const OpaqueDataBuffer *data, const unsigned int *shape, const unsigned int ndims,
             std::string mode) {
  auto dtype = data->getDataBuffer()->getDataType();
  BUILD_SINGLE_SELECTOR(dtype,cnpy::npy_save,(fname,data->getDataBuffer()->primary(),shape,ndims,mode),SD_COMMON_TYPES);
}


void sortByKey(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y,bool descending) {
  #ifdef __cpp_exceptions
  try {
    auto xType = x->dataType();
    auto yType = y->dataType();
    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortByKey(x, y, descending),
                          SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto xType = x->dataType();
    auto yType = y->dataType();
    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortByKey(x, y, descending),
                          SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
  #endif
}

void sortByValue(sd::Pointer *extraPointers, OpaqueNDArray x,OpaqueNDArray y, bool descending) {
  #ifdef __cpp_exceptions
  try {
    auto xType = x->dataType();
    auto yType = y->dataType();
    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortByValue(x, y, descending),
                          SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto xType = x->dataType();
    auto yType = y->dataType();
    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortByValue(x, y, descending),
                          SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
  #endif
}

void sortTadByKey(sd::Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y,
                  OpaqueNDArray dimension, bool descending) {
  #ifdef __cpp_exceptions
  try {
    auto xType = x->dataType();
    auto yType = y->dataType();
    auto dimensionLength = dimension->lengthOf();
    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortTadByValue(x, y, dimension, descending), SD_NUMERIC_TYPES,
                          SD_NUMERIC_TYPES);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto xType = x->dataType();
    auto yType = y->dataType();
    auto dimensionLength = dimension->lengthOf();
    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortTadByValue(x, y, dimension, descending), SD_NUMERIC_TYPES,
                          SD_NUMERIC_TYPES);
  #endif
}
void sortTadByValue(sd::Pointer *extraPointers, OpaqueNDArray x,
                    OpaqueNDArray y,OpaqueNDArray dimension, bool descending) {
  #ifdef __cpp_exceptions
  try {
    auto xType = x->dataType();
    auto yType = y->dataType();
    auto dimensionLength = dimension->lengthOf();
    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortTadByValue(x, y, dimension, descending), SD_NUMERIC_TYPES,
                          SD_NUMERIC_TYPES);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto xType = x->dataType();
    auto yType = y->dataType();
    auto dimensionLength = dimension->lengthOf();
    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortTadByValue(x, y, dimension, descending), SD_NUMERIC_TYPES,
                          SD_NUMERIC_TYPES);
  #endif
}


void execIndexReduceScalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,void *extraParams,
                           OpaqueNDArray z) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execIndexReduceScalar(nullptr, opNum,
                                               x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                               extraParams,
                                               z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execIndexReduceScalar(nullptr, opNum,
                                               x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                               extraParams,
                                               z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  #endif
}

void execIndexReduce(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,
                     void *extraParams,
                     OpaqueNDArray z, OpaqueNDArray dimension
) {
  #ifdef __cpp_exceptions
  try {
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    auto hTADShapeInfo = tadPack->primaryShapeInfo();
    auto hTADOffsets = tadPack->primaryOffsets();

    NativeOpExecutioner::execIndexReduce(nullptr, opNum,
                                         x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                         extraParams,
                                         z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                         dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                         hTADShapeInfo, hTADOffsets);

  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    auto hTADShapeInfo = tadPack->primaryShapeInfo();
    auto hTADOffsets = tadPack->primaryOffsets();

    NativeOpExecutioner::execIndexReduce(nullptr, opNum,
                                         x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                         extraParams,
                                         z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                         dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                         hTADShapeInfo, hTADOffsets);

  #endif
}

void execBroadcast(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y,
                   OpaqueNDArray z,void *extraInfo, OpaqueNDArray dimension) {
  #ifdef __cpp_exceptions
  try {
    auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                      dimension->bufferAsT<sd::LongType>(),
                                                                      dimension->lengthOf());
    auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(),
                                                                      dimension->bufferAsT<sd::LongType>(),
                                                                      dimension->lengthOf());

#if defined(PRINT_INDICES)
    printf("broadcast exec tad full x\n");
    shape::printShapeInfo(x->shapeInfo());
    printf("broadcast exec tad full y\n");
    shape::printShapeInfo(y->shapeInfo());
    printf("broadcast exec tad full z\n");
    shape::printShapeInfo(z->shapeInfo());
#endif
    auto hTADShapeInfo = tadPackX->primaryShapeInfo();
    auto hTADOffsets = tadPackX->primaryOffsets();
    auto hTADShapeInfoZ = tadPackZ->primaryShapeInfo();
    auto hTADOffsetsZ = tadPackZ->primaryOffsets();

    NativeOpExecutioner::execBroadcast(nullptr, opNum,
                                       x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                       y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                       z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                       dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                       hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);

  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                      dimension->bufferAsT<sd::LongType>(),
                                                                      dimension->lengthOf());
    auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(),
                                                                      dimension->bufferAsT<sd::LongType>(),
                                                                      dimension->lengthOf());

#if defined(PRINT_INDICES)
    printf("broadcast exec tad full x\n");
    shape::printShapeInfo(x->shapeInfo());
    printf("broadcast exec tad full y\n");
    shape::printShapeInfo(y->shapeInfo());
    printf("broadcast exec tad full z\n");
    shape::printShapeInfo(z->shapeInfo());
#endif
    auto hTADShapeInfo = tadPackX->primaryShapeInfo();
    auto hTADOffsets = tadPackX->primaryOffsets();
    auto hTADShapeInfoZ = tadPackZ->primaryShapeInfo();
    auto hTADOffsetsZ = tadPackZ->primaryOffsets();

    NativeOpExecutioner::execBroadcast(nullptr, opNum,
                                       x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                       y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                       z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                       dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                       hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);

  #endif
}

void execPairwiseTransform(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y,
                           OpaqueNDArray z, void *extraParams) {
  #ifdef __cpp_exceptions
  try {
    /**
     * TODO: look in to offsets here as left over change from ndarrays being available?
     */
    NativeOpExecutioner::execPairwiseTransform(nullptr, opNum,
                                               x->bufferWithOffset(x->offset()),
                                               x->shapeInfo(),
                                               x->specialBufferWithOffset(x->offset()),
                                               x->specialShapeInfo(),
                                               y->bufferWithOffset(y->offset()),
                                               y->shapeInfo(),
                                               y->specialBufferWithOffset(y->offset()),
                                               y->specialShapeInfo(),
                                               z->bufferWithOffset(z->offset()),
                                               z->shapeInfo(),
                                               const_cast<void *>(z->specialBufferWithOffset(z->offset())),
                                               z->specialShapeInfo(),
                                               extraParams);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    /**
     * TODO: look in to offsets here as left over change from ndarrays being available?
     */
    NativeOpExecutioner::execPairwiseTransform(nullptr, opNum,
                                               x->bufferWithOffset(x->offset()),
                                               x->shapeInfo(),
                                               x->specialBufferWithOffset(x->offset()),
                                               x->specialShapeInfo(),
                                               y->bufferWithOffset(y->offset()),
                                               y->shapeInfo(),
                                               y->specialBufferWithOffset(y->offset()),
                                               y->specialShapeInfo(),
                                               z->bufferWithOffset(z->offset()),
                                               z->shapeInfo(),
                                               const_cast<void *>(z->specialBufferWithOffset(z->offset())),
                                               z->specialShapeInfo(),
                                               extraParams);
  #endif
}

void execReduceFloat(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,
                     void *extraParams, OpaqueNDArray z) {
  #ifdef __cpp_exceptions
  try {
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceFloat: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceFloat: null shapeInfo in input arrays");
      return;
    }

    NativeOpExecutioner::execReduceFloatScalar(nullptr, opNum,
                                               x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                               extraParams,
                                               z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
    checkAndCleanupCaches();
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceFloat: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceFloat: null shapeInfo in input arrays");
      return;
    }

    NativeOpExecutioner::execReduceFloatScalar(nullptr, opNum,
                                               x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                               extraParams,
                                               z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
    checkAndCleanupCaches();
  #endif
}

void execReduceSame(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,
                    void *extraParams,OpaqueNDArray z) {
  #ifdef __cpp_exceptions
  try {
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceSame: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceSame: null shapeInfo in input arrays");
      return;
    }

    NativeOpExecutioner::execReduceSameScalar(nullptr, opNum,
                                              x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                              extraParams,
                                              z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
    checkAndCleanupCaches();
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceSame: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceSame: null shapeInfo in input arrays");
      return;
    }

    NativeOpExecutioner::execReduceSameScalar(nullptr, opNum,
                                              x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                              extraParams,
                                              z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
    checkAndCleanupCaches();
  #endif
}

void execReduceBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams,
                    OpaqueNDArray z, OpaqueNDArray dimension) {
  #ifdef __cpp_exceptions
  try {
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr || dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceBool: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr || dimension->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceBool: null shapeInfo in input arrays");
      return;
    }

    // Removed unused TAD pack creation that was causing cache bloat and memory leaks
    // The NativeOpExecutioner::execReduceBool handles TAD operations internally
    NativeOpExecutioner::execReduceBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(),
                                        x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), z->shapeInfo(),
                                        z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<sd::LongType> (), dimension->lengthOf());
    checkAndCleanupCaches();
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr || dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceBool: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr || dimension->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceBool: null shapeInfo in input arrays");
      return;
    }

    // Removed unused TAD pack creation that was causing cache bloat and memory leaks
    // The NativeOpExecutioner::execReduceBool handles TAD operations internally
    NativeOpExecutioner::execReduceBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(),
                                        x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), z->shapeInfo(),
                                        z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<sd::LongType> (), dimension->lengthOf());
    checkAndCleanupCaches();
  #endif
}

void execReduceLong(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams,
                    OpaqueNDArray z, OpaqueNDArray dimension) {
  #ifdef __cpp_exceptions
  try {
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr || dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr || dimension->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong: null shapeInfo in input arrays");
      return;
    }

    // Removed unused TAD pack creation that was causing cache bloat and memory leaks
    // The NativeOpExecutioner::execReduceLong handles TAD operations internally
    NativeOpExecutioner::execReduceLong(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(),
                                        x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), z->shapeInfo(),
                                        z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<sd::LongType> (), dimension->lengthOf());
    checkAndCleanupCaches();
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr || dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr || dimension->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong: null shapeInfo in input arrays");
      return;
    }

    // Removed unused TAD pack creation that was causing cache bloat and memory leaks
    // The NativeOpExecutioner::execReduceLong handles TAD operations internally
    NativeOpExecutioner::execReduceLong(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(),
                                        x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), z->shapeInfo(),
                                        z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<sd::LongType> (), dimension->lengthOf());
    checkAndCleanupCaches();
  #endif
}

void execReduceFloat2(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,void *extraParams,
                      OpaqueNDArray z, OpaqueNDArray dimension) {
  #ifdef __cpp_exceptions
  try {
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr || dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceFloat2: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr || dimension->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceFloat2: null shapeInfo in input arrays");
      return;
    }

    std::vector<sd::LongType>  dimensions(dimension->lengthOf());
    for(sd::LongType i = 0; i < dimension->lengthOf(); i++) {
      sd::LongType curr = dimension->e<sd::LongType>(i);
      if(curr < 0) {
        curr += x->rankOf();
      }
      dimensions[i] = curr;
    }
    const sd::LongType *zShapeInfoH = z->shapeInfo();
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo()) && z->lengthOf() != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims = (z->lengthOf() != 1) ?
                                  sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<sd::LongType>();

    NativeOpExecutioner::execReduceFloat(nullptr, opNum,
                                         x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                         extraParams,
                                         z->buffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                         dims->data(), dims->size());

    delete dims;

    checkAndCleanupCaches();

  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr || dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceFloat2: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr || dimension->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceFloat2: null shapeInfo in input arrays");
      return;
    }

    std::vector<sd::LongType>  dimensions(dimension->lengthOf());
    for(sd::LongType i = 0; i < dimension->lengthOf(); i++) {
      sd::LongType curr = dimension->e<sd::LongType>(i);
      if(curr < 0) {
        curr += x->rankOf();
      }
      dimensions[i] = curr;
    }
    const sd::LongType *zShapeInfoH = z->shapeInfo();
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo()) && z->lengthOf() != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims = (z->lengthOf() != 1) ?
                                  sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<sd::LongType>();

    NativeOpExecutioner::execReduceFloat(nullptr, opNum,
                                         x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                         extraParams,
                                         z->buffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                         dims->data(), dims->size());

    delete dims;

    checkAndCleanupCaches();

  #endif
}

void execReduceBool2(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,
                     void *extraParams,
                     OpaqueNDArray z, OpaqueNDArray dimension) {
  #ifdef __cpp_exceptions
  try {
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr || dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceBool2: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr || dimension->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceBool2: null shapeInfo in input arrays");
      return;
    }

    std::vector<sd::LongType> dimensions(dimension->lengthOf());
    for(sd::LongType i = 0; i < dimension->lengthOf(); i++) {
      sd::LongType curr = dimension->e<sd::LongType>(i);
      if(curr < 0) {
        curr += x->rankOf();
      }
      dimensions[i] = curr;
    }

    const sd::LongType *zShapeInfoH = z->shapeInfo();
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo())) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims = (z->lengthOf() != 1) ?
                                  sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<sd::LongType>();

    NativeOpExecutioner::execReduceBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());

    delete dims;

    checkAndCleanupCaches();

  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr || dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceBool2: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr || dimension->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceBool2: null shapeInfo in input arrays");
      return;
    }

    std::vector<sd::LongType> dimensions(dimension->lengthOf());
    for(sd::LongType i = 0; i < dimension->lengthOf(); i++) {
      sd::LongType curr = dimension->e<sd::LongType>(i);
      if(curr < 0) {
        curr += x->rankOf();
      }
      dimensions[i] = curr;
    }

    const sd::LongType *zShapeInfoH = z->shapeInfo();
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo())) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims = (z->lengthOf() != 1) ?
                                  sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<sd::LongType>();

    NativeOpExecutioner::execReduceBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());

    delete dims;

    checkAndCleanupCaches();

  #endif
}

void execReduceSame2(sd::Pointer *extraPointers, int opNum,
                     OpaqueNDArray x,void *extraParams,
                     OpaqueNDArray z, OpaqueNDArray dimension) {
  #ifdef __cpp_exceptions
  try {
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr || dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceSame2: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr || dimension->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceSame2: null shapeInfo in input arrays");
      return;
    }

    std::vector<sd::LongType> dimensions(dimension->lengthOf());
    for(sd::LongType i = 0; i < dimension->lengthOf(); i++) {
      sd::LongType curr = dimension->e<sd::LongType>(i);
      if(curr < 0) {
        curr += x->rankOf();
      }
      dimensions[i] = curr;
    }

    const sd::LongType *zShapeInfoH = z->shapeInfo();
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo()) && z->lengthOf() != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims = (z->lengthOf() != 1) ?
                                  sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<sd::LongType>();

    NativeOpExecutioner::execReduceSame(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());

    delete dims;

    checkAndCleanupCaches();

  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    // Validate input pointers to prevent segfault
    if (x == nullptr || z == nullptr || dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceSame2: null pointer in input parameters");
      return;
    }
    if (x->shapeInfo() == nullptr || z->shapeInfo() == nullptr || dimension->shapeInfo() == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceSame2: null shapeInfo in input arrays");
      return;
    }

    std::vector<sd::LongType> dimensions(dimension->lengthOf());
    for(sd::LongType i = 0; i < dimension->lengthOf(); i++) {
      sd::LongType curr = dimension->e<sd::LongType>(i);
      if(curr < 0) {
        curr += x->rankOf();
      }
      dimensions[i] = curr;
    }

    const sd::LongType *zShapeInfoH = z->shapeInfo();
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo()) && z->lengthOf() != 1) {
      auto zPack = sd::ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims = (z->lengthOf() != 1) ?
                                  sd::ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<sd::LongType>();

    NativeOpExecutioner::execReduceSame(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());

    delete dims;

    checkAndCleanupCaches();

  #endif
}

void execReduceLong2(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,
                     void *extraParams,
                     OpaqueNDArray z, OpaqueNDArray dimension) {
  #ifdef __cpp_exceptions
  try {
    // Validate input pointers to prevent segfault
    if (x == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: input array x is null");
      return;
    }
    if (z == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: output array z is null");
      return;
    }
    if (dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: dimension array is null");
      return;
    }

    // If we validate first (call shapeInfo()), then cache later (call shapeInfo() again),
    // the pointer could become invalid between the two calls, causing SIGSEGV.
    // By caching once and validating the cached value, we ensure consistency.
    const sd::LongType *xShapeInfoH = x->shapeInfo();
    const sd::LongType *xShapeInfoD = x->specialShapeInfo();
    void *xBuffer = x->buffer();
    void *xSpecialBuffer = x->specialBuffer();

    void *zBuffer = z->buffer();
    const sd::LongType *zShapeInfoH = z->shapeInfo();
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();
    const sd::LongType zLength = z->lengthOf();

    if (xShapeInfoH == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: input array x has null shapeInfo");
      return;
    }
    if (zShapeInfoH == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: output array z has null shapeInfo");
      return;
    }

    void *dimensionBuffer = dimension->buffer();
    sd::DataBuffer *dimensionDb = dimension->getDataBuffer();
    if (dimensionBuffer == nullptr || dimensionDb == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: dimension array has null buffer");
      return;
    }

    const sd::LongType xRank = shape::rank(xShapeInfoH);
    const sd::DataType dimType = dimension->dataType();
    if (dimType != sd::DataType::INT32 && dimType != sd::DataType::INT64) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      std::string err = "execReduceLong2: unsupported dimension buffer data type: ";
      err += sd::DataTypeUtils::asString(dimType);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(err.c_str());
      return;
    }

    const sd::LongType dimensionLength =
        static_cast<sd::LongType>(dimensionDb->getLenInBytes() / sd::DataTypeUtils::sizeOf(dimType));

    // Extract dimension values directly from the raw buffer. Do not rely on dimension->shapeInfo()
    // because some callers mutate or free the dimension shape buffer once the NDArray is created.
    std::vector<sd::LongType> dimensions(dimensionLength);
    if (dimensionLength > 0) {
      if (dimType == sd::DataType::INT32) {
        auto dimensionData = reinterpret_cast<int *>(dimensionBuffer);
        for (sd::LongType i = 0; i < dimensionLength; i++) {
          sd::LongType curr = static_cast<sd::LongType>(dimensionData[i]);
          if (curr < 0) {
            curr += xRank;
          }
          dimensions[i] = curr;
        }
      } else {
        auto dimensionData = reinterpret_cast<sd::LongType *>(dimensionBuffer);
        for (sd::LongType i = 0; i < dimensionLength; i++) {
          sd::LongType curr = dimensionData[i];
          if (curr < 0) {
            curr += xRank;
          }
          dimensions[i] = curr;
        }
      }
    }

    // Validate output shape matches expected dimensions after reduction
    // If ranks don't match, this indicates a shape mismatch from the calling layer
    // DO NOT attempt to reshape - the buffer and shape must match
    if (shape::rank(xShapeInfoH) - dimensionLength != shape::rank(zShapeInfoH) && zLength != 1) {
      std::string errorMsg = "execReduceLong2: Output shape rank mismatch. ";
      errorMsg += "Input rank: " + std::to_string(shape::rank(xShapeInfoH));
      errorMsg += ", reduction dimensions: " + std::to_string(dimensionLength);
      errorMsg += ", expected output rank: " + std::to_string(shape::rank(xShapeInfoH) - dimensionLength);
      errorMsg += ", but got output rank: " + std::to_string(shape::rank(zShapeInfoH));
      THROW_EXCEPTION(errorMsg.c_str());
    }

    std::vector<sd::LongType> *dims = (zLength != 1) ?
                                  sd::ShapeUtils::evalDimsForReduceOp(shape::rank(xShapeInfoH), &dimensions) :
                                  new std::vector<sd::LongType>();

    NativeOpExecutioner::execReduceLong(nullptr, opNum,
                                        xBuffer, xShapeInfoH, xSpecialBuffer, xShapeInfoD,
                                        extraParams,
                                        zBuffer, zShapeInfoH, nullptr, nullptr,
                                        dims->data(), dims->size());

    delete dims;

    checkAndCleanupCaches();

  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    // Validate input pointers to prevent segfault
    if (x == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: input array x is null");
      return;
    }
    if (z == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: output array z is null");
      return;
    }
    if (dimension == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: dimension array is null");
      return;
    }

    // If we validate first (call shapeInfo()), then cache later (call shapeInfo() again),
    // the pointer could become invalid between the two calls, causing SIGSEGV.
    // By caching once and validating the cached value, we ensure consistency.
    const sd::LongType *xShapeInfoH = x->shapeInfo();
    const sd::LongType *xShapeInfoD = x->specialShapeInfo();
    void *xBuffer = x->buffer();
    void *xSpecialBuffer = x->specialBuffer();

    void *zBuffer = z->buffer();
    const sd::LongType *zShapeInfoH = z->shapeInfo();
    const sd::LongType *zShapeInfoD = z->specialShapeInfo();
    const sd::LongType zLength = z->lengthOf();

    if (xShapeInfoH == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: input array x has null shapeInfo");
      return;
    }
    if (zShapeInfoH == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: output array z has null shapeInfo");
      return;
    }

    void *dimensionBuffer = dimension->buffer();
    sd::DataBuffer *dimensionDb = dimension->getDataBuffer();
    if (dimensionBuffer == nullptr || dimensionDb == nullptr) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("execReduceLong2: dimension array has null buffer");
      return;
    }

    const sd::LongType xRank = shape::rank(xShapeInfoH);
    const sd::DataType dimType = dimension->dataType();
    if (dimType != sd::DataType::INT32 && dimType != sd::DataType::INT64) {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      std::string err = "execReduceLong2: unsupported dimension buffer data type: ";
      err += sd::DataTypeUtils::asString(dimType);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(err.c_str());
      return;
    }

    const sd::LongType dimensionLength =
        static_cast<sd::LongType>(dimensionDb->getLenInBytes() / sd::DataTypeUtils::sizeOf(dimType));

    // Extract dimension values directly from the raw buffer. Do not rely on dimension->shapeInfo()
    // because some callers mutate or free the dimension shape buffer once the NDArray is created.
    std::vector<sd::LongType> dimensions(dimensionLength);
    if (dimensionLength > 0) {
      if (dimType == sd::DataType::INT32) {
        auto dimensionData = reinterpret_cast<int *>(dimensionBuffer);
        for (sd::LongType i = 0; i < dimensionLength; i++) {
          sd::LongType curr = static_cast<sd::LongType>(dimensionData[i]);
          if (curr < 0) {
            curr += xRank;
          }
          dimensions[i] = curr;
        }
      } else {
        auto dimensionData = reinterpret_cast<sd::LongType *>(dimensionBuffer);
        for (sd::LongType i = 0; i < dimensionLength; i++) {
          sd::LongType curr = dimensionData[i];
          if (curr < 0) {
            curr += xRank;
          }
          dimensions[i] = curr;
        }
      }
    }

    // Validate output shape matches expected dimensions after reduction
    // If ranks don't match, this indicates a shape mismatch from the calling layer
    // DO NOT attempt to reshape - the buffer and shape must match
    if (shape::rank(xShapeInfoH) - dimensionLength != shape::rank(zShapeInfoH) && zLength != 1) {
      std::string errorMsg = "execReduceLong2: Output shape rank mismatch. ";
      errorMsg += "Input rank: " + std::to_string(shape::rank(xShapeInfoH));
      errorMsg += ", reduction dimensions: " + std::to_string(dimensionLength);
      errorMsg += ", expected output rank: " + std::to_string(shape::rank(xShapeInfoH) - dimensionLength);
      errorMsg += ", but got output rank: " + std::to_string(shape::rank(zShapeInfoH));
      THROW_EXCEPTION(errorMsg.c_str());
    }

    std::vector<sd::LongType> *dims = (zLength != 1) ?
                                  sd::ShapeUtils::evalDimsForReduceOp(shape::rank(xShapeInfoH), &dimensions) :
                                  new std::vector<sd::LongType>();

    NativeOpExecutioner::execReduceLong(nullptr, opNum,
                                        xBuffer, xShapeInfoH, xSpecialBuffer, xShapeInfoD,
                                        extraParams,
                                        zBuffer, zShapeInfoH, nullptr, nullptr,
                                        dims->data(), dims->size());

    delete dims;

    checkAndCleanupCaches();

  #endif
}


void execReduce3Tad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams,OpaqueNDArray y,
                    OpaqueNDArray z, OpaqueNDArray dimension) {
  #ifdef __cpp_exceptions
  try {
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    auto hTADShapeInfo = tadPack->primaryShapeInfo();
    auto hTADOffsets = tadPack->primaryOffsets();

    NativeOpExecutioner::execReduce3TAD(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                        hTADShapeInfo, hTADOffsets, nullptr, nullptr);

    checkAndCleanupCaches();

  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    auto hTADShapeInfo = tadPack->primaryShapeInfo();
    auto hTADOffsets = tadPack->primaryOffsets();

    NativeOpExecutioner::execReduce3TAD(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                        hTADShapeInfo, hTADOffsets, nullptr, nullptr);

    checkAndCleanupCaches();

  #endif
}

void execScalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z,
                OpaqueNDArray scalar, void *extraParams) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execScalar(nullptr, opNum,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                    extraParams);
    checkAndCleanupCaches();
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execScalar(nullptr, opNum,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                    extraParams);
    checkAndCleanupCaches();
  #endif
}

void execScalarBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z,
                    OpaqueNDArray scalar, void *extraParams) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execScalarBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                        extraParams);
    checkAndCleanupCaches();
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execScalarBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                        extraParams);
    checkAndCleanupCaches();
  #endif
}

void execSummaryStatsScalar(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,
                            void *extraParams,
                            OpaqueNDArray z,  bool biasCorrected) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execSummaryStatsScalar(nullptr, opNum,
                                                x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                                extraParams,
                                                z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                                biasCorrected);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execSummaryStatsScalar(nullptr, opNum,
                                                x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                                extraParams,
                                                z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                                biasCorrected);
  #endif
}

void execSummaryStats(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,
                      OpaqueNDArray z, void *extraParams, bool biasCorrected) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execSummaryStats(nullptr, opNum,
                                          x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                          extraParams,
                                          z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                          biasCorrected);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execSummaryStats(nullptr, opNum,
                                          x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                          extraParams,
                                          z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                          biasCorrected);
  #endif
}

void execSummaryStatsTad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,
                         void *extraParams,OpaqueNDArray z, OpaqueNDArray dimension,
                         bool biasCorrected) {
  #ifdef __cpp_exceptions
  try {
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->shapeOf(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execSummaryStats(nullptr, opNum,
                                          x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                          extraParams,
                                          z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                          dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                          tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                          biasCorrected);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->shapeOf(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execSummaryStats(nullptr, opNum,
                                          x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                          extraParams,
                                          z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                          dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                          tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                          biasCorrected);
  #endif
}

void execTransformFloat(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,  void *extraParams,OpaqueNDArray z) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execTransformFloat(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                            x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                            z->specialShapeInfo(), extraParams);
    checkAndCleanupCaches();
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execTransformFloat(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                            x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                            z->specialShapeInfo(), extraParams);
    checkAndCleanupCaches();
  #endif
}

void execTransformSame(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,void *extraParams, OpaqueNDArray z) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execTransformSame(nullptr, opNum,
                                           x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                           z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                           extraParams, nullptr, nullptr);
    checkAndCleanupCaches();
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execTransformSame(nullptr, opNum,
                                           x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                           z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                           extraParams, nullptr, nullptr);
    checkAndCleanupCaches();
  #endif
}

void execTransformBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,void *extraParams, OpaqueNDArray z) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execTransformBool(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                           x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                           z->specialShapeInfo(), extraParams);
    checkAndCleanupCaches();
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execTransformBool(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                           x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                           z->specialShapeInfo(), extraParams);
    checkAndCleanupCaches();
  #endif
}

void execTransformAny(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,void *extraParams, OpaqueNDArray z) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execTransformAny(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                          x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                          z->specialShapeInfo(), extraParams, false);
    checkAndCleanupCaches();
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execTransformAny(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                          x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                          z->specialShapeInfo(), extraParams, false);
    checkAndCleanupCaches();
  #endif
}

void execTransformStrict(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x,void *extraParams, OpaqueNDArray z) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execTransformStrict(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                             x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                             z->specialShapeInfo(), extraParams);
    checkAndCleanupCaches();
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execTransformStrict(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                             x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                             z->specialShapeInfo(), extraParams);
    checkAndCleanupCaches();
  #endif
}

void execReduce3All(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, OpaqueNDArray dimension, void *extraParams) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execReduce3All(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                        nullptr, nullptr, nullptr, nullptr);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execReduce3All(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                        nullptr, nullptr, nullptr, nullptr);
  #endif
}

void execRandom(sd::Pointer *extraPointers, int opNum, sd::Pointer state, OpaqueNDArray z,
                void *extraArguments) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  #endif
}

void execRandom3(sd::Pointer *extraPointers, int opNum, sd::Pointer state, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z,
                 void *extraArguments) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  #endif
}

void execScalarTad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z,
                   OpaqueNDArray scalar,void *extraParams, OpaqueNDArray dimension) {
  #ifdef __cpp_exceptions
  try {
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execScalar(nullptr, opNum,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    extraParams,
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                    dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                    tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                    tadPack->primaryShapeInfo(), tadPack->primaryOffsets());
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execScalar(nullptr, opNum,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    extraParams,
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                    dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                    tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                    tadPack->primaryShapeInfo(), tadPack->primaryOffsets());
  #endif
}

void execScalarBoolTad(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z,
                       OpaqueNDArray scalar, void *extraParams,OpaqueNDArray dimension) {
  #ifdef __cpp_exceptions
  try {
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execScalarBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                        dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                        tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                        tadPack->primaryShapeInfo(), tadPack->primaryOffsets());
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execScalarBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                        dimension->bufferAsT<sd::LongType>(), dimension->lengthOf(),
                                        tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                        tadPack->primaryShapeInfo(), tadPack->primaryOffsets());
  #endif
}


void execPairwiseTransformBool(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y,
                               void *extraParams,OpaqueNDArray z) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execPairwiseBoolTransform(nullptr, opNum,
                                                   x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                                   y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                                   z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                                   extraParams);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execPairwiseBoolTransform(nullptr, opNum,
                                                   x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                                   y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                                   z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                                   extraParams);
  #endif
}




void execRandom2(sd::Pointer *extraPointers, int opNum, sd::Pointer state,
                 OpaqueNDArray x, OpaqueNDArray z, void *extraArguments) {
  #ifdef __cpp_exceptions
  try {
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  } catch (std::exception &e) {
   sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
   sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
  #else
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  #endif
}



int binaryLevel() {
#ifdef CPU_FEATURES

  #if defined(F_X64)
  return 1;
#elif defined(F_AVX2)
  return 2;
#elif defined(F_AVX512)
  return 3;
#else
  return 0;
#endif

#else
  return 0;
#endif
}

int optimalLevel() {
#ifdef CPU_FEATURES
  auto features = cpu_features::GetX86Info().features;

  if (features.avx && features.avx2 && features.avx512f && features.avx512vl && features.avx512bw &&
      features.avx512dq && features.avx512cd)
    return 3;
  else if (features.avx && features.avx2)
    return 2;
  else
    return 1;

#else
  return 0;
#endif
}

bool isMinimalRequirementsMet() {
#ifdef CPU_FEATURES
  auto features = cpu_features::GetX86Info().features;

#if defined(F_X64)
  return true;
#elif defined(F_AVX2)
  return features.avx && features.avx2;
#elif defined(F_AVX512)
  // we're optimizing for skylake-avx512 features, so we'll check those out
  return features.avx && features.avx2 && features.avx512f && features.avx512vl && features.avx512bw &&
         features.avx512dq && features.avx512cd;
#else
  return true;
#endif

#else
  return true;
#endif
}

bool isOptimalRequirementsMet() {
#ifdef CPU_FEATURES
  auto b = ::binaryLevel();
  auto o = ::optimalLevel();

  if (b == o)
    return true;
  else
    return false;
#else
  return true;
#endif
}


template <typename T>
void _printHostBuffer(OpaqueDataBuffer *buffer, sd::LongType offset) {
  auto xType = buffer->dataBuffer()->getDataType();
  sd::LongType len = buffer->dataBuffer()->getNumElements();
  auto buff = buffer->dataBuffer()->template primaryAsT<T>();
  sd_printf("Data type %s: ", sd::DataTypeUtils::asString(xType).c_str());
  sd_printf("Host buffer: ",0);
  for(int i = offset; i < len; i++) {
    sd_printf("%f ",(double) buff[i]);
  }

  sd_printf("\n",0);
}
void printDeviceBuffer(OpaqueDataBuffer *buffer)  {
  printDeviceBuffer(buffer, 0);
}
void printDeviceBuffer(OpaqueDataBuffer *buffer, sd::LongType offset) {
  if(buffer->special() != nullptr) {
    sd_printf("Device sd::Pointer address: %d\n", buffer->special());
  } else {
    sd_printf("Device sd::Pointer address: none\n",0);
  }

  if(buffer->primary() != nullptr) {
    sd_printf("Host sd::Pointer address: %d\n", buffer->primary());
  } else  {
    sd_printf("Host sd::Pointer address: none\n",0);
  }

  auto xType = buffer->dataBuffer()->getDataType();
  BUILD_SINGLE_SELECTOR(xType, _printHostBuffer,(buffer,offset),SD_COMMON_TYPES);


}




BUILD_SINGLE_TEMPLATE( void pullRowsGeneric,
                      (OpaqueNDArray, OpaqueNDArray, const int, OpaqueNDArray, sd::LongType),
                      SD_COMMON_TYPES);



BUILD_SINGLE_TEMPLATE( void tearGeneric,
                      (void *, sd::LongType const *, sd::Pointer *, sd::LongType const *, sd::LongType const *,sd::LongType const *),
                      SD_COMMON_TYPES);

BUILD_SINGLE_TEMPLATE( void shuffleGeneric,
                      (OpaqueNDArrayArr, OpaqueNDArrayArr, int, int *,sd::LongType *, sd::LongType),
                      SD_COMMON_TYPES);
