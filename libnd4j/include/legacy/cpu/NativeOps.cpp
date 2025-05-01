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
#include <sys/types.h>

#include <execution/Threads.h>
#include <graph/Context.h>
#include <graph/ResultWrapper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>

#include <ops/declarable/OpRegistrator.h>
#include <ops/specials.h>
#include <system/Environment.h>
#ifdef CPU_FEATURES
#include <cpuinfo_x86.h>
#endif

#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>
using namespace sd;



//these are mainly for cuda
sd::Pointer lcScalarPointer(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcReductionPointer(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcAllocationPointer(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcExecutionStream(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcCopyStream(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcBlasHandle(OpaqueLaunchContext lc) { return nullptr; }

sd::Pointer lcSolverHandle(OpaqueLaunchContext lc) { return nullptr; }


void execBroadcastBool(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                       NDArray *z,void *extraParams, NDArray *dimension) {
  try {
    auto tadPackX = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                      dimension->bufferAsT<sd::LongType>(),
                                                                      dimension->lengthOf());
    auto tadPackZ = ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(),
                                                                      dimension->bufferAsT<sd::LongType>(),
                                                                      dimension->lengthOf());

    auto hTADShapeInfo = tadPackX->primaryShapeInfo();
    auto hTADOffsets = tadPackX->primaryOffsets();
    auto hTADShapeInfoZ = tadPackZ->primaryShapeInfo();
    auto hTADOffsetsZ = tadPackZ->primaryOffsets();

    NDArray::prepareSpecialUse({z}, {x, y});

    NativeOpExecutioner::execBroadcastBool(nullptr, opNum,
                                           x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                           y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                           z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                           extraParams,
                                           dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                           hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);

    NDArray::registerSpecialUse({z}, {x, y});

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
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
void execReduce3(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams,OpaqueNDArray y, OpaqueNDArray z) {
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
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
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
void execReduce3Scalar(Pointer *extraPointers, int opNum, OpaqueNDArray x,  void *extraParams ,OpaqueNDArray y, OpaqueNDArray z) {
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
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


bool isBlasVersionMatches(int major, int minor, int build) { return true; }




/**
 * This is dummy method for JNI compatibility
 * Since we'll use this from java, jni compiler would like to have method no matter what.
 */
void initializeDevicesAndFunctions() {}

void initializeFunctions(Pointer *functions) { BlasHelper::getInstance().initializeFunctions(functions); }

/**
 * This method acquires memory chunk of requested size on host side
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param flags optional parameter
 */
Pointer mallocHost(LongType memorySize, int flags) {
#if defined(SD_ALIGNED_ALLOC)
  return static_cast<Pointer *>(
      aligned_alloc(SD_DESIRED_ALIGNMENT, (memorySize + SD_DESIRED_ALIGNMENT - 1) & (-SD_DESIRED_ALIGNMENT)));
#else
  return reinterpret_cast<Pointer>(new int8_t[memorySize]);
#endif
}

/**
 * This method acquires memory chunk of requested size on specified device
 *
 * PLEASE NOTE: This method is NOT supported and has NO effect in CPU-based backend.
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
 * @param flags optional parameter
 */
Pointer mallocDevice(LongType memorySize, int deviceId, int flags) {
  // not supported
  return 0L;
}

/**
 * This method releases previously allocated host memory space
 *
 * @param pointer pointer that'll be freed
 */
int freeHost(Pointer pointer) {
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
 * @param pointer pointer that'll be freed
 * @param ptrToDeviceId pointer to deviceId.
 */
int freeDevice(Pointer pointer, int deviceId) {
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

Pointer createContext() { return 0L; }

Pointer createStream() { return 0L; }

Pointer createEvent() { return 0L; }
int getDeviceBlockThreshold(int deviceId) { return 0; }
int getDeviceMajor(int deviceId) { return 0; }
int getDeviceSharedThreshold(int deviceId) {return 0; }
int getDeviceMinor(int deviceId) { return 0; }
int getDeviceId(void* deviceId) { return 0; }

int registerEvent(Pointer event, Pointer stream) { return 0L; }

int setDevice(int deviceId) { return 0L; }

LongType getDeviceFreeMemory(int deviceId) { return 0L; }

LongType getDeviceFreeMemoryDefault() { return 0L; }

LongType getDeviceTotalMemory(int deviceId) { return 0L; }

int memcpySync(Pointer dst, Pointer src, LongType size, int flags, Pointer reserved) { return 0L; }

int memcpyAsync(Pointer dst, Pointer src, LongType size, int flags, Pointer reserved) { return 0L; }

int memsetSync(Pointer dst, int value, LongType size, int flags, Pointer reserved) { return 0L; }

int memsetAsync(Pointer dst, int value, LongType size, int flags, Pointer reserved) { return 0L; }

int destroyEvent(Pointer event) { return 0L; }

int streamSynchronize(Pointer stream) { return 0L; }

int eventSynchronize(Pointer event) { return 0L; }

int getAvailableDevices() { return 0L; }

void enableDebugMode(bool reallyEnable) { Environment::getInstance().setDebug(reallyEnable); }

void enableVerboseMode(bool reallyEnable) { Environment::getInstance().setVerbose(reallyEnable); }

void setGridLimit(int gridSize) {
  // no-op
}

void prescanArrayRecursive(sd::Pointer *extras, int *dZ, int *dX, int numElements, int level) {
  THROW_EXCEPTION("prescanArrayRecursive Not implemented");
}



int memcpyConstantAsync(LongType dst, Pointer src, LongType size, int flags, Pointer reserved) {
  // no-op
  return 0L;
}

Pointer getConstantSpace() {
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
  int _threads = math::sd_max<int>(1, elementsPerThread);
  _threads = math::sd_min<int>(_threads, Environment::getInstance().maxThreads());

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

      for (LongType i = 0; i < tadLength; i++) {
        hZ[zOffset + i] = hX[xOffset + i];
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, n, 1, _threads);
}
void tryPointer(sd::Pointer extra, sd::Pointer p, int len) {
  try {
    auto buf = reinterpret_cast<int8_t *>(p);
    int cnt = 0;
    for (int i = 0; i < len; i++) cnt += buf[cnt];
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void pullRows(sd::Pointer *extraPointers,
              OpaqueNDArray x,
              OpaqueNDArray z,
              sd::LongType n,
              OpaqueNDArray indexes,
              sd::LongType dimension) {
  try {
    auto xType = ArrayOptions::dataType(x->shapeInfo());

    BUILD_SINGLE_SELECTOR(xType, pullRowsGeneric, (x, z, n, indexes, dimension), SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
template <typename T>
void tearGeneric(void *vx, LongType const *hXShapeInfo, Pointer *targets, LongType const *hZShapeInfo,
                 LongType const *tadShapeInfo, LongType const *tadOffsets) {
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

      for (LongType j = 0; j < tadLength; j++) {
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

void tear(Pointer *extraPointers, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
          LongType const *dXShapeInfo, Pointer *targets, LongType const *hZShapeInfo,
          LongType const *tadShapeInfo, LongType const *tadOffsets) {
  try {
    auto xType = ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, tearGeneric,
                          (dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, targets, hZShapeInfo, tadShapeInfo, tadOffsets),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
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
        for (LongType r = 0; r < xLength; r++) {
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

          math::sd_swap<T>(hX2[xOffset], hZ2[zOffset]);
        }
      } else {


        sd::LongType tadShapeInfoRank = shape::rank(tadOnlyShapeInfoX);
        sd::LongType *tadShapeInfoShape = shape::shapeOf(tadOnlyShapeInfoX);
        sd::LongType *tadShapeInfoStride = shape::stride(tadOnlyShapeInfoX);

        sd::LongType zTadShapeInfoRank = shape::rank(tadOnlyShapeInfoZ);
        sd::LongType *zTadShapeInfoShape = shape::shapeOf(tadOnlyShapeInfoZ);
        sd::LongType *zTadShapeInfoStride = shape::stride(tadOnlyShapeInfoZ);
        for (LongType r = 0; r < numTads; r++) {
          if (shuffleMap[r] < 0) continue;

          auto oldOffsetX = tadOffsetsX[r];
          auto newOffsetZ = tadOffsetsZ[shuffleMap[r]];

          auto rX = hX2 + oldOffsetX;
          auto rZ = hZ2 + newOffsetZ;

          for (LongType i = 0; i < tadLength; i++) {
            sd::LongType xCoords[SD_MAX_RANK];
            sd::LongType zCoords[SD_MAX_RANK];
            sd::LongType xOffset;
            sd::LongType zOffset;

            INDEX2COORDS(i, tadShapeInfoRank, tadShapeInfoShape, xCoords);
            COORDS2INDEX(tadShapeInfoRank,tadShapeInfoStride, xCoords, xOffset);
            INDEX2COORDS(i, zTadShapeInfoRank,zTadShapeInfoShape, zCoords);
            COORDS2INDEX(zTadShapeInfoRank, zTadShapeInfoStride, zCoords, zOffset);

            math::sd_swap<T>(rX[xOffset], rZ[zOffset]);
          }
        }
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, N);
}

void shuffle(Pointer *extras,
             OpaqueNDArrayArr x,
             OpaqueNDArrayArr z,
             int N,
             OpaqueNDArray dimension,
             OpaqueNDArray shuffleMap) {
  try {
    auto dimensionData = reinterpret_cast<LongType *>(dimension->buffer());
    auto dimensionLength = shape::length(dimension->shapeInfo());

    auto xType = ArrayOptions::dataType(x[0]->shapeInfo());

    BUILD_SINGLE_SELECTOR(xType, shuffleGeneric, (x, z, N, reinterpret_cast<int *>(shuffleMap->buffer()), dimensionData, dimensionLength), SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

bool isExperimentalEnabled() { return Environment::getInstance().isExperimentalBuild(); }

void setOmpMinThreads(int threads) {
  // TODO: to be implemented
}

int getDevice() { return 0; }



char *name;
bool nameSet = false;

const char *getDeviceName(int deviceId) {
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
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }

  return name;
}



void execRandom(Pointer *extraPointers, int opNum, Pointer state, OpaqueDataBuffer *dbZ,
                const LongType *hZShapeInfo, const LongType *dZShapeInfo, void *extraArguments) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo,
                                    extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom3(Pointer *extraPointers, int opNum, Pointer state, OpaqueDataBuffer *dbX,
                 const LongType *hXShapeInfo, const LongType *dXShapeInfo, OpaqueDataBuffer *dbY,
                 const LongType *hYShapeInfo, const LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                 const LongType *hZShapeInfo, const LongType *dZShapeInfo, void *extraArguments) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                    dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ != nullptr ? dbZ->primary() : nullptr,
                                    hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom2(Pointer *extraPointers, int opNum, Pointer state, OpaqueDataBuffer *dbX,
                 const LongType *hXShapeInfo, const LongType *dXShapeInfo, OpaqueDataBuffer *dbZ,
                 const LongType *hZShapeInfo, const LongType *dZShapeInfo, void *extraArguments) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                    dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

Pointer initRandom(Pointer *extraPointers, long seed, long bufferSize, Pointer ptrToBuffer) {
  try {
    auto generator = new sd::graph::RandomGenerator(seed, seed);

    return (Pointer)generator;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }
}

void refreshBuffer(Pointer *extraPointers, long seed, Pointer ptrRandom) {
  auto generator = reinterpret_cast<sd::graph::RandomGenerator *>(ptrRandom);

  generator->setStates(seed);
}

void reSeedBuffer(Pointer *extraPointers, long seed, Pointer ptrRandom) {
  auto generator = reinterpret_cast<sd::graph::RandomGenerator *>(ptrRandom);

  generator->setStates(seed);
}

void destroyRandom(Pointer ptrBuffer) {
  auto buffer = reinterpret_cast<sd::graph::RandomGenerator *>(ptrBuffer);
  delete buffer;
}

/**
 * Return the length of a shape buffer
 * based on the pointer
 * @param buffer  the buffer pointer to check
 * @return
 */
int lengthForShapeBufferPointer(Pointer buffer) {
  auto shapeBuffer = reinterpret_cast<LongType *>(buffer);
  return shape::shapeInfoLength(shape::rank(shapeBuffer));
}

/**
 * The pointer to get the address for
 *
 * @param address the address to get the pointer
 * @return the pointer for the given address
 */

Pointer pointerForAddress(LongType address) { return reinterpret_cast<Pointer>(address); }

void sort(Pointer *extraPointers, OpaqueNDArray x, bool descending) {
  try {
    NativeOpExecutioner::execSort(x, descending);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTad(Pointer *extraPointers, OpaqueNDArray  x,
             LongType *dimension, LongType dimensionLength,
             LongType *tadShapeInfo,  LongType *tadOffsets, bool descending) {
  try {
    NativeOpExecutioner::execSort(x, dimension, dimensionLength, descending);

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}



Status execCustomOp2(Pointer *extraPointers, LongType hash, OpaqueContext *context) {
  try {
    auto op = ops::OpRegistrator::getInstance().getOperation(hash);
    return op->execute(context);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
    return Status::VALIDATION;
  }
}



void setShapeBuffer(LongType *inputShapeData,DataType dt,LongType *bufferToSet,char order,int elementWiseStride,bool isEmpty,bool isView) {
  if(inputShapeData == nullptr)
    THROW_EXCEPTION("setShapeBuffer: inputShapeData is null");

  if(bufferToSet == nullptr)
    THROW_EXCEPTION("setShapeBuffer: bufferToSet is null");
  LongType  rank = inputShapeData[0];
  if(rank > SD_MAX_RANK || rank < 0)
    THROW_EXCEPTION("Invalid rank for shape buffer.");
  std::vector<LongType> shape;
  std::vector<LongType> strides;
  //shape, stride, data type
  for(LongType i = 1; i < rank * 2 + 1; i++) {
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

  ArrayOptions::setDataType(bufferToSet,dt);
  if(isView) {
    ArrayOptions::toggleIsView(bufferToSet);
  }
  if(!ArrayOptions::isEmpty(inputShapeData) && isEmpty) {
    ArrayOptions::toggleIsEmpty(bufferToSet);
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


void sortByKey(Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y,bool descending) {
  try {
    auto xType = x->dataType();
    auto yType = y->dataType();

    BUILD_DOUBLE_SELECTOR(xType, yType, DoubleMethods, ::sortByKey(x, y, descending),
                          SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortByValue(Pointer *extraPointers, OpaqueNDArray x,OpaqueNDArray y, bool descending) {
  try {
    auto xType = x->dataType();
    auto yType = y->dataType();

    BUILD_DOUBLE_SELECTOR(xType, yType, DoubleMethods, ::sortByValue(x, y, descending),
                          SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTadByKey(Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray y,
                  OpaqueNDArray dimension, bool descending) {
  try {
    auto xType = x->dataType();
    auto yType = y->dataType();
    auto dimensionLength = dimension->lengthOf();
    BUILD_DOUBLE_SELECTOR(xType, yType, DoubleMethods, ::sortTadByValue(x, y, dimension, descending), SD_NUMERIC_TYPES,
                          SD_NUMERIC_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
void sortTadByValue(Pointer *extraPointers, OpaqueNDArray x,
                    OpaqueNDArray y,OpaqueNDArray dimension, bool descending) {
  try {
    auto xType = x->dataType();
    auto yType = y->dataType();
    auto dimensionLength = dimension->lengthOf();
    BUILD_DOUBLE_SELECTOR(xType, yType, DoubleMethods, ::sortTadByValue(x, y, dimension, descending), SD_NUMERIC_TYPES,
                          SD_NUMERIC_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void execIndexReduceScalar(Pointer *extraPointers, int opNum, NDArray *x,void *extraParams,
                           NDArray *z) {
  try {
    NativeOpExecutioner::execIndexReduceScalar(nullptr, opNum,
                                               x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                               extraParams,
                                               z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execIndexReduce(Pointer *extraPointers, int opNum, NDArray *x,
                     void *extraParams,
                     NDArray *z, NDArray *dimension
) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    auto hTADShapeInfo = tadPack->primaryShapeInfo();
    auto hTADOffsets = tadPack->primaryOffsets();

    NativeOpExecutioner::execIndexReduce(nullptr, opNum,
                                         x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                         extraParams,
                                         z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                         dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                         hTADShapeInfo, hTADOffsets);

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execBroadcast(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                   NDArray *z,void *extraInfo, NDArray *dimension) {
  try {
    auto tadPackX = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                      dimension->bufferAsT<sd::LongType>(),
                                                                      dimension->lengthOf());
    auto tadPackZ = ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(),
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
                                       dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                       hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execPairwiseTransform(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                           NDArray *z, void *extraParams) {
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
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceFloat(Pointer *extraPointers, int opNum, NDArray *x,
                     void *extraParams, NDArray *z) {
  try {
    NativeOpExecutioner::execReduceFloatScalar(nullptr, opNum,
                                               x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                               extraParams,
                                               z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceSame(Pointer *extraPointers, int opNum, NDArray *x,
                    void *extraParams,NDArray *z) {
  try {
    NativeOpExecutioner::execReduceSameScalar(nullptr, opNum,
                                              x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                              extraParams,
                                              z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceBool(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams,
                    OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->shapeOf(),
                                                                     dimension->lengthOf());

    auto hTADShapeInfo = tadPack->primaryShapeInfo();
    auto hTADOffsets = tadPack->primaryOffsets();

    NativeOpExecutioner::execReduceBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(),
                                        x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), z->shapeInfo(),
                                        z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<LongType>(), dimension->lengthOf());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceLong(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams,
                    OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    auto hTADShapeInfo = tadPack->primaryShapeInfo();
    auto hTADOffsets = tadPack->primaryOffsets();

    NativeOpExecutioner::execReduceLong(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(),
                                        x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), z->shapeInfo(),
                                        z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<LongType>(), dimension->lengthOf());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceFloat2(Pointer *extraPointers, int opNum, NDArray *x,void *extraParams,
                      NDArray *z, NDArray *dimension) {
  try {
    std::vector<LongType> dimensions(dimension->lengthOf());
    for(LongType i = 0; i < dimension->lengthOf(); i++) {
      dimensions[i] = dimension->e<LongType>(i);
    }

    const LongType *zShapeInfoH = z->shapeInfo();
    const LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo()) && z->lengthOf() != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims = (z->lengthOf() != 1) ?
                                  ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<LongType>();

    NativeOpExecutioner::execReduceFloat(nullptr, opNum,
                                         x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                         extraParams,
                                         z->buffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                         dims->data(), dims->size());

    delete dims;

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceBool2(Pointer *extraPointers, int opNum, NDArray *x,
                     void *extraParams,
                     NDArray *z, NDArray *dimension) {
  try {
    std::vector<LongType> dimensions(dimension->lengthOf());
    for(LongType i = 0; i < dimension->lengthOf(); i++) {
      dimensions[i] = dimension->e<LongType>(i);
    }

    const LongType *zShapeInfoH = z->shapeInfo();
    const LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo())) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims = (z->lengthOf() != 1) ?
                                  ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<LongType>();

    NativeOpExecutioner::execReduceBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());

    delete dims;

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceSame2(Pointer *extraPointers, int opNum,
                     NDArray *x,void *extraParams,
                     NDArray *z, NDArray *dimension) {
  try {
    std::vector<LongType> dimensions(dimension->lengthOf());
    for(LongType i = 0; i < dimension->lengthOf(); i++) {
      dimensions[i] = dimension->e<LongType>(i);
    }

    const LongType *zShapeInfoH = z->shapeInfo();
    const LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo()) && z->lengthOf() != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims = (z->lengthOf() != 1) ?
                                  ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<LongType>();

    NativeOpExecutioner::execReduceSame(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());

    delete dims;

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceLong2(Pointer *extraPointers, int opNum, NDArray *x,
                     void *extraParams,
                     NDArray *z, NDArray *dimension) {
  try {
    std::vector<LongType> dimensions(dimension->lengthOf());
    for(LongType i = 0; i < dimension->lengthOf(); i++) {
      dimensions[i] = dimension->e<LongType>(i);
    }

    const LongType *zShapeInfoH = z->shapeInfo();
    const LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo()) && z->lengthOf() != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims = (z->lengthOf() != 1) ?
                                  ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<LongType>();

    NativeOpExecutioner::execReduceLong(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());

    delete dims;

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void execReduce3Tad(Pointer *extraPointers, int opNum, NDArray *x, void *extraParams,NDArray *y,
                    NDArray *z, NDArray *dimension) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    auto hTADShapeInfo = tadPack->primaryShapeInfo();
    auto hTADOffsets = tadPack->primaryOffsets();

    NativeOpExecutioner::execReduce3TAD(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                        hTADShapeInfo, hTADOffsets, nullptr, nullptr);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalar(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                NDArray *scalar, void *extraParams) {
  try {
    NativeOpExecutioner::execScalar(nullptr, opNum,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                    extraParams);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalarBool(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                    NDArray *scalar, void *extraParams) {
  try {
    NativeOpExecutioner::execScalarBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                        extraParams);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execSummaryStatsScalar(Pointer *extraPointers, int opNum, NDArray *x,
                            void *extraParams,
                            NDArray *z,  bool biasCorrected) {
  try {
    NativeOpExecutioner::execSummaryStatsScalar(nullptr, opNum,
                                                x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                                extraParams,
                                                z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                                biasCorrected);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execSummaryStats(Pointer *extraPointers, int opNum, NDArray *x,
                      NDArray *z, void *extraParams, bool biasCorrected) {
  try {
    NativeOpExecutioner::execSummaryStats(nullptr, opNum,
                                          x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                          extraParams,
                                          z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                          biasCorrected);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execSummaryStatsTad(Pointer *extraPointers, int opNum, NDArray *x,
                         void *extraParams,NDArray *z, NDArray *dimension,
                         bool biasCorrected) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->shapeOf(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execSummaryStats(nullptr, opNum,
                                          x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                          extraParams,
                                          z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                          dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                          tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                          biasCorrected);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformFloat(Pointer *extraPointers, int opNum, NDArray *x,  void *extraParams,NDArray *z) {
  try {
    NativeOpExecutioner::execTransformFloat(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                            x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                            z->specialShapeInfo(), extraParams);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformSame(Pointer *extraPointers, int opNum, NDArray *x,void *extraParams, NDArray *z) {
  try {
    NativeOpExecutioner::execTransformSame(nullptr, opNum,
                                           x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                           z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                           extraParams, nullptr, nullptr);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformBool(Pointer *extraPointers, int opNum, NDArray *x,void *extraParams, NDArray *z) {
  try {
    NativeOpExecutioner::execTransformBool(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                           x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                           z->specialShapeInfo(), extraParams);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformAny(Pointer *extraPointers, int opNum, NDArray *x,void *extraParams, NDArray *z) {
  try {
    NativeOpExecutioner::execTransformAny(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                          x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                          z->specialShapeInfo(), extraParams, false);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformStrict(Pointer *extraPointers, int opNum, NDArray *x,void *extraParams, NDArray *z) {
  try {
    NativeOpExecutioner::execTransformStrict(nullptr, opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                             x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                             z->specialShapeInfo(), extraParams);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduce3All(sd::Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, OpaqueNDArray dimension, void *extraParams) {
  try {
    NativeOpExecutioner::execReduce3All(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                        nullptr, nullptr, nullptr, nullptr);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom(Pointer *extraPointers, int opNum, Pointer state, NDArray *z,
                void *extraArguments) {
  try {
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom3(Pointer *extraPointers, int opNum, Pointer state, NDArray *x, NDArray *y, NDArray *z,
                 void *extraArguments) {
  try {
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalarTad(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                   NDArray *scalar,void *extraParams, NDArray *dimension) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execScalar(nullptr, opNum,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    extraParams,
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                    dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                    tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                    tadPack->primaryShapeInfo(), tadPack->primaryOffsets());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalarBoolTad(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                       NDArray *scalar, void *extraParams,NDArray *dimension) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->bufferAsT<sd::LongType>(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execScalarBool(nullptr, opNum,
                                        x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                        dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                        tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                        tadPack->primaryShapeInfo(), tadPack->primaryOffsets());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void execPairwiseTransformBool(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                               void *extraParams,NDArray *z) {
  try {
    NativeOpExecutioner::execPairwiseBoolTransform(nullptr, opNum,
                                                   x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                                   y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                                   z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                                   extraParams);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}




void execRandom2(Pointer *extraPointers, int opNum, Pointer state,
                 NDArray *x, NDArray *z, void *extraArguments) {
  try {
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
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
  LongType len = buffer->dataBuffer()->getNumElements();
  auto buff = buffer->dataBuffer()->template primaryAsT<T>();
  sd_printf("Data type %s: ", DataTypeUtils::asString(xType).c_str());
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
    sd_printf("Device pointer address: %d\n", buffer->special());
  } else {
    sd_printf("Device pointer address: none\n",0);
  }

  if(buffer->primary() != nullptr) {
    sd_printf("Host pointer address: %d\n", buffer->primary());
  } else  {
    sd_printf("Host pointer address: none\n",0);
  }

  auto xType = buffer->dataBuffer()->getDataType();
  BUILD_SINGLE_SELECTOR(xType, _printHostBuffer,(buffer,offset),SD_COMMON_TYPES_ALL);


}




BUILD_SINGLE_TEMPLATE(template void pullRowsGeneric,
                      (OpaqueNDArray, OpaqueNDArray, const int, OpaqueNDArray, sd::LongType),
                      SD_COMMON_TYPES);



BUILD_SINGLE_TEMPLATE(template void tearGeneric,
                      (void *, LongType const *, Pointer *, LongType const *, LongType const *,
                          LongType const *),
                      SD_COMMON_TYPES);

BUILD_SINGLE_TEMPLATE(template void shuffleGeneric,
                      (OpaqueNDArrayArr, OpaqueNDArrayArr, int, int *,sd::LongType *, sd::LongType),
                      SD_COMMON_TYPES);
