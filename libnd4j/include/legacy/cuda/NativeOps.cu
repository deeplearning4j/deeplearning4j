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

#include <cuda.h>
#include <exceptions/cuda_exception.h>
#include <exceptions/datatype_exception.h>
#include <execution/AffinityManager.h>

#include <helpers/BlasHelper.h>
#include <helpers/CudaLaunchHelper.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/StringUtils.h>
#include <helpers/threshold.h>
#include <system/Environment.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/NativeOps.h>
#include <loops/reduce_bool.h>
#include <loops/reduce_long.h>
#include <loops/scalar.h>
#include <loops/transform_any.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/OpExecutionLogger.h>
#include <graph/OpContextLifecycleTracker.h>
#include <ops/specials_cuda.h>
#include <system/buffer.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>


#include <curand.h>
#include <helpers/DebugHelper.h>

#include <execution/cuda/LaunchDims.h>
#include <loops/special_kernels.h>

#include "../../array/ShapeList.h"
#include "../../helpers/shape.h"
#include "../../ops/declarable/DeclarableOp.h"
#include "../../system/common.h"
#include "../NativeOpExecutioner.h"
#include "../NativeOps.h"
#include <system/type_boilerplate.h>
#include <loops/special_kernels.h>
#include <system/selective_rendering.h>
#include <execution/LaunchContext.h>
cudaDeviceProp *deviceProperties;
cudaFuncAttributes *funcAttributes = new cudaFuncAttributes[64];
int blockLimit = 128;
int maxThreads = 512;
bool allowedP2P = false;
bool supportedP2P = false;

// TadPack lifetime registry - keeps shared_ptr<TadPack> alive for TadPacks returned to Java
// Without this, when ConstantTadHelper::tadForDimensions() returns shared_ptr<TadPack>,
// but tadOnlyShapeInfo() returns raw TadPack*, the local shared_ptr goes out of scope
// NOTE: TadPack registry is now in NativeOpsHelpers_DataBuffers.cpp (shared between CPU and CUDA)



//note we only include this if we're running gcc linux
//and should not be enabled in default builds.
#if defined(SD_GCC_FUNCTRACE)
#include <cxxabi.h>  // needed  __cxa_demangle
#include <dlfcn.h>   // needed for dladdr

#include "exceptions/backward.hpp"
#include "execution/cuda/LaunchDims.h"


//note this is outside extern C. This is fine.


#endif





int minThreads = 32;




// this method just does type conversion in fancy way
int getDeviceId(sd::Pointer ptrToDeviceId) { return (int)(sd::LongType)ptrToDeviceId; }

// execCustomOp2 moved to NativeOps_customOp.cu for SD_GCC_FUNCTRACE builds


sd::Pointer lcScalarPointer(OpaqueLaunchContext lc) { return lc->getScalarPointer(); }

sd::Pointer lcReductionPointer(OpaqueLaunchContext lc) { return lc->getReductionPointer(); }

sd::Pointer lcAllocationPointer(OpaqueLaunchContext lc) { return lc->getAllocationPointer(); }

sd::Pointer lcExecutionStream(OpaqueLaunchContext lc) { return lc->getCudaStream(); }

sd::Pointer lcCopyStream(OpaqueLaunchContext lc) { return lc->getCudaSpecialStream(); }

sd::Pointer lcBlasHandle(OpaqueLaunchContext lc) { return lc->getCublasHandle(); }

sd::Pointer lcSolverHandle(OpaqueLaunchContext lc) { return lc->getCusolverHandle(); }


/*
 * Basic CUDA constants here: number of blocks per MP
 */
int getDeviceBlockThreshold(int deviceId) {
  int ccMinor = deviceProperties[deviceId].minor;
  int ccMajor = deviceProperties[deviceId].major;

  int blockThreshold = 8;

  if (ccMajor >= 5)
    blockThreshold = 32;
  else if (ccMajor == 3)
    blockThreshold = 16;
  else if (ccMajor < 3)
    blockThreshold = 8;

  return blockThreshold;
}

/*
 * This message returns shared memory threshold value. default overflow ratio is 0.3
 */
int getDeviceSharedThreshold(int deviceId) {
  int ccMinor = deviceProperties[deviceId].minor;
  int ccMajor = deviceProperties[deviceId].major;

  // please note threshold isn't multiple of 32, and that's NOT a mistake

  int shmemThreshold;
  if (ccMajor == 6 && ccMinor == 0)
    shmemThreshold = 65536;
  else if (ccMajor == 6 && ccMinor == 1)
    shmemThreshold = 49152;
  else if (ccMajor == 5 && ccMinor == 2)
    shmemThreshold = 98304;
  else if (ccMajor == 5)
    shmemThreshold = 65536;
  else if (ccMajor == 3 && ccMinor == 7)
    shmemThreshold = 114688;
  else
    shmemThreshold = 49152;

  return shmemThreshold / 0.3;
}

sd::buffer::Buffer<sd::LongType> *createScalarBuffer(cudaStream_t stream) {
  auto scalarShapeInfo = shape::createScalarShapeInfo();
  auto buff = sd::buffer::createBuffer(scalarShapeInfo, shape::shapeInfoLength(2), stream);
  copyDataToGpu(&buff, stream);
  return buff;
}


template <typename T>
SD_KERNEL SD_INLINE void _printBuffers(void* buffer, sd::LongType bufferLength) {
  T * inputBuffer = reinterpret_cast<T *>(buffer);
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid == 0) {
    printf("DEVICE buffer: ");
  }
  const auto step = gridDim.x * blockDim.x;
  for (int t = tid; t < bufferLength; t += step) {
    if(t == 0) {
      printf("DEVICE buffer: ");
    }
    printf(" %f ",(double) inputBuffer[t]);
    if(t == bufferLength - 1) {
      printf("\n");
    }
  }



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
  BUILD_SINGLE_SELECTOR(xType, _printHostBuffer,(buffer,offset),SD_COMMON_TYPES);


}

template <typename T>
void _printDeviceBuffer(OpaqueDataBuffer *buffer) {
  auto xType = buffer->dataBuffer()->getDataType();
  sd::LongType len = buffer->dataBuffer()->getNumElements();
  _printBuffers<T><<<256, 512, 1024>>>(buffer->special(),len);
  cudaDeviceSynchronize();
  sd::DebugHelper::checkGlobalErrorCode("print device buffer(...) failed");


}

void printDeviceBuffer(OpaqueDataBuffer *buffer) {
  auto xType = buffer->dataBuffer()->getDataType();
  sd_printf("Data type %s: ", sd::DataTypeUtils::asString(xType).c_str());

  if(buffer->special() != nullptr) {
    sd_printf("Device pointer address: %d\n", reinterpret_cast<sd::LongType>(buffer->special()));
  } else {
    sd_printf("Device pointer address: none\n",0);
  }
  BUILD_SINGLE_SELECTOR(xType, _printDeviceBuffer,(buffer),SD_COMMON_TYPES);


  if(buffer->primary() != nullptr) {
    sd_printf("Host pointer address: %d\n",  reinterpret_cast<sd::LongType>(buffer->primary()));
  } else  {
    sd_printf("Host pointer address: none\n",0);
  }


}

// Explicit template instantiations for _printDeviceBuffer
BUILD_SINGLE_TEMPLATE(void _printDeviceBuffer, (OpaqueDataBuffer *buffer), SD_COMMON_TYPES);

// Explicit template instantiations for _printHostBuffer
BUILD_SINGLE_TEMPLATE(void _printHostBuffer, (OpaqueDataBuffer *buffer, sd::LongType offset), SD_COMMON_TYPES);



// execPairwiseTransform, execPairwiseTransformBool, execSummaryStatsScalar
// moved to NativeOps_pairwise.cu for SD_GCC_FUNCTRACE builds
// execBroadcastBool, execBroadcast moved to NativeOps_broadcast.cu for SD_GCC_FUNCTRACE builds

// execReduceFloat, execReduceSame, execReduceSame2, execReduceLong, execReduceLong2,
// execReduceBool, execReduceBool2, execReduceFloat2, execIndexReduce, execIndexReduceScalar,
// execTransformSame, execTransformBool, execTransformAny, execTransformStrict, execTransformFloat
// moved to NativeOps_reduce.cu and NativeOps_transform.cu for SD_GCC_FUNCTRACE builds

void checkP2P() {
  int curDevice = 0;

  cudaGetDevice(&curDevice);

  int devCnt = 0;
  cudaGetDeviceCount(&devCnt);

  if (curDevice < 0 && curDevice > devCnt) curDevice = 0;

  bool tempSupport = true;

  if (devCnt > 1) {
    for (int dX = 0; dX < devCnt; dX++) {
      for (int dY = 0; dY < devCnt; dY++) {
        if (dX == dY) continue;

        int canAccess = 0;
        cudaSetDevice(dX);

        cudaDeviceCanAccessPeer(&canAccess, dX, dY);

        if (!canAccess) {
          tempSupport = false;
          break;
        }
      }
    }

    supportedP2P = tempSupport;

    cudaSetDevice(curDevice);
  } else {
    // if we have only 1 device - we say that we support P2P, since all data will be on 1 device
    supportedP2P = true;
  }
}

void enableP2P(bool enable) {
  if (enable == allowedP2P) return;

  int curDevice = 0;

  cudaGetDevice(&curDevice);

  int devCnt = 0;
  cudaGetDeviceCount(&devCnt);

  if (curDevice < 0 && curDevice > devCnt) curDevice = 0;

  if (devCnt > 1) {
    for (int dX = 0; dX < devCnt; dX++) {
      for (int dY = 0; dY < devCnt; dY++) {
        if (dX == dY) continue;

        int canAccess = 0;
        cudaSetDevice(dX);

        cudaDeviceCanAccessPeer(&canAccess, dX, dY);

        if (canAccess) {
          if (enable) {
            cudaDeviceEnablePeerAccess(dY, 0);
          } else {
            cudaDeviceDisablePeerAccess(dY);
          }
        } else {
          if (sd::Environment::getInstance().isVerbose()) printf("Peer access [%i] -> [%i] isn't possible\n", dX, dY);
        }
      }
    }

    cudaSetDevice(curDevice);
  }

  allowedP2P = enable;

  cudaSetDevice(curDevice);
}

bool isP2PAvailable() { return supportedP2P; }

// initializeDevicesAndFunctions moved to NativeOps_utils.cu for SD_GCC_FUNCTRACE builds

/**
 * Initialize the shape cache early to prevent race conditions during static initialization.
 * This ensures ConstantShapeHelper and its internal DirectShapeTrie are fully initialized
 * before any multi-threaded access occurs.
 *
 * Safe to call multiple times - subsequent calls are no-ops.
 */
void initializeShapeCache() {
  sd::ConstantShapeHelper::getInstance();
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

void initializeFunctions(sd::Pointer *functions) { sd::BlasHelper::getInstance().initializeDeviceFunctions(functions);
}


/**
 * This method acquires memory chunk of requested size on host side
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param flags optional parameter
 */
sd::Pointer mallocHost(sd::LongType memorySize, int flags) {
  sd::Pointer pointer;
  // cudaHostAllocMapped |cudaHostAllocPortable
  auto res = cudaHostAlloc(reinterpret_cast<void **>(&pointer), memorySize + 8, cudaHostAllocDefault);
  if (res != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(res);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaHostAlloc failed");
  }

  return reinterpret_cast<int8_t *>(pointer);
}

/**
 * This method acquires memory chunk of requested size on specified device
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
 * @param flags optional parameter
 */
sd::Pointer mallocDevice(sd::LongType memorySize, int deviceId, int flags) {
  sd::Pointer pointer;
  auto res = cudaMalloc(reinterpret_cast<void **>(&pointer), memorySize + 8);
  if (res != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(res);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMalloc failed");
  }

  return reinterpret_cast<int8_t *>(pointer);
}

/**
 * This method releases previously allocated host memory space
 *
 * @param pointer pointer that'll be freed
 */
int freeHost(sd::Pointer pointer) {
/*  auto res = cudaFreeHost(reinterpret_cast<void *>(pointer));
  if (res != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(res);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaFreeHost failed");
  }*/

  return 1L;
}

/**
 * This method releases previously allocated memory space on device
 *
 * @param pointer pointer that'll be freed
 * @param deviceId device where the pointer was allocated
 */
int freeDevice(sd::Pointer pointer, int deviceId) {
  if (pointer == nullptr) {
    return 1L;  // Nothing to free
  }

  // Get current device to restore later
  int currentDevice = 0;
  cudaGetDevice(&currentDevice);

  // Set to the correct device if different
  if (deviceId != currentDevice) {
    cudaError_t setDevErr = cudaSetDevice(deviceId);
    if (setDevErr != cudaSuccess) {
      cudaGetLastError();  // Clear error
      sd_debug("freeDevice: Failed to set device %d, using current device%s\n", deviceId, "");
    }
  }

  cudaError_t syncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess) {
    sd_debug("freeDevice: cudaDeviceSynchronize failed: %s%s\n", cudaGetErrorString(syncErr), "");
    cudaGetLastError();  // Clear error state
  }

  auto res = cudaFree(reinterpret_cast<void *>(pointer));

  // Restore original device if we switched
  if (deviceId != currentDevice) {
    cudaSetDevice(currentDevice);
  }

  // we're intentionally skipping error 1 (cudaErrorInvalidValue for null pointer)
  if (res != 0 && res != 1) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(res);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaFree failed");
  }

  return res == 0 ? 1L : 0L;
}

sd::Pointer createContext() { return 0L; }

sd::Pointer createStream() {
  auto stream = new cudaStream_t();
  auto dZ = cudaStreamCreate(stream);
  if (dZ != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaStreamCreate failed");
  }

  return stream;
}

sd::Pointer createEvent() {
  sd::Pointer nativeEvent = (sd::Pointer)malloc(sizeof(cudaEvent_t));

  CHECK_ALLOC(nativeEvent, "Failed to allocate new CUDA event buffer", sizeof(cudaEvent_t));

  auto dZ = cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t *>(&nativeEvent), cudaEventDisableTiming);
  if (dZ != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaEventCreateWithFlags failed");
  }

  return nativeEvent;
}

int registerEvent(sd::Pointer event, sd::Pointer stream) {
  auto pEvent = reinterpret_cast<cudaEvent_t *>(&event);
  auto pStream = reinterpret_cast<cudaStream_t *>(stream);

  auto dZ = cudaEventRecord(*pEvent, *pStream);
  if (dZ != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaEventRecord failed");
  }

  return 1;
}

int setDevice(int deviceId) {
  sd::AffinityManager::setCurrentDevice(deviceId);
  return 1;
}
void setAvailableDevices(int *devices, int size) {
  std::vector<int> devs;
  for (int i = 0; i < size; i++) devs.push_back(devices[i]);
  sd::AffinityManager::setAvailableDevices(devs);
}


sd::LongType getDeviceFreeMemoryDefault() {
  size_t memFree = 0;
  size_t memTotal = 0;

  cudaMemGetInfo(&memFree, &memTotal);

  return (sd::LongType)memFree;
}

sd::LongType getDeviceFreeMemory(int device) {
  int orig = -1;

  cudaGetDevice(&orig);

  if (device >= 0 && device != orig) {
    cudaSetDevice(device);
  }

  size_t memFree = 0;
  size_t memTotal = 0;

  cudaMemGetInfo(&memFree, &memTotal);

  if (device >= 0 && device != orig) {
    cudaSetDevice(orig);
  }

  return (sd::LongType)memFree;
}

sd::LongType getDeviceTotalMemory(int device) {
  int orig = -1;

  cudaGetDevice(&orig);

  if (device >= 0 && device != orig) {
    cudaSetDevice(device);
  }
  size_t memFree = 0;
  size_t memTotal = 0;

  cudaMemGetInfo(&memFree, &memTotal);

  if (device >= 0 && device != orig) {
    cudaSetDevice(orig);
  }

  return (sd::LongType)memTotal;
}

int memcpySync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) {
  cudaMemcpyKind kind;

  switch (flags) {
    case 0: {
      kind = cudaMemcpyHostToHost;
    } break;
    case 1: {
      kind = cudaMemcpyHostToDevice;
    } break;
    case 2: {
      kind = cudaMemcpyDeviceToHost;
    } break;
    case 3: {
      kind = cudaMemcpyDeviceToDevice;
    } break;
    default: {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("UNDEFNED MEMCPY");
      return 0;
    }
  }

  auto dZ = cudaMemcpy(reinterpret_cast<void *>(dst), const_cast<const void *>(reinterpret_cast<void *>(src)),
                       static_cast<size_t>(size), kind);
  if (dZ != 0) {
    printf("Failed on [%p] -> [%p], size: [%i], direction: [%i], dZ: [%i]\n", src, dst, size, flags,
           static_cast<int>(dZ));
    fflush(stdout);
    fflush(stderr);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMemcpy failed");
    return 0;
  }

  return 1;
}

int memcpyAsync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) {
  auto pStream = reinterpret_cast<cudaStream_t *>(reserved);

  cudaMemcpyKind kind;


  switch (flags) {
    case 0: {
      kind = cudaMemcpyHostToHost;
    } break;
    case 1: {
      kind = cudaMemcpyHostToDevice;
    } break;
    case 2: {
      kind = cudaMemcpyDeviceToHost;
    } break;
    case 3: {
      kind = cudaMemcpyDeviceToDevice;
    } break;
    default: {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("UNDEFINED MEMCPY");
      return 0;
    }
  }

  auto dZ = cudaMemcpyAsync(reinterpret_cast<void *>(dst), const_cast<const void *>(reinterpret_cast<void *>(src)),
                            static_cast<size_t>(size), kind, *pStream);

  if (dZ != 0) {
    printf("Failed on [%p] -> [%p], size: [%i], direction: [%i], dZ: [%i]\n", src, dst, size, flags,
           static_cast<int>(dZ));

    fflush(stdout);
    fflush(stderr);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMemcpyAsync failed");
    return 0;
  }


  return 1;
}

int memsetSync(sd::Pointer dst, int value, sd::LongType size, int flags, sd::Pointer reserved) {
  auto dZ = cudaMemset(reinterpret_cast<void *>(dst), value, static_cast<size_t>(size));
  if (dZ != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMemset failed");
  }

  return 1;
}

int memsetAsync(sd::Pointer dst, int value, sd::LongType size, int flags, sd::Pointer reserved) {
  auto pStream = reinterpret_cast<cudaStream_t *>(reserved);

  auto dZ = cudaMemsetAsync(reinterpret_cast<void *>(dst), value, static_cast<size_t>(size), *pStream);
  if (dZ != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMemsetAsync failed");
  }

  return 1;
}

int destroyEvent(sd::Pointer event) {
  auto pEvent = reinterpret_cast<cudaEvent_t *>(&event);
  auto dZ = cudaEventDestroy(*pEvent);
  if (dZ != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaEventDestroy failed");
  }

  return 1;
}

int streamSynchronize(sd::Pointer stream) {
  auto pStream = reinterpret_cast<cudaStream_t *>(stream);

  auto dZ = cudaStreamSynchronize(*pStream);
  if (dZ != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaStreamSynchronize failed");
  }

  return 1L;
}

int eventSynchronize(sd::Pointer event) {
  auto pEvent = reinterpret_cast<cudaEvent_t *>(&event);

  auto dZ = cudaEventSynchronize(*pEvent);
  if (dZ != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaEventSynchronize failed");
  }

  return 1L;
}

int getAvailableDevices() {
  int devCnt = 0;
  cudaGetDeviceCount(&devCnt);
  return devCnt;
}

void enableDebugMode(bool reallyEnable) { sd::Environment::getInstance().setDebug(reallyEnable); }

void setGridLimit(int gridSize) {
  if (gridSize > 8192) gridSize = 8192;
  if (gridSize < 1) gridSize = 1;
  blockLimit = gridSize;
}

int ompGetMaxThreads() { return maxThreads; }

int ompGetNumThreads() { return maxThreads; }

void setOmpNumThreads(int threads) {
  if (threads > 1024) threads = 1024;
  if (threads < 32) threads = 32;
  maxThreads = threads;
}

/**
 * Sets the number of threads used by OpenBLAS for BLAS operations.
 * On CUDA backend, this is a no-op since we use cuBLAS, not OpenBLAS.
 */
void setOpenBlasThreads(int threads) {
  // No-op on CUDA - we use cuBLAS, not OpenBLAS
  // But still track the setting in Environment for consistency
  sd::Environment::getInstance().setOpenBlasThreads(threads);
}

/**
 * Gets the number of threads OpenBLAS is configured to use.
 * On CUDA backend, returns 0 since we use cuBLAS.
 */
int getOpenBlasThreads() {
  return sd::Environment::getInstance().getOpenBlasThreads();
}

/**
 * Check if BLAS call serialization is enabled.
 * On CUDA backend, this is typically not needed since cuBLAS handles threading internally.
 */
bool isSerializeBlasCalls() {
  return sd::Environment::getInstance().isSerializeBlasCalls();
}

/**
 * Enable or disable BLAS call serialization.
 * On CUDA backend, this is typically not needed since cuBLAS handles threading internally.
 */
void setSerializeBlasCalls(bool serialize) {
  sd::Environment::getInstance().setSerializeBlasCalls(serialize);
}

void enableVerboseMode(bool reallyEnable) { sd::Environment::getInstance().setVerbose(reallyEnable); }

int getDeviceMajor(int device) { return deviceProperties[device].major; }

int getDeviceMinor(int device) { return deviceProperties[device].minor; }

const char *getDeviceName(int device) { return deviceProperties[device].name; }



void saveNpy(std::string fname, const OpaqueDataBuffer *data, const unsigned int *shape, const unsigned int ndims,
             std::string mode) {
  auto dtype = data->getDataBuffer()->getDataType();
  BUILD_SINGLE_SELECTOR(dtype,cnpy::npy_save,(fname,data->getDataBuffer()->primary(),shape,ndims,mode),SD_COMMON_TYPES);
}


// NOTE: tadOnlyShapeInfo is implemented in NativeOpsHelpers_DataBuffers.cpp
// which is shared between CPU and CUDA builds. The implementation there correctly
// takes OpaqueDataBuffer* and extracts the LongType* via hXShapeInfo->primary().


int memcpyConstantAsync(sd::LongType dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) {
  cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(reserved);

  cudaMemcpyKind kind;

  DEBUG_KERNEL(pStream, -1);

  switch (flags) {
    case 0: {
      kind = cudaMemcpyHostToHost;
    } break;
    case 1: {
      kind = cudaMemcpyHostToDevice;
    } break;
    case 2: {
      kind = cudaMemcpyDeviceToHost;
    }
    case 3: {
      kind = cudaMemcpyDeviceToDevice;
    } break;
  }
  auto dZ = cudaMemcpyToSymbolAsync(getConstantSpace(), const_cast<const void *>(src), size, dst, kind, *pStream);
  if (dZ != 0) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMemcpyToSymbolAsync failed");
  }

  return 1;
}

sd::Pointer getConstantSpace() {
return sd::ConstantHelper::getInstance().getConstantSpace();
}

// pullRows moved to NativeOps_utils.cu for SD_GCC_FUNCTRACE builds




bool isExperimentalEnabled() { return sd::Environment::getInstance().isExperimentalBuild(); }

// shuffle moved to NativeOps_shuffle.cu for SD_GCC_FUNCTRACE builds

void setOmpMinThreads(int threads) {
  minThreads = sd::math::sd_max<int>(32, threads);
  minThreads = sd::math::sd_min<int>(maxThreads, minThreads);
}

int getDevice() { return sd::AffinityManager::currentDeviceId(); }

// execSummaryStats, execSummaryStatsTad moved to NativeOps_summaryStats.cu for SD_GCC_FUNCTRACE builds

// execReduce3, execReduce3Tad, execReduce3Scalar moved to NativeOps_reduce3.cu for SD_GCC_FUNCTRACE builds

// execScalar, execScalarBool, execScalarTad, execScalarBoolTad moved to NativeOps_scalar.cu for SD_GCC_FUNCTRACE builds

// execRandom, execRandom2, execRandom3, initRandom, destroyRandom, refreshBuffer, reSeedBuffer
// moved to NativeOps_random.cu for SD_GCC_FUNCTRACE builds

/**
 * Return the length of a shape buffer
 * based on the pointer
 * @param buffer  the buffer pointer to check
 * @return
 */
int lengthForShapeBufferPointer(sd::Pointer buffer) {
  auto shapeBuffer = reinterpret_cast<sd::LongType *>(buffer);
  return shape::shapeInfoLength(shape::rank(shapeBuffer));
}

/**
 * The pointer to get the address for
 *
 * @param address the address to get the pointer
 * @return the pointer for the given address
 */

sd::Pointer pointerForAddress(sd::LongType address) { return reinterpret_cast<sd::Pointer>(address); }



void prescanArrayRecursive(sd::Pointer *extras, int *dZ, int *dX, int numElements, int level) {
  auto stream = reinterpret_cast<cudaStream_t *>(extras[1]);
  auto g_scanBlockSums = reinterpret_cast<int **>(extras[2]);

  int blockSize = 512;  // max size of the thread blocks
  int numBlocks = sd::math::sd_max<int>(1, static_cast<int>(ceil(static_cast<float>(numElements) / (2.f * blockSize))));
  int numThreads;

  if (numBlocks > 1)
    numThreads = blockSize;
  else if (sd::isPowerOfTwo(numElements))
    numThreads = numElements / 2;
  else
    numThreads = sd::floorPow2(numElements);

  int numEltsPerBlock = numThreads * 2;

  // if this is a non-power-of-2 array, the last block will be non-full
  // compute the smallest power of 2 able to compute its scan.
  int numEltsLastBlock = numElements - (numBlocks - 1) * numEltsPerBlock;
  int numThreadsLastBlock = sd::math::sd_max<int>(1, numEltsLastBlock / 2);
  int np2LastBlock = 0;
  int sharedMemLastBlock = 0;

  if (numEltsLastBlock != numEltsPerBlock) {
    np2LastBlock = 1;

    if (!sd::isPowerOfTwo(numEltsLastBlock)) numThreadsLastBlock = sd::floorPow2(numEltsLastBlock);

    unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
    sharedMemLastBlock = sizeof(int) * (2 * numThreadsLastBlock + extraSpace);
  }

  // padding space is used to avoid shared memory bank conflicts
  int extraSpace = numEltsPerBlock / NUM_BANKS;
  int sharedMemSize = sizeof(int) * (numEltsPerBlock + extraSpace);

  // setup execution parameters
  // if NP2, we process the last block separately
  dim3 grid(sd::math::sd_max<int>(1, numBlocks - np2LastBlock), 1, 1);
  dim3 threads(numThreads, 1, 1);
  dim3 gridOnes(1, 1, 1);
  dim3 threadsOnes(numThreadsLastBlock, 1, 1);

  if (sharedMemSize < 2048) sharedMemSize = 2048;

  if (sharedMemLastBlock < 2048) sharedMemLastBlock = 2048;

  // execute the scan
  if (numBlocks > 1) {
    sd::prescanLauncher<true, false>(grid, threads, sharedMemSize, stream, dZ, dX, g_scanBlockSums[level],
                                     numThreads * 2, 0, 0);
    if (np2LastBlock) {
      sd::prescanLauncher<true, true>(gridOnes, threadsOnes, sharedMemLastBlock, stream, dZ, dX, g_scanBlockSums[level],
                                      numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
    }

    // After scanning all the sub-blocks, we are mostly done.  But now we
    // need to take all of the last values of the sub-blocks and scan those.
    // This will give us a new value that must be sdded to each block to
    // get the final results.
    // recursive (CPU) call
    prescanArrayRecursive(extras, g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level + 1);

    sd::uniformAdd<<<grid, threads, 1024, *stream>>>(dZ, g_scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);
    sd::DebugHelper::checkGlobalErrorCode("uniform addfailed(...) failed");

    if (np2LastBlock) {
      sd::uniformAdd<<<1, numThreadsLastBlock, 1024, *stream>>>(dZ, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1,
          numElements - numEltsLastBlock);
      sd::DebugHelper::checkGlobalErrorCode("concat general case failed(...) failed");

    }
  } else if (sd::isPowerOfTwo(numElements)) {
    sd::prescanLauncher<false, false>(grid, threads, sharedMemSize, stream, dZ, dX, 0, numThreads * 2, 0, 0);

  } else {
    sd::prescanLauncher<false, true>(grid, threads, sharedMemSize, stream, dZ, dX, 0, numElements, 0, 0);
  }

  sd::DebugHelper::checkErrorCode(stream, "prescanArray(...) failed");
}

// execReduce3All moved to NativeOps_reduce3.cu for SD_GCC_FUNCTRACE builds

// sort, sortByKey, sortByValue, sortTadByKey, sortTadByValue, sortTad
// moved to NativeOps_sort.cu for SD_GCC_FUNCTRACE builds

// tryPointerKernel, tryPointer moved to NativeOps_utils.cu for SD_GCC_FUNCTRACE builds



bool isBlasVersionMatches(int major, int minor, int build) {
  auto result = major == sd::Environment::getInstance()._blasMajorVersion &&
                minor == sd::Environment::getInstance()._blasMinorVersion &&
                build == sd::Environment::getInstance()._blasPatchVersion;

  if (!result) {
    sd_printf("CUDA/cuBLAS version mismatch. Expected: %i.%i.%i but got %i.%i.%i instead\n",
              sd::Environment::getInstance()._blasMajorVersion, sd::Environment::getInstance()._blasMinorVersion,
              sd::Environment::getInstance()._blasPatchVersion, major, minor, build);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(152);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("CUDA/cuBLAS version mismatch");
  }

  return result;
}


void setGraphContextCudaContext(Context *ptr, void *stream, void *reductionPointer,
                                void *allocationPointer) {
  ptr->setCudaContext(stream, reductionPointer, allocationPointer);
}




int binaryLevel() { return 0; }

int optimalLevel() { return 0; }

bool isMinimalRequirementsMet() { return true; }

bool isOptimalRequirementsMet() { return true; }








void setShapeBuffer(sd::LongType *inputShapeData,sd::DataType dt,sd::LongType *bufferToSet,char order,int elementWiseStride,bool isEmpty,bool isView) {
  if (inputShapeData == nullptr) THROW_EXCEPTION("setShapeBuffer: inputShapeData is null");

  if (bufferToSet == nullptr) THROW_EXCEPTION("setShapeBuffer: bufferToSet is null");
  sd::LongType rank = inputShapeData[0];
  if (rank > SD_MAX_RANK || rank < 0) THROW_EXCEPTION("Invalid rank for shape buffer.");
  std::vector<sd::LongType> shape;
  std::vector<sd::LongType> strides;
  // shape, stride, data type
  for (sd::LongType i = 1; i < rank * 2 + 1; i++) {
    if (i <= rank) {
      shape.push_back(inputShapeData[i]);
    } else if (shape.size() == rank) {
      strides.push_back(inputShapeData[i]);
    }
  }

  auto len = shape::shapeInfoLength(rank);
  for (int i = 0; i < len; i++) {
    bufferToSet[i] = inputShapeData[i];
  }

  sd::ArrayOptions::setDataType(bufferToSet, dt);
  if (isView) {
    sd::ArrayOptions::toggleIsView(bufferToSet);
  }
  if (!sd::ArrayOptions::isEmpty(inputShapeData) && isEmpty) {
    sd::ArrayOptions::toggleIsEmpty(bufferToSet);
  }

  if (rank == 0) {
    // detect when the shape buffer values are unset.
    auto len = shape::shapeInfoLength(rank);
    // min number of values in a shape info buffer
    bool allZero = true;
    for (int i = 0; i < len; i++) {
      if (bufferToSet[i] != 0) {
        allZero = false;
        break;
      }
    }

    if (allZero) {
      THROW_EXCEPTION("Found shape buffer with all zero values. Values likely unset.");
    }
  }
}

////////////////////////////////////////////////////////////////////////
// CUDA-specific clearLastError implementation
// Clears both the ErrorReference and the CUDA runtime error state
void clearLastError() {
  // Clear the ErrorReference
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");

  // Clear any pending CUDA errors first
  cudaError_t err = cudaGetLastError();

  // If there was a serious error, we need to be more aggressive about cleanup
  if (err != cudaSuccess) {
    // Try to synchronize the device - this may help flush bad state
    cudaDeviceSynchronize();
    cudaGetLastError();  // Clear any error from synchronize

    // If the context was really corrupted, try to reset the device streams
    // by getting a fresh stream from the context
    auto ctx = sd::LaunchContext::defaultContext();
    if (ctx != nullptr) {
      // Force stream synchronization
      auto stream = ctx->getCudaStream();
      if (stream != nullptr) {
        cudaStreamSynchronize(*stream);
        cudaGetLastError();
      }
    }
  }
}

