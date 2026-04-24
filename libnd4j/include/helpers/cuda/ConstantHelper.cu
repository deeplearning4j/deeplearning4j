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
//  @author raver119@gmail.com
//
#include <array/DataBuffer.h>
#include <array/DataTypeUtils.h>
#include <array/PrimaryPointerDeallocator.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <exceptions/cuda_exception.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <helpers/ConstantHelper.h>
#include <helpers/logger.h>
#include <helpers/shape.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/specials.h>
#include <ops/impl/specials_double.hpp>
#include <system/selective_rendering.h>
#define CONSTANT_LIMIT 49152

__constant__ char deviceConstantMemory[CONSTANT_LIMIT];

namespace sd {

namespace {
SD_INLINE cudaStream_t captureSafeStreamOrDefault() {
  if (tl_graphExecutionActive && tl_graphCaptureStream != nullptr) {
    return tl_graphCaptureStream;
  }
  auto* streamPtr = LaunchContext::defaultContext()->getCudaStream();
  return (streamPtr != nullptr) ? *streamPtr : nullptr;
}
}  // namespace

void * ConstantHelper::getConstantSpace() {
  // Always use memory pool for constant space
  // The __constant__ memory approach via cudaGetSymbolAddress was causing issues
  // with CUDA module registration timing, leading to error 400 on kernel launches.
  // Using regular device memory is functionally equivalent and more reliable.
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  void* ptr = memory::CudaMemoryPool::getInstance().allocate(CONSTANT_LIMIT, deviceId, nullptr);
  if (ptr == nullptr) {
    cudaGetLastError();  // Clear error state
    throw cuda_exception::build("Failed to allocate constant space", cudaErrorMemoryAllocation);
  }
  return ptr;
}

int ConstantHelper::getCurrentDevice() { return AffinityManager::currentDeviceId(); }

int ConstantHelper::getNumberOfDevices() { return AffinityManager::numberOfDevices(); }

ConstantHelper::ConstantHelper() {
  // Force CUDA runtime initialization by making a simple API call
  // This ensures the CUDA context and module registration happen before we use any CUDA features
  cudaFree(0);

  // Clear any stale CUDA errors from previous operations
  cudaGetLastError();

  auto initialDevice = getCurrentDevice();

  auto numDevices = getNumberOfDevices();
  _devicePointers.resize(numDevices);
  _deviceOffsets.resize(numDevices);
  _cache.resize(numDevices);
  _counters.resize(numDevices);

  // filling all pointers
  for (int e = 0; e < numDevices; e++) {
    auto res = cudaSetDevice(e);
    if (res != 0) {
      cudaGetLastError();  // Clear error before throwing
      throw cuda_exception::build("cudaSetDevice failed", res);
    }
    auto constant = getConstantSpace();

    SD_MAP_IMPL<ConstantDescriptor, ConstantHolder *> devCache;

    _devicePointers[e] = constant;
    _deviceOffsets[e] = 0;
    _cache[e] = devCache;
    _counters[e] = 0L;
  }

  //
  auto res = cudaSetDevice(initialDevice);
  if (res != 0) {
    cudaGetLastError();  // Clear error before throwing
    throw cuda_exception::build("Final cudaSetDevice failed", res);
  }

  // Clear any errors that may have accumulated
  cudaGetLastError();
}

ConstantHelper::~ConstantHelper() {
  for (const auto &v : _cache) {
    for (const auto &c : v) {
      delete c.second;
    }
  }
}

ConstantHelper &ConstantHelper::getInstance() {
  static ConstantHelper instance;
  return instance;
}

void *ConstantHelper::replicatePointer(void *src, size_t numBytes, memory::Workspace *workspace) {
  std::lock_guard<std::mutex> lock(_mutex);

  auto deviceId = getCurrentDevice();
  Pointer constantPtr = nullptr;
  LongType constantOffset = 0L;
  if (_devicePointers[deviceId] == 0) {
    auto constant = getConstantSpace();

    // filling default ptr, which will be 0 probably
    _devicePointers[deviceId] = constant;
    _deviceOffsets[deviceId] = 0;
    constantPtr = constant;
  } else {
    constantPtr = _devicePointers[deviceId];
    constantOffset = _deviceOffsets[deviceId];
  }

  int8_t *ptr = nullptr;
  bool usedPinnedHost = false;
  if (workspace == nullptr) {
    // Constant shape buffers MUST be on the correct device or GPU-accessible from all devices.
    // If allocateFailover places a shape buffer on a different device, non-P2P GPUs can't
    // access it → CUDA error 700 (illegal memory access). Fix: trim + retry, then fall back
    // to pinned host memory which is accessible from ALL GPUs.
    int actualDevice = deviceId;
    size_t allocSize = numBytes + SD_ALLOC_PADDING;
    // During CUDA graph capture, allocations MUST use the captured stream.
    // Using nullptr (legacy default stream) causes implicit sync with the captured
    // stream, invalidating the capture (error 901).
    cudaStream_t allocStream = nullptr;
    if (tl_graphExecutionActive) {
      allocStream = captureSafeStreamOrDefault();
    }
    ptr = reinterpret_cast<int8_t*>(
        memory::CudaMemoryPool::getInstance().allocate(allocSize, deviceId, allocStream, &actualDevice));
    if (ptr == nullptr && tl_graphExecutionActive) {
      // During CUDA graph capture, CudaMemoryPool::allocate() uses the capture
      // workspace (bump allocator). If it returned nullptr, the workspace is
      // exhausted. The fallback paths (trimPool, cudaMallocHost, pool.free) are
      // all synchronous or use nullptr stream, which would poison the capture
      // stream (error 901). Abort immediately — the caller's capture segment
      // will fall back to slot-by-slot execution.
      THROW_EXCEPTION("[DEVICE] replicatePointer: capture workspace exhausted during CUDA graph capture. "
                      "Increase via -Dnd4j.dsp.captureWorkspaceMb=512 or ND4J_DSP_CAPTURE_WORKSPACE_MB=512");
    }
    if (ptr == nullptr) {
      // Shape buffers are tiny (~200 bytes). Failure likely means a stale CUDA
      // error is blocking allocations, not true OOM. Clear errors, trim pool, retry.
      cudaGetLastError();
      memory::CudaMemoryPool::getInstance().trimPool(deviceId);
      ptr = reinterpret_cast<int8_t*>(
          memory::CudaMemoryPool::getInstance().allocate(allocSize, deviceId, allocStream, &actualDevice));
    }
    if (ptr == nullptr) {
      // Pool retry failed — fall back to pinned host memory which is
      // GPU-accessible from ALL devices. Shape buffers are read-only constants
      // so pinned host is safe and avoids OOM for trivial allocations.
      cudaGetLastError();
      auto hostRes = cudaMallocHost(reinterpret_cast<void**>(&ptr), allocSize);
      if (hostRes != cudaSuccess || ptr == nullptr) {
        cudaGetLastError();
        THROW_EXCEPTION("[DEVICE] replicatePointer allocation failed (pool + pinned host both failed)");
      }
      // Register in hostAllocations_ so CudaMemoryPool::free() can route to cudaFreeHost
      memory::CudaMemoryPool::getInstance().registerHostAllocation(ptr, allocSize);
      usedPinnedHost = true;
      sd_debug("replicatePointer: pool alloc failed for device %d, using pinned host (%zu bytes)\n",
               deviceId, allocSize);
    }
    if (actualDevice != deviceId) {
      // Wrong device: free and fall back to pinned host memory immediately.
      // Shape/constant buffers are tiny (~200 bytes). Using pinned host memory
      // (cudaMallocHost) is safe because it's GPU-accessible from ALL devices,
      // including non-P2P GPUs. This avoids expensive cudaDeviceSynchronize()
      // calls that would block all GPU compute for a tiny allocation.
      memory::CudaMemoryPool::getInstance().free(ptr, actualDevice, nullptr);
      ptr = nullptr;

      auto hostRes = cudaMallocHost(reinterpret_cast<void**>(&ptr), allocSize);
      if (hostRes != cudaSuccess || ptr == nullptr) {
        cudaGetLastError();
        THROW_EXCEPTION("[DEVICE] replicatePointer: pinned host fallback allocation failed");
      }
      // Register in hostAllocations_ so CudaMemoryPool::free() can route to cudaFreeHost
      memory::CudaMemoryPool::getInstance().registerHostAllocation(ptr, allocSize);
      usedPinnedHost = true;
      sd_debug("replicatePointer: device %d OOM, using pinned host for constant (%zu bytes)\n",
               deviceId, numBytes);
    }
  } else {
    size_t allocSize = numBytes + SD_ALLOC_PADDING;
    ptr = reinterpret_cast<int8_t*>(workspace->allocateBytes(memory::MemoryType::DEVICE, allocSize));
    if (ptr == nullptr) {
      THROW_EXCEPTION("[DEVICE] replicatePointer workspace allocation failed");
    }
  }

  if (usedPinnedHost) {
    // Host-to-host copy for pinned memory (no CUDA context needed)
    memcpy(ptr, src, numBytes);
  } else {
    if (tl_graphExecutionActive) {
      // During CUDA graph capture, synchronous cudaMemcpy on the legacy default stream
      // implicitly syncs with ALL named streams (including the captured stream), causing
      // capture invalidation (error 901). Use cudaMemcpyAsync on the CAPTURED stream
      // so it becomes a recorded graph node.
      //
      //  The H2D memcpy node bakes the source address into the graph. If `src`
      // points to stack/temporary memory, graph replay will read garbage (the stack frame
      // is gone). Copy src into the capture host workspace (persistent pinned memory) first,
      // matching the pattern in DataBuffer::syncToSpecial().
      cudaStream_t capturedStream = captureSafeStreamOrDefault();
      void* h2dSource = src;
      if (tl_captureHostWorkspace != nullptr) {
        size_t aligned = (numBytes + 255) & ~255ULL;
        if (tl_captureHostWorkspaceOffset + aligned <= tl_captureHostWorkspaceSize) {
          void* pinnedCopy = static_cast<char*>(tl_captureHostWorkspace) + tl_captureHostWorkspaceOffset;
          tl_captureHostWorkspaceOffset += aligned;
          std::memcpy(pinnedCopy, src, numBytes);
          h2dSource = pinnedCopy;
        }
        // If workspace exhausted, fall through to use src directly — best effort
      }
      auto res = cudaMemcpyAsync(ptr, h2dSource, numBytes, cudaMemcpyHostToDevice, capturedStream);
      if (res != 0) {
        std::string errorMessage = "cudaMemcpyAsync (graph capture) failed with error code " + std::to_string(res);
        THROW_EXCEPTION(errorMessage.c_str());
      }
    } else {
      auto res = cudaMemcpy(ptr, src, numBytes, cudaMemcpyHostToDevice);
      if (res != 0) {
        std::string errorMessage = "cudaMemcpy failed with error code " + std::to_string(res);
        auto lastError = cudaGetLastError();
        if (lastError != cudaSuccess) {
          errorMessage += "; last error: " + std::string(cudaGetErrorString(lastError));
        }
        THROW_EXCEPTION(errorMessage.c_str());
      }
    }
  }

  constantPtr = ptr;
  return reinterpret_cast<int8_t *>(constantPtr) + constantOffset;
}

ConstantDataBuffer *ConstantHelper::constantBuffer(const ConstantDescriptor &descriptor, DataType dataType) {
  const auto deviceId = getCurrentDevice();

  // all cache modifications are synchronous
  _mutexHolder.lock();

  if (_cache[deviceId].count(descriptor) == 0) {
    _cache[deviceId][descriptor] = new ConstantHolder();
  }
  auto holder = _cache[deviceId][descriptor];

  // release cache lock
  _mutexHolder.unlock();

  ConstantDataBuffer *result;

  // access to this holder instance is synchronous
  std::lock_guard<std::mutex> lock(*holder->mutex());

  if (holder->hasBuffer(dataType)) {
    result = holder->getConstantDataBuffer(dataType);
  } else {
    auto numBytes = descriptor.length() * DataTypeUtils::sizeOf(dataType);
    auto cbuff = std::make_shared<PointerWrapper>(new int8_t[numBytes], std::make_shared<PointerDeallocator>());
    _counters[deviceId] += numBytes;

    // create buffer with this dtype
    if (descriptor.isFloat()) {
      BUILD_DOUBLE_SELECTOR(
          sd::DataType::DOUBLE, dataType, SpecialTypeConverter::convertGeneric,
          (nullptr, const_cast<double *>(descriptor.floatValues().data()), descriptor.length(), cbuff->pointer()),
          (DOUBLE, double), SD_COMMON_TYPES);
    } else if (descriptor.isInteger()) {
      BUILD_DOUBLE_SELECTOR(sd::DataType::INT64, dataType, SpecialTypeConverter::convertGeneric,
                            (nullptr, const_cast<LongType *>(descriptor.integerValues().data()),
                                descriptor.length(), cbuff->pointer()),
                            (INT64, LongType), SD_COMMON_TYPES);
    }

    // we don't have deallocator here.
    // TODO: we probably want to make use deallocator here, if we're not using constant memory
    auto dbuff = std::make_shared<PointerWrapper>(
        replicatePointer(cbuff->pointer(), descriptor.length() * DataTypeUtils::sizeOf(dataType)));

    ConstantDataBuffer *dataBuffer = new ConstantDataBuffer(cbuff, dbuff, descriptor.length(), dataType);

    holder->addBuffer(*dataBuffer, dataType);
    result = holder->getConstantDataBuffer(dataType);
  }

  return result;
}

LongType ConstantHelper::getCachedAmount(int deviceId) {
  int numDevices = getNumberOfDevices();
  if (deviceId > numDevices || deviceId < 0)
    return 0L;
  else
    return _counters[deviceId];
}

// Explicit template instantiations for SpecialTypeConverter::convertGeneric
// These are needed because BUILD_DOUBLE_SELECTOR expands to call these with (DOUBLE, SD_COMMON_TYPES) and (INT64, SD_COMMON_TYPES)
// #define INSTANTIATE_CONVERT_DOUBLE(T) template void SpecialTypeConverter::convertGeneric<double, GET_SECOND(T)>(sd::Pointer*, void*, sd::LongType, void*);
// ITERATE_LIST((SD_COMMON_TYPES), INSTANTIATE_CONVERT_DOUBLE)
template void SpecialTypeConverter::convertGeneric<double, bool>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, float16>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, bfloat16>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, float>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, double>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, int8_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, uint8_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, int16_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, int32_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, sd::LongType>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, uint16_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, uint32_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<double, uint64_t>(sd::Pointer*, void*, sd::LongType, void*);

// #define INSTANTIATE_CONVERT_LONG(T) template void SpecialTypeConverter::convertGeneric<LongType, GET_SECOND(T)>(sd::Pointer*, void*, sd::LongType, void*);
// ITERATE_LIST((SD_COMMON_TYPES), INSTANTIATE_CONVERT_LONG)
template void SpecialTypeConverter::convertGeneric<LongType, bool>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, float16>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, bfloat16>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, float>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, double>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, int8_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, uint8_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, int16_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, int32_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, sd::LongType>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, uint16_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, uint32_t>(sd::Pointer*, void*, sd::LongType, void*);
template void SpecialTypeConverter::convertGeneric<LongType, uint64_t>(sd::Pointer*, void*, sd::LongType, void*);

}  // namespace sd
