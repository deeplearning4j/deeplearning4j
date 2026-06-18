/* ******************************************************************************
 *
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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <array/DataTypeUtils.h>
#include <climits>

#include <execution/AffinityManager.h>
#include <memory/MemoryCounter.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <system/CanaryConstants.h>
#include <system/Environment.h>
#include <system/env_functions.h>
#include <system/op_boilerplate.h>
#include <system/type_boilerplate.h>
#include <helpers/TransferMetrics.h>
#include <graph/DspDiagnostics.h>
#include <chrono>

#include "../DataBuffer.h"
#include "helpers/DebugHelper.h"

#if defined(SD_GCC_FUNCTRACE)
#include <array/DataBufferLifecycleTracker.h>
#endif

namespace sd {

SD_TLS_EXPORT thread_local DataBufferThreadState tl_dataBufferState;

namespace {
SD_INLINE cudaStream_t captureSafeStreamOrDefault() {
  if (tl_graphExecutionActive && tl_graphCaptureStream != nullptr) {
    return reinterpret_cast<cudaStream_t>(tl_graphCaptureStream);
  }
  auto* streamPtr = LaunchContext::defaultContext()->getCudaStream();
  return (streamPtr != nullptr) ? *streamPtr : nullptr;
}

SD_INLINE cudaStream_t asyncTransferStream(bool switchedDevice) {
  if (tl_graphExecutionActive) {
    cudaStream_t captureStream = captureSafeStreamOrDefault();
    return captureStream != nullptr ? captureStream : cudaStreamPerThread;
  }

  if (!switchedDevice) {
    if (tl_dspExecutionStream != nullptr) {
      return reinterpret_cast<cudaStream_t>(tl_dspExecutionStream);
    }
    auto* streamPtr = LaunchContext::defaultContext()->getCudaStream();
    if (streamPtr != nullptr && *streamPtr != nullptr) {
      return *streamPtr;
    }
  }

  return cudaStreamPerThread;
}

SD_INLINE void throwCudaStatus(const char* caller, cudaError_t err) {
  std::string msg = std::string(caller) + ": " +
                    cudaGetErrorString(err) + " (" +
                    std::to_string(static_cast<int>(err)) + ")";
  THROW_EXCEPTION(msg.c_str());
}

struct ThreadDspCompletionEvent {
  cudaEvent_t event = nullptr;
  bool recorded = false;
  int deviceId = -1;

  ~ThreadDspCompletionEvent() {
    if (event == nullptr) return;
    int currentDevice = -1;
    bool switched = false;
    if (cudaGetDevice(&currentDevice) == cudaSuccess && deviceId >= 0 && currentDevice != deviceId) {
      if (cudaSetDevice(deviceId) == cudaSuccess) {
        switched = true;
      }
    }
    cudaEventDestroy(event);
    if (switched && currentDevice >= 0) {
      cudaSetDevice(currentDevice);
    }
    event = nullptr;
    recorded = false;
    deviceId = -1;
  }

  /**
   * Ensure the completion event exists and belongs to the current CUDA device.
   * Returns true on success, false if the CUDA context is corrupted (e.g., after
   * an OOM-triggered illegal memory access). Callers must check the return value
   * and skip event recording on failure.
   */
  bool ensureForCurrentDevice() {
    int currentDevice = -1;
    auto devErr = cudaGetDevice(&currentDevice);
    if (devErr != cudaSuccess) {
      // CUDA context is broken — clear the sticky error and bail.
      cudaGetLastError();
      sd_printf("ThreadDspCompletionEvent: cudaGetDevice failed: %s — skipping completion event\n",
                cudaGetErrorString(devErr));
      return false;
    }
    if (event != nullptr && deviceId != currentDevice) {
      cudaEventDestroy(event);
      event = nullptr;
      recorded = false;
      deviceId = -1;
    }
    if (event == nullptr) {
      auto eventErr = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      if (eventErr != cudaSuccess) {
        // Clear the sticky error. This typically happens when a prior op caused
        // an illegal memory access (error 700) and the CUDA context is unusable.
        cudaGetLastError();
        sd_printf("ThreadDspCompletionEvent: cudaEventCreateWithFlags failed: %s — skipping completion event\n",
                  cudaGetErrorString(eventErr));
        return false;
      }
      deviceId = currentDevice;
    }
    return true;
  }
};

thread_local ThreadDspCompletionEvent tl_dspCompletionEvent;

SD_INLINE void waitForLastDspCompletionIfNeeded(cudaStream_t stream) {
  if (tl_graphExecutionActive || tl_dspExecutionStream != nullptr) return;
  if (!tl_dspCompletionEvent.recorded || tl_dspCompletionEvent.event == nullptr) return;
  auto waitErr = cudaStreamWaitEvent(stream, tl_dspCompletionEvent.event, 0);
  if (waitErr != cudaSuccess) {
    throwCudaStatus("DataBuffer: cudaStreamWaitEvent(last DSP completion) failed", waitErr);
  }
}

/**
 * Check if an allocation that failed over to a different device landed on a non-peer device.
 * Returns the effective device ID for the buffer.
 *
 * cudaMallocManaged memory (used by CudaMemoryPool for non-peer failover) is accessible
 * from ALL devices via UVA page migration — no replacement needed. Regular cudaMalloc
 * on a non-peer device would cause CUDA error 700, but the pool's non-peer path always
 * uses cudaMallocManaged so this case should not arise.
 *
 * @param buffer        Pointer to the allocated buffer (unchanged for managed/peer memory)
 * @param allocSize     Size of the allocation in bytes
 * @param requestedDevice  The device the caller wanted the allocation on
 * @param actualDevice     The device the allocation actually landed on
 * @param caller        Name of the calling function for diagnostics
 * @return The effective device ID (actualDevice — managed memory is accessible from requestedDevice via UVA)
 */
int handleNonPeerFailover(void*& buffer, size_t allocSize, int requestedDevice, int actualDevice, const char* caller) {
  if (actualDevice == requestedDevice) return actualDevice;
  if (memory::CudaMemoryPool::getInstance().isPeerAccessEnabled(requestedDevice, actualDevice)) return actualDevice;

  // Non-peer cross-device allocation. Check if the pointer is managed memory
  // (cudaMallocManaged) — if so, it's accessible from all devices via UVA page migration.
  cudaPointerAttributes attrs;
  cudaError_t queryErr = cudaPointerGetAttributes(&attrs, buffer);
  if (queryErr == cudaSuccess && attrs.type == cudaMemoryTypeManaged) {
    // Managed memory — accessible from all devices. Advise prefetch to requesting device
    // for better first-access latency (driver will migrate pages on demand regardless).
    cudaMemPrefetchAsync(buffer, allocSize, requestedDevice, nullptr);
    if (sd::Environment::getInstance().isVerbose()) {
      sd_printf("%s: Non-peer failover (device %d → %d) — managed memory, accessible via UVA (no pinned replacement needed)\n",
                caller, requestedDevice, actualDevice);
    }
    return actualDevice;
  }
  if (queryErr != cudaSuccess) {
    cudaGetLastError();  // Clear the error
  }

  // Non-managed, non-peer device memory — would cause CUDA error 700 on requestedDevice.
  // This path should not be hit with the current pool implementation (which uses
  // cudaMallocManaged for non-peer failover), but handle it defensively with pinned host.
  if (sd::Environment::getInstance().isVerbose()) {
    sd_printf("%s: Non-peer failover (device %d → %d) — non-managed memory, switching to pinned host\n",
              caller, requestedDevice, actualDevice);
  }

  void* pinnedPtr = nullptr;
  cudaError_t err = cudaMallocHost(&pinnedPtr, allocSize);
  if (err != cudaSuccess || pinnedPtr == nullptr) {
    std::string msg = std::string(caller) + ": pinned host allocation also failed for " + std::to_string(allocSize) + " bytes";
    THROW_EXCEPTION(msg.c_str());
  }

  void* oldBuffer = buffer;
  cudaSetDevice(actualDevice);
  memory::CudaMemoryPool::getInstance().free(oldBuffer, actualDevice, nullptr);
  cudaSetDevice(requestedDevice);

  memory::CudaMemoryPool::getInstance().registerHostAllocation(pinnedPtr, allocSize);
  buffer = pinnedPtr;
  return requestedDevice;
}
}  // namespace

void dspPublishThreadCompletionEvent(void* streamPtr) {
  if (streamPtr == nullptr) return;

  // Clear any sticky CUDA error from a prior failed op (e.g., OOM during DSP
  // slot execution). Without this, cudaEventCreateWithFlags / cudaEventRecord
  // will fail with the propagated error and crash the process.
  auto stickyErr = cudaGetLastError();
  if (stickyErr != cudaSuccess) {
    sd_printf("dspPublishThreadCompletionEvent: cleared sticky CUDA error: %s — skipping completion event\n",
              cudaGetErrorString(stickyErr));
    tl_dspCompletionEvent.recorded = false;
    return;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamPtr);
  if (!tl_dspCompletionEvent.ensureForCurrentDevice()) {
    // CUDA context is broken — ensureForCurrentDevice already logged why.
    return;
  }
  auto err = cudaEventRecord(tl_dspCompletionEvent.event, stream);
  if (err != cudaSuccess) {
    cudaGetLastError();  // clear
    sd_printf("dspPublishThreadCompletionEvent: cudaEventRecord failed: %s — skipping\n",
              cudaGetErrorString(err));
    tl_dspCompletionEvent.recorded = false;
    return;
  }
  tl_dspCompletionEvent.recorded = true;
}

void DataBuffer::expand(const uint64_t size) {
  throwIfFrozen("expand");
  if (size > _lenInBytes) {
    // allocate new buffer
    int8_t* newBuffer = nullptr;
    int8_t* newSpecialBuffer = nullptr;
    auto currentDeviceId = AffinityManager::currentDeviceId();
    // Use _specialDeviceId for the old buffer since we're freeing the special buffer
    // This may differ from _deviceId due to failover during OOM
    auto oldDeviceId = _specialDeviceId.load();
    if (oldDeviceId < 0) {
      oldDeviceId = _deviceId.load();  // Fallback for legacy code
    }

    // Allocate new buffer, tracking actual device in case of failover
    int actualExpandDevice = currentDeviceId;
    if (_workspace == nullptr) {
      size_t allocSize = size + 8;
      newSpecialBuffer = reinterpret_cast<int8_t*>(
          memory::CudaMemoryPool::getInstance().allocate(allocSize, currentDeviceId, nullptr, &actualExpandDevice));
      if (newSpecialBuffer == nullptr) {
        THROW_EXCEPTION("[DEVICE] expand allocation failed");
      }
      void* expandBuf = reinterpret_cast<void*>(newSpecialBuffer);
      actualExpandDevice = handleNonPeerFailover(expandBuf, allocSize, currentDeviceId, actualExpandDevice, "DataBuffer::expand");
      newSpecialBuffer = reinterpret_cast<int8_t*>(expandBuf);
    } else {
      size_t allocSize = size + 8;
      newSpecialBuffer = reinterpret_cast<int8_t*>(
          _workspace->allocateBytes(memory::MemoryType::DEVICE, allocSize));
    }
#if defined(SD_GCC_FUNCTRACE)
    array::DataBufferLifecycleTracker::getInstance().recordAllocation(
        newSpecialBuffer, size, _dataType,array::BufferType::SPECIAL, this, _workspace != nullptr);
#endif

    // copy data from existing buffer
    size_t hostAllocSize = size + (_workspace == nullptr ? static_cast<size_t>(HOST_ALLOC_PADDING) : 0);
    if (_primaryBuffer != nullptr) {
      // there's non-zero chance that primary buffer doesn't exist yet
      ALLOCATE(newBuffer, _workspace, hostAllocSize, int8_t);
      if (_lenInBytes > 0) {
        std::memcpy(newBuffer, _primaryBuffer, _lenInBytes);
      }

      // Write canary values in the padding region (same as allocatePrimary)
      if (_workspace == nullptr) {
        uint64_t* canary = reinterpret_cast<uint64_t*>(newBuffer + size);
        for (size_t i = 0; i < (static_cast<size_t>(HOST_ALLOC_PADDING) / sizeof(uint64_t)); i++) {
          canary[i] = sd::CanaryConstants::DATA_BUFFER_CANARY;
        }
      }

      if (_isOwnerPrimary) {
#if defined(SD_GCC_FUNCTRACE)
        array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
            _primaryBuffer,array::BufferType::PRIMARY);
#endif
        auto ipb = reinterpret_cast<int8_t*>(_primaryBuffer);
        RELEASE(ipb, _workspace);
      }

      _primaryBuffer = newBuffer;
      _isOwnerPrimary = true;
#if defined(SD_GCC_FUNCTRACE)
      array::DataBufferLifecycleTracker::getInstance().recordAllocation(
          _primaryBuffer, size, _dataType,array::BufferType::PRIMARY, this, _workspace != nullptr);
#endif
    }

    // Cross-device copy — use cudaStreamPerThread to avoid error 906 when
    // another thread is mid-CUDA-graph-capture on the same device.
    cudaMemcpyAsync(newSpecialBuffer, _specialBuffer, _lenInBytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);

    if (_isOwnerSpecial && _specialBuffer != nullptr) {
      // Switch to old device to release memory
      if (oldDeviceId != currentDeviceId && oldDeviceId >= 0) {
        cudaSetDevice(oldDeviceId);
      }

#if defined(SD_GCC_FUNCTRACE)
      array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
          _specialBuffer,array::BufferType::SPECIAL);
#endif
      auto isb = reinterpret_cast<int8_t*>(_specialBuffer);
      // Use device-aware free - critical for multi-GPU correctness
      RELEASE_SPECIAL_WITH_DEVICE(isb, oldDeviceId, _workspace);

      // Switch back to current device
      if (oldDeviceId != currentDeviceId && oldDeviceId >= 0) {
        cudaSetDevice(currentDeviceId);
      }
    }

    _specialBuffer = newSpecialBuffer;
    _lenInBytes = size;
    _specialAllocBytes = size + 8;  // match actual allocation (size + 8)
    if (_primaryBuffer != nullptr) _primaryAllocBytes = hostAllocSize;

    // If CudaMemoryPool returned a capture-workspace interior pointer (bump-allocated
    // during CUDA graph capture), the workspace lifecycle manages this memory — not
    // this DataBuffer. Setting _isOwnerSpecial=true would cause deleteSpecial() to
    // call cudaFreeAsync on a workspace interior pointer after the workspace is
    // unregistered → "illegal memory access" (error 700) → CUDA context corruption.
    // Mirror the allocateSpecial() workspace path: set _isOwnerSpecial=false so
    // deleteSpecial() skips the free entirely (workspace is freed as a whole unit).
    if (_workspace == nullptr &&
        memory::CudaMemoryPool::getInstance().isInCaptureWorkspace(newSpecialBuffer)) {
      _isOwnerSpecial = false;
    } else {
      _isOwnerSpecial = true;
    }

    // Store actual device where memory was allocated (may differ from currentDeviceId after failover)
    _deviceId.store(actualExpandDevice);
    _specialDeviceId.store(actualExpandDevice);
  }
}

DataBuffer DataBuffer::dup() {
  DataBuffer result;
  result._dataType = _dataType;
  result._lenInBytes = _lenInBytes;
  result._primaryAllocBytes = 0;
  result._specialAllocBytes = 0;
  // Don't copy buffer pointers - allocateBuffers will create new ones
  result._primaryBuffer = nullptr;
  result._specialBuffer = nullptr;
  // Don't copy ownership flags - we'll own the new buffers
  result._isOwnerPrimary = false;
  result._isOwnerSpecial = false;
  result.allocateBuffers(true);
  result.copyCounters(*this);
  result.copyBufferFrom(*this);
  return result;
}

template <typename T>
void* DataBuffer::primaryAtOffset(const LongType offset) {
  // Validate buffer integrity before returning pointer to prevent use-after-free crashes in BLAS/cuBLAS
  validateIntegrity();

  if(_primaryBuffer == nullptr)
    return nullptr;
  T *type = reinterpret_cast<T*>(_primaryBuffer);
  return reinterpret_cast<void *>(type + offset);
}
template <typename T>
void* DataBuffer::specialAtOffset(const LongType offset) {
  // Validate buffer integrity before returning pointer to prevent use-after-free crashes
  validateIntegrity();

  if(_specialBuffer == nullptr)
    return nullptr;
  T *type = reinterpret_cast<T*>(_specialBuffer);
  return reinterpret_cast<void *>(type + offset);
}

// Explicit template instantiations for primaryAtOffset and specialAtOffset
// (ITERATE_LIST macro doesn't work with MSVC old preprocessor on Windows CUDA)
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<bool>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<float16>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<bfloat16>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<float>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<double>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<int8_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<uint8_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<int16_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<int32_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<sd::LongType>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<uint16_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<uint32_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<uint64_t>(sd::LongType offset);

template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<bool>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<float16>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<bfloat16>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<float>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<double>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<int8_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<uint8_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<int16_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<int32_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<sd::LongType>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<uint16_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<uint32_t>(sd::LongType offset);
template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<uint64_t>(sd::LongType offset);


template <typename T>
void _printHostBuffer(DataBuffer* buffer, long offset) {
  sd::LongType len = buffer->getNumElements();
  auto buff = buffer->template primaryAsT<T>();


  sd::LongType limit = len;
  if (limit == -1 || limit >= buffer->getNumElements()) {
    limit = buffer->getNumElements();
  }

  const char* msg = nullptr;
  if (msg != nullptr) {
    printf("%s: ", msg);
  } else {
    printf("[");
  }

  sd::DataType dataType = buffer->getDataType();
  auto baseOffset = offset;
  if (dataType == sd::DataType::DOUBLE || dataType == sd::DataType::FLOAT32) {
    for (sd::LongType e = baseOffset; e < limit; e++) {
      if (e > offset) printf(", ");
      if (dataType == sd::DataType::DOUBLE) {
        printf("%.15f", buff[e]);
      } else {
        printf("%.15f", static_cast<float>(buff[e]));
      }
    }
  } else if (dataType == sd::DataType::INT64 || dataType == sd::DataType::UINT64 ||
             dataType == sd::DataType::INT32 || dataType == sd::DataType::UINT32) {
    for (sd::LongType e = baseOffset; e < limit; e++) {
      if (dataType == sd::DataType::INT64 || dataType == sd::DataType::UINT64) {
        printf("%lld", static_cast<long long>(buff[e]));
      } else {
        printf("%d", static_cast<int>(buff[e]));
      }

      if (e < limit - 1) {
        printf(", ");
      }
    }
  } else if (dataType == sd::DataType::BOOL) {
    for (sd::LongType e = baseOffset; e < limit; e++) {
      if (static_cast<bool>(buff[e])) {
        printf("true");
      } else {
        printf("false");
      }

      if (e < limit - 1) {
        printf(", ");
      }
    }
  } else if (dataType == sd::DataType::UTF8 || dataType == sd::DataType::UTF16 ||
             dataType == sd::DataType::UTF32) {
    for (sd::LongType e = baseOffset; e < limit; e++) {
      printf("\"%s\"", reinterpret_cast<const char*>(&buff[e]));
      if (e < limit - 1) {
        printf(", ");
      }
    }
  }

  printf("]\n");
  fflush(stdout);
}

void DataBuffer::printHostDevice(long offset) {
  THROW_EXCEPTION("");
}

void DataBuffer::printSpecialAllocationTraces() {
  //no op on purpose
}

void DataBuffer::showBufferLimited() {

}

template <typename T>
SD_KERNEL void printDeviceBufferKernel(void* buffer, sd::LongType offset, sd::LongType length) {
  T* typedBuffer = reinterpret_cast<T*>(buffer);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[ ");
    for (sd::LongType i = offset; i < offset + length; i++) {
      // Cast to double for consistent formatting
      printf("%g ", (double)typedBuffer[i]);
    }
    printf("]");
  }
}

BUILD_SINGLE_TEMPLATE( SD_KERNEL void printDeviceBufferKernel,(void* buffer, sd::LongType offset, sd::LongType length),SD_COMMON_TYPES);


// Wrapper function to launch the kernel
template <typename T>
void launchPrintDeviceBufferKernel(void* buffer, sd::LongType offset, sd::LongType length) {
  // Cache the stream reference for consistent usage
  auto stream = LaunchContext::defaultContext()->getCudaStream();
  printDeviceBufferKernel<T><<<1, 1, 32*1024, *stream>>>(
      buffer, offset, length);
  cudaStreamSynchronize(*stream);
  sd::DebugHelper::checkErrorCode(stream, "printBufferDebug kernel failed");
}
BUILD_SINGLE_TEMPLATE( void launchPrintDeviceBufferKernel,(void* buffer, sd::LongType offset, sd::LongType length),SD_COMMON_TYPES);


template <typename T>
void DataBuffer::printHostBufferContent(void* buffer, sd::LongType offset, sd::LongType length) {
  T* typedBuffer = reinterpret_cast<T*>(buffer);

  sd_printf("[ ", 0);
  for (sd::LongType i = offset; i < offset + length; i++) {
    // For numeric types, cast to double for consistent formatting
    if (std::is_arithmetic<T>::value) {
      sd_printf("%g ", (double)typedBuffer[i]);
    } else {
      // For non-numeric types, print as hex
      sd_printf("0x%x ", *reinterpret_cast<int*>(&typedBuffer[i]));
    }
  }
  sd_printf("]", 0);
}
BUILD_SINGLE_TEMPLATE( SD_LIB_EXPORT void DataBuffer::printHostBufferContent,(void* buffer, sd::LongType offset, sd::LongType length),SD_COMMON_TYPES);


// DataBuffer implementation for .cu file
void DataBuffer::printBufferDebug(const char* msg, sd::LongType offset, sd::LongType limit) {
  if (msg) sd_printf("%s:\n", msg);

  // Print metadata
  sd_printf("DataBuffer: DataType=%s, Length=%lld elements, DeviceId=%d\n",
            DataTypeUtils::asString(_dataType).c_str(), (long long)getNumElements(), deviceId());

  // Print host buffer content
  if (_primaryBuffer != nullptr) {
    sd_printf("Host buffer (@%p): ", _primaryBuffer);

    sd::LongType len = getNumElements();
    sd::LongType printLen = limit < 0 ? len : std::min(len - offset, limit);

    // Print based on datatype
    BUILD_SINGLE_SELECTOR(_dataType, printHostBufferContent,
                          (_primaryBuffer, offset, printLen), SD_COMMON_TYPES);

    if (offset + printLen < len) sd_printf("... ", 0);
    sd_printf("\n", 0);
  } else {
    sd_printf("Host buffer: nullptr\n", 0);
  }

  // Print device buffer using kernel
  if (_specialBuffer != nullptr) {
    sd_printf("Device buffer (@%p): ", _specialBuffer);

    sd::LongType len = getNumElements();
    sd::LongType printLen = limit < 0 ? len : std::min(len - offset, limit);

    // Launch kernel through wrapper function
    BUILD_SINGLE_SELECTOR(_dataType, launchPrintDeviceBufferKernel,
                          (_specialBuffer, offset, printLen), SD_COMMON_TYPES);

    sd_printf("\n", 0);
  } else {
    sd_printf("Device buffer: nullptr\n", 0);
  }

  // Print sync state counters
  sd_printf("Sync state: _counter=%lld, _writePrimary=%lld, _writeSpecial=%lld, _readPrimary=%lld, _readSpecial=%lld\n",
            (long long)_counter.load(), (long long)_writePrimary.load(), (long long)_writeSpecial.load(),
            (long long)_readPrimary.load(), (long long)_readSpecial.load());
  sd_printf("isPrimaryActual=%d, isSpecialActual=%d\n", isPrimaryActual(), isSpecialActual());
}



void DataBuffer::showCounters(const char* msg1, const char* msg2) {
  sd_debug("%s %s || primary %p special %p :: wP: %d wS: %d rP: %d rS: %d\n", msg1, msg2, _primaryBuffer,
           _specialBuffer, (int)_writePrimary.load(), (int)_writeSpecial.load(), (int)_readPrimary.load(),
           (int)_readSpecial.load());
}
////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateSpecial() {
  // Fast path: if special buffer already exists on the correct device,
  // no mutation occurs — skip the frozen guard entirely.
  if (_specialBuffer != nullptr) {
    if (isConstant) {
      return;
    }
    auto currentDeviceId = AffinityManager::currentDeviceId();
    auto bufferDeviceId = _deviceId.load();
    if (bufferDeviceId == currentDeviceId) {
      return;  // Already allocated on correct device — no-op
    }
    // Buffer exists but on wrong device — migrate requires mutation.
    // If frozen, skip: the buffer is pinned to its freeze-time device.
    if (_frozenRefCount.load(std::memory_order_relaxed) > 0) {
      return;
    }
    migrate();
    return;
  }
  // Actual allocation needed — guard against frozen mutation
  throwIfFrozen("allocateSpecial");

  if (_lenInBytes == 0) {
    // Use getLenInBytes() which handles scalar fallback (sizeOfElement for 0-length buffers)
    auto computedLen = getLenInBytes();
    if (computedLen > 0) {
      // Scalar or uninitialized buffer — fix _lenInBytes and proceed with allocation
      _lenInBytes = computedLen;
    } else {
      std::string errorMessage;
      errorMessage += "DataBuffer::allocateSpecial: ";
      errorMessage += "Special buffer is already allocated";
      errorMessage += " or length is 0";
      errorMessage += "Length is: ";
      errorMessage += std::to_string(computedLen);
      errorMessage += "Special buffer is nullptr : ";
      errorMessage += std::to_string(_specialBuffer == nullptr);
      THROW_EXCEPTION(errorMessage.c_str());
    }
  }
#if defined(SD_GCC_FUNCTRACE)
  if(sd::env_isFuncTracePrintAllocate()) {
    allocationStackTraceSpecial = new StackTrace();
    allocationStackTraceSpecial->load_here();
  }

#endif

  if (_specialBuffer == nullptr) {
    auto deviceId = AffinityManager::currentDeviceId();

    cudaError_t pendingErr = cudaGetLastError();
    if (pendingErr != cudaSuccess) {
      // Log the error but continue - clearing it may allow allocation to proceed
      sd_debug("DataBuffer::allocateSpecial: Cleared pending CUDA error before allocation: %s\n",
               cudaGetErrorString(pendingErr));
    }

    if (_workspace == nullptr) {
      if (!memory::MemoryCounter::getInstance().validate(getLenInBytes())) {
        std::string errorMessage;
        errorMessage += "DataBuffer::allocateSpecial: ";
        errorMessage += "Requested amount exceeds device limits";
        errorMessage += "DeviceId: ";
        errorMessage += std::to_string(deviceId);
        errorMessage += "Device limit: ";
        errorMessage += std::to_string(memory::MemoryCounter::getInstance().deviceLimit(deviceId));
        errorMessage += "Requested amount: ";
        errorMessage += std::to_string(getLenInBytes());
        errorMessage += "Special buffer is nullptr : ";
        errorMessage += std::to_string(_specialBuffer == nullptr);
        THROW_EXCEPTION(errorMessage.c_str());
      }
    }

    // Allocate device memory, tracking which device it actually ends up on
    // (failover may place it on a different GPU or pinned host memory)
    int actualDevice = deviceId;
    if (_workspace == nullptr) {
      size_t allocSize = getLenInBytes() + 8;

      // ── Capture workspace bump allocation ────────────────────────────────
      // During CUDA graph capture, allocate from the pre-allocated capture
      // workspace (bump allocator) instead of CudaMemoryPool::allocate().
      //
      // WHY: CudaMemoryPool uses cudaMallocAsync, which creates MemAlloc
      // graph nodes. The async pool reuses freed GPU addresses within the
      // same step — two ops can get the same address if the first op's
      // output is freed before the second allocates. The captured graph
      // bakes these overlapping addresses. On replay, the pool may resolve
      // differently → address mismatch → wrong data flows between ops.
      // This was the root cause of the 49.6% accuracy mega-graph bug
      // (commit 321884f564).
      //
      // The capture workspace is a single contiguous GPU allocation made
      // BEFORE capture begins. Bump-allocating from it guarantees:
      //   1. No MemAlloc/MemFree graph nodes (no async pool interaction)
      //   2. Unique non-overlapping addresses (monotonic bump pointer)
      //   3. Addresses persist across replay (workspace lives for graph lifetime)
      //
      // Same pattern as PointersManager::allocateDevMem() (line 66-76).
      // deleteSpecial() already handles workspace-interior buffers correctly
      // (lines 1060-1079: checks isInCaptureWorkspace, skips CUDA free).
      if (tl_graphExecutionActive && tl_captureWorkspace != nullptr) {
        size_t aligned = (allocSize + 255) & ~255ULL;  // 256-byte alignment
        if (tl_captureWorkspaceOffset + aligned <= tl_captureWorkspaceSize) {
          _specialBuffer = reinterpret_cast<int8_t*>(
              static_cast<char*>(tl_captureWorkspace) + tl_captureWorkspaceOffset);
          tl_captureWorkspaceOffset += aligned;
          // Workspace-allocated: NOT owned by this DataBuffer (workspace
          // lifecycle manages the memory). Set _isOwnerSpecial=false below
          // so deleteSpecial() doesn't try to free an interior pointer.
          _isOwnerSpecial = false;
          _specialAllocBytes = allocSize;
          _specialDeviceId.store(deviceId);
          _deviceId.store(deviceId);
          tl_dspAllocBytes += getLenInBytes();
          tl_dspAllocCount++;
#if defined(SD_GCC_FUNCTRACE)
          array::DataBufferLifecycleTracker::getInstance().recordAllocation(
              _specialBuffer, getLenInBytes(), getDataType(),
              array::BufferType::SPECIAL, this, true /*workspace*/);
#endif
          DSP_DIAG(MEMORY, "CAPTURE_WORKSPACE_ALLOC: %zu bytes @ offset=%zu/%zu ptr=%p",
                   allocSize, tl_captureWorkspaceOffset - aligned,
                   tl_captureWorkspaceSize, (void*)_specialBuffer);
          // Skip the pool allocation, failover, and memory counter paths below.
          // Jump to the end of the allocation block (after _isOwnerSpecial is set).
          goto alloc_done;
        }
        // Workspace exhausted — fall through to pool allocation.
        // This is NOT ideal during capture (creates MemAlloc nodes), but
        // prevents hard failure. The capture workspace size should be tuned
        // to avoid this (Environment::dspCaptureWorkspaceMb).
        DSP_DIAG(MEMORY, "CAPTURE_WORKSPACE_EXHAUSTED: need %zu, remaining %zu/%zu — "
                 "falling through to cudaMallocAsync (will create MemAlloc graph node)",
                 aligned, tl_captureWorkspaceSize - tl_captureWorkspaceOffset,
                 tl_captureWorkspaceSize);
      }

      // During CUDA graph capture, allocations MUST use the captured stream.
      // Using nullptr (legacy default stream) causes implicit sync with the captured
      // stream in DEFAULT scheduling mode, invalidating the capture (error 901).
      // LaunchContext is guaranteed to be initialized during capture (we're executing ops).
      cudaStream_t allocStream = nullptr;
      if (tl_graphExecutionActive) {
        allocStream = captureSafeStreamOrDefault();
      }
      _specialBuffer = reinterpret_cast<int8_t*>(
          memory::CudaMemoryPool::getInstance().allocate(allocSize, deviceId, allocStream, &actualDevice));
      if (_specialBuffer == nullptr) {
        THROW_EXCEPTION("[DEVICE] allocation failed");
      }
      // If failover landed on a non-peer device, switch to pinned host (UVA-accessible).
      void* specialBuf = reinterpret_cast<void*>(_specialBuffer);
      actualDevice = handleNonPeerFailover(specialBuf, allocSize, deviceId, actualDevice, "DataBuffer::allocateSpecial");
      _specialBuffer = reinterpret_cast<int8_t*>(specialBuf);
    } else {
      size_t allocSize = getLenInBytes() + 8;
      _specialBuffer = reinterpret_cast<int8_t*>(
          _workspace->allocateBytes(memory::MemoryType::DEVICE, allocSize));
      actualDevice = deviceId;  // workspace allocations stay on requested device
    }
    _isOwnerSpecial = true;
    _specialAllocBytes = getLenInBytes() + 8;  // match actual allocation (getLenInBytes() + 8)
    tl_dspAllocBytes += getLenInBytes();
    tl_dspAllocCount++;

    // Store the ACTUAL device where special buffer was allocated, not the requested device
    // This is critical for multi-GPU: failover may allocate on a different GPU
    _deviceId.store(actualDevice);
    _specialDeviceId.store(actualDevice);

#if defined(SD_GCC_FUNCTRACE)
    // Record SPECIAL (device) buffer allocation
    array::DataBufferLifecycleTracker::getInstance().recordAllocation(
        _specialBuffer, getLenInBytes(), getDataType(),
       array::BufferType::SPECIAL, this, _workspace != nullptr);
#endif

    if (_workspace == nullptr) {
      memory::MemoryCounter::getInstance().countIn(actualDevice >= 0 ? actualDevice : deviceId, getLenInBytes());
      memory::MemoryCounter::getInstance().countIn(memory::MemoryType::DEVICE, getLenInBytes());
    }
alloc_done:  // Target for capture-workspace goto (skips pool alloc + counters already set)
    (void)0;  // Empty statement after label (C++ requires statement after label)
  } else if(getLenInBytes() == 0) {
    std::string errorMessage;
    errorMessage += "DataBuffer::allocateSpecial: ";
    errorMessage += "Special buffer is already allocated";
    errorMessage += " or length is 0";
    errorMessage += "Length is: ";
    errorMessage += std::to_string(getLenInBytes());
    errorMessage += "Special buffer is nullptr : ";
    errorMessage += std::to_string(_specialBuffer == nullptr);
    THROW_EXCEPTION(errorMessage.c_str());
  }
}

void DataBuffer::waitForSpecialWriteEvent(void* streamPtr) const {
  if (!_writeEventRecorded.load(std::memory_order_acquire) || _writeEvent == nullptr) return;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamPtr);
  auto err = cudaStreamWaitEvent(stream, reinterpret_cast<cudaEvent_t>(_writeEvent), 0);
  if (err != cudaSuccess) {
    throwCudaStatus("DataBuffer::waitForSpecialWriteEvent: cudaStreamWaitEvent failed", err);
  }
}

void DataBuffer::recordSpecialWriteEvent(void* streamPtr) const {
  // During DSP execution, all ops run on the same tl_dspExecutionStream so
  // ordering is guaranteed by the stream itself. Per-buffer write events are
  // unnecessary and dangerous: temporary buffers allocated during device-drift
  // (cudaGetDevice() returning the wrong device because LaunchContext init
  // iterates all devices) get _deviceId on device 0 while the DSP stream is on
  // device 1, causing cudaEventRecord error 400 (invalid resource handle).
  // Cross-stream sync after DSP completes is handled by
  // dspPublishThreadCompletionEvent() — same pattern as graph capture mode.
  if (tl_dspExecutionStream != nullptr) {
    return;
  }

  // During CUDA graph capture, per-buffer events are also skipped.
  if (tl_graphExecutionActive && tl_graphCaptureStream != nullptr) {
    return;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamPtr);

  int targetDevice = -1;
  auto devErr = cudaGetDevice(&targetDevice);
  if (devErr != cudaSuccess) {
    throwCudaStatus("DataBuffer::recordSpecialWriteEvent: cudaGetDevice failed", devErr);
  }

  int eventDevice = _writeEventDeviceId.load(std::memory_order_acquire);
  if (_writeEvent != nullptr && eventDevice != targetDevice) {
    // Event is on wrong device — destroy on its original device and recreate
    if (eventDevice >= 0 && eventDevice != targetDevice) {
      cudaSetDevice(eventDevice);
      cudaEventDestroy(reinterpret_cast<cudaEvent_t>(_writeEvent));
      cudaSetDevice(targetDevice);
    } else {
      cudaEventDestroy(reinterpret_cast<cudaEvent_t>(_writeEvent));
    }
    _writeEvent = nullptr;
    _writeEventRecorded.store(false, std::memory_order_release);
    _writeEventDeviceId.store(-1, std::memory_order_release);
  }

  if (_writeEvent == nullptr) {
    cudaEvent_t evt = nullptr;
    auto createErr = cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
    if (createErr != cudaSuccess) {
      throwCudaStatus("DataBuffer::recordSpecialWriteEvent: cudaEventCreateWithFlags failed", createErr);
    }
    _writeEvent = reinterpret_cast<void*>(evt);
    _writeEventDeviceId.store(targetDevice, std::memory_order_release);
  }

  auto recordErr = cudaEventRecord(reinterpret_cast<cudaEvent_t>(_writeEvent), stream);
  if (recordErr != cudaSuccess) {
    throwCudaStatus("DataBuffer::recordSpecialWriteEvent: cudaEventRecord failed", recordErr);
  }
  _writeEventRecorded.store(true, std::memory_order_release);
}

void DataBuffer::clearSpecialWriteEvent() const {
  if (_writeEvent == nullptr) return;

  int currentDevice = -1;
  bool switchedDevice = false;
  int eventDevice = _writeEventDeviceId.load(std::memory_order_acquire);
  if (cudaGetDevice(&currentDevice) == cudaSuccess && eventDevice >= 0 && currentDevice != eventDevice) {
    if (cudaSetDevice(eventDevice) == cudaSuccess) {
      switchedDevice = true;
    }
  }
  cudaEventDestroy(reinterpret_cast<cudaEvent_t>(_writeEvent));
  if (switchedDevice && currentDevice >= 0) {
    cudaSetDevice(currentDevice);
  }
  _writeEvent = nullptr;
  _writeEventRecorded.store(false, std::memory_order_release);
  _writeEventDeviceId.store(-1, std::memory_order_release);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToPrimary(const LaunchContext* context, const bool forceSync) {
  if (_specialBuffer == nullptr || _lenInBytes == 0 || closed) {
    return;
  }

  // Constant buffers (shape DataBuffers created via dbCreateConstantExternalDataBuffer)
  // are initialized with valid host data (shapeCopy) and must NEVER be overwritten by
  // D2H transfers. The counter state for these buffers has _writeSpecial=1 and all
  // other counters=0, so isPrimaryActual() permanently returns false, causing every
  // syncToPrimary call to proceed. If the GPU shape pointer (_specialBuffer) becomes
  // stale (e.g., after DirectShapeTrie::clearCache() frees it), the cudaMemcpyAsync
  // would corrupt shapeCopy with whatever GPU memory happens to be at that address —
  // causing ConstantShapeBuffer data corruption and downstream crashes/wrong results.
  // EXCEPTION: forceSync=true bypasses this guard. Java setCloseable(false) marks
  // data buffers as constant to protect them from deallocation, but those buffers
  // still need D2H sync for serialization and cross-device migration.
  if (isConstant && !forceSync) {
    return;
  }

  // During graph execution (CUDA Graphs capture, oneDNN Graph, ACL Dynamic Fusion),
  // D2H transfers are forbidden — for CUDA, they create illegal dependencies between
  // the capture stream and the legacy stream; for CPU graphs, they cause unnecessary
  // data movement. Data stays on the compute device; the warmup pass already computed
  // any needed host-side values.
  // EXCEPTION: forceSync=true bypasses this guard. Cross-device migration (Java
  // CudaZeroHandler.memcpy) calls dbForceSyncToPrimary which sets forceSync=true.
  // Without this exception, migration during DSP execution sees zeros because the
  // D2H from the source device is skipped.
  if (tl_graphExecutionActive && !forceSync) {
    return;
  }

  // During DSP composite replay, gap ops run with frozen shapes — all shape
  // values are cached and don't need D2H sync. syncToPrimary triggers
  // cudaStreamSynchronize(stream=0) which is a full GPU pipeline drain
  // (~30-50us per call). With 818 reshape_no_copy calls per step, this adds
  // ~40ms/step of pure sync overhead. Skip D2H entirely during replay.
  // EXCEPTION: forceSync=true bypasses this guard for the same reason as above.
  if (tl_dspReplayActive && !forceSync) {
    return;
  }

  if (isPrimaryActual() && !forceSync) {
    return;
  }

  allocatePrimary();

  // If primary buffer exists but is undersized (e.g., setPrimaryBuffer was called
  // with a smaller size, then setSpecialBuffer increased _lenInBytes), reallocate
  // to prevent buffer overrun during cudaMemcpy.
  if (_primaryBuffer != nullptr && _primaryAllocBytes > 0 && _primaryAllocBytes < getLenInBytes()) {
    if (_isOwnerPrimary) {
      auto ipb = reinterpret_cast<int8_t*>(_primaryBuffer);
      RELEASE(ipb, _workspace);
    }
    _primaryBuffer = nullptr;
    _primaryAllocBytes = 0;
    allocatePrimary();
  }

  // Use _specialDeviceId for operations on the special buffer
  auto bufferDeviceId = _specialDeviceId.load();
  if (bufferDeviceId < 0) {
    bufferDeviceId = _deviceId.load();  // Fallback for legacy code
  }
  auto currentDeviceId = AffinityManager::currentDeviceId();

  // Verify actual device of the GPU pointer. _deviceId metadata can get out of sync
  // with the actual pointer location during cross-device DSP execution (e.g., when
  // allocateFailover places data on a different device, or when Java-side device
  // routing overrides the expected allocation device). Using the wrong device causes
  // cudaMemcpyAsync to fail with cudaErrorInvalidValue.
  cudaPointerAttributes ptrAttrs;
  auto attrRes = cudaPointerGetAttributes(&ptrAttrs, _specialBuffer);
  if (attrRes == cudaSuccess && ptrAttrs.type == cudaMemoryTypeDevice) {
    if (ptrAttrs.device != bufferDeviceId) {
      bufferDeviceId = ptrAttrs.device;
    }
  } else {
    cudaGetLastError(); // clear any error from the query
  }

  bool switchedDevice = false;
  if (currentDeviceId != bufferDeviceId) {
    cudaSetDevice(bufferDeviceId);
    switchedDevice = true;
  }

  // Always use stream 0 (default stream) for D2H transfers.
  // Rationale: context->getCudaStream() accesses thread-local ContextBuffers, which
  // reinitializes when it detects a device change (release old streams + create new
  // ones on the current device). The freshly created stream has NO ordering relationship
  // with prior cudaMallocAsync pool allocations or kernel writes, causing
  // cudaMemcpyAsync to fail with cudaErrorInvalidValue on pool-allocated buffers.
  // This happens during cross-device DSP execution where ops run on device 1 but the
  // thread was originally on device 0. Stream 0 avoids this because it implicitly
  // orders after all prior operations on the device.
  // Performance impact is minimal since syncToPrimary always calls cudaStreamSynchronize
  // after the memcpy, making it a blocking call regardless of which stream is used.
  cudaStream_t stream = 0;

  cudaError_t res;
  if (_writeEventRecorded.load()) {
    waitForSpecialWriteEvent(stream);
    _writeEventRecorded.store(false);  // Reset for next write
  }

  // Track D2H transfer
  auto startTime = std::chrono::high_resolution_clock::now();

  // Use async memcpy - works best with pinned (page-locked) host memory
  // With CudaPinnedMemoryPool, _primaryBuffer is pinned, enabling true async DMA
  res = cudaMemcpyAsync(_primaryBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToHost, stream);
  if (res != cudaSuccess) {
    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
    int actualDev = -1;
    cudaGetDevice(&actualDev);
    std::string errorMessage;
    errorMessage += "DataBuffer::syncToPrimary: cudaMemcpyAsync failed: ";
    errorMessage += std::to_string(getLenInBytes());
    errorMessage += " ";
    errorMessage += cudaGetErrorString(res);
    errorMessage += " (buffer device: ";
    errorMessage += std::to_string(bufferDeviceId);
    errorMessage += ", actual cuda device: ";
    errorMessage += std::to_string(actualDev);
    errorMessage += ", primary=";
    errorMessage += std::to_string(reinterpret_cast<uintptr_t>(_primaryBuffer));
    errorMessage += ", special=";
    errorMessage += std::to_string(reinterpret_cast<uintptr_t>(_specialBuffer));
    errorMessage += ", forceSync=";
    errorMessage += std::to_string(forceSync);
    errorMessage += ", stream=";
    errorMessage += std::to_string(reinterpret_cast<uintptr_t>(stream));
    // Query actual device ownership of the special buffer pointer
    cudaPointerAttributes ptrAttrs;
    auto attrRes = cudaPointerGetAttributes(&ptrAttrs, _specialBuffer);
    if (attrRes == cudaSuccess) {
      errorMessage += ", ptrDevice=";
      errorMessage += std::to_string(ptrAttrs.device);
      errorMessage += ", ptrType=";
      errorMessage += std::to_string(static_cast<int>(ptrAttrs.type));
    } else {
      errorMessage += ", ptrQuery=FAILED(";
      errorMessage += cudaGetErrorString(attrRes);
      errorMessage += ")";
      cudaGetLastError(); // clear error
    }
    errorMessage += ")";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  // Must synchronize after async D2H to ensure data is available on host
  // This is required because the caller expects data to be ready after this call
  res = cudaStreamSynchronize(stream);
  if (res != cudaSuccess) {
    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
    std::string errorMessage;
    errorMessage += "DataBuffer::syncToPrimary: stream sync failed: ";
    errorMessage += cudaGetErrorString(res);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
  TransferMetrics::getInstance().recordTransfer(TransferType::DEVICE_TO_HOST, getLenInBytes(),
                                                 durationNs, bufferDeviceId, -1);

  // DSP diagnostics: D2H transfer
  DSP_DIAG(TRANSFER, "D2H: %lld bytes, dev=%d, duration=%.2f us",
           (long long)getLenInBytes(), bufferDeviceId, durationNs / 1000.0);

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  readPrimary();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToSpecial(const bool forceSync) {
  // Nothing to do for null or zero-length buffers
  if (_primaryBuffer == nullptr || _lenInBytes == 0) return;

  if (isSpecialActual() && !forceSync) {
    return;
  }

  // If the host buffer was never written to, there is nothing meaningful to sync.
  // This prevents copying uninitialized host garbage to the device for newly created
  // output arrays (e.g., createUninitialized), which would overwrite valid device data
  // or create stale values that survive kernel execution due to stream ordering.
  //
  // Exception: when ALL counters are zero (neither host nor device is "actual") AND
  // the device buffer was never written to (_writeSpecial == 0), a host buffer with
  // valid data from external initialization (e.g., Java/JNI via DataBuffer ctor or
  // SameDiff.dup()) would never get synced. In this case, the host data should be
  // considered the source of truth — the device buffer is freshly allocated (zeros)
  // and needs the host data copied.
  if (!forceSync && !isPrimaryActual()) {
    // Only skip if the device was explicitly written at some point, meaning it has
    // real data and the host doesn't. When _writeSpecial == 0, the device buffer
    // was never written — fall through to do the H2D copy.
    if (_writeSpecial.load() > 0) {
      return;
    }
  }

  // During CUDA graph capture, use a capture-safe path:
  // - Skip cudaPointerGetAttributes (synchronous query breaks capture)
  // - Use the captured stream instead of stream 0 (default stream syncs with captured stream)
  // - Skip cudaStreamSynchronize (explicit sync breaks capture)
  //
  //  Allocate a PINNED HOST COPY of the data and use it as the H2D source.
  // The original _primaryBuffer may be freed after capture (e.g., for temporary axis/dimension
  // parameter arrays in gap op Contexts). The CUDA graph's H2D memcpy node bakes the source
  // address — if that address points to freed memory, graph replay reads garbage data.
  // The pinned copy is added to tl_capturedHostPtrs and persists for the graph's lifetime.
  if (tl_graphExecutionActive) {
    allocateSpecial();
    if (_specialBuffer == nullptr) return;

    // If device data is already current, skip recording the H2D memcpy node.
    // Capture buffers (for placeholder external inputs) have both host and device
    // data initialized before capture. Recording a redundant H2D bakes a pinned
    // workspace address into the graph. On replay, the D2D copy correctly updates
    // the capture buffer with fresh data BEFORE graph launch, but then the graph's
    // H2D node overwrites it with stale pinned-workspace data from capture time.
    // Skipping the H2D when device is already current prevents this stale overwrite.
    // Temporary arrays (axis/dimension parameters) have isSpecialActual()=false
    // and still get the pinned workspace H2D treatment they need.
    if (isSpecialActual()) {
      return;
    }

    // Use capture host workspace bump allocator for persistent pinned copy.
    // The H2D memcpy node bakes the source address — if _primaryBuffer is freed
    // after capture, graph replay reads garbage. The pinned copy in the workspace
    // persists for the graph's lifetime (freed when replay handle is destroyed).
    void* h2dSource = _primaryBuffer;
    if (tl_captureHostWorkspace != nullptr) {
      size_t aligned = (getLenInBytes() + 255) & ~255ULL;
      if (tl_captureHostWorkspaceOffset + aligned <= tl_captureHostWorkspaceSize) {
        void* pinnedCopy = static_cast<char*>(tl_captureHostWorkspace) + tl_captureHostWorkspaceOffset;
        tl_captureHostWorkspaceOffset += aligned;
        std::memcpy(pinnedCopy, _primaryBuffer, getLenInBytes());
        h2dSource = pinnedCopy;
      }
      // If workspace exhausted, fall through to use _primaryBuffer directly
    }

    cudaStream_t capturedStream = captureSafeStreamOrDefault();
    auto res = cudaMemcpyAsync(_specialBuffer, h2dSource, getLenInBytes(),
                               cudaMemcpyHostToDevice, capturedStream);
    DSP_DIAG(EXECUTE, "CAPTURE_H2D(DataBuffer): size=%zu src=%p dst=%p isPinned=%d stream=%p",
             getLenInBytes(), h2dSource, _specialBuffer,
             (h2dSource != _primaryBuffer) ? 1 : 0, (void*)capturedStream);
    if (res != cudaSuccess) {
      cudaGetLastError();  // Clear error
    }
    writeSpecial();
    return;
  }

  allocateSpecial();

  // If special buffer exists but is undersized, reallocate to prevent overrun
  if (_specialBuffer != nullptr && _specialAllocBytes > 0 && _specialAllocBytes < getLenInBytes()) {
    if (_isOwnerSpecial) {
      //  Get device ID BEFORE releasing - buffer may be on different device due to failover
      auto bufferDeviceId = _specialDeviceId.load();
      if (bufferDeviceId < 0) {
        bufferDeviceId = _deviceId.load();
      }
      auto currentDeviceId = AffinityManager::currentDeviceId();
      bool switchedDevice = false;
      if (currentDeviceId != bufferDeviceId && bufferDeviceId >= 0) {
        cudaSetDevice(bufferDeviceId);
        switchedDevice = true;
      }
      auto isb = reinterpret_cast<int8_t*>(_specialBuffer);
      RELEASE_SPECIAL_WITH_DEVICE(isb, bufferDeviceId, _workspace);
      if (switchedDevice) {
        cudaSetDevice(currentDeviceId);
      }
    }
    _specialBuffer = nullptr;
    _specialAllocBytes = 0;
    allocateSpecial();
  }

  // Use _specialDeviceId for operations on the special buffer
  auto bufferDeviceId = _specialDeviceId.load();
  if (bufferDeviceId < 0) {
    bufferDeviceId = _deviceId.load();  // Fallback for legacy code
  }
  auto currentDeviceId = AffinityManager::currentDeviceId();

  // Verify actual device of the GPU pointer (same as syncToPrimary — see comment there)
  cudaPointerAttributes ptrAttrsSync;
  auto attrResSync = cudaPointerGetAttributes(&ptrAttrsSync, _specialBuffer);
  if (attrResSync == cudaSuccess && ptrAttrsSync.type == cudaMemoryTypeDevice) {
    if (ptrAttrsSync.device != bufferDeviceId) {
      bufferDeviceId = ptrAttrsSync.device;
    }
  } else {
    // cudaPointerGetAttributes failed — _specialBuffer is NOT a valid CUDA pointer.
    // This can happen if the buffer was freed/corrupted between allocations.
    // Previously we silently continued and passed the bogus pointer to cudaMemcpyAsync,
    // which caused SIGSEGV inside libcuda.so when the driver tried to resolve the address.
    cudaGetLastError();  // clear sticky error
    DSP_DIAG(MEMORY, "syncToSpecial: _specialBuffer=%p failed cudaPointerGetAttributes "
             "(err=%d: %s). len=%zu deviceId=%d isClosed=%d — attempting realloc",
             _specialBuffer, static_cast<int>(attrResSync), cudaGetErrorString(attrResSync),
             getLenInBytes(), bufferDeviceId, isClosed() ? 1 : 0);
    cudaGetLastError();  // clear error from the GetErrorString
    // Attempt to reallocate the special buffer on the correct device
    _specialBuffer = nullptr;
    _specialAllocBytes = 0;
    allocateSpecial();
    if (_specialBuffer == nullptr) {
      DSP_DIAG(MEMORY, "syncToSpecial: reallocation of _specialBuffer FAILED — aborting H2D copy");
      return;  // Cannot proceed without a valid device buffer
    }
    DSP_DIAG(MEMORY, "syncToSpecial: reallocated _specialBuffer=%p (%zu bytes)",
             _specialBuffer, _specialAllocBytes);
    // Re-verify the new allocation
    auto attrResRetry = cudaPointerGetAttributes(&ptrAttrsSync, _specialBuffer);
    if (attrResRetry != cudaSuccess) {
      cudaGetLastError();
      DSP_DIAG(MEMORY, "syncToSpecial: reallocated buffer also failed pointer check — aborting");
      return;
    }
    if (ptrAttrsSync.type == cudaMemoryTypeDevice) {
      bufferDeviceId = ptrAttrsSync.device;
    }
  }

  bool switchedDevice = false;
  if (currentDeviceId != bufferDeviceId) {
    cudaSetDevice(bufferDeviceId);
    switchedDevice = true;
  }

  // Track H2D transfer
  auto startTime = std::chrono::high_resolution_clock::now();

  cudaStream_t stream = asyncTransferStream(switchedDevice);

  // Validate stream and check for latent CUDA errors before cudaMemcpyAsync.
  // Catches: stale/destroyed streams, latent errors from prior ops.
  {
    cudaError_t prevErr = cudaGetLastError();
    if (prevErr != cudaSuccess) {
      DSP_DIAG(TRANSFER, "syncToSpecial: LATENT CUDA error before memcpy: %d (%s) "
               "dst=%p src=%p len=%zu stream=%p device=%d",
               (int)prevErr, cudaGetErrorString(prevErr),
               _specialBuffer, _primaryBuffer, getLenInBytes(), (void*)stream, bufferDeviceId);
      cudaGetLastError();  // clear so cudaMemcpyAsync doesn't inherit it
    }
    if (stream != 0) {
      // Skip cudaStreamQuery when the stream is being captured.
      // cudaStreamQuery is ILLEGAL during CUDA graph capture — it returns
      // cudaErrorStreamCaptureUnsupported (900) which permanently invalidates
      // the capture stream, causing all subsequent operations (cuLaunchKernel,
      // cudaMemcpyAsync, etc.) to fail with error 901 ("operation failed due
      // to a previous error during capture").
      bool skipStreamQuery = false;
      {
        cudaStreamCaptureStatus capStatus = cudaStreamCaptureStatusNone;
        cudaError_t capErr = cudaStreamIsCapturing(stream, &capStatus);
        if (capErr == cudaSuccess && capStatus != cudaStreamCaptureStatusNone) {
          skipStreamQuery = true;
        }
      }
      if (!skipStreamQuery) {
        cudaError_t streamErr = cudaStreamQuery(stream);
        // cudaSuccess or cudaErrorNotReady both mean stream is valid
        if (streamErr != cudaSuccess && streamErr != cudaErrorNotReady) {
          DSP_DIAG(TRANSFER, "syncToSpecial: INVALID stream=%p err=%d (%s) — falling back to per-thread stream. "
                   "dst=%p src=%p len=%zu device=%d",
                   (void*)stream, (int)streamErr, cudaGetErrorString(streamErr),
                   _specialBuffer, _primaryBuffer, getLenInBytes(), bufferDeviceId);
          cudaGetLastError();  // clear
          stream = cudaStreamPerThread;
        }
      }
    }
  }

  auto res = cudaMemcpyAsync(_specialBuffer, _primaryBuffer, getLenInBytes(), cudaMemcpyHostToDevice, stream);
  if (res != cudaSuccess) {
    // Restore device before throwing
    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
    std::string errorMessage;
    errorMessage += "Failed to copy dataBuffer::syncToSpecial: ";
    errorMessage += std::to_string(getLenInBytes());
    errorMessage += " ";
    errorMessage += cudaGetErrorString(res);
    errorMessage += " (buffer device: ";
    errorMessage += std::to_string(bufferDeviceId);
    errorMessage += ")";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  recordSpecialWriteEvent(stream);

  auto endTime = std::chrono::high_resolution_clock::now();
  auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
  TransferMetrics::getInstance().recordTransfer(TransferType::HOST_TO_DEVICE, getLenInBytes(),
                                                 durationNs, -1, bufferDeviceId);

  // DSP diagnostics: H2D transfer
  DSP_DIAG(TRANSFER, "H2D: %lld bytes, dev=%d, duration=%.2f us",
           (long long)getLenInBytes(), bufferDeviceId, durationNs / 1000.0);

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  readSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteSpecial() {
  clearSpecialWriteEvent();

  if (_isOwnerSpecial && _specialBuffer != nullptr && getLenInBytes() != 0) {
    // During CUDA graph capture, skip ALL GPU memory frees for non-workspace buffers.
    // Recording cudaFreeAsync for graph-external memory creates MemFree graph nodes
    // that reference stale addresses on cudaGraphLaunch → SIGSEGV.
    // The memory stays allocated; it will be freed when the NDArray is deleted
    // AFTER capture completes (via flushPendingClose or normal destruction).
    // Workspace-allocated memory is managed by the workspace lifecycle, not freed individually.
    if (tl_graphExecutionActive && _workspace == nullptr) {
      // Check if this buffer is within the capture workspace (bump-allocated during capture)
      bool inWorkspace = false;
      if (tl_captureWorkspace != nullptr) {
        char* wsStart = static_cast<char*>(tl_captureWorkspace);
        char* wsEnd = wsStart + tl_captureWorkspaceSize;
        char* p = static_cast<char*>(_specialBuffer);
        inWorkspace = (p >= wsStart && p < wsEnd);
      }
      if (!inWorkspace) {
        // Non-workspace buffer — do NOT free, do NOT reset pointer.
        // Caller (NDArray destructor or flushPendingClose) will retry after capture.
        tl_dspFreeSkipCount++;
        return;
      }
      // Workspace buffer — fall through to reset pointer (no actual CUDA free needed)
      _specialBuffer = nullptr;
      _specialDeviceId.store(-1);
      _isOwnerSpecial = false;
      return;
    }

    // Use the tracked device ID where special buffer was actually allocated
    // This is critical for multi-GPU: buffer may have been allocated on a different device
    // due to failover during OOM, and we must free from the correct device context
    auto bufferDeviceId = _specialDeviceId.load();
    if (bufferDeviceId < 0) {
      // Fallback to _deviceId if _specialDeviceId not set (legacy code path)
      bufferDeviceId = _deviceId.load();
    }

    auto currentDeviceId = AffinityManager::currentDeviceId();
    bool switchedDevice = false;

    if (currentDeviceId != bufferDeviceId && bufferDeviceId >= 0) {
      cudaSetDevice(bufferDeviceId);
      switchedDevice = true;
    }

    auto p = reinterpret_cast<int8_t*>(_specialBuffer);

    // Secondary guard: check if this buffer is within an active capture workspace.
    // The primary guard (line 968) only works DURING capture (tl_graphExecutionActive).
    // After capture ends, workspace-interior pointers from intermediates that outlive
    // capture would reach RELEASE_SPECIAL_WITH_DEVICE → CudaMemoryPool::free() →
    // cudaFreeAsync on interior pointer → "invalid argument" → cudaFree fallback →
    // CUDA context corruption. Short-circuit here to avoid all of that.
    if (memory::CudaMemoryPool::getInstance().isInCaptureWorkspace(_specialBuffer)) {
      _specialBuffer = nullptr;
      _specialDeviceId.store(-1);
      _isOwnerSpecial = false;
      if (switchedDevice) cudaSetDevice(currentDeviceId);
      return;
    }

#if defined(SD_GCC_FUNCTRACE)
    // Record SPECIAL (device) buffer deallocation before releasing
    array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
        _specialBuffer,array::BufferType::SPECIAL);
#endif
    // Use device-aware free - critical for multi-GPU correctness
    tl_dspFreeBytes += getLenInBytes();
    tl_dspFreeCount++;
    RELEASE_SPECIAL_WITH_DEVICE(p, bufferDeviceId, _workspace);

    // count out towards DataBuffer device, only if we're not in workspace
    if (_workspace == nullptr) {
      sd::memory::MemoryCounter::getInstance().countOut(bufferDeviceId, getLenInBytes());
      sd::memory::MemoryCounter::getInstance().countOut(sd::memory::MemoryType::DEVICE, getLenInBytes());
    }

    // Restore original device if we switched
    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
  }

  // Always reset pointer and ownership flag after delete, regardless of whether we owned it
  // This prevents stale pointers from causing allocateSpecial() to skip allocation
  _specialBuffer = nullptr;
  _specialDeviceId.store(-1);
  _isOwnerSpecial = false;
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::freeGpuOnly() {
  throwIfFrozen("freeGpuOnly");
  deleteSpecial();
  deletePrimary();
  closed = true;
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::freeGpuOnStream(void* stream) {
  throwIfFrozen("freeGpuOnStream");
  // Free GPU (special) buffer on the specified CUDA stream.
  // This is used by DSP mid-execution flushing to free on the execution stream
  // instead of the default stream 0. When allocations and frees happen on the
  // same stream, the pool can immediately reuse freed memory for new allocations
  // on that stream without cross-stream synchronization.
  if (_isOwnerSpecial && _specialBuffer != nullptr && getLenInBytes() != 0) {
    //  Use _specialDeviceId to get the device where buffer was actually allocated
    // This may differ from _deviceId due to failover during OOM
    auto bufferDeviceId = _specialDeviceId.load();
    if (bufferDeviceId < 0) {
      // Fallback to _deviceId if _specialDeviceId not set (legacy code path)
      bufferDeviceId = _deviceId.load();
    }
    auto currentDeviceId = AffinityManager::currentDeviceId();
    bool switchedDevice = false;

    if (currentDeviceId != bufferDeviceId) {
      cudaSetDevice(bufferDeviceId);
      switchedDevice = true;
    }

    auto p = reinterpret_cast<void*>(_specialBuffer);
    // The 'stream' parameter is a cudaStream_t* (pointer to stream handle), NOT the
    // stream handle itself. Must dereference to get the actual cudaStream_t handle.
    // If we switched devices, the caller's stream belongs to the wrong device.
    // Use nullptr (default stream on this device) when cross-device.
    cudaStream_t freeStream = nullptr;
    if (!switchedDevice && stream != nullptr) {
      freeStream = *reinterpret_cast<cudaStream_t*>(stream);
    }
    memory::CudaMemoryPool::getInstance().free(p, bufferDeviceId, freeStream);

    // count out towards DataBuffer device, only if we're not in workspace
    if (_workspace == nullptr) {
      sd::memory::MemoryCounter::getInstance().countOut(bufferDeviceId, getLenInBytes());
      sd::memory::MemoryCounter::getInstance().countOut(sd::memory::MemoryType::DEVICE, getLenInBytes());
    }

    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
  }

  _specialBuffer = nullptr;
  _isOwnerSpecial = false;

  // Free host buffer and set closed=true so the destructor's deleteBuffers() is a no-op.
  deletePrimary();
  closed = true;
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setCountersToZero() {
  _counter.store(0L);
  _writePrimary.store(0L);
  _writeSpecial.store(0L);
  _readPrimary.store(0L);
  _readSpecial.store(0L);

  // Write events are created lazily by async copy paths. Constructor/reset does
  // not allocate CUDA objects.
  _writeEventRecorded.store(false);
  _writeEventDeviceId.store(-1);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyCounters(const DataBuffer& other) {
  _counter.store(other._counter);
  _writePrimary.store(other._readSpecial);
  _writeSpecial.store(other._readPrimary);
  _readPrimary.store(other._writeSpecial);
  _readSpecial.store(other._writePrimary);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFrom(const DataBuffer& other, size_t sizeToCopyinBytes, const sd::LongType offsetThis,
                                const sd::LongType offsetOther) {  // copies only to special buffer

  if (other._primaryBuffer == nullptr && other._specialBuffer == nullptr) {
    return;
  }

  if (sizeToCopyinBytes == 0) {
    sizeToCopyinBytes = other.getLenInBytes();
  }
  if (sizeToCopyinBytes == 0) {
    return;
  }

  // Use _specialDeviceId for operations on the special buffer
  auto bufferDeviceId = _specialDeviceId.load();
  if (bufferDeviceId < 0) {
    bufferDeviceId = _deviceId.load();  // Fallback for legacy code
  }
  auto currentDeviceId = AffinityManager::currentDeviceId();
  bool switchedDevice = false;

  if (currentDeviceId != bufferDeviceId) {
    cudaSetDevice(bufferDeviceId);
    switchedDevice = true;
  }

  int res = 0;
  cudaStream_t copyStream = asyncTransferStream(switchedDevice);
  if (other.isPrimaryActual()) {
    // Always use cudaMemcpyAsync with a specific stream instead of synchronous cudaMemcpy.
    // Synchronous cudaMemcpy uses the legacy default stream (stream 0) which implicitly
    // synchronizes with ALL other streams. If any stream on the device has active or
    // recently-invalidated CUDA graph capture, cudaMemcpy fails with error 906
    // (cudaErrorStreamCaptureImplicit). Using cudaMemcpyAsync on a per-thread stream
    // avoids this because cudaStreamPerThread doesn't implicitly sync with named streams.
    res = cudaMemcpyAsync(
        static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
        static_cast<const int8_t*>(other._primaryBuffer) + offsetOther * DataTypeUtils::sizeOfElement(other._dataType),
        sizeToCopyinBytes, cudaMemcpyHostToDevice, copyStream);
    other.readPrimary();
  } else {
    waitForLastDspCompletionIfNeeded(copyStream);
    other.waitForSpecialWriteEvent(copyStream);
    res = cudaMemcpyAsync(
        static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
        static_cast<const int8_t*>(other._specialBuffer) + offsetOther * DataTypeUtils::sizeOfElement(other._dataType),
        sizeToCopyinBytes, cudaMemcpyDeviceToDevice, copyStream);
    other.readSpecial();
  }

  if (res == 0) {
    recordSpecialWriteEvent(copyStream);
  }

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  if (res != 0) {
    if (other.isPrimaryActual()) {
      throwCudaStatus("DataBuffer::copyBufferFrom: cudaMemcpy_cudaMemcpyHostToDevice failed", static_cast<cudaError_t>(res));
    } else {
      throwCudaStatus("DataBuffer::copyBufferFrom: cudaMemcpy_cudaMemcpyDeviceToDevice failed", static_cast<cudaError_t>(res));
    }
  }

  writeSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFromHost(const void* hostBuffer, size_t sizeToCopyinBytes, const sd::LongType offsetThis,
                                    const sd::LongType offsetHostBuffer) {  // copies only to special buffer

  if (hostBuffer == nullptr) return;

  if (sizeToCopyinBytes == 0) sizeToCopyinBytes = getLenInBytes();
  if (sizeToCopyinBytes == 0) return;

  // Use _specialDeviceId for operations on the special buffer
  auto bufferDeviceId = _specialDeviceId.load();
  if (bufferDeviceId < 0) {
    bufferDeviceId = _deviceId.load();  // Fallback for legacy code
  }
  auto currentDeviceId = AffinityManager::currentDeviceId();
  bool switchedDevice = false;

  if (currentDeviceId != bufferDeviceId) {
    cudaSetDevice(bufferDeviceId);
    switchedDevice = true;
  }

  cudaStream_t copyStream = asyncTransferStream(switchedDevice);
  int res = cudaMemcpyAsync(
      static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
      static_cast<const int8_t*>(hostBuffer) + offsetHostBuffer * DataTypeUtils::sizeOfElement(_dataType),
      sizeToCopyinBytes, cudaMemcpyHostToDevice, copyStream);

  if (res == 0) {
    recordSpecialWriteEvent(copyStream);
  }

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  if (res != 0)
    throwCudaStatus("DataBuffer::copyBufferFromHost: cudaMemcpy_cudaMemcpyHostToDevice failed", static_cast<cudaError_t>(res));

  writeSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setSpecial(void* special, const bool isOwnerSpecial) {
  deleteSpecial();
  _specialBuffer = special;
  _isOwnerSpecial = isOwnerSpecial;

  if (special != nullptr) {
    // Determine the ACTUAL device the pointer lives on, not the thread's current device.
    // Multi-GPU routing can allocate on device 0 while the thread affinity says device 1.
    // Using AffinityManager::currentDeviceId() here would store the wrong device, causing
    // migrate() to attempt cudaMemcpyPeer from a device the pointer isn't actually on.
    int actualDevice = AffinityManager::currentDeviceId();  // fallback
    cudaPointerAttributes ptrAttrs;
    auto attrRes = cudaPointerGetAttributes(&ptrAttrs, special);
    if (attrRes == cudaSuccess && ptrAttrs.type == cudaMemoryTypeDevice) {
      actualDevice = ptrAttrs.device;
    } else if (attrRes != cudaSuccess) {
      cudaGetLastError();  // Clear error
    }
    _deviceId.store(actualDevice);
    _specialDeviceId.store(actualDevice);
  }
}

void DataBuffer::replaceSpecialBuffer(void* newPtr, bool isOwner) {
  throwIfFrozen("replaceSpecialBuffer");
  // Replace _specialBuffer WITHOUT calling deleteSpecial() — caller handles old pointer.
  // This is used for weight migration: old pool pointer already freed via cudaFreeAsync,
  // new direct pointer allocated via cudaMalloc.
  clearSpecialWriteEvent();
  _specialBuffer = newPtr;
  _isOwnerSpecial = isOwner;
  if (newPtr != nullptr) {
    // Determine actual device from CUDA pointer attributes, not thread affinity.
    int actualDevice = AffinityManager::currentDeviceId();  // fallback
    cudaPointerAttributes ptrAttrs;
    auto attrRes = cudaPointerGetAttributes(&ptrAttrs, newPtr);
    if (attrRes == cudaSuccess && ptrAttrs.type == cudaMemoryTypeDevice) {
      actualDevice = ptrAttrs.device;
    } else if (attrRes != cudaSuccess) {
      cudaGetLastError();  // Clear error
    }
    _deviceId.store(actualDevice);
    _specialDeviceId.store(actualDevice);
  }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateBuffers(const bool allocBoth) {  // always allocate special buffer only (cuda case)
  allocateSpecial();

  if (allocBoth) allocatePrimary();
}

////////////////////////////////////////////////////////////////////////
  void DataBuffer::setToZeroBuffers(const bool both) {
  if(getLenInBytes() < 1 || special() == nullptr)
    return;

  // Use _specialDeviceId for operations on the special buffer
  auto bufferDeviceId = _specialDeviceId.load();
  if (bufferDeviceId < 0) {
    bufferDeviceId = _deviceId.load();  // Fallback for legacy code
  }
  cudaPointerAttributes zeroPtrAttrs;
  auto zeroAttrRes = cudaPointerGetAttributes(&zeroPtrAttrs, special());
  if (zeroAttrRes == cudaSuccess && zeroPtrAttrs.type == cudaMemoryTypeDevice) {
    bufferDeviceId = zeroPtrAttrs.device;
  } else if (zeroAttrRes != cudaSuccess) {
    cudaGetLastError();
  }
  auto currentDeviceId = AffinityManager::currentDeviceId();
  if (tl_graphExecutionActive && tl_graphCaptureStream != nullptr) {
    int cudaDeviceId = -1;
    auto devErr = cudaGetDevice(&cudaDeviceId);
    if (devErr == cudaSuccess && cudaDeviceId >= 0) {
      currentDeviceId = cudaDeviceId;
    } else if (devErr != cudaSuccess) {
      cudaGetLastError();
    }
  }
  bool switchedDevice = false;

  if (currentDeviceId != bufferDeviceId) {
    // During CUDA graph capture, cudaSetDevice is NOT allowed — it invalidates capture.
    // Skip device switch during capture; the buffer address is still valid as a GPU pointer
    // regardless of which device is "current" (CUDA unified address space).
    if (tl_graphExecutionActive) {
      // Don't switch devices during capture — use the capture stream on the current device
    } else {
      cudaSetDevice(bufferDeviceId);
      switchedDevice = true;
    }
  }

  // During CUDA graph capture, record a memset graph node on the capture stream.
  // cudaMemsetAsync on the capture stream is legal and records a memset node in the
  // graph — this is the correct behavior for zeroing buffers that the graph will use.
  // For cross-device buffers during capture, skip the zero (the op will overwrite anyway,
  // and cross-device memset nodes cause cudaGraphInstantiate "invalid argument" errors).
  cudaStream_t stream;
  if (tl_graphExecutionActive && tl_graphCaptureStream != nullptr) {
    // During capture, use the capture stream to record a memset node.
    // Cross-device buffers: skip zero entirely (cross-device nodes break instantiation).
    if (currentDeviceId != bufferDeviceId) {
      // Cross-device during capture: skip zero, op will write correct values
      DSP_DIAG(MEMORY,
               "CAPTURE_MEMSET_SKIP_CROSS_DEVICE: buf=%p bytes=%lld currentDev=%d bufferDev=%d stream=%p",
               special(), (long long)getLenInBytes(), currentDeviceId, bufferDeviceId,
               (void*)tl_graphCaptureStream);
      if (switchedDevice) {
        cudaSetDevice(currentDeviceId);
      }
      writeSpecial();
      return;
    }
    stream = reinterpret_cast<cudaStream_t>(tl_graphCaptureStream);
  } else {
    // Cache the stream reference - must obtain AFTER device switch so we get the correct device's stream
    stream = captureSafeStreamOrDefault();
  }
  auto res = cudaMemsetAsync(special(), 0, getLenInBytes(), stream);

  if (res == 901 || res == 906) {
    // cudaErrorStreamCaptureImplicit / cudaErrorStreamCaptureInvalidated
    // Captures diagnostics: stream state, capture flags, buffer details
    cudaStreamCaptureStatus capStatus = cudaStreamCaptureStatusNone;
    unsigned long long capId = 0;
    cudaStreamGetCaptureInfo(stream, &capStatus, &capId);
    cudaGetLastError();
    char msg[512];
    snprintf(msg, sizeof(msg),
             "setToZeroBuffers: pool-level capture error — "
             "err=%d stream=%p capStatus=%d capGraphId=%llu "
             "graphExecActive=%d graphCaptureStream=%p "
             "buf=%p bytes=%lld switchDev=%d dstDev=%d srcDev=%d",
             (int)res, (void*)stream, (int)capStatus, capId,
             (int)tl_graphExecutionActive, (void*)tl_graphCaptureStream,
             special(), (long long)getLenInBytes(),
             (int)switchedDevice, currentDeviceId, bufferDeviceId);
    THROW_EXCEPTION(msg);
  }

  if (res != cudaSuccess) {
    // cudaMemsetAsync fails on pinned host pointers (from OOM fallback allocation).
    // Pinned host memory is CPU-accessible, so fall back to memset on the host side.
    // Use cudaPointerGetAttributes to definitively check if the pointer is host memory.
    cudaGetLastError();  // Clear the error before proceeding
    cudaPointerAttributes ptrAttrs;
    cudaError_t attrRes = cudaPointerGetAttributes(&ptrAttrs, special());
    bool isHostPtr = (attrRes == cudaSuccess && ptrAttrs.type == cudaMemoryTypeHost);
    if (!isHostPtr) {
      // Also check the pool's pinned host tracking as a fallback
      isHostPtr = sd::memory::CudaMemoryPool::getInstance().isPinnedHostAllocation(special());
    }
    if (attrRes != cudaSuccess) cudaGetLastError();  // Clear attribute query error
    if (isHostPtr) {
      memset(special(), 0, getLenInBytes());
    } else if (static_cast<int>(res) == 201 || static_cast<int>(res) == 200 ||
               res == cudaErrorInvalidResourceHandle) {
      // Error 201 = cudaErrorDeviceUninitialized (invalid device context)
      // Error 200 = cudaErrorInvalidDevice
      // cudaErrorInvalidResourceHandle = stale stream handle
      //
      // The CUDA primary context on this device has become invalid. This happens
      // when a stale async error corrupts the context, or when thread-local state
      // loses its CUDA context association. CudaMemoryPool::allocate handles this
      // via allocateFailover(), but cudaMemsetAsync had no equivalent recovery.
      //
      // Recovery: re-establish the context via cudaSetDevice, get a fresh stream,
      // and retry. If that still fails, fall back to synchronous cudaMemset.
      sd_printf("DataBuffer::setToZeroBuffers: cudaMemsetAsync failed with %s (%d) on device %d — "
                "attempting context recovery\n",
                cudaGetErrorString(res), static_cast<int>(res), bufferDeviceId);
      cudaGetLastError();  // Clear sticky error

      // Re-establish CUDA primary context
      cudaError_t setDevErr = cudaSetDevice(bufferDeviceId);
      if (setDevErr != cudaSuccess) {
        cudaGetLastError();
      }

      // Get a fresh stream after context recovery
      cudaStream_t freshStream = captureSafeStreamOrDefault();
      if (freshStream == nullptr) {
        freshStream = cudaStreamPerThread;
      }

      // Retry cudaMemsetAsync with recovered context
      cudaError_t retryRes = cudaMemsetAsync(special(), 0, getLenInBytes(), freshStream);
      if (retryRes == cudaSuccess) {
        sd_debug("DataBuffer::setToZeroBuffers: context recovery succeeded on device %d\n",
                 bufferDeviceId);
        stream = freshStream;
        res = cudaSuccess;
      } else {
        cudaGetLastError();  // Clear retry error
        // Last resort: synchronous cudaMemset (uses default stream, no explicit stream needed)
        cudaError_t syncRes = cudaMemset(special(), 0, getLenInBytes());
        if (syncRes == cudaSuccess) {
          sd_debug("DataBuffer::setToZeroBuffers: synchronous cudaMemset fallback succeeded "
                   "on device %d\n", bufferDeviceId);
          res = cudaSuccess;
        } else {
          cudaGetLastError();  // Clear sync error
          sd_printf("DataBuffer::setToZeroBuffers: all recovery attempts failed on device %d "
                    "(async=%s(%d), sync=%s(%d))\n",
                    bufferDeviceId,
                    cudaGetErrorString(retryRes), static_cast<int>(retryRes),
                    cudaGetErrorString(syncRes), static_cast<int>(syncRes));
          if (switchedDevice) {
            cudaSetDevice(currentDeviceId);
          }
          throwCudaStatus("DataBuffer::setToZeroBuffers: cudaMemsetAsync failed", res);
        }
      }
    } else {
      if (switchedDevice) {
        cudaSetDevice(currentDeviceId);
      }
      throwCudaStatus("DataBuffer::setToZeroBuffers: cudaMemsetAsync failed", res);
    }
  }

  if (res == cudaSuccess) {
    recordSpecialWriteEvent(stream);
  }

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  writeSpecial();

  if (both) {
    memset(primary(), 0, getLenInBytes());
    readPrimary();
  }
}



/////////////////////////


template <typename T>
void memcpyWithT(DataBuffer* dst, DataBuffer* src, sd::LongType startingOffset, sd::LongType dstOffset, sd::LongType n) {
  auto sizeOfElement = DataTypeUtils::sizeOfElement(src->getDataType());
  // Calculate copy size in bytes, accounting for offsets
  sd::LongType srcAvailable = src->getLenInBytes() - startingOffset * sizeOfElement;
  sd::LongType dstAvailable = dst->getLenInBytes() - dstOffset * sizeOfElement;
  sd::LongType copyBytes;
  if (n > 0) {
    copyBytes = n * sizeOfElement;
  } else {
    // When n=0, copy as much as fits (min of available src and dst)
    copyBytes = srcAvailable < dstAvailable ? srcAvailable : dstAvailable;
  }
  // Clamp to available space to prevent overruns
  if (copyBytes > srcAvailable) copyBytes = srcAvailable;
  if (copyBytes > dstAvailable) copyBytes = dstAvailable;
  if (copyBytes <= 0) return;

  auto dstDeviceId = dst->deviceId();
  auto currentDeviceId = AffinityManager::currentDeviceId();
  bool switchedDevice = false;

  if (currentDeviceId != dstDeviceId) {
    cudaSetDevice(dstDeviceId);
    switchedDevice = true;
  }

  // Cache the stream reference - must obtain AFTER device switch
  cudaStream_t stream = asyncTransferStream(switchedDevice);

  cudaError_t res = cudaSuccess;
  if (src->isSpecialActual()) {
    waitForLastDspCompletionIfNeeded(stream);
    src->waitForSpecialWriteEvent(stream);
    res = cudaMemcpyAsync(dst->specialAtOffset<T>(dstOffset), src->specialAtOffset<T>(startingOffset), copyBytes, cudaMemcpyDeviceToDevice,
                          stream);
  } else if (src->isPrimaryActual()) {
    res = cudaMemcpyAsync(dst->specialAtOffset<T>(dstOffset), src->specialAtOffset<T>(startingOffset), copyBytes, cudaMemcpyHostToDevice,
                          stream);
  }

  if (res != cudaSuccess) {
    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
    throwCudaStatus("DataBuffer::memcpy: cudaMemcpyAsync failed", res);
  }

  // No stream sync needed here - subsequent GPU operations on same stream will
  // automatically wait for memcpy to complete. This eliminates unnecessary CPU-GPU sync.
  // The writeSpecial() below will track the write event for cross-thread sync if needed.
  dst->recordSpecialWriteEvent(stream);

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  dst->writeSpecial();
}
BUILD_SINGLE_TEMPLATE(void memcpyWithT, (DataBuffer* dst, DataBuffer* src, sd::LongType startingOffset, sd::LongType dstOffset, sd::LongType n), SD_COMMON_TYPES);

void DataBuffer::memcpy(DataBuffer* dst, DataBuffer* src,
                        sd::LongType startingOffset, sd::LongType dstOffset, sd::LongType n) {
  BUILD_SINGLE_SELECTOR(src->getDataType(), memcpyWithT, (dst, src, startingOffset, dstOffset, n), SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::migrate() {
  if (isConstant) {
    return;
  }

  // During CUDA graph capture, migration is forbidden. It involves synchronous
  // driver queries (cudaPointerGetAttributes), cross-device copies (cudaMemcpyPeer),
  // and new allocations — all of which poison the capture stream. Buffers must be
  // pre-positioned on the correct device before capture begins. Skip silently.
  if (tl_graphExecutionActive) {
    return;
  }

  // When this buffer is registered in a frozen NativeDynamicShapePlan (as an
  // external input or retained weight), its _specialBuffer address is baked into
  // frozen slot contexts and/or CUDA graph replay handles. Migrating would free
  // the old pointer and allocate a new one on a different device, leaving the
  // frozen plan with a dangling address → SIGSEGV on next replay.
  // Skip silently: the buffer is already on the correct device (pinned there at
  // freeze time). The device-ID mismatch that triggered syncToDevice → migrate
  // is from a stale NDArray._deviceId, not a genuine need to relocate data.
  if (_frozenRefCount.load(std::memory_order_relaxed) > 0) {
    return;
  }

  auto currentDeviceId = AffinityManager::currentDeviceId();
  // Use _specialDeviceId for the old buffer since we're migrating the special buffer
  // This may differ from _deviceId due to failover during OOM
  auto oldDeviceId = _specialDeviceId.load();
  if (oldDeviceId < 0) {
    oldDeviceId = _deviceId.load();  // Fallback for legacy code
  }

  // Don't migrate if already on the target device
  if (oldDeviceId == currentDeviceId && _specialBuffer != nullptr) {
    return;
  }

  // Guard against zero-length buffers — cudaMemcpy with 0 bytes returns
  // "invalid argument" on some CUDA versions (e.g. zero-dim arrays like [1,3,0,64])
  if (_lenInBytes == 0) {
    return;
  }

  // Validate metadata hasn't been corrupted by heap overruns.
  // Without jemalloc, C++ buffer overruns can stomp on adjacent DataBuffer objects,
  // corrupting _specialDeviceId, _lenInBytes, or _specialBuffer pointer.
  int numDevices = 0;
  cudaGetDeviceCount(&numDevices);
  if (oldDeviceId >= numDevices || currentDeviceId >= numDevices) {
    sd_printf("DataBuffer::migrate: CORRUPTED device IDs detected! oldDeviceId=%d, currentDeviceId=%d, numDevices=%d. Skipping migration.\n",
              oldDeviceId, currentDeviceId, numDevices);
    return;
  }

  // Validate _lenInBytes is reasonable (max 16GB per buffer)
  constexpr size_t MAX_MIGRATE_BYTES = 16ULL * 1024 * 1024 * 1024;
  if (_lenInBytes > MAX_MIGRATE_BYTES) {
    sd_printf("DataBuffer::migrate: CORRUPTED _lenInBytes=%zu (>16GB). Skipping migration.\n", _lenInBytes);
    return;
  }

  // Validate _lenInBytes is consistent with _dataType. A buffer smaller than one
  // element of its declared type is metadata corruption (e.g., heap overrun stomping
  // _lenInBytes from a valid value to 1). Skip rather than copying 1 byte via
  // cudaMemcpy — that fails with "invalid argument" when the source pointer is
  // inside a CUDA allocation smaller than the advertised size.
  {
    auto elementSize = DataTypeUtils::sizeOfElement(_dataType);
    if (elementSize > 0 && _lenInBytes < elementSize) {
      sd_printf("DataBuffer::migrate: CORRUPTED _lenInBytes=%zu < sizeOfElement(%d)=%zu for dataType=%d. "
                "primary=%p special=%p oldDevice=%d currentDevice=%d. Skipping migration.\n",
                _lenInBytes, static_cast<int>(_dataType), elementSize,
                static_cast<int>(_dataType),
                _primaryBuffer, _specialBuffer, oldDeviceId, currentDeviceId);
      return;
    }
  }

  // Validate _specialBuffer pointer using cudaPointerGetAttributes.
  // If the pointer is invalid or on a different device than expected,
  // use the actual device from CUDA rather than our potentially-corrupted metadata.
  if (_specialBuffer != nullptr) {
    cudaPointerAttributes ptrAttrs;
    auto attrRes = cudaPointerGetAttributes(&ptrAttrs, _specialBuffer);
    if (attrRes != cudaSuccess) {
      // Pointer is not recognized by CUDA — corrupted or already freed.
      cudaGetLastError();  // Clear the error
      sd_printf("DataBuffer::migrate: INVALID _specialBuffer=%p (cudaPointerGetAttributes failed: %s). Skipping migration.\n",
                _specialBuffer, cudaGetErrorString(attrRes));
      return;
    }
    // Check if CUDA reports a different device than our metadata
    if (ptrAttrs.type == cudaMemoryTypeDevice && ptrAttrs.device != oldDeviceId) {
      sd_printf("DataBuffer::migrate: Device mismatch! metadata says device %d, CUDA says device %d for ptr=%p. Using CUDA device.\n",
                oldDeviceId, ptrAttrs.device, _specialBuffer);
      oldDeviceId = ptrAttrs.device;
      // If corrected device matches target, no migration needed
      if (oldDeviceId == currentDeviceId) {
        return;
      }
    }
  }

  // Clear any previous CUDA errors to ensure clean state
  cudaError_t prevErr = cudaGetLastError();
  if (prevErr != cudaSuccess) {
    sd_debug("DataBuffer::migrate: Cleared previous CUDA error before migration: %s\n", cudaGetErrorString(prevErr));
  }

  // Verify we're on the expected device before starting
  int actualDevice = -1;
  cudaGetDevice(&actualDevice);
  if (actualDevice != currentDeviceId) {
    cudaSetDevice(currentDeviceId);
  }

  memory::Workspace* newWorkspace = nullptr;
  void* newBuffer;
  void* oldBuffer = _specialBuffer;  // Save old buffer pointer for deallocation

  // Start timing the transfer
  auto startTime = std::chrono::high_resolution_clock::now();

  // Allocate on current (target) device, tracking actual device in case of failover
  int actualMigrateDevice = currentDeviceId;
  {
    size_t allocSize = getLenInBytes() + 8;
    newBuffer = reinterpret_cast<void*>(
        memory::CudaMemoryPool::getInstance().allocate(allocSize, currentDeviceId, nullptr, &actualMigrateDevice));
    if (newBuffer == nullptr) {
      THROW_EXCEPTION("[DEVICE] migrate allocation failed");
    }
  }

  // Use actual allocation device for all copy operations. CudaMemoryPool::allocate()
  // may fail over to a different device than requested (e.g., device 1 full → device 0).
  // Without this, we'd try cudaMemcpy H2D to a pointer on device 0 while cudaSetDevice(1)
  // is active → "invalid argument" error.
  int targetDevice = actualMigrateDevice;
  if (targetDevice != currentDeviceId) {
    sd_printf("DataBuffer::migrate: Allocation failed over from device %d to device %d for %zu bytes\n",
              currentDeviceId, targetDevice, getLenInBytes());

    // If failover landed on a non-peer device, switch to pinned host (UVA-accessible).
    size_t migrateAllocSize = getLenInBytes() + 8;
    targetDevice = handleNonPeerFailover(newBuffer, migrateAllocSize, currentDeviceId, targetDevice, "DataBuffer::migrate");
    actualMigrateDevice = targetDevice;
  }

  if (_specialBuffer != nullptr) {
    // Copy from old device to new device
    if (oldDeviceId != targetDevice && oldDeviceId >= 0) {
      // Belt-and-suspenders: re-validate source pointer device right before the copy.
      // Metadata (_specialDeviceId / _deviceId) can become stale if setSpecial() was
      // called from a thread whose affinity doesn't match the pointer's actual device.
      // The earlier pre-check at line ~1572 may have been too far from this copy point.
      {
        cudaPointerAttributes preCopyAttrs;
        auto preCopyRes = cudaPointerGetAttributes(&preCopyAttrs, _specialBuffer);
        if (preCopyRes == cudaSuccess && preCopyAttrs.type == cudaMemoryTypeDevice) {
          if (preCopyAttrs.device != oldDeviceId) {
            sd_printf("DataBuffer::migrate: PRE-COPY device correction! metadata=%d, CUDA=%d for ptr=%p, bytes=%zu\n",
                      oldDeviceId, preCopyAttrs.device, _specialBuffer, getLenInBytes());
            oldDeviceId = preCopyAttrs.device;
            // If corrected source == target, skip the copy entirely — just use same-device memcpy
            if (oldDeviceId == targetDevice) {
              cudaSetDevice(targetDevice);
              auto res = cudaMemcpy(newBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToDevice);
              if (res != cudaSuccess) {
                std::string err = "DataBuffer::migrate: same-device cudaMemcpy D2D failed after correction! Error: " +
                                  std::string(cudaGetErrorString(res)) + ", bytes: " + std::to_string(getLenInBytes());
                THROW_EXCEPTION(err.c_str());
              }
              goto copyDone;
            }
          }
        } else if (preCopyRes != cudaSuccess) {
          cudaGetLastError();  // Clear error
        }
      }

      // Cross-device copy via cudaMemcpyPeer — handles staging internally
      // (uses peer DMA if available, otherwise stages through host automatically).
      // Synchronize source device first to ensure prior operations complete.
      auto setRes = cudaSetDevice(oldDeviceId);
      if (setRes != cudaSuccess) {
        cudaSetDevice(targetDevice);
        std::string err = "DataBuffer::migrate: Failed to switch to source device " + std::to_string(oldDeviceId) +
                          ": " + std::string(cudaGetErrorString(setRes));
        THROW_EXCEPTION(err.c_str());
      }

      // getCudaStream() may trigger ContextBuffers initialization which can change the
      // active CUDA device (via allocateFailover). We must restore oldDeviceId afterward.
      auto srcStream = sd::LaunchContext::defaultContext()->getCudaStream();
      if (srcStream != nullptr)
        cudaStreamSynchronize(*srcStream);

      // Restore source device — getCudaStream() may have changed it via ContextBuffers init.
      cudaSetDevice(oldDeviceId);
      cudaGetLastError();  // clear any sticky error from ContextBuffers init

      auto peerRes = cudaMemcpyPeer(newBuffer, targetDevice, _specialBuffer, oldDeviceId, getLenInBytes());
      if (peerRes != cudaSuccess) {
        cudaGetLastError();  // clear sticky error from failed memcpyPeer

        // Re-query pointer attributes — the source pointer may actually reside on
        // targetDevice despite metadata claiming oldDeviceId. This happens when
        // CudaMemoryPool failover placed an allocation on a different device than
        // _specialDeviceId records, or when the async pool's physical placement
        // differs from the logical device.
        cudaPointerAttributes retryAttrs;
        auto retryRes = cudaPointerGetAttributes(&retryAttrs, _specialBuffer);
        cudaGetLastError();

        if (retryRes == cudaSuccess && retryAttrs.type == cudaMemoryTypeDevice &&
            retryAttrs.device == targetDevice) {
          // Source pointer is actually on the target device — use same-device D2D copy
          sd_printf("DataBuffer::migrate: cudaMemcpyPeer failed (metadata device=%d), but ptr is on target device %d. Using same-device D2D copy.\n",
                    oldDeviceId, targetDevice);
          cudaSetDevice(targetDevice);
          auto d2dRes = cudaMemcpy(newBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToDevice);
          if (d2dRes != cudaSuccess) {
            std::string err = "DataBuffer::migrate: same-device D2D fallback failed! Error: " +
                              std::string(cudaGetErrorString(d2dRes)) + ", bytes: " + std::to_string(getLenInBytes());
            THROW_EXCEPTION(err.c_str());
          }
          goto copyDone;
        }

        cudaSetDevice(targetDevice);
        std::string err = "DataBuffer::migrate: cudaMemcpyPeer failed! Error: " + std::string(cudaGetErrorString(peerRes)) +
                          ", bytes: " + std::to_string(getLenInBytes()) +
                          ", from device " + std::to_string(oldDeviceId) + " to device " + std::to_string(targetDevice);
        if (retryRes == cudaSuccess) {
          err += ", ptrAttrs: type=" + std::to_string(retryAttrs.type) +
                 " device=" + std::to_string(retryAttrs.device);
        } else {
          err += ", ptr validation FAILED (pointer likely corrupted or freed)";
        }
        THROW_EXCEPTION(err.c_str());
      }

      // cudaMemcpyPeer is synchronous — no additional sync needed
      cudaSetDevice(targetDevice);
    } else {
      // Same device copy or unknown source device
      cudaSetDevice(targetDevice);
      auto res = cudaMemcpy(newBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToDevice);
      if (res != cudaSuccess) {
        std::string err = "DataBuffer::migrate: cudaMemcpy D2D failed! Error: " + std::string(cudaGetErrorString(res)) +
                          ", bytes: " + std::to_string(getLenInBytes()) + ", device " + std::to_string(targetDevice);
        THROW_EXCEPTION(err.c_str());
      }
    }
  } else if (_primaryBuffer != nullptr) {
    // Copy from host to device if no special buffer exists
    cudaSetDevice(targetDevice);
    auto res = cudaMemcpy(newBuffer, _primaryBuffer, getLenInBytes(), cudaMemcpyHostToDevice);
    if (res != cudaSuccess) {
      std::string err = "DataBuffer::migrate: cudaMemcpy H2D failed! Error: " + std::string(cudaGetErrorString(res)) +
                        ", bytes: " + std::to_string(getLenInBytes()) + ", to device " + std::to_string(targetDevice);
      THROW_EXCEPTION(err.c_str());
    }
  }

  copyDone:  // Target for pre-copy device correction (same-device copy after metadata fix)

  auto endTime = std::chrono::high_resolution_clock::now();
  auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();

  // Record transfer metrics
  if (oldBuffer != nullptr) {
    // Device to device transfer (possibly peer-to-peer)
    TransferType transferType = (oldDeviceId != targetDevice) ?
        TransferType::PEER_TO_PEER : TransferType::DEVICE_TO_DEVICE;
    TransferMetrics::getInstance().recordTransfer(transferType, getLenInBytes(), durationNs,
                                                   oldDeviceId, targetDevice);

    // DSP diagnostics: D2D migrate
    DSP_DIAG(TRANSFER, "D2D migrate: %lld bytes, from=%d to=%d, duration=%.2f us",
             (long long)getLenInBytes(), oldDeviceId, targetDevice, durationNs / 1000.0);
  } else if (_primaryBuffer != nullptr) {
    // Host to device transfer
    TransferMetrics::getInstance().recordTransfer(TransferType::HOST_TO_DEVICE, getLenInBytes(),
                                                   durationNs, -1, targetDevice);

    // DSP diagnostics: H2D via migrate
    DSP_DIAG(TRANSFER, "H2D migrate: %lld bytes, to=%d, duration=%.2f us",
             (long long)getLenInBytes(), targetDevice, durationNs / 1000.0);
  }

  if (_isOwnerSpecial && oldBuffer != nullptr) {
    // Switch to old device to release memory
    if (oldDeviceId != targetDevice && oldDeviceId >= 0) {
      cudaSetDevice(oldDeviceId);
    }

    auto p = reinterpret_cast<int8_t*>(oldBuffer);
    // Use device-aware free - critical for multi-GPU correctness
    RELEASE_SPECIAL_WITH_DEVICE(p, oldDeviceId, _workspace);

    // Switch back to target device (where new buffer lives)
    if (oldDeviceId != targetDevice && oldDeviceId >= 0) {
      cudaSetDevice(targetDevice);
    }
  }

  // If CudaMemoryPool returned a capture-workspace interior pointer during CUDA
  // graph capture (bump-allocated from the shared workspace), the workspace
  // lifecycle manages this memory — not this DataBuffer. Setting _isOwnerSpecial=true
  // would cause deleteSpecial() to call cudaFreeAsync on a workspace interior pointer
  // after unregisterCaptureWorkspace → "illegal memory access" (error 700).
  if (_workspace == nullptr &&
      memory::CudaMemoryPool::getInstance().isInCaptureWorkspace(newBuffer)) {
    _isOwnerSpecial = false;
  } else {
    _isOwnerSpecial = true;
  }
   _specialBuffer = newBuffer;

   // Store actual device where memory was allocated (may differ after failover)
   _deviceId.store(actualMigrateDevice);
   _specialDeviceId.store(actualMigrateDevice);  // Also update _specialDeviceId for consistency

  // Restore caller's expected device context. The caller called migrate() expecting
  // to remain on currentDeviceId. Even though the buffer may have ended up on a
  // different device (failover), the caller's CUDA context should be preserved.
  int restoreDev = -1;
  cudaGetDevice(&restoreDev);
  if (restoreDev != currentDeviceId) {
    cudaSetDevice(currentDeviceId);
  }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::writePrimary() const { _writePrimary = ++_counter; }
void DataBuffer::writeSpecial() const {
  _writeSpecial = ++_counter;

  // Event recording is intentionally omitted. syncToPrimary() uses stream 0
  // (the legacy default stream) which implicitly synchronizes with ALL other
  // streams on the device, making event-based ordering redundant.
  // Removing event creation also eliminates heap allocations (new cudaEvent_t)
  // in the hot path that are vulnerable to corrupted heap metadata from native
  // op buffer overruns.
}
void DataBuffer::readPrimary() const { _readPrimary = ++_counter; }
void DataBuffer::readSpecial() const { _readSpecial = ++_counter; }
bool DataBuffer::isPrimaryActual() const {
  return (_writePrimary.load() > _writeSpecial.load() || _readPrimary.load() > _writeSpecial.load());
}
bool DataBuffer::isSpecialActual() const {
  return (_writeSpecial.load() > _writePrimary.load() || _readSpecial.load() > _writePrimary.load());
}

}  // namespace sd
