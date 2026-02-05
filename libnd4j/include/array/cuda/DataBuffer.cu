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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <array/DataTypeUtils.h>
#include <exceptions/allocation_exception.h>
#include <exceptions/cuda_exception.h>
#include <execution/AffinityManager.h>
#include <memory/MemoryCounter.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <system/op_boilerplate.h>
#include <system/type_boilerplate.h>
#include <helpers/TransferMetrics.h>
#include <chrono>

#include "../DataBuffer.h"
#include "helpers/DebugHelper.h"

#if defined(SD_GCC_FUNCTRACE)
#include <array/DataBufferLifecycleTracker.h>
#endif

namespace sd {
void DataBuffer::expand(const uint64_t size) {
  if (size > _lenInBytes) {
    // allocate new buffer
    int8_t* newBuffer = nullptr;
    int8_t* newSpecialBuffer = nullptr;
    auto currentDeviceId = AffinityManager::currentDeviceId();
    auto oldDeviceId = _deviceId.load();  // Save old device ID for releasing old buffer

    // Allocate new buffer, tracking actual device in case of failover
    int actualExpandDevice = currentDeviceId;
    if (_workspace == nullptr) {
      size_t allocSize = size + 8;
      newSpecialBuffer = reinterpret_cast<int8_t*>(
          memory::CudaMemoryPool::getInstance().allocate(allocSize, currentDeviceId, nullptr, &actualExpandDevice));
      if (newSpecialBuffer == nullptr) {
        THROW_EXCEPTION("[DEVICE] expand allocation failed");
      }
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
    if (_primaryBuffer != nullptr) {
      // there's non-zero chance that primary buffer doesn't exist yet
      ALLOCATE(newBuffer, _workspace, size, int8_t);
      std::memcpy(newBuffer, _primaryBuffer, _lenInBytes);

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

    // Cross-device copy
    cudaMemcpy(newSpecialBuffer, _specialBuffer, _lenInBytes, cudaMemcpyDeviceToDevice);

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
      RELEASE_SPECIAL(isb, _workspace);

      // Switch back to current device
      if (oldDeviceId != currentDeviceId && oldDeviceId >= 0) {
        cudaSetDevice(currentDeviceId);
      }
    }

    _specialBuffer = newSpecialBuffer;
    _lenInBytes = size;
    _specialAllocBytes = size;
    if (_primaryBuffer != nullptr) _primaryAllocBytes = size;
    _isOwnerSpecial = true;

    // Store actual device where memory was allocated (may differ from currentDeviceId after failover)
    _deviceId.store(actualExpandDevice);
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

#define PRIMARYOFFSET(T) template SD_LIB_EXPORT void* DataBuffer::primaryAtOffset<GET_SECOND(T)>(sd::LongType offset);
ITERATE_LIST((SD_COMMON_TYPES),PRIMARYOFFSET)

#define SPECIALOFFSET(T) template SD_LIB_EXPORT void* DataBuffer::specialAtOffset<GET_SECOND(T)>(sd::LongType offset);
ITERATE_LIST((SD_COMMON_TYPES),SPECIALOFFSET)


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

BUILD_SINGLE_TEMPLATE( SD_LIB_EXPORT  SD_KERNEL void printDeviceBufferKernel,(void* buffer, sd::LongType offset, sd::LongType length),SD_COMMON_TYPES);


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
BUILD_SINGLE_TEMPLATE( SD_LIB_EXPORT void launchPrintDeviceBufferKernel,(void* buffer, sd::LongType offset, sd::LongType length),SD_COMMON_TYPES);


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
  if (_specialBuffer != nullptr) {
    if (isConstant) {
      // Constant buffers are cached and shared - don't migrate them
      return;
    }
    auto currentDeviceId = AffinityManager::currentDeviceId();
    auto bufferDeviceId = _deviceId.load();
    if (bufferDeviceId != currentDeviceId) {
      // Buffer exists but on wrong device - migrate it
      migrate();
    }
    return;
  }

  if (_lenInBytes == 0) {
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
#if defined(SD_GCC_FUNCTRACE)
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
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
      _specialBuffer = reinterpret_cast<int8_t*>(
          memory::CudaMemoryPool::getInstance().allocate(allocSize, deviceId, nullptr, &actualDevice));
      if (_specialBuffer == nullptr) {
        THROW_EXCEPTION("[DEVICE] allocation failed");
      }
    } else {
      size_t allocSize = getLenInBytes() + 8;
      _specialBuffer = reinterpret_cast<int8_t*>(
          _workspace->allocateBytes(memory::MemoryType::DEVICE, allocSize));
      actualDevice = deviceId;  // workspace allocations stay on requested device
    }
    _isOwnerSpecial = true;
    _specialAllocBytes = getLenInBytes();

    // Store the ACTUAL device where memory was allocated, not the requested device
    _deviceId.store(actualDevice);

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

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToPrimary(const LaunchContext* context, const bool forceSync) {
  if (_specialBuffer == nullptr || _lenInBytes == 0 || closed) {
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

  auto bufferDeviceId = _deviceId.load();
  auto currentDeviceId = AffinityManager::currentDeviceId();
  bool switchedDevice = false;

  if (currentDeviceId != bufferDeviceId) {
    cudaSetDevice(bufferDeviceId);
    switchedDevice = true;
  }

  // Get the stream for async transfer
  cudaStream_t stream = context != nullptr ? *context->getCudaStream() : 0;

  // Use event-based synchronization for cross-thread correctness.
  // The problem: if kernel K runs on Thread A's stream and syncToPrimary() is called
  // from Thread B, the old code would sync Thread B's stream (useless) and then memcpy
  // while kernel K might still be running on Thread A's stream.
  //
  // Solution: writeSpecial() records an event on the actual kernel stream. Here we wait
  // on that event on our stream, which properly synchronizes regardless of which thread is calling.
  cudaError_t res;
  if (_writeEvent != nullptr && _writeEventRecorded.load()) {
    cudaEvent_t* event = reinterpret_cast<cudaEvent_t*>(_writeEvent);
    // Make our stream wait for the write event (non-blocking on CPU)
    res = cudaStreamWaitEvent(stream, *event, 0);
    if (res != cudaSuccess) {
      // Fallback to event sync if stream wait fails
      res = cudaEventSynchronize(*event);
    }
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
    std::string errorMessage;
    errorMessage += "DataBuffer::syncToPrimary: cudaMemcpyAsync failed: ";
    errorMessage += std::to_string(getLenInBytes());
    errorMessage += " ";
    errorMessage += cudaGetErrorString(res);
    errorMessage += " (buffer device: ";
    errorMessage += std::to_string(bufferDeviceId);
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

  allocateSpecial();

  // If special buffer exists but is undersized, reallocate to prevent overrun
  if (_specialBuffer != nullptr && _specialAllocBytes > 0 && _specialAllocBytes < getLenInBytes()) {
    if (_isOwnerSpecial) {
      auto isb = reinterpret_cast<int8_t*>(_specialBuffer);
      RELEASE_SPECIAL(isb, _workspace);
    }
    _specialBuffer = nullptr;
    _specialAllocBytes = 0;
    allocateSpecial();
  }

  auto bufferDeviceId = _deviceId.load();
  auto currentDeviceId = AffinityManager::currentDeviceId();
  bool switchedDevice = false;

  if (currentDeviceId != bufferDeviceId) {
    cudaSetDevice(bufferDeviceId);
    switchedDevice = true;
  }

  // Track H2D transfer
  auto startTime = std::chrono::high_resolution_clock::now();

  // Use the compute stream from LaunchContext for proper stream ordering.
  // Using stream 0 would create unnecessary serialization with all other streams
  // in legacy default stream mode, and would be incorrect under per-thread default streams.
  cudaStream_t stream = 0;
  auto ctx = LaunchContext::defaultContext();
  if (ctx != nullptr && ctx->getCudaStream() != nullptr) {
    stream = *ctx->getCudaStream();
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

  // Synchronize to ensure data is on device before returning
  // Kernels that use this buffer will be scheduled after this sync
  res = cudaStreamSynchronize(stream);
  if (res != cudaSuccess) {
    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
    std::string errorMessage;
    errorMessage += "DataBuffer::syncToSpecial: stream sync failed: ";
    errorMessage += cudaGetErrorString(res);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
  TransferMetrics::getInstance().recordTransfer(TransferType::HOST_TO_DEVICE, getLenInBytes(),
                                                 durationNs, -1, bufferDeviceId);

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  readSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteSpecial() {
  if (_isOwnerSpecial && _specialBuffer != nullptr && getLenInBytes() != 0) {
    auto bufferDeviceId = _deviceId.load();
    auto currentDeviceId = AffinityManager::currentDeviceId();
    bool switchedDevice = false;

    if (currentDeviceId != bufferDeviceId) {
      cudaSetDevice(bufferDeviceId);
      switchedDevice = true;
    }

    auto p = reinterpret_cast<int8_t*>(_specialBuffer);
#if defined(SD_GCC_FUNCTRACE)
    // Record SPECIAL (device) buffer deallocation before releasing
    array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
        _specialBuffer,array::BufferType::SPECIAL);
#endif
    RELEASE_SPECIAL(p, _workspace);

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
  _isOwnerSpecial = false;
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setCountersToZero() {
  _counter.store(0L);
  _writePrimary.store(0L);
  _writeSpecial.store(0L);
  _readPrimary.store(0L);
  _readSpecial.store(0L);

  // Initialize or reset the write event for cross-thread synchronization
  if (_writeEvent == nullptr) {
    cudaEvent_t* event = new cudaEvent_t();
    // cudaEventDisableTiming for better performance since we don't need timing
    auto res = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
    if (res != cudaSuccess) {
      delete event;
      // Non-fatal: fall back to stream sync if event creation fails
      _writeEvent = nullptr;
    } else {
      _writeEvent = event;
    }
  }
  _writeEventRecorded.store(false);
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

  auto bufferDeviceId = _deviceId.load();
  auto currentDeviceId = AffinityManager::currentDeviceId();
  bool switchedDevice = false;

  if (currentDeviceId != bufferDeviceId) {
    cudaSetDevice(bufferDeviceId);
    switchedDevice = true;
  }

  int res = 0;
  if (other.isPrimaryActual()) {
    res = cudaMemcpy(
        static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
        static_cast<const int8_t*>(other._primaryBuffer) + offsetOther * DataTypeUtils::sizeOfElement(other._dataType),
        sizeToCopyinBytes, cudaMemcpyHostToDevice);
    other.readPrimary();
  } else {
    res = cudaMemcpy(
        static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
        static_cast<const int8_t*>(other._specialBuffer) + offsetOther * DataTypeUtils::sizeOfElement(other._dataType),
        sizeToCopyinBytes, cudaMemcpyDeviceToDevice);
    other.readSpecial();
  }

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  if (res != 0) {
    if (other.isPrimaryActual()) {
      throw cuda_exception::build("DataBuffer::copyBufferFrom: cudaMemcpy_cudaMemcpyHostToDevice failed!", res);
    } else {
      throw cuda_exception::build("DataBuffer::copyBufferFrom: cudaMemcpy_cudaMemcpyDeviceToDevice failed!", res);
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

  auto bufferDeviceId = _deviceId.load();
  auto currentDeviceId = AffinityManager::currentDeviceId();
  bool switchedDevice = false;

  if (currentDeviceId != bufferDeviceId) {
    cudaSetDevice(bufferDeviceId);
    switchedDevice = true;
  }

  auto res =
      cudaMemcpy(static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
                 static_cast<const int8_t*>(hostBuffer) + offsetHostBuffer * DataTypeUtils::sizeOfElement(_dataType),
                 sizeToCopyinBytes, cudaMemcpyHostToDevice);

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  if (res != 0)
    throw cuda_exception::build("DataBuffer::copyBufferFromHost: cudaMemcpy_cudaMemcpyHostToDevice failed!", res);

  writeSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setSpecial(void* special, const bool isOwnerSpecial) {
  deleteSpecial();
  _specialBuffer = special;
  _isOwnerSpecial = isOwnerSpecial;

  if (special != nullptr) {
    _deviceId.store(AffinityManager::currentDeviceId());
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

  auto bufferDeviceId = _deviceId.load();
  auto currentDeviceId = AffinityManager::currentDeviceId();
  bool switchedDevice = false;

  if (currentDeviceId != bufferDeviceId) {
    cudaSetDevice(bufferDeviceId);
    switchedDevice = true;
  }

  // Cache the stream reference - must obtain AFTER device switch so we get the correct device's stream
  auto stream = LaunchContext::defaultContext()->getCudaStream();
  auto res = cudaMemsetAsync(special(), 0, getLenInBytes(), *stream);

  if (res != cudaSuccess) {
    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
    throw cuda_exception::build("DataBuffer::setToZeroBuffers: cudaMemsetAsync failed!", res);
  }

  // Record event for cross-thread synchronization
  // No need to sync stream here - subsequent GPU operations on same stream will
  // automatically wait for memset to complete. This eliminates unnecessary CPU-GPU sync.
  if (_writeEvent != nullptr) {
    cudaEvent_t* event = reinterpret_cast<cudaEvent_t*>(_writeEvent);
    cudaEventRecord(*event, *stream);
    _writeEventRecorded.store(true);
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
void memcpyWithT(DataBuffer* dst, DataBuffer* src, sd::LongType startingOffset, sd::LongType dstOffset) {
  if (src->getLenInBytes() > dst->getLenInBytes())
    THROW_EXCEPTION("DataBuffer::memcpy: Source data buffer is larger than destination");

  auto dstDeviceId = dst->deviceId();
  auto currentDeviceId = AffinityManager::currentDeviceId();
  bool switchedDevice = false;

  if (currentDeviceId != dstDeviceId) {
    cudaSetDevice(dstDeviceId);
    switchedDevice = true;
  }

  // Cache the stream reference - must obtain AFTER device switch
  auto stream = LaunchContext::defaultContext()->getCudaStream();

  cudaError_t res = cudaSuccess;
  if (src->isSpecialActual()) {
    res = cudaMemcpyAsync(dst->specialAtOffset<T>(dstOffset), src->specialAtOffset<T>(startingOffset), src->getLenInBytes(), cudaMemcpyDeviceToDevice,
                          *stream);
  } else if (src->isPrimaryActual()) {
    res = cudaMemcpyAsync(dst->specialAtOffset<T>(dstOffset), src->specialAtOffset<T>(startingOffset), src->getLenInBytes(), cudaMemcpyHostToDevice,
                          *stream);
  }

  if (res != cudaSuccess) {
    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
    throw cuda_exception::build("DataBuffer::memcpy: cudaMemcpyAsync failed!", res);
  }

  // No stream sync needed here - subsequent GPU operations on same stream will
  // automatically wait for memcpy to complete. This eliminates unnecessary CPU-GPU sync.
  // The writeSpecial() below will track the write event for cross-thread sync if needed.

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  dst->writeSpecial();
}
BUILD_SINGLE_TEMPLATE(void memcpyWithT, (DataBuffer* dst, DataBuffer* src, sd::LongType startingOffset, sd::LongType dstOffset), SD_COMMON_TYPES);

void DataBuffer::memcpy(DataBuffer* dst, DataBuffer* src,
                        sd::LongType startingOffset, sd::LongType dstOffset) {
  BUILD_SINGLE_SELECTOR(src->getDataType(), memcpyWithT, (dst, src, startingOffset, dstOffset), SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::migrate() {
  if (isConstant) {
    return;
  }

  auto currentDeviceId = AffinityManager::currentDeviceId();
  auto oldDeviceId = _deviceId.load();

  // Don't migrate if already on the target device
  if (oldDeviceId == currentDeviceId && _specialBuffer != nullptr) {
    return;
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

  if (_specialBuffer != nullptr) {
    // Copy from old device to new device
    if (oldDeviceId != currentDeviceId && oldDeviceId >= 0) {
      // Cross-device copy - stage through host memory for reliability
      void* hostStaging = nullptr;
      auto allocRes = cudaMallocHost(&hostStaging, getLenInBytes());
      if (allocRes != cudaSuccess) {
        std::string err = "DataBuffer::migrate: cudaMallocHost for staging failed! Error: " +
                          std::string(cudaGetErrorString(allocRes)) +
                          ", bytes: " + std::to_string(getLenInBytes()) +
                          ", from device " + std::to_string(oldDeviceId) + " to device " + std::to_string(currentDeviceId);
        THROW_EXCEPTION(err.c_str());
      }

      // Copy from source device to host - need to be on source device for this
      auto setRes = cudaSetDevice(oldDeviceId);
      if (setRes != cudaSuccess) {
        cudaFreeHost(hostStaging);
        cudaSetDevice(currentDeviceId);
        std::string err = "DataBuffer::migrate: Failed to switch to source device " + std::to_string(oldDeviceId) +
                          ": " + std::string(cudaGetErrorString(setRes));
        THROW_EXCEPTION(err.c_str());
      }

      // Synchronize the default stream on source device to ensure prior operations complete
      auto srcStream = sd::LaunchContext::defaultContext()->getCudaStream();
      if (srcStream != nullptr)
        cudaStreamSynchronize(*srcStream);

      auto d2hRes = cudaMemcpy(hostStaging, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToHost);
      if (d2hRes != cudaSuccess) {
        cudaFreeHost(hostStaging);
        cudaSetDevice(currentDeviceId);
        std::string err = "DataBuffer::migrate: D2H copy failed! Error: " + std::string(cudaGetErrorString(d2hRes)) +
                          ", bytes: " + std::to_string(getLenInBytes()) + ", from device " + std::to_string(oldDeviceId);
        THROW_EXCEPTION(err.c_str());
      }

      // Switch back to target device for H2D copy
      setRes = cudaSetDevice(currentDeviceId);
      if (setRes != cudaSuccess) {
        cudaFreeHost(hostStaging);
        std::string err = "DataBuffer::migrate: Failed to switch to target device " + std::to_string(currentDeviceId) +
                          ": " + std::string(cudaGetErrorString(setRes));
        THROW_EXCEPTION(err.c_str());
      }

      auto h2dRes = cudaMemcpy(newBuffer, hostStaging, getLenInBytes(), cudaMemcpyHostToDevice);
      cudaFreeHost(hostStaging);
      if (h2dRes != cudaSuccess) {
        std::string err = "DataBuffer::migrate: H2D copy failed! Error: " + std::string(cudaGetErrorString(h2dRes)) +
                          ", bytes: " + std::to_string(getLenInBytes()) + ", to device " + std::to_string(currentDeviceId);
        THROW_EXCEPTION(err.c_str());
      }

      // cudaMemcpy is synchronous, no additional sync needed
    } else {
      // Same device copy or unknown source device
      auto res = cudaMemcpy(newBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToDevice);
      if (res != cudaSuccess) {
        std::string err = "DataBuffer::migrate: cudaMemcpy D2D failed! Error: " + std::string(cudaGetErrorString(res)) +
                          ", bytes: " + std::to_string(getLenInBytes()) + ", device " + std::to_string(currentDeviceId);
        THROW_EXCEPTION(err.c_str());
      }
    }
  } else if (_primaryBuffer != nullptr) {
    // Copy from host to device if no special buffer exists
    auto res = cudaMemcpy(newBuffer, _primaryBuffer, getLenInBytes(), cudaMemcpyHostToDevice);
    if (res != cudaSuccess) {
      std::string err = "DataBuffer::migrate: cudaMemcpy H2D failed! Error: " + std::string(cudaGetErrorString(res)) +
                        ", bytes: " + std::to_string(getLenInBytes()) + ", to device " + std::to_string(currentDeviceId);
      THROW_EXCEPTION(err.c_str());
    }
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();

  // Record transfer metrics
  if (oldBuffer != nullptr) {
    // Device to device transfer (possibly peer-to-peer)
    TransferType transferType = (oldDeviceId != currentDeviceId) ?
        TransferType::PEER_TO_PEER : TransferType::DEVICE_TO_DEVICE;
    TransferMetrics::getInstance().recordTransfer(transferType, getLenInBytes(), durationNs,
                                                   oldDeviceId, currentDeviceId);
  } else if (_primaryBuffer != nullptr) {
    // Host to device transfer
    TransferMetrics::getInstance().recordTransfer(TransferType::HOST_TO_DEVICE, getLenInBytes(),
                                                   durationNs, -1, currentDeviceId);
  }

  if (_isOwnerSpecial && oldBuffer != nullptr) {
    // Switch to old device to release memory
    if (oldDeviceId != currentDeviceId && oldDeviceId >= 0) {
      cudaSetDevice(oldDeviceId);
    }

    auto p = reinterpret_cast<int8_t*>(oldBuffer);
    RELEASE_SPECIAL(p, _workspace);

    // Switch back to current device
    if (oldDeviceId != currentDeviceId && oldDeviceId >= 0) {
      cudaSetDevice(currentDeviceId);
    }
  }

  _isOwnerSpecial = true;
  _specialBuffer = newBuffer;

  // Store actual device where memory was allocated (may differ after failover)
  _deviceId.store(actualMigrateDevice);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::writePrimary() const { _writePrimary = ++_counter; }
void DataBuffer::writeSpecial() const {
  _writeSpecial = ++_counter;

  // Record an event on the current stream so that syncToPrimary() can properly
  // synchronize even when called from a different thread with a different stream.
  // This fixes the cross-thread synchronization bug where CUDA kernels complete
  // on Stream A but syncToPrimary() syncs on Thread B's stream (useless).

  // Create event on demand if it doesn't exist (handles DataBuffers created
  // before this fix was added, or if setCountersToZero() wasn't called)
  if (_writeEvent == nullptr) {
    cudaEvent_t* event = new cudaEvent_t();
    auto res = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
    if (res != cudaSuccess) {
      delete event;
      // Non-fatal: fall back to stream sync if event creation fails
      return;
    }
    _writeEvent = event;  // _writeEvent is mutable
  }

  cudaEvent_t* event = reinterpret_cast<cudaEvent_t*>(_writeEvent);
  cudaStream_t* stream = reinterpret_cast<cudaStream_t*>(
      LaunchContext::defaultContext()->getCudaStream());
  if (stream != nullptr && *stream != nullptr) {
    auto res = cudaEventRecord(*event, *stream);
    if (res == cudaSuccess) {
      _writeEventRecorded.store(true);
    }
  }
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
