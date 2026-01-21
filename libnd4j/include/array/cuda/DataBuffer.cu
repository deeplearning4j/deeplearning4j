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

    // Allocate new buffer on current device
    ALLOCATE_SPECIAL(newSpecialBuffer, _workspace, size, int8_t);
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
    _isOwnerSpecial = true;

    _deviceId.store(currentDeviceId);
  }
}

DataBuffer DataBuffer::dup() {
  DataBuffer result;
  result._dataType = _dataType;
  result._lenInBytes = _lenInBytes;
  result._primaryBuffer = _primaryBuffer;
  result._specialBuffer = _specialBuffer;
  result._isOwnerPrimary = _isOwnerPrimary;
  result._isOwnerSpecial = _isOwnerSpecial;
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

    ALLOCATE_SPECIAL(_specialBuffer, _workspace, getLenInBytes(), int8_t);
    _isOwnerSpecial = true;

    _deviceId.store(deviceId);

#if defined(SD_GCC_FUNCTRACE)
    // Record SPECIAL (device) buffer allocation
    array::DataBufferLifecycleTracker::getInstance().recordAllocation(
        _specialBuffer, getLenInBytes(), getDataType(),
       array::BufferType::SPECIAL, this, _workspace != nullptr);
#endif

    if (_workspace == nullptr) {
      memory::MemoryCounter::getInstance().countIn(deviceId, getLenInBytes());
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

  auto bufferDeviceId = _deviceId.load();
  auto currentDeviceId = AffinityManager::currentDeviceId();
  bool switchedDevice = false;

  if (currentDeviceId != bufferDeviceId) {
    cudaSetDevice(bufferDeviceId);
    switchedDevice = true;
  }

  auto res = cudaStreamSynchronize(*context->getCudaStream());
  if (res != 0)  {
    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
    std::string errorMessage;
    errorMessage += "DataBuffer::syncToPrimary: cudaStreamSynchronize failed: ";
    errorMessage += std::to_string(getLenInBytes());
    errorMessage += " ";
    errorMessage += cudaGetErrorString(res);
    errorMessage += " (buffer device: ";
    errorMessage += std::to_string(bufferDeviceId);
    errorMessage += ")";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  // Track D2H transfer
  auto startTime = std::chrono::high_resolution_clock::now();

  res = cudaMemcpy(_primaryBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToHost);
  if (res != 0) {
    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
    std::string errorMessage;
    errorMessage += "DataBuffer::syncToPrimary: cudaMemcpy failed: ";
    errorMessage += std::to_string(getLenInBytes());
    errorMessage += " ";
    errorMessage += cudaGetErrorString(res);
    errorMessage += " (buffer device: ";
    errorMessage += std::to_string(bufferDeviceId);
    errorMessage += ")";
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

  auto bufferDeviceId = _deviceId.load();
  auto currentDeviceId = AffinityManager::currentDeviceId();
  bool switchedDevice = false;

  if (currentDeviceId != bufferDeviceId) {
    cudaSetDevice(bufferDeviceId);
    switchedDevice = true;
  }

  // Track H2D transfer
  auto startTime = std::chrono::high_resolution_clock::now();

  auto res = cudaMemcpy(_specialBuffer, _primaryBuffer, getLenInBytes(), cudaMemcpyHostToDevice);
  if (res != 0) {
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
    _specialBuffer = nullptr;
    _isOwnerSpecial = false;

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
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setCountersToZero() {
  _counter.store(0L);
  _writePrimary.store(0L);
  _writeSpecial.store(0L);
  _readPrimary.store(0L);
  _readSpecial.store(0L);
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
  cudaMemsetAsync(special(), 0, getLenInBytes(), *stream);
  auto res = cudaStreamSynchronize(*stream);

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  if (res != 0) throw cuda_exception::build("DataBuffer::setToZeroBuffers: streamSync failed!", res);

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

  int res = 0;
  if (src->isSpecialActual()) {
    res = cudaMemcpyAsync(dst->specialAtOffset<T>(dstOffset), src->specialAtOffset<T>(startingOffset), src->getLenInBytes(), cudaMemcpyDeviceToDevice,
                          *stream);
  } else if (src->isPrimaryActual()) {
    res = cudaMemcpyAsync(dst->specialAtOffset<T>(dstOffset), src->specialAtOffset<T>(startingOffset), src->getLenInBytes(), cudaMemcpyHostToDevice,
                          *stream);
  }

  if (res != 0) {
    if (switchedDevice) {
      cudaSetDevice(currentDeviceId);
    }
    throw cuda_exception::build("DataBuffer::memcpy: cudaMemcpyAsync failed!", res);
  }

  res = cudaStreamSynchronize(*stream);

  // Restore original device if we switched
  if (switchedDevice) {
    cudaSetDevice(currentDeviceId);
  }

  if (res != 0) throw cuda_exception::build("DataBuffer::memcpy: streamSync failed!", res);

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

  memory::Workspace* newWorkspace = nullptr;
  void* newBuffer;
  void* oldBuffer = _specialBuffer;  // Save old buffer pointer for deallocation

  // Start timing the transfer
  auto startTime = std::chrono::high_resolution_clock::now();

  // Allocate on current (target) device
  ALLOCATE_SPECIAL(newBuffer, newWorkspace, getLenInBytes(), int8_t);

  if (_specialBuffer != nullptr) {
    // Copy from old device to new device (cross-device copy)
    auto res = cudaMemcpy(newBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToDevice);
    if (res != 0) throw cuda_exception::build("DataBuffer::migrate: cudaMemcpy failed!", res);
  } else if (_primaryBuffer != nullptr) {
    // Copy from host to device if no special buffer exists
    auto res = cudaMemcpy(newBuffer, _primaryBuffer, getLenInBytes(), cudaMemcpyHostToDevice);
    if (res != 0) throw cuda_exception::build("DataBuffer::migrate: cudaMemcpy H2D failed!", res);
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

  // Update device ID to the current device
  _deviceId.store(currentDeviceId);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::writePrimary() const { _writePrimary = ++_counter; }
void DataBuffer::writeSpecial() const { _writeSpecial = ++_counter; }
void DataBuffer::readPrimary() const { _readPrimary = ++_counter; }
void DataBuffer::readSpecial() const { _readSpecial = ++_counter; }
bool DataBuffer::isPrimaryActual() const {
  return (_writePrimary.load() > _writeSpecial.load() || _readPrimary.load() > _writeSpecial.load());
}
bool DataBuffer::isSpecialActual() const {
  return (_writeSpecial.load() > _writePrimary.load() || _readSpecial.load() > _writePrimary.load());
}

}  // namespace sd
