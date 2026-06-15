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
#include <array/DataBuffer.h>
#include <array/DataBufferLifecycleTracker.h>
#include <array/DataTypeUtils.h>
#include <types/types.h>
#include <system/type_boilerplate.h>


namespace sd {

// Definition of thread-local graph execution flag (declared in DataBuffer.h)
// Set during graph segment execution (oneDNN Graph, ACL Dynamic Fusion).
// SD_TLS_EXPORT needed on Windows/MinGW so __emutls symbols are exported from DLL.
SD_TLS_EXPORT thread_local bool tl_graphExecutionActive = false;

// Thread-local accumulator for pinned host buffers during graph capture (CUDA only).
// On CPU builds, this is unused but must be defined to satisfy the linker.
SD_TLS_EXPORT thread_local std::vector<void*> tl_capturedHostPtrs;
SD_TLS_EXPORT thread_local std::unordered_map<uint64_t, void*> tl_captureReplicateCache;

// CPU stubs for CUDA graph capture workspace variables (declared in DataBuffer.h)
SD_TLS_EXPORT thread_local void* tl_captureWorkspace = nullptr;
SD_TLS_EXPORT thread_local size_t tl_captureWorkspaceSize = 0;
SD_TLS_EXPORT thread_local size_t tl_captureWorkspaceOffset = 0;

// CPU stubs for cuBLAS workspace thread-locals (declared in DataBuffer.h)
SD_TLS_EXPORT thread_local void*  tl_cublasWorkspacePtr = nullptr;
SD_TLS_EXPORT thread_local size_t tl_cublasWorkspaceSize = 0;

// CPU stubs for DSP slot-exec alloc/free accounting thread-locals (declared in DataBuffer.h).
// NativeDynamicShapePlan_segments.cpp references them unconditionally; the CUDA definitions
// live in array/cuda/DataBuffer.cu and are not compiled on CPU builds.
SD_TLS_EXPORT thread_local long long tl_dspAllocBytes = 0;
SD_TLS_EXPORT thread_local long long tl_dspFreeBytes = 0;
SD_TLS_EXPORT thread_local int tl_dspAllocCount = 0;
SD_TLS_EXPORT thread_local int tl_dspFreeCount = 0;
SD_TLS_EXPORT thread_local int tl_dspFreeSkipCount = 0;

void DataBuffer::replaceSpecialBuffer(void* newPtr, bool isOwner) {
  // No-op on CPU: there is no device (special) buffer to replace.
}

void DataBuffer::expand(const uint64_t size) {
  throwIfFrozen("expand");
  if (static_cast<LongType>(size) > _lenInBytes) {
    // allocate new buffer
    int8_t* newBuffer = nullptr;
    static constexpr size_t HOST_ALLOC_PADDING = 65536;
    size_t allocSize = size + (_workspace == nullptr ? HOST_ALLOC_PADDING : 0);
    ALLOCATE(newBuffer, _workspace, allocSize, int8_t);
#if defined(SD_GCC_FUNCTRACE)
    // Track the new allocation before swapping pointers
    sd::array::DataBufferLifecycleTracker::getInstance().recordAllocation(
        newBuffer, size, _dataType, sd::array::BufferType::PRIMARY, this, _workspace != nullptr);
#endif

    // copy data from existing buffer
    if (_primaryBuffer != nullptr && _lenInBytes > 0) {
      std::memcpy(newBuffer, _primaryBuffer, _lenInBytes);
    }

    // Write canary values in the padding region (same as allocatePrimary)
    if (_workspace == nullptr) {
      uint64_t* canary = reinterpret_cast<uint64_t*>(newBuffer + size);
      for (size_t i = 0; i < (HOST_ALLOC_PADDING / sizeof(uint64_t)); i++) {
        canary[i] = 0xDEADBEEFCAFEBABEULL;
      }
    }

    if (_isOwnerPrimary) {
#if defined(SD_GCC_FUNCTRACE)
      // Record deallocation of the old primary buffer before releasing it
      sd::array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
          _primaryBuffer, sd::array::BufferType::PRIMARY);
#endif
      RELEASE(reinterpret_cast<int8_t*>(_primaryBuffer), _workspace);
    }

    _primaryBuffer = newBuffer;
    _lenInBytes = size;
    _primaryAllocBytes = allocSize;
    _isOwnerPrimary = true;
  }
}

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
BUILD_SINGLE_TEMPLATE(void DataBuffer::printHostBufferContent,(void* buffer, sd::LongType offset, sd::LongType length),SD_COMMON_TYPES);




// DataBuffer implementation for .cpp file
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

  // Print device info but not content (CPU version)
  if (_specialBuffer != nullptr) {
    sd_printf("Device buffer (@%p): [Not accessible from CPU]\n", _specialBuffer);
  } else {
    sd_printf("Device buffer: nullptr\n", 0);
  }

#if defined(SD_CUDA)
  // Print sync state counters
  sd_printf("Sync state: _counter=%lld, _writePrimary=%lld, _writeSpecial=%lld, _readPrimary=%lld, _readSpecial=%lld\n",
            (long long)_counter.load(), (long long)_writePrimary.load(), (long long)_writeSpecial.load(),
            (long long)_readPrimary.load(), (long long)_readSpecial.load());
  sd_printf("isPrimaryActual=%d, isSpecialActual=%d\n", isPrimaryActual(), isSpecialActual());
#endif
}

// Helper template to print host buffer content
template <typename T>
void printHostBufferContent(void* buffer, sd::LongType offset, sd::LongType length) {
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

void DataBuffer::printSpecialAllocationTraces() {
  //no op on purpose
}


////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateBuffers(const bool allocBoth) {  // always allocate primary buffer only (cpu case)
  allocatePrimary();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFrom(const DataBuffer& other,
                                size_t sizeToCopyinBytes,
                                const sd::LongType offsetThis,
                                const sd::LongType offsetOther) {
  if(other._dataType != _dataType) {
    THROW_EXCEPTION("DataBuffer::copyBufferFrom: data types of buffers are different");
  }
  if (sizeToCopyinBytes == 0) {
    LongType otherBytes = other.getLenInBytes() - offsetOther;
    LongType thisBytes = getLenInBytes() - offsetThis;
    sizeToCopyinBytes = otherBytes < thisBytes ? otherBytes : thisBytes;
  }
  if (sizeToCopyinBytes == 0) return;
  if(static_cast<LongType>(sizeToCopyinBytes) > other._lenInBytes - offsetOther) {
    std::string errorMessage;
    errorMessage = "DataBuffer::copyBufferFrom: size to copy is larger than source buffer ";
    errorMessage += std::to_string(sizeToCopyinBytes);
    errorMessage += " > ";
    errorMessage += std::to_string(other._lenInBytes - offsetOther);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if(sizeToCopyinBytes > getLenInBytes() - offsetThis) {
    std::string errorMessage;
    errorMessage = "DataBuffer::copyBufferFrom: size to copy is larger than destination buffer ";
    errorMessage += std::to_string(sizeToCopyinBytes);
    errorMessage += " > ";
    errorMessage += std::to_string(getLenInBytes() - offsetThis);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (other._primaryBuffer != nullptr) {
    auto sizeOfElement = DataTypeUtils::sizeOfElement(_dataType);
    auto sizeOfOtherElement = DataTypeUtils::sizeOfElement(_dataType);
    if(sizeOfElement != sizeOfOtherElement) {
      THROW_EXCEPTION("DataBuffer::copyBufferFrom: size of elements in buffers are different");
    }
    std::memcpy(
        static_cast<int8_t*>(_primaryBuffer) + offsetThis * sizeOfElement,
        static_cast<const int8_t*>(other._primaryBuffer) + offsetOther * sizeOfOtherElement,
        sizeToCopyinBytes);
  }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFromHost(const void* hostBuffer, size_t sizeToCopyinBytes, const sd::LongType offsetThis,
                                    const sd::LongType offsetHostBuffer) {
  if (sizeToCopyinBytes == 0) sizeToCopyinBytes = getLenInBytes();
  if (sizeToCopyinBytes == 0) return;

  if (hostBuffer != nullptr)
    std::memcpy(static_cast<int8_t*>(_primaryBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
                static_cast<const int8_t*>(hostBuffer) + offsetHostBuffer * DataTypeUtils::sizeOfElement(_dataType),
                sizeToCopyinBytes);
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

  std::memcpy(dst->primaryAtOffset<T>(dstOffset), src->primaryAtOffset<T>(startingOffset), copyBytes);
  dst->readPrimary();
}

void DataBuffer::memcpy(DataBuffer* dst, DataBuffer* src,
                        sd::LongType startingOffset, sd::LongType dstOffset, sd::LongType n) {
  BUILD_SINGLE_SELECTOR(dst->_dataType, memcpyWithT,(dst, src, startingOffset, dstOffset, n),
                        SD_COMMON_TYPES);

  dst->readPrimary();
}




////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteSpecial() {
  // Always reset pointer and ownership flag for consistency with CUDA version
  // Even though CPU doesn't have a special buffer, this ensures clean state
  _specialBuffer = nullptr;
  _isOwnerSpecial = false;
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::freeGpuOnly() {
  throwIfFrozen("freeGpuOnly");
  // CPU: no GPU buffer to free. On CPU this is called during JVM shutdown
  // (via dbFreeBuffersOnly) to avoid free() crashing on corrupted heap metadata.
  // deletePrimary() now skips free() when canaries are corrupted, so it's safe
  // to call here. deleteSpecial() is a no-op on CPU.
  deleteSpecial();
  deletePrimary();
  closed = true;
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::freeGpuOnStream(void* stream) {
  // CPU: stream parameter is ignored, just delegates to freeGpuOnly
  freeGpuOnly();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToPrimary(const LaunchContext* context, const bool forceSync) {}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setCountersToZero() {}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyCounters(const DataBuffer& other) {}

void DataBuffer::writePrimary() const {}
void DataBuffer::writeSpecial() const {}
void DataBuffer::readPrimary() const {}
void DataBuffer::readSpecial() const {}
bool DataBuffer::isPrimaryActual() const { return true; }
bool DataBuffer::isSpecialActual() const { return false; }
void DataBuffer::showBufferLimited() {}



DataBuffer DataBuffer::dup() {
  DataBuffer result;
  result._dataType = _dataType;
  result._lenInBytes = _lenInBytes;
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

////////////////////////////////////////////////////////////////////////
void DataBuffer::setSpecial(void* special, const bool isOwnerSpecial) {}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setToZeroBuffers(const bool both) { memset(primary(), 0, getLenInBytes()); }

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToSpecial(const bool forceSync) {}

////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateSpecial() {}

////////////////////////////////////////////////////////////////////////
void DataBuffer::migrate() {}


template <typename T>
void _printHostBuffer(DataBuffer* buffer, long offset) {
  sd::LongType len = buffer->getNumElements();
  auto buff = buffer->template primaryAsT<T>();


  sd::LongType limit = len;
  if (limit == -1 || limit >= static_cast<LongType>(buffer->getNumElements())) {
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
        double val = static_cast<double>(buff[e]);
        printf("%.15f",val);
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
  auto xType = getDataType();
  BUILD_SINGLE_SELECTOR(xType, _printHostBuffer,(this,offset),SD_COMMON_TYPES);

}


void DataBuffer::showCounters(const char* msg1, const char* msg2) {

}
template <typename T>
void* DataBuffer::primaryAtOffset(const LongType offset) {
  if(_primaryBuffer == nullptr)
    return nullptr;
  T *type = reinterpret_cast<T*>(_primaryBuffer);
  return reinterpret_cast<void *>(type + offset);
}

#define PRIMARYOFFSET(T) template void* DataBuffer::primaryAtOffset<GET_SECOND(T)>(const LongType offset);
ITERATE_LIST((SD_COMMON_TYPES),PRIMARYOFFSET)

template <typename T>
void* DataBuffer::specialAtOffset(const LongType offset) {
  if(_specialBuffer == nullptr)
    return nullptr;
  T *type = reinterpret_cast<T*>(_specialBuffer);
  return reinterpret_cast<void *>(type + offset);
}

#define SPECIALOFFSET(T) template void* DataBuffer::specialAtOffset<GET_SECOND(T)>(const LongType offset);
ITERATE_LIST((SD_COMMON_TYPES),SPECIALOFFSET)
}  // namespace sd
