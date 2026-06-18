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
#include <array/DataTypeUtils.h>
#include <execution/AffinityManager.h>
#include <helpers/logger.h>
#include <memory/MemoryCounter.h>
#include <system/CanaryConstants.h>
#include <system/Environment.h>
#include <system/PointerValidation.h>
#include <system/env_functions.h>
#include <sstream>

#if defined(SD_GCC_FUNCTRACE)
#include <array/DataBufferLifecycleTracker.h>
#endif

namespace sd {
///// IMPLEMENTATION OF COMMON METHODS /////

////////////////////////////////////////////////////////////////////////
// default constructor
DataBuffer::DataBuffer() {
  if(sd::env_isLogNativeNDArrayCreation()) {
    printf("DataBuffer::DataBuffer() default constructor\n");
    fflush(stdout);
  }
  _primaryBuffer = nullptr;
  _specialBuffer = nullptr;
  _lenInBytes = 0;
  _dataType = INT8;
  _workspace = nullptr;
  _isOwnerPrimary = false;
  _isOwnerSpecial = false;
  _deviceId = AffinityManager::currentDeviceId();
#if defined(SD_GCC_FUNCTRACE)
  // - Stack trace capture via backward-cpp's backtrace() is NOT safe during early JVM initialization
  // - The JVM's memory mappings and signal handlers aren't fully set up yet
  // - This causes SIGSEGV crashes at addresses like 0x7f647edc2000 inside glibc internals
  // - Session #953's try-catch doesn't work when C++ exceptions are disabled (common for performance)
  // - DataBufferLifecycleTracker already captures stack traces separately for leak detection
  // - The creationStackTrace was redundant and only used for constructor error messages
  // - Solution: Leave creationStackTrace as nullptr (getCreationTraceAsString() handles this gracefully)
  // - This eliminates crashes while preserving all leak detection functionality
  creationStackTrace = nullptr;
#endif
  setCountersToZero();
}

////////////////////////////////////////////////////////////////////////
// copy constructor
DataBuffer::DataBuffer(const DataBuffer& other) {
  if(other._dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION("DataBuffer constructor: dataType is UNKNOWN !");
  }
  if(sd::env_isLogNativeNDArrayCreation()) {
    printf("DataBuffer::DataBuffer(const DataBuffer& other) copy constructor\n");
    fflush(stdout);
  }
  _lenInBytes = other._lenInBytes;
  _primaryAllocBytes = other._primaryAllocBytes;
  _specialAllocBytes = other._specialAllocBytes;
  _dataType = other._dataType;
  _workspace = other._workspace;
#if defined(SD_GCC_FUNCTRACE)
  // Don't share stack traces - they will be created fresh when we allocate
  allocationStackTracePrimary = nullptr;
  allocationStackTraceSpecial = nullptr;
  creationStackTrace = nullptr;
#endif
  _primaryBuffer = other._primaryBuffer;
  _specialBuffer = other._specialBuffer;

#if defined(SD_GCC_FUNCTRACE)
  // - Stack trace capture via backward-cpp's backtrace() is NOT safe during early JVM initialization
  // - The JVM's memory mappings and signal handlers aren't fully set up yet
  // - This causes SIGSEGV crashes at addresses like 0x7f647edc2000 inside glibc internals
  // - Session #953's try-catch doesn't work when C++ exceptions are disabled (common for performance)
  // - DataBufferLifecycleTracker already captures stack traces separately for leak detection
  // - The creationStackTrace was redundant and only used for constructor error messages
  // - Solution: Leave creationStackTrace as nullptr (getCreationTraceAsString() handles this gracefully)
  // - This eliminates crashes while preserving all leak detection functionality
  creationStackTrace = nullptr;
#endif

  _deviceId.store(other._deviceId.load());

  setCountersToZero();

  allocateBuffers();
  copyBufferFrom(other);
}

////////////////////////////////////////////////////////////////////////
DataBuffer::DataBuffer(void* primary, void* special, const size_t lenInBytes, const DataType dataType,
                       const bool isOwnerPrimary, const bool isOwnerSpecial, memory::Workspace* workspace) {
  if(dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION("DataBuffer constructor: dataType is UNKNOWN !");
  }
  if(sd::env_isLogNativeNDArrayCreation()) {
    printf(
        "DataBuffer::DataBuffer(void* primary, void* special, const size_t lenInBytes, const DataType dataType, const bool isOwnerPrimary, const bool isOwnerSpecial, memory::Workspace* workspace) constructor\n");
    fflush(stdout);
  }
  _primaryBuffer = primary;
  _specialBuffer = special;
  _lenInBytes = lenInBytes;
  _dataType = dataType;
  _workspace = workspace;
  _isOwnerPrimary = isOwnerPrimary;
  _isOwnerSpecial = isOwnerSpecial;
  _deviceId = AffinityManager::currentDeviceId();
#if defined(SD_GCC_FUNCTRACE)
  // - Stack trace capture via backward-cpp's backtrace() is NOT safe during early JVM initialization
  // - The JVM's memory mappings and signal handlers aren't fully set up yet
  // - This causes SIGSEGV crashes at addresses like 0x7f647edc2000 inside glibc internals
  // - Session #953's try-catch doesn't work when C++ exceptions are disabled (common for performance)
  // - DataBufferLifecycleTracker already captures stack traces separately for leak detection
  // - The creationStackTrace was redundant and only used for constructor error messages
  // - Solution: Leave creationStackTrace as nullptr (getCreationTraceAsString() handles this gracefully)
  // - This eliminates crashes while preserving all leak detection functionality
  creationStackTrace = nullptr;
#endif
  setCountersToZero();

  if (primary != nullptr) {
    readPrimary();
  }
  if (special != nullptr) {
    readSpecial();
  }
}

////////////////////////////////////////////////////////////////////////
DataBuffer::DataBuffer(void* primary, const size_t lenInBytes, const DataType dataType, const bool isOwnerPrimary,
                       memory::Workspace* workspace)
    : DataBuffer(primary, nullptr, lenInBytes, dataType, isOwnerPrimary, false, workspace) {
  if(dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION("DataBuffer constructor: dataType is UNKNOWN !");
  }

  if(sd::env_isLogNativeNDArrayCreation()) {
    printf("DataBuffer::DataBuffer(void* primary, const size_t lenInBytes, const DataType dataType, const bool isOwnerPrimary, memory::Workspace* workspace) constructor\n");
    fflush(stdout);
  }

  if(primary != nullptr)
    syncToSpecial(true);

#if defined(SD_GCC_FUNCTRACE)
  // - Stack trace capture via backward-cpp's backtrace() is NOT safe during early JVM initialization
  // - The JVM's memory mappings and signal handlers aren't fully set up yet
  // - This causes SIGSEGV crashes at addresses like 0x7f647edc2000 inside glibc internals
  // - Session #953's try-catch doesn't work when C++ exceptions are disabled (common for performance)
  // - DataBufferLifecycleTracker already captures stack traces separately for leak detection
  // - The creationStackTrace was redundant and only used for constructor error messages
  // - Solution: Leave creationStackTrace as nullptr (getCreationTraceAsString() handles this gracefully)
  // - This eliminates crashes while preserving all leak detection functionality
  creationStackTrace = nullptr;
#endif
}

////////////////////////////////////////////////////////////////////////
// copies data from hostBuffer to own memory buffer
DataBuffer::DataBuffer(const void* hostBuffer, const DataType dataType, const size_t lenInBytes,
                       memory::Workspace* workspace) {
  if(dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION("DataBuffer constructor: dataType is UNKNOWN !");
  }

  if(sd::env_isLogNativeNDArrayCreation()) {
    printf("DataBuffer::DataBuffer(const void* hostBuffer, const DataType dataType, const size_t lenInBytes, memory::Workspace* workspace) constructor\n");
    fflush(stdout);
  }
  if (hostBuffer == nullptr) {
#if defined(SD_GCC_FUNCTRACE)
    std::string traceInfo = getCreationTraceAsString();
    std::string errorMsg = "DataBuffer constructor: can't be initialized with nullptr host buffer !";
    if (!traceInfo.empty()) {
      errorMsg += "\n\nDataBuffer allocation trace:\n" + traceInfo;
    }
    THROW_EXCEPTION(errorMsg.c_str());
#else
    THROW_EXCEPTION("DataBuffer constructor: can't be initialized with nullptr host buffer !");
#endif
  }
  if (lenInBytes == 0) {
#if defined(SD_GCC_FUNCTRACE)
    std::string traceInfo = getCreationTraceAsString();
    std::string errorMsg = "DataBuffer constructor: can't be initialized with zero length !";
    if (!traceInfo.empty()) {
      errorMsg += "\n\nDataBuffer allocation trace:\n" + traceInfo;
    }
    THROW_EXCEPTION(errorMsg.c_str());
#else
    THROW_EXCEPTION("DataBuffer constructor: can't be initialized with zero length !");
#endif
  }

  _primaryBuffer = nullptr;
  _specialBuffer = nullptr;
  _lenInBytes = lenInBytes;
  _dataType = dataType;
  _workspace = workspace;

  _deviceId = AffinityManager::currentDeviceId();

  setCountersToZero();

  allocateBuffers();

  copyBufferFromHost(hostBuffer, lenInBytes);

#if defined(SD_GCC_FUNCTRACE)
  // - Stack trace capture via backward-cpp's backtrace() is NOT safe during early JVM initialization
  // - The JVM's memory mappings and signal handlers aren't fully set up yet
  // - This causes SIGSEGV crashes at addresses like 0x7f647edc2000 inside glibc internals
  // - Session #953's try-catch doesn't work when C++ exceptions are disabled (common for performance)
  // - DataBufferLifecycleTracker already captures stack traces separately for leak detection
  // - The creationStackTrace was redundant and only used for constructor error messages
  // - Solution: Leave creationStackTrace as nullptr (getCreationTraceAsString() handles this gracefully)
  // - This eliminates crashes while preserving all leak detection functionality
  creationStackTrace = nullptr;
#endif
}

////////////////////////////////////////////////////////////////////////
DataBuffer::DataBuffer(const sd::LongType lenInBytes, const DataType dataType, memory::Workspace* workspace,
                       const bool allocBoth) {

  if(dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION("DataBuffer constructor: dataType is UNKNOWN !");
  }

  if(sd::env_isLogNativeNDArrayCreation()) {
    printf("DataBuffer::DataBuffer(const size_t lenInBytes, const DataType dataType, memory::Workspace* workspace, const bool allocBoth) constructor\n");
    fflush(stdout);
  }



  _dataType = dataType;
  _workspace = workspace;
  _lenInBytes = lenInBytes;

  _primaryBuffer = nullptr;
  _specialBuffer = nullptr;
  _isOwnerPrimary = false;
  _isOwnerSpecial = false;

  _deviceId = AffinityManager::currentDeviceId();

  setCountersToZero();

  allocateBuffers(allocBoth);
  writeSpecial();

#if defined(SD_GCC_FUNCTRACE)
  // - Stack trace capture via backward-cpp's backtrace() is NOT safe during early JVM initialization
  // - The JVM's memory mappings and signal handlers aren't fully set up yet
  // - This causes SIGSEGV crashes at addresses like 0x7f647edc2000 inside glibc internals
  // - Session #953's try-catch doesn't work when C++ exceptions are disabled (common for performance)
  // - DataBufferLifecycleTracker already captures stack traces separately for leak detection
  // - The creationStackTrace was redundant and only used for constructor error messages
  // - Solution: Leave creationStackTrace as nullptr (getCreationTraceAsString() handles this gracefully)
  // - This eliminates crashes while preserving all leak detection functionality
  creationStackTrace = nullptr;
#endif

}

////////////////////////////////////////////////////////////////////////
// move constructor
DataBuffer::DataBuffer(DataBuffer&& other) {

  if(other._dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION("DataBuffer constructor: dataType is UNKNOWN !");
  }

  if(sd::env_isLogNativeNDArrayCreation()) {
    printf("DataBuffer::DataBuffer(DataBuffer&& other) move constructor\n");
    fflush(stdout);
  }
  _primaryBuffer = other._primaryBuffer;
  _specialBuffer = other._specialBuffer;
  _lenInBytes = other._lenInBytes;
  _primaryAllocBytes = other._primaryAllocBytes;
  _specialAllocBytes = other._specialAllocBytes;
  _dataType = other._dataType;
  _workspace = other._workspace;
  _isOwnerPrimary = other._isOwnerPrimary;
  _isOwnerSpecial = other._isOwnerSpecial;
  _deviceId.store(other._deviceId);
  _specialDeviceId.store(other._specialDeviceId.load());  // Also copy special device ID for multi-GPU

  copyCounters(other);
  _writeEvent = other._writeEvent;
  _writeEventRecorded.store(other._writeEventRecorded.load(std::memory_order_acquire),
                            std::memory_order_release);
  _writeEventDeviceId.store(other._writeEventDeviceId.load(std::memory_order_acquire),
                            std::memory_order_release);
  other._writeEvent = nullptr;
  other._writeEventRecorded.store(false, std::memory_order_release);
  other._writeEventDeviceId.store(-1, std::memory_order_release);
#if defined(SD_GCC_FUNCTRACE)
  allocationStackTracePrimary = other.allocationStackTracePrimary;
  allocationStackTraceSpecial = other.allocationStackTraceSpecial;
  creationStackTrace = other.creationStackTrace;
  // Transfer ownership - null out the source pointers to prevent double-free
  other.allocationStackTracePrimary = nullptr;
  other.allocationStackTraceSpecial = nullptr;
  other.creationStackTrace = nullptr;
#endif
  other._primaryBuffer = other._specialBuffer = nullptr;
  other.setAllocFlags(false, false);
  other._lenInBytes = 0;
  other._primaryAllocBytes = 0;
  other._specialAllocBytes = 0;

#if defined(SD_GCC_FUNCTRACE)
  // - Stack trace capture via backward-cpp's backtrace() is NOT safe during early JVM initialization
  // - The JVM's memory mappings and signal handlers aren't fully set up yet
  // - This causes SIGSEGV crashes at addresses like 0x7f647edc2000 inside glibc internals
  // - Session #953's try-catch doesn't work when C++ exceptions are disabled (common for performance)
  // - DataBufferLifecycleTracker already captures stack traces separately for leak detection
  // - The creationStackTrace was redundant and only used for constructor error messages
  // - Solution: Leave creationStackTrace as nullptr (getCreationTraceAsString() handles this gracefully)
  // - This eliminates crashes while preserving all leak detection functionality
  creationStackTrace = nullptr;
#endif
}

////////////////////////////////////////////////////////////////////////
// assignment operator
DataBuffer& DataBuffer::operator=(const DataBuffer& other) {
  if(other._dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION("DataBuffer assignment operator: dataType is UNKNOWN !");
  }
  if(sd::env_isLogNativeNDArrayCreation()) {
    printf("DataBuffer::operator=(const DataBuffer& other) assignment operator\n");
    fflush(stdout);
  }
  if (this == &other) return *this;

  deleteBuffers();

  _lenInBytes = other._lenInBytes;
  _primaryAllocBytes = other._primaryAllocBytes;
  _specialAllocBytes = other._specialAllocBytes;
  _dataType = other._dataType;
  _workspace = other._workspace;

  allocateBuffers();
  copyBufferFrom(other);
#if defined(SD_GCC_FUNCTRACE)
  // - Stack trace capture via backward-cpp's backtrace() is NOT safe during early JVM initialization
  // - The JVM's memory mappings and signal handlers aren't fully set up yet
  // - This causes SIGSEGV crashes at addresses like 0x7f647edc2000 inside glibc internals
  // - Session #953's try-catch doesn't work when C++ exceptions are disabled (common for performance)
  // - DataBufferLifecycleTracker already captures stack traces separately for leak detection
  // - The creationStackTrace was redundant and only used for constructor error messages
  // - Solution: Leave creationStackTrace as nullptr (getCreationTraceAsString() handles this gracefully)
  // - This eliminates crashes while preserving all leak detection functionality
  creationStackTrace = nullptr;
#endif
  return *this;
}

////////////////////////////////////////////////////////////////////////
// move assignment operator
DataBuffer& DataBuffer::operator=(DataBuffer&& other) noexcept {
  if(other._dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION("DataBuffer move assignment operator: dataType is UNKNOWN !");
  }

  if(sd::env_isLogNativeNDArrayCreation()) {
    printf("DataBuffer::operator=(DataBuffer&& other) move assignment operator\n");
    fflush(stdout);
  }
  if (this == &other) return *this;

  deleteBuffers();

  _primaryBuffer = other._primaryBuffer;
  _specialBuffer = other._specialBuffer;
  _lenInBytes = other._lenInBytes;
  _primaryAllocBytes = other._primaryAllocBytes;
  _specialAllocBytes = other._specialAllocBytes;
  _dataType = other._dataType;
  _workspace = other._workspace;
  _isOwnerPrimary = other._isOwnerPrimary;
  _isOwnerSpecial = other._isOwnerSpecial;
  _deviceId.store(other._deviceId);
  _specialDeviceId.store(other._specialDeviceId.load());  // Also copy special device ID for multi-GPU

  copyCounters(other);
  _writeEvent = other._writeEvent;
  _writeEventRecorded.store(other._writeEventRecorded.load(std::memory_order_acquire),
                            std::memory_order_release);
  _writeEventDeviceId.store(other._writeEventDeviceId.load(std::memory_order_acquire),
                            std::memory_order_release);
  other._writeEvent = nullptr;
  other._writeEventRecorded.store(false, std::memory_order_release);
  other._writeEventDeviceId.store(-1, std::memory_order_release);

#if defined(SD_GCC_FUNCTRACE)
  allocationStackTracePrimary = other.allocationStackTracePrimary;
  allocationStackTraceSpecial = other.allocationStackTraceSpecial;
  creationStackTrace = other.creationStackTrace;
  // Transfer ownership - null out the source pointers to prevent double-free
  other.allocationStackTracePrimary = nullptr;
  other.allocationStackTraceSpecial = nullptr;
  other.creationStackTrace = nullptr;
#endif

  other._primaryBuffer = other._specialBuffer = nullptr;
  other.setAllocFlags(false, false);
  other._lenInBytes = 0;
  other._primaryAllocBytes = 0;
  other._specialAllocBytes = 0;
#if defined(SD_GCC_FUNCTRACE)
  // - Stack trace capture via backward-cpp's backtrace() is NOT safe during early JVM initialization
  // - The JVM's memory mappings and signal handlers aren't fully set up yet
  // - This causes SIGSEGV crashes at addresses like 0x7f647edc2000 inside glibc internals
  // - Session #953's try-catch doesn't work when C++ exceptions are disabled (common for performance)
  // - DataBufferLifecycleTracker already captures stack traces separately for leak detection
  // - The creationStackTrace was redundant and only used for constructor error messages
  // - Solution: Leave creationStackTrace as nullptr (getCreationTraceAsString() handles this gracefully)
  // - This eliminates crashes while preserving all leak detection functionality
  creationStackTrace = nullptr;
#endif
  return *this;
}


void DataBuffer::markConstant(bool reallyConstant) {
  isConstant = reallyConstant;
}

////////////////////////////////////////////////////////////////////////
// Frozen-phase mutation guard.
// Throws immediately if this buffer is registered in one or more frozen
// NativeDynamicShapePlan contexts. Call from any mutator that would change
// the identity of the backing storage (reallocate, free, setPrimary/Special,
// replaceSpecial, expand, migrate, close, etc.). Content-only writes
// (writePrimary/writeSpecial/syncTo*) must NOT call this — they don't change
// the underlying pointer.
void DataBuffer::throwIfFrozen(const char* op) const {
  int refCount = _frozenRefCount.load(std::memory_order_relaxed);
  if (refCount > 0) {
    char msg[384];
    snprintf(msg, sizeof(msg),
             "DataBuffer LIFECYCLE VIOLATION: %s called on frozen DataBuffer %p "
             "(frozenRefCount=%d, primary=%p, special=%p, lenInBytes=%lld) - "
             "mutation of identity during frozen-phase execution would invalidate "
             "baked-in GPU addresses held by frozen slot contexts / CUDA graph replay handles",
             op ? op : "<unknown>",
             static_cast<const void*>(this),
             refCount,
             _primaryBuffer,
             _specialBuffer,
             static_cast<long long>(_lenInBytes));
    THROW_EXCEPTION(msg);
  }
}

////////////////////////////////////////////////////////////////////////
// Validation method following DirectShapeTrie pattern
// Checks for use-after-free, corrupted pointers, and invalid state
void DataBuffer::validateIntegrity() const {
  // Check magic number first - if wrong, pointer is dangling/corrupted
  if (_magicNumber != MAGIC_NUMBER) {
    // Magic number doesn't match - this is a freed/corrupted DataBuffer!
    std::stringstream ss;
    ss << "DataBuffer integrity check FAILED!\n";
    ss << "  Expected magic number: 0x" << std::hex << MAGIC_NUMBER << "\n";
    ss << "  Actual magic number: 0x" << std::hex << _magicNumber << "\n";
    ss << "  Likely causes:\n";
    ss << "    1. Use-after-free: DataBuffer was deleted but pointer still used\n";
    ss << "    2. Corrupted pointer: Pointer points to invalid memory\n";
    ss << "    3. Uninitialized memory: DataBuffer was never properly constructed\n";
    ss << "  This indicates a SERIOUS BUG in buffer lifecycle management!\n";
    ss << "  Check where this DataBuffer pointer came from and ensure it's still valid.\n";
    THROW_EXCEPTION(ss.str().c_str());
  }

  // Check if buffer has been closed
  if (closed) {
    std::stringstream ss;
    ss << "DataBuffer integrity check FAILED!\n";
    ss << "  Buffer has been closed (freed) but is still being accessed\n";
    ss << "  Magic number is valid (0x" << std::hex << _magicNumber << ") but closed flag is true\n";
    ss << "  This indicates use-after-close: buffer was explicitly closed but pointer retained\n";
    THROW_EXCEPTION(ss.str().c_str());
  }

  // Sanity check data type
  if (_dataType == DataType::UNKNOWN) {
    std::stringstream ss;
    ss << "DataBuffer integrity check FAILED!\n";
    ss << "  DataType is UNKNOWN - buffer was not properly initialized\n";
    THROW_EXCEPTION(ss.str().c_str());
  }

  // Sanity check length (negative or excessively large values indicate corruption)
  if (_lenInBytes < 0 || _lenInBytes > (1LL << 40)) {  // 1TB limit
    std::stringstream ss;
    ss << "DataBuffer integrity check FAILED!\n";
    ss << "  Length is invalid: " << _lenInBytes << " bytes\n";
    ss << "  Valid range is 0 to " << (1LL << 40) << " bytes (1TB)\n";
    ss << "  This indicates memory corruption\n";
    THROW_EXCEPTION(ss.str().c_str());
  }

  // Validate canary values to detect buffer overruns.
  // FIX: Skip canary check for zero-length buffers - the canary would be at offset 0
  // (start of buffer), and any write to the buffer would corrupt it.
  if (_workspace == nullptr && _lenInBytes > 0 && _primaryBuffer != nullptr && _primaryAllocBytes > _lenInBytes) {
    const uint64_t* canary = reinterpret_cast<const uint64_t*>(
        static_cast<const int8_t*>(_primaryBuffer) + _lenInBytes);
    size_t numCanaries = (static_cast<size_t>(HOST_ALLOC_PADDING) / sizeof(uint64_t));
    for (size_t i = 0; i < numCanaries; i++) {
      if (canary[i] != sd::CanaryConstants::DATA_BUFFER_CANARY) {
        std::stringstream ss;
        ss << "DataBuffer integrity check FAILED - BUFFER OVERRUN DETECTED!\n";
        ss << "  Canary value at offset " << (i * sizeof(uint64_t)) << " is corrupted\n";
        ss << "  Expected: 0xDEADBEEFCAFEBABE\n";
        ss << "  Actual: 0x" << std::hex << canary[i] << "\n";
        ss << std::dec;
        ss << "  Buffer size: " << _lenInBytes << " bytes (allocBytes=" << _primaryAllocBytes << ")\n";
        ss << "  DataBuffer this=" << (void*)this << " primaryBuffer=" << _primaryBuffer << "\n";
        ss << "  DataType=" << (int)_dataType << " isConstant=" << isConstant << " workspace=" << (void*)_workspace << "\n";
        ss << "  This indicates an operation wrote past the end of the buffer!\n";
        THROW_EXCEPTION(ss.str().c_str());
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////
void* DataBuffer::primary() {
  return _primaryBuffer;
}

////////////////////////////////////////////////////////////////////////
void* DataBuffer::special() {
  return _specialBuffer;
}

////////////////////////////////////////////////////////////////////////
DataType DataBuffer::getDataType() { return _dataType; }

////////////////////////////////////////////////////////////////////////
size_t DataBuffer::getLenInBytes() const {
  // Check if buffer has been closed/freed
  if(closed) {
    return 0;
  }
  //we need minimum 1 for scalars
  if(_lenInBytes == 0) {
   if(_dataType == DataType::UNKNOWN) {
     THROW_EXCEPTION("DataBuffer getLenInBytes: dataType is UNKNOWN !");
   }
    return DataTypeUtils::sizeOfElement(_dataType);
  }
  return _lenInBytes;
}
size_t DataBuffer::getNumElements()   {
  return _lenInBytes / DataTypeUtils::sizeOfElement(getDataType());
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::allocatePrimary() {
  // Fast path: if primary buffer already exists, no mutation — skip frozen guard.
  if (_primaryBuffer != nullptr) return;
  // Allocating HOST (primary) memory does NOT change the device pointer (_specialBuffer)
  // that frozen DSP plans bake into GPU kernel arguments / CUDA graph replay handles.
  // Only block mutations that change _specialBuffer (the GPU address).
  // This allows safe host-side readback of frozen GPU-only buffers (e.g. when a
  // different plan shares a weight buffer that was frozen by another plan).
#if defined(SD_GCC_FUNCTRACE)
  // DataBufferLifecycleTracker already captures allocations for leak detection
  if(allocationStackTracePrimary != nullptr) {
    delete allocationStackTracePrimary;
    allocationStackTracePrimary = nullptr;
  }
#endif
  {
    auto deviceId = AffinityManager::currentDeviceId();
    // check if this allocation won't bring us above limit
    if (_workspace == nullptr) {
      // Proactive soft limit check: reject early if system RAM usage exceeds threshold.
      // This prevents cumulative exhaustion from many small allocations (e.g. DSP warmup)
      // that individually succeed but collectively push the system into swap/OOM.
      if (!memory::MemoryCounter::getInstance().validateSoftLimit(getLenInBytes())) {
        std::string __alloc_msg = std::string("Allocation would breach CPU soft memory limit") +
            "; Limit bytes: [" + std::to_string(memory::MemoryCounter::getInstance().allocatedGroup(memory::MemoryType::HOST)) +
            "]; Requested bytes: [" + std::to_string(getLenInBytes()) + "]";
        THROW_EXCEPTION(__alloc_msg.c_str());
      }

      if (sd::env_isCPU()) {
        // on cpu backend we validate against device 0 for now
        if (!memory::MemoryCounter::getInstance().validate(getLenInBytes())) {
          std::string __alloc_msg = std::string("Requested amount exceeds HOST device limits") +
              "; Limit bytes: [" + std::to_string(memory::MemoryCounter::getInstance().deviceLimit(deviceId)) +
              "]; Requested bytes: [" + std::to_string(getLenInBytes()) + "]";
          THROW_EXCEPTION(__alloc_msg.c_str());
        }
      } else {
        // in heterogenuous mode we validate against device group
        if (!memory::MemoryCounter::getInstance().validateGroup(memory::MemoryType::HOST, getLenInBytes())) {
          std::string __alloc_msg = std::string("Requested amount exceeds HOST group limits") +
              "; Limit bytes: [" + std::to_string(memory::MemoryCounter::getInstance().groupLimit(memory::MemoryType::HOST)) +
              "]; Requested bytes: [" + std::to_string(getLenInBytes()) + "]";
          THROW_EXCEPTION(__alloc_msg.c_str());
        }
      }
    }



    // Add padding for non-workspace heap allocations. C++ ops can overrun output
    // buffers by a few bytes, corrupting adjacent glibc malloc chunk headers.
    // Workspace allocations use bump allocation where overruns are harmless.
    size_t allocSize = getLenInBytes() + (_workspace == nullptr ? static_cast<size_t>(HOST_ALLOC_PADDING) : 0);
    ALLOCATE(_primaryBuffer, _workspace, allocSize, int8_t);
    _isOwnerPrimary = true;
    _primaryAllocBytes = allocSize;

    // Write canary values at end of padding to detect overruns
    if (_workspace == nullptr && _primaryBuffer != nullptr) {
      uint64_t* canary = reinterpret_cast<uint64_t*>(
          static_cast<int8_t*>(_primaryBuffer) + getLenInBytes());
      for (size_t i = 0; i < (static_cast<size_t>(HOST_ALLOC_PADDING) / sizeof(uint64_t)); i++) {
        canary[i] = sd::CanaryConstants::DATA_BUFFER_CANARY;
      }
    }

    // count in towards current deviceId if we're not in workspace mode
    if (_workspace == nullptr) {
      if (sd::env_isCPU())  // we don't want this counter to be added to CUDA device
        memory::MemoryCounter::getInstance().countIn(deviceId, getLenInBytes());

      memory::MemoryCounter::getInstance().countIn(memory::MemoryType::HOST, getLenInBytes());
    }

#if defined(SD_GCC_FUNCTRACE)
    // Record allocation in lifecycle tracker
    array::DataBufferLifecycleTracker::getInstance().recordAllocation(
        _primaryBuffer, getLenInBytes(), getDataType(),
        array::BufferType::PRIMARY, this, _workspace != nullptr);
#endif
  }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setAllocFlags(const bool isOwnerPrimary, const bool isOwnerSpecial) {
  _isOwnerPrimary = isOwnerPrimary;
  _isOwnerSpecial = isOwnerSpecial;
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::deletePrimary() {
#if defined(SD_GCC_FUNCTRACE)
  printPrimaryAllocationStackTraces();

#endif
  if (_isOwnerPrimary && _primaryBuffer != nullptr) {
    // Check canary values before freeing to detect buffer overruns.
    // allocatePrimary() writes 0xDEADBEEFCAFEBABE in the HOST_ALLOC_PADDING region
    // after the data. If any canary is corrupted, a C++ op wrote past this buffer.
    // FIX: Skip canary check for zero-length buffers - the canary would be at offset 0
    // (start of buffer), and any write to the buffer would corrupt it.
    bool canaryCorrupted = false;
    if (_workspace == nullptr && _lenInBytes > 0 && _primaryAllocBytes > 0 && _primaryAllocBytes > _lenInBytes) {
      auto canary = reinterpret_cast<uint64_t*>(
          static_cast<int8_t*>(_primaryBuffer) + _lenInBytes);
      size_t paddingBytes = _primaryAllocBytes - _lenInBytes;
      size_t canaryCount = paddingBytes / sizeof(uint64_t);
      size_t checkCount = (canaryCount > 16) ? 16 : canaryCount;  // check first 128 bytes
      for (size_t i = 0; i < checkCount; i++) {
        if (canary[i] != sd::CanaryConstants::DATA_BUFFER_CANARY) {
          canaryCorrupted = true;
          if (sd::Environment::getInstance().isDebug()) {
            fprintf(stderr, "\n!!! CANARY CORRUPTED in deletePrimary — LEAKING BUFFER TO PREVENT CRASH !!!\n");
            fprintf(stderr, "  buffer=%p, lenInBytes=%zu, allocBytes=%zu, dtype=%d\n",
                    _primaryBuffer, _lenInBytes, _primaryAllocBytes, static_cast<int>(_dataType));
            fprintf(stderr, "  First corrupted canary at offset %zu (byte offset %zu from data end)\n",
                    i, i * sizeof(uint64_t));
            fprintf(stderr, "  Canary values: ");
            for (size_t j = 0; j < checkCount && j < 8; j++) {
              fprintf(stderr, "%016lx ", static_cast<unsigned long>(canary[j]));
            }
            fprintf(stderr, "\n");
            fflush(stderr);
          }
          break;
        }
      }
    }

    if (canaryCorrupted) {
      // Buffer overrun detected: the op that used this buffer wrote past its end,
      // corrupting the canary region and potentially the adjacent malloc chunk header.
      // Calling free() on this buffer would crash with "double free or corruption (!prev)"
      // because glibc validates chunk metadata during free(). Instead, we intentionally
      // leak the buffer — a small memory leak is far better than a process crash.
      // The stderr message above identifies which buffer was corrupted for debugging.
      _primaryBuffer = nullptr;
      _isOwnerPrimary = false;
    } else if(sd::env_isDeletePrimary()) {
#if defined(SD_GCC_FUNCTRACE)
      // Record deallocation before releasing memory
      array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
          _primaryBuffer, array::BufferType::PRIMARY);
#endif
      auto p = reinterpret_cast<int8_t*>(_primaryBuffer);
      RELEASE(p, _workspace);
      // Always nullify pointer and clear ownership flag, regardless of isDeletePrimary
      _primaryBuffer = nullptr;
      _isOwnerPrimary = false;
    } else {
      _primaryBuffer = nullptr;
      _isOwnerPrimary = false;
    }

    // count out towards DataBuffer device, only if we're not in workspace
    if (_workspace == nullptr) {
      if (sd::env_isCPU()) memory::MemoryCounter::getInstance().countOut(_deviceId, getLenInBytes());

      memory::MemoryCounter::getInstance().countOut(memory::MemoryType::HOST, getLenInBytes());
    }
  }
}

void DataBuffer::printPrimaryAllocationStackTraces() {
#if defined(SD_GCC_FUNCTRACE)

#endif

}

////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteBuffers() {
  if(isConstant || closed) {
    return;
  }

  // NOTE: intentionally no throwIfFrozen() here. The destructor calls
  // deleteBuffers() directly, and throwing from a destructor would call
  // std::terminate. The frozen guard is enforced at the public close()
  // entry point and at every mutator that replaces pointers
  // (setPrimaryBuffer, setSpecialBuffer, replaceSpecialBuffer, expand,
  // migrate, freeGpuOnly, freeGpuOnStream). If a buffer reaches the
  // destructor while still frozen, that's a separate lifetime bug that
  // should be caught by the release-path checks in NativeDynamicShapePlan.
  std::lock_guard<std::mutex> lock(_deleteMutex);
  deletePrimary();
  deleteSpecial();

  // Clean up stack traces to prevent memory leak
#if defined(SD_GCC_FUNCTRACE)
  if(allocationStackTracePrimary != nullptr) {
    delete allocationStackTracePrimary;
    allocationStackTracePrimary = nullptr;
  }
  if(allocationStackTraceSpecial != nullptr) {
    delete allocationStackTraceSpecial;
    allocationStackTraceSpecial = nullptr;
  }
  if(creationStackTrace != nullptr) {
    delete creationStackTrace;
    creationStackTrace = nullptr;
  }
#endif

  closed = true;
  _lenInBytes = 0;
}

////////////////////////////////////////////////////////////////////////
DataBuffer::~DataBuffer() {
  // Clear magic number to detect use-after-free
  // If anyone tries to use this buffer after destruction, validateIntegrity() will catch it
  _magicNumber = MAGIC_DESTROYED;
  deleteBuffers();
}


void DataBuffer::setPrimaryBuffer(void* buffer, size_t length) {
  throwIfFrozen("setPrimaryBuffer");
  std::lock_guard<std::mutex> lock(_deleteMutex);
#if defined(SD_GCC_FUNCTRACE)
  // DataBufferLifecycleTracker already captures allocations for leak detection
  if(allocationStackTracePrimary != nullptr) {
    delete allocationStackTracePrimary;
    allocationStackTracePrimary = nullptr;
  }
#endif
  _primaryBuffer = buffer;
  _isOwnerPrimary = false;  // External buffer - caller manages lifetime (JavaCPP Pointer, workspace, etc.)
  _lenInBytes = length * DataTypeUtils::sizeOf(_dataType);
  _primaryAllocBytes = _lenInBytes;
}

void DataBuffer::setSpecialBuffer(void* buffer, size_t length) {
  throwIfFrozen("setSpecialBuffer");
  std::lock_guard<std::mutex> lock(_deleteMutex);
#if defined(SD_GCC_FUNCTRACE)
  // DataBufferLifecycleTracker already captures allocations for leak detection
  if(allocationStackTraceSpecial != nullptr) {
    delete allocationStackTraceSpecial;
    allocationStackTraceSpecial = nullptr;
  }
#endif
  this->setSpecial(buffer, false);
  _lenInBytes = length * DataTypeUtils::sizeOf(_dataType);
  _specialAllocBytes = _lenInBytes;
}

void DataBuffer::setDataType(DataType dataType) {
  if(dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION("DataBuffer setDataType: dataType is UNKNOWN !");
  }
  _dataType = dataType;
}

void DataBuffer::printAllocationTrace() {
  if(closed) {
    printf("DataBuffer::printAllocationTrace() - buffer is closed\n");
    fflush(stdout);
  }
#if defined(SD_GCC_FUNCTRACE)
  //print whether each stack trace is null or not:
  Printer p;
  if(allocationStackTracePrimary != nullptr) {
    p.print(*allocationStackTracePrimary);
  }
  if(allocationStackTraceSpecial != nullptr) {
    p.print(*allocationStackTraceSpecial);
  }
  if(creationStackTrace != nullptr) {
    p.print(*creationStackTrace);
  }
#endif
}

std::string DataBuffer::getCreationTraceAsString() const {
#if defined(SD_GCC_FUNCTRACE)
  if (creationStackTrace == nullptr || creationStackTrace->size() == 0) {
    return "";
  }

  std::ostringstream oss;
  backward::TraceResolver resolver;
  resolver.load_stacktrace(*creationStackTrace);

  for (size_t i = 0; i < creationStackTrace->size(); ++i) {
    const backward::ResolvedTrace &trace = resolver.resolve((*creationStackTrace)[i]);

    // Format: #frame function_name at source_file:line
    oss << "#" << i << " ";

    if (!trace.object_function.empty()) {
      oss << trace.object_function;
    } else {
      oss << "???";
    }

    if (!trace.source.filename.empty()) {
      oss << " at " << trace.source.filename;
      if (trace.source.line > 0) {
        oss << ":" << trace.source.line;
      }
    }

    oss << "\n";
  }

  return oss.str();
#else
  return "";
#endif
}

int DataBuffer::deviceId() const { return _deviceId.load(); }

void DataBuffer::close() {
  throwIfFrozen("close");
  this->deleteBuffers();
}

void DataBuffer::setDeviceId(int deviceId) { _deviceId = deviceId; }

void DataBuffer::resetCounters() {
  _writePrimary.store(0);
  _writeSpecial.store(0);
  _readPrimary.store(0);
  _readSpecial.store(0);
}
}  // namespace sd
