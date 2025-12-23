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
#include <exceptions/allocation_exception.h>
#include <execution/AffinityManager.h>
#include <helpers/logger.h>
#include <memory/MemoryCounter.h>
#include <sstream>

#if defined(SD_GCC_FUNCTRACE)
#include <array/DataBufferLifecycleTracker.h>
#endif

namespace sd {
///// IMPLEMENTATION OF COMMON METHODS /////

////////////////////////////////////////////////////////////////////////
// default constructor
DataBuffer::DataBuffer() {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
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
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    printf("DataBuffer::DataBuffer(const DataBuffer& other) copy constructor\n");
    fflush(stdout);
  }
  _lenInBytes = other._lenInBytes;
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
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
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

  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
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

  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
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

  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
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

  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    printf("DataBuffer::DataBuffer(DataBuffer&& other) move constructor\n");
    fflush(stdout);
  }
  _primaryBuffer = other._primaryBuffer;
  _specialBuffer = other._specialBuffer;
  _lenInBytes = other._lenInBytes;
  _dataType = other._dataType;
  _workspace = other._workspace;
  _isOwnerPrimary = other._isOwnerPrimary;
  _isOwnerSpecial = other._isOwnerSpecial;
  _deviceId.store(other._deviceId);

  copyCounters(other);
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
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    printf("DataBuffer::operator=(const DataBuffer& other) assignment operator\n");
    fflush(stdout);
  }
  if (this == &other) return *this;

  deleteBuffers();

  _lenInBytes = other._lenInBytes;
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

  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    printf("DataBuffer::operator=(DataBuffer&& other) move assignment operator\n");
    fflush(stdout);
  }
  if (this == &other) return *this;

  deleteBuffers();

  _primaryBuffer = other._primaryBuffer;
  _specialBuffer = other._specialBuffer;
  _lenInBytes = other._lenInBytes;
  _dataType = other._dataType;
  _workspace = other._workspace;
  _isOwnerPrimary = other._isOwnerPrimary;
  _isOwnerSpecial = other._isOwnerSpecial;

  copyCounters(other);

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
#if defined(SD_GCC_FUNCTRACE)
  // DataBufferLifecycleTracker already captures allocations for leak detection
  if(allocationStackTracePrimary != nullptr) {
    delete allocationStackTracePrimary;
    allocationStackTracePrimary = nullptr;
  }
#endif
  if (_primaryBuffer == nullptr) {
    auto deviceId = AffinityManager::currentDeviceId();
    // check if this allocation won't bring us above limit
    if (_workspace == nullptr) {
      if (Environment::getInstance().isCPU()) {
        // on cpu backend we validate against device 0 for now
        if (!memory::MemoryCounter::getInstance().validate(getLenInBytes()))
          THROW_EXCEPTION(allocation_exception::build("Requested amount exceeds HOST device limits",
                                            memory::MemoryCounter::getInstance().deviceLimit(deviceId),
                                            getLenInBytes()).what());
      } else {
        // in heterogenuous mode we validate against device group
        if (!memory::MemoryCounter::getInstance().validateGroup(memory::MemoryType::HOST, getLenInBytes()))
          THROW_EXCEPTION(allocation_exception::build(
              "Requested amount exceeds HOST group limits",
              memory::MemoryCounter::getInstance().groupLimit(memory::MemoryType::HOST), getLenInBytes()).what());
      }
    }



    ALLOCATE(_primaryBuffer, _workspace, getLenInBytes(), int8_t);
    _isOwnerPrimary = true;

    // count in towards current deviceId if we're not in workspace mode
    if (_workspace == nullptr) {
      if (Environment::getInstance().isCPU())  // we don't want this counter to be added to CUDA device
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
    auto p = reinterpret_cast<int8_t*>(_primaryBuffer);

    if(Environment::getInstance().isDeletePrimary()) {
#if defined(SD_GCC_FUNCTRACE)
      // Record deallocation before releasing memory
      array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
          _primaryBuffer, array::BufferType::PRIMARY);
#endif
      RELEASE(p, _workspace);
      _primaryBuffer = nullptr;
    }

    _isOwnerPrimary = false;

    // count out towards DataBuffer device, only if we're not in workspace
    if (_workspace == nullptr) {
      if (Environment::getInstance().isCPU()) memory::MemoryCounter::getInstance().countOut(_deviceId, getLenInBytes());

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
  _magicNumber = 0xDEADBEEF;
  deleteBuffers();
}


void DataBuffer::setPrimaryBuffer(void* buffer, size_t length) {
  std::lock_guard<std::mutex> lock(_deleteMutex);
#if defined(SD_GCC_FUNCTRACE)
  // DataBufferLifecycleTracker already captures allocations for leak detection
  if(allocationStackTracePrimary != nullptr) {
    delete allocationStackTracePrimary;
    allocationStackTracePrimary = nullptr;
  }
#endif
  _primaryBuffer = buffer;
  _isOwnerPrimary = false;
  _lenInBytes = length * DataTypeUtils::sizeOf(_dataType);
}

void DataBuffer::setSpecialBuffer(void* buffer, size_t length) {
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

void DataBuffer::close() { this->deleteBuffers(); }

void DataBuffer::setDeviceId(int deviceId) { _deviceId = deviceId; }
}  // namespace sd
