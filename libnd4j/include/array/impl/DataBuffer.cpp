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
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    creationStackTrace = new StackTrace();
    creationStackTrace->load_here();
  }

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
  allocationStackTracePrimary = other.allocationStackTracePrimary;
  allocationStackTraceSpecial = other.allocationStackTraceSpecial;
#endif
  _primaryBuffer = other._primaryBuffer;
  _specialBuffer = other._specialBuffer;

#if defined(SD_GCC_FUNCTRACE)
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    creationStackTrace = new StackTrace();
    creationStackTrace->load_here();
  }

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
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    creationStackTrace = new StackTrace();
    creationStackTrace->load_here();
  }

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
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    creationStackTrace = new StackTrace();
    creationStackTrace->load_here();
  }

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
  if (hostBuffer == nullptr)
    THROW_EXCEPTION("DataBuffer constructor: can't be initialized with nullptr host buffer !");
  if (lenInBytes == 0) THROW_EXCEPTION("DataBuffer constructor: can't be initialized with zero length !");

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
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    creationStackTrace = new StackTrace();
    creationStackTrace->load_here();
  }

#endif
}

////////////////////////////////////////////////////////////////////////
DataBuffer::DataBuffer(const size_t lenInBytes, const DataType dataType, memory::Workspace* workspace,
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

  _deviceId = AffinityManager::currentDeviceId();

  setCountersToZero();

  allocateBuffers(allocBoth);
  writeSpecial();

#if defined(SD_GCC_FUNCTRACE)
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    creationStackTrace = new StackTrace();
    creationStackTrace->load_here();
  }

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
#endif
  other._primaryBuffer = other._specialBuffer = nullptr;
  other.setAllocFlags(false, false);
  other._lenInBytes = 0;

#if defined(SD_GCC_FUNCTRACE)
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    creationStackTrace = new StackTrace();
    creationStackTrace->load_here();
  }

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
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    creationStackTrace = new StackTrace();
    creationStackTrace->load_here();
  }

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
  _isOwnerPrimary = false;
  _isOwnerSpecial = false;

  copyCounters(other);

  other._primaryBuffer = other._specialBuffer = nullptr;
  other.setAllocFlags(false, false);
  other._lenInBytes = 0;
#if defined(SD_GCC_FUNCTRACE)
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    creationStackTrace = new StackTrace();
    creationStackTrace->load_here();
  }

#endif
  return *this;
}


void DataBuffer::markConstant(bool reallyConstant) {
  isConstant = reallyConstant;
}
////////////////////////////////////////////////////////////////////////
void* DataBuffer::primary() { return _primaryBuffer; }

////////////////////////////////////////////////////////////////////////
void* DataBuffer::special() { return _specialBuffer; }

////////////////////////////////////////////////////////////////////////
DataType DataBuffer::getDataType() { return _dataType; }

////////////////////////////////////////////////////////////////////////
size_t DataBuffer::getLenInBytes() const {
  //we need minimum 1 for scalars
  if(_lenInBytes == 0)
    return DataTypeUtils::sizeOfElement(_dataType);
  return _lenInBytes;
}
size_t DataBuffer::getNumElements()   {
  return _lenInBytes / DataTypeUtils::sizeOfElement(getDataType());
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::allocatePrimary() {
#if defined(SD_GCC_FUNCTRACE)
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    allocationStackTracePrimary = new StackTrace();
    allocationStackTracePrimary->load_here();
  }

#endif
  if (_primaryBuffer == nullptr) {
    auto deviceId = AffinityManager::currentDeviceId();
    // check if this allocation won't bring us above limit
    if (_workspace == nullptr) {
      if (Environment::getInstance().isCPU()) {
        // on cpu backend we validate against device 0 for now
        if (!memory::MemoryCounter::getInstance().validate(getLenInBytes()))
          throw allocation_exception::build("Requested amount exceeds HOST device limits",
                                            memory::MemoryCounter::getInstance().deviceLimit(deviceId),
                                            getLenInBytes());
      } else {
        // in heterogenuous mode we validate against device group
        if (!memory::MemoryCounter::getInstance().validateGroup(memory::MemoryType::HOST, getLenInBytes()))
          throw allocation_exception::build(
              "Requested amount exceeds HOST group limits",
              memory::MemoryCounter::getInstance().groupLimit(memory::MemoryType::HOST), getLenInBytes());
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
  closed = true;
  _lenInBytes = 0;
}

////////////////////////////////////////////////////////////////////////
DataBuffer::~DataBuffer() { deleteBuffers(); }

void DataBuffer::setPrimaryBuffer(void* buffer, size_t length) {
  std::lock_guard<std::mutex> lock(_deleteMutex);
#if defined(SD_GCC_FUNCTRACE)
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    if(allocationStackTracePrimary != nullptr) {
      delete allocationStackTracePrimary;
      allocationStackTracePrimary = nullptr;
    }
    allocationStackTracePrimary = new StackTrace();
    allocationStackTracePrimary->load_here();
  }
#endif
  _primaryBuffer = buffer;
  _isOwnerPrimary = false;
  _lenInBytes = length * DataTypeUtils::sizeOf(_dataType);
}

void DataBuffer::setSpecialBuffer(void* buffer, size_t length) {
  std::lock_guard<std::mutex> lock(_deleteMutex);
#if defined(SD_GCC_FUNCTRACE)
  if(Environment::getInstance().isFuncTracePrintAllocate()) {
    if(allocationStackTraceSpecial != nullptr) {
      delete allocationStackTraceSpecial;
      allocationStackTraceSpecial = nullptr;
    }
    allocationStackTraceSpecial = new StackTrace();
    allocationStackTraceSpecial->load_here();
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


int DataBuffer::deviceId() const { return _deviceId.load(); }

void DataBuffer::close() { this->deleteBuffers(); }

void DataBuffer::setDeviceId(int deviceId) { _deviceId = deviceId; }
}  // namespace sd
