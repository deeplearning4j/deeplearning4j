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
#include "../DataBuffer.h"

namespace sd {
void DataBuffer::expand(const uint64_t size) {
  if (size > _lenInBytes) {
    // allocate new buffer
    int8_t* newBuffer = nullptr;
    int8_t* newSpecialBuffer = nullptr;
    ALLOCATE_SPECIAL(newSpecialBuffer, _workspace, size, int8_t);

    // copy data from existing buffer
    if (_primaryBuffer != nullptr) {
      // there's non-zero chance that primary buffer doesn't exist yet
      ALLOCATE(newBuffer, _workspace, size, int8_t);
      std::memcpy(newBuffer, _primaryBuffer, _lenInBytes);

      if (_isOwnerPrimary) {
        auto ipb = reinterpret_cast<int8_t*>(_primaryBuffer);
        RELEASE(ipb, _workspace);
      }

      _primaryBuffer = newBuffer;
      _isOwnerPrimary = true;
    }

    cudaMemcpy(newSpecialBuffer, _specialBuffer, _lenInBytes, cudaMemcpyDeviceToDevice);

    if (_isOwnerSpecial) {
      auto isb = reinterpret_cast<int8_t*>(_specialBuffer);
      RELEASE_SPECIAL(isb, _workspace);
    }

    _specialBuffer = newSpecialBuffer;
    _lenInBytes = size;
    _isOwnerSpecial = true;
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
  T *type = reinterpret_cast<T*>(_primaryBuffer);
  return reinterpret_cast<void *>(type + offset);
}
template <typename T>
void* DataBuffer::specialAtOffset(const LongType offset) {
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

void DataBuffer::showCounters(const char* msg1, const char* msg2) {
  sd_debug("%s %s || primary %p special %p :: wP: %d wS: %d rP: %d rS: %d\n", msg1, msg2, _primaryBuffer,
           _specialBuffer, (int)_writePrimary.load(), (int)_writeSpecial.load(), (int)_readPrimary.load(),
           (int)_readSpecial.load());
}
////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateSpecial() {
  if (_specialBuffer != nullptr) {
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
  if (isPrimaryActual() && !forceSync) {
    return;
  }

  allocatePrimary();

  auto res = cudaStreamSynchronize(*context->getCudaStream());
  if (res != 0)  {
    std::string errorMessage;
    errorMessage += "DataBuffer::syncToPrimary: cudaStreamSynchronize failed: ";
    errorMessage += std::to_string(getLenInBytes());
    errorMessage += cudaGetErrorString(res);
    errorMessage += "Special buffer is nullptr : ";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  res = cudaMemcpy(_primaryBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToHost);
  if (res != 0) {
        std::string errorMessage;
        errorMessage += "DataBuffer::syncToPrimary: cudaMemcpy failed: ";
        errorMessage += std::to_string(getLenInBytes());
        errorMessage += cudaGetErrorString(res);
        errorMessage += "Special buffer is nullptr : ";
        errorMessage += std::to_string(_specialBuffer == nullptr);
        THROW_EXCEPTION(errorMessage.c_str());
  }

  readPrimary();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToSpecial(const bool forceSync) {
  // in this case there's nothing to do here
  if (_primaryBuffer == nullptr) return;

  if (isSpecialActual() && !forceSync) {
    return;
  }

  allocateSpecial();

  auto res = cudaMemcpy(_specialBuffer, _primaryBuffer, getLenInBytes(), cudaMemcpyHostToDevice);
  if (res != 0) {
    std::string errorMessage;
    errorMessage += "Failed to copy dataBuffer::syncToSpecial: ";
    errorMessage += std::to_string(getLenInBytes());
    errorMessage += cudaGetErrorString(res);
    THROW_EXCEPTION(errorMessage.c_str());

  }

  readSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteSpecial() {
  if (_isOwnerSpecial && _specialBuffer != nullptr && getLenInBytes() != 0) {
    auto p = reinterpret_cast<int8_t*>(_specialBuffer);
    RELEASE_SPECIAL(p, _workspace);
    _specialBuffer = nullptr;
    _isOwnerSpecial = false;

    // count out towards DataBuffer device, only if we're not in workspace
    if (_workspace == nullptr) {
      sd::memory::MemoryCounter::getInstance().countOut(_deviceId, getLenInBytes());
      sd::memory::MemoryCounter::getInstance().countOut(sd::memory::MemoryType::DEVICE, getLenInBytes());
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

  if (other.isPrimaryActual()) {
    auto res = cudaMemcpy(
        static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
        static_cast<const int8_t*>(other._primaryBuffer) + offsetOther * DataTypeUtils::sizeOfElement(other._dataType),
        sizeToCopyinBytes, cudaMemcpyHostToDevice);
    if (res != 0)
      throw cuda_exception::build("DataBuffer::copyBufferFrom: cudaMemcpy_cudaMemcpyHostToDevice failed!", res);
    other.readPrimary();
  } else {
    auto res = cudaMemcpy(
        static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
        static_cast<const int8_t*>(other._specialBuffer) + offsetOther * DataTypeUtils::sizeOfElement(other._dataType),
        sizeToCopyinBytes, cudaMemcpyDeviceToDevice);
    if (res != 0)
      throw cuda_exception::build("DataBuffer::copyBufferFrom: cudaMemcpy_cudaMemcpyDeviceToDevice failed!", res);
    other.readSpecial();
  }

  writeSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFromHost(const void* hostBuffer, size_t sizeToCopyinBytes, const sd::LongType offsetThis,
                                    const sd::LongType offsetHostBuffer) {  // copies only to special buffer

  if (hostBuffer == nullptr) return;

  if (sizeToCopyinBytes == 0) sizeToCopyinBytes = getLenInBytes();
  if (sizeToCopyinBytes == 0) return;

  auto res =
      cudaMemcpy(static_cast<int8_t*>(_specialBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType),
                 static_cast<const int8_t*>(hostBuffer) + offsetHostBuffer * DataTypeUtils::sizeOfElement(_dataType),
                 sizeToCopyinBytes, cudaMemcpyHostToDevice);
  if (res != 0)
    throw cuda_exception::build("DataBuffer::copyBufferFromHost: cudaMemcpy_cudaMemcpyHostToDevice failed!", res);

  writeSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setSpecial(void* special, const bool isOwnerSpecial) {
  deleteSpecial();
  _specialBuffer = special;
  _isOwnerSpecial = isOwnerSpecial;
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
  cudaMemsetAsync(special(), 0, getLenInBytes(), *LaunchContext::defaultContext()->getCudaStream());
  auto res = cudaStreamSynchronize(*LaunchContext::defaultContext()->getCudaStream());
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

  int res = 0;
  if (src->isSpecialActual()) {
    res = cudaMemcpyAsync(dst->specialAtOffset<T>(dstOffset), src->specialAtOffset<T>(startingOffset), src->getLenInBytes(), cudaMemcpyDeviceToDevice,
                          *LaunchContext::defaultContext()->getCudaStream());
  } else if (src->isPrimaryActual()) {
    res = cudaMemcpyAsync(dst->specialAtOffset<T>(dstOffset), src->specialAtOffset<T>(startingOffset), src->getLenInBytes(), cudaMemcpyHostToDevice,
                          *LaunchContext::defaultContext()->getCudaStream());
  }

  if (res != 0) throw cuda_exception::build("DataBuffer::memcpy: cudaMemcpyAsync failed!", res);

  res = cudaStreamSynchronize(*LaunchContext::defaultContext()->getCudaStream());
  if (res != 0) throw cuda_exception::build("DataBuffer::memcpy: streamSync failed!", res);

  dst->writeSpecial();
}

void DataBuffer::memcpy(DataBuffer* dst, DataBuffer* src,
                        sd::LongType startingOffset, sd::LongType dstOffset) {
  BUILD_SINGLE_TEMPLATE(memcpyWithT,(dst, src, startingOffset, dstOffset),
                        SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::migrate() {
  memory::Workspace* newWorkspace = nullptr;
  void* newBuffer;
  ALLOCATE_SPECIAL(newBuffer, newWorkspace, getLenInBytes(), int8_t);
  auto res = cudaMemcpy(newBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToDevice);
  if (res != 0) throw cuda_exception::build("DataBuffer::migrate: cudaMemcpyAsync failed!", res);

  if (_isOwnerSpecial) {
    // now we're releasing original buffer
    RELEASE_SPECIAL(_specialBuffer, _workspace);
  }

  _isOwnerSpecial = true;
  _specialBuffer = newBuffer;
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
