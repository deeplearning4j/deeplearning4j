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
#include <types/types.h>
#include <system/type_boilerplate.h>


namespace sd {
void DataBuffer::expand(const uint64_t size) {
  if (static_cast<LongType>(size) > _lenInBytes) {
    // allocate new buffer
    int8_t* newBuffer = nullptr;
    ALLOCATE(newBuffer, _workspace, size, int8_t);

    // copy data from existing buffer
    std::memcpy(newBuffer, _primaryBuffer, _lenInBytes);

    if (_isOwnerPrimary) {
      RELEASE(reinterpret_cast<int8_t*>(_primaryBuffer), _workspace);
    }

    _primaryBuffer = newBuffer;
    _lenInBytes = size;
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
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT void DataBuffer::printHostBufferContent,(void* buffer, sd::LongType offset, sd::LongType length),SD_COMMON_TYPES);




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

#if defined(__CUDABLAS__)
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
void memcpyWithT(DataBuffer* dst, DataBuffer* src, sd::LongType startingOffset, sd::LongType dstOffset) {
  if(src->getLenInBytes() != dst->getLenInBytes()) {
    THROW_EXCEPTION("DataBuffer::memcpy: source and destination buffers have different length in bytes");
  }

  std::memcpy(dst->primaryAtOffset<T>(dstOffset), src->primaryAtOffset<T>(startingOffset), src->getLenInBytes());
  dst->readPrimary();
}

void DataBuffer::memcpy(DataBuffer* dst, DataBuffer* src,
                        sd::LongType startingOffset, sd::LongType dstOffset) {
  BUILD_SINGLE_TEMPLATE(memcpyWithT,(dst, src, startingOffset, dstOffset),
                        SD_COMMON_TYPES);

  dst->readPrimary();
}




////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteSpecial() {}

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
  result._primaryBuffer = _primaryBuffer;
  result._specialBuffer = _specialBuffer;
  result._isOwnerPrimary = _isOwnerPrimary;
  result._isOwnerSpecial = _isOwnerSpecial;
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
  BUILD_SINGLE_SELECTOR(xType, _printHostBuffer,(this,offset),SD_COMMON_TYPES_ALL);

}


void DataBuffer::showCounters(const char* msg1, const char* msg2) {

}
template <typename T>
void* DataBuffer::primaryAtOffset(const LongType offset) {
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
