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
  if (size > _lenInBytes) {
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

void DataBuffer::printSpecialAllocationTraces() {
  //no op on purpose
}


////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateBuffers(const bool allocBoth) {  // always allocate primary buffer only (cpu case)
  allocatePrimary();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFrom(const DataBuffer& other, size_t sizeToCopyinBytes, const sd::LongType offsetThis,
                                const sd::LongType offsetOther) {
  if (sizeToCopyinBytes == 0) sizeToCopyinBytes = other.getLenInBytes();
  if (sizeToCopyinBytes == 0) return;

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

void DataBuffer::memcpyPointer(std::shared_ptr<DataBuffer>   dst, std::shared_ptr<DataBuffer>  src) {
  if (src->_lenInBytes > dst->_lenInBytes) {
    std::string errorMessage;
    errorMessage = "DataBuffer::memcpy: Source data buffer is larger than destination";
    errorMessage += std::to_string(src->_lenInBytes);
    errorMessage += " > ";
    errorMessage += std::to_string(dst->_lenInBytes);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  std::memcpy(dst->_primaryBuffer, src->_primaryBuffer, src->_lenInBytes);
  dst->readPrimary();
}

void DataBuffer::memcpy(const DataBuffer dst, const DataBuffer src) {
  if (src._lenInBytes > dst._lenInBytes) {
    std::string errorMessage;
    errorMessage = "DataBuffer::memcpy: Source data buffer is larger than destination";
    errorMessage += std::to_string(src._lenInBytes);
    errorMessage += " > ";
    errorMessage += std::to_string(dst._lenInBytes);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  std::memcpy(dst._primaryBuffer, src._primaryBuffer, src._lenInBytes);
  dst.readPrimary();
}

void DataBuffer::memcpy(const DataBuffer *dst, const DataBuffer *src) {
  if (src->_lenInBytes > dst->_lenInBytes) {
    std::string errorMessage;
    errorMessage = "DataBuffer::memcpy: Source data buffer is larger than destination";
    errorMessage += std::to_string(src->_lenInBytes);
    errorMessage += " > ";
    errorMessage += std::to_string(dst->_lenInBytes);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  std::memcpy(dst->_primaryBuffer, src->_primaryBuffer, src->_lenInBytes);
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
void _printHostBuffer(DataBuffer *buffer) {
  sd::LongType len = buffer->getNumElements();
  auto buff = buffer->template primaryAsT<T>();
  sd_printf("Host buffer: address %p ",buffer->primary());
  for(int i = 0; i < len; i++) {
    sd_printf("%f ",(double) buff[i]);
  }

  sd_printf("\n",0);
}




void DataBuffer::printHostDevice() {
  auto xType = getDataType();
  BUILD_SINGLE_SELECTOR(xType, _printHostBuffer,(this),SD_COMMON_TYPES_ALL);

}


void DataBuffer::showCounters(const char* msg1, const char* msg2) {

}

}  // namespace sd
