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
//
#include <array/DataTypeUtils.h>
#include <array/InteropDataBuffer.h>
#include <execution/AffinityManager.h>
#include <helpers/logger.h>

namespace sd {
InteropDataBuffer::InteropDataBuffer(InteropDataBuffer* dataBuffer, uint64_t length) {
 if(dataBuffer == nullptr) {
        THROW_EXCEPTION("InteropDataBuffer::InteropDataBuffer(InteropDataBuffer& dataBuffer, uint64_t length, uint64_t offset) - dataBuffer is nullptr");
 }
  if(dataBuffer->_dataBuffer->getDataType() == DataType::UNKNOWN)
    THROW_EXCEPTION("InteropDataBuffer::InteropDataBuffer(InteropDataBuffer& dataBuffer, uint64_t length, uint64_t offset) - dataBuffer has unknown data type");
  _dataBuffer = dataBuffer->dataBuffer();
  _dataType = dataBuffer->_dataType;

}

InteropDataBuffer::InteropDataBuffer(DataBuffer * databuffer) {
  _dataBuffer = databuffer;
  _dataType = databuffer->getDataType();
  if(_dataType == DataType::UNKNOWN) {
    THROW_EXCEPTION(
        "InteropDataBuffer::InteropDataBuffer(size_t lenInBytes, DataType dtype, bool allocateBoth) - data type is unknown");
  }
}

InteropDataBuffer::InteropDataBuffer(size_t lenInBytes, DataType dtype, bool allocateBoth) {
  if(dtype == DataType::UNKNOWN) {
    THROW_EXCEPTION(
        "InteropDataBuffer::InteropDataBuffer(size_t lenInBytes, DataType dtype, bool allocateBoth) - data type is unknown");
  }

  if (lenInBytes == 0) {
    _dataBuffer = nullptr;
    this->_dataType = dtype;

  } else {
    //note this should be size in bytes hence why we multiply the number of elements by the size of the data type
    _dataBuffer = new DataBuffer(lenInBytes, dtype, nullptr, allocateBoth);
    this->_dataType = dtype;
  }
}


void InteropDataBuffer::printDbAllocationTrace() {
  if(_dataBuffer == nullptr)
    return;
  _dataBuffer->printAllocationTrace();
}

void InteropDataBuffer::markOwner(bool owner) {
  this->owner = owner;
  this->_dataBuffer->_isOwnerPrimary = owner;
  this->_dataBuffer->_isOwnerSpecial = owner;
}

DataBuffer * InteropDataBuffer::getDataBuffer() const { return _dataBuffer; }

DataBuffer * InteropDataBuffer::dataBuffer() {
  if(_dataBuffer == nullptr || _dataBuffer == nullptr) {
    return nullptr;
  }
  return _dataBuffer;
}



void* InteropDataBuffer::primary() const {
  if(_dataBuffer->primary() == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<int8_t*>(_dataBuffer->primary());
}

void* InteropDataBuffer::special() const {
  if(_dataBuffer == nullptr)
    return nullptr;
  if(_dataBuffer->special() == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<int8_t*>(_dataBuffer->special());
}

void InteropDataBuffer::setPrimary(void* ptr, size_t length) {
  if(_dataBuffer == nullptr)
    THROW_EXCEPTION("InteropDataBuffer::setPrimary() - _dataBuffer is nullptr");
  _dataBuffer->setPrimaryBuffer(ptr, length);
}

void InteropDataBuffer::setSpecial(void* ptr, size_t length) {
  if(_dataBuffer == nullptr)
    THROW_EXCEPTION("InteropDataBuffer::setSpecial() - _dataBuffer is nullptr");
  _dataBuffer->setSpecialBuffer(ptr, length);
}



int InteropDataBuffer::deviceId() const { return _dataBuffer->deviceId(); }

int InteropDataBuffer::useCount() const {
  return 1;
}

void InteropDataBuffer::registerSpecialUse(const std::vector<const InteropDataBuffer*>& writeList,
                                           const std::vector<const InteropDataBuffer*>& readList) {
  for (const auto& v : writeList) {
    if (v == nullptr) continue;

    v->getDataBuffer()->writeSpecial();
  }
}

void InteropDataBuffer::prepareSpecialUse(const std::vector<const InteropDataBuffer*>& writeList,
                                          const std::vector<const InteropDataBuffer*>& readList,
                                          bool synchronizeWritables) {
  auto currentDeviceId = AffinityManager::currentDeviceId();
  for (const auto& v : readList) {
    if (v == nullptr) continue;

    if (v->getDataBuffer()->deviceId() != currentDeviceId) v->getDataBuffer()->migrate();

    v->getDataBuffer()->syncToSpecial();
  }

  // we don't tick write list, only ensure the same device affinity
  for (const auto& v : writeList) {
    if (v == nullptr) continue;

    // special case for legacy ops - views can be updated on host side, thus original array can be not updated
    if (!v->getDataBuffer()->isSpecialActual()) v->getDataBuffer()->syncToSpecial();

    if (v->getDataBuffer()->deviceId() != currentDeviceId) v->getDataBuffer()->migrate();
  }
}

void InteropDataBuffer::registerPrimaryUse(const std::vector<const InteropDataBuffer*>& writeList,
                                           const std::vector<const InteropDataBuffer*>& readList) {

}

void InteropDataBuffer::preparePrimaryUse(const std::vector<const InteropDataBuffer*>& writeList,
                                          const std::vector<const InteropDataBuffer*>& readList,
                                          bool synchronizeWritables) {

}

void InteropDataBuffer::expand(size_t newlength) {
  _dataBuffer->expand(newlength * DataTypeUtils::sizeOf(_dataBuffer->getDataType()));
}

void InteropDataBuffer::setDeviceId(int deviceId) { _dataBuffer->setDeviceId(deviceId); }
}  // namespace sd
