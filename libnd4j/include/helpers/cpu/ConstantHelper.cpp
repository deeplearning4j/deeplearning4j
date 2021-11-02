/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
//  @author raver119@gmail.com
//

#ifndef __CUDABLAS__
#include <array/PrimaryPointerDeallocator.h>
#include <execution/AffinityManager.h>
#include <helpers/ConstantHelper.h>
#include <loops/type_conversions.h>
#include <system/type_boilerplate.h>
#include <types/types.h>

#include <cstring>

namespace sd {

ConstantHelper::ConstantHelper() {
  int numDevices = getNumberOfDevices();
  _cache.resize(numDevices);
  _counters.resize(numDevices);
  for (int e = 0; e < numDevices; e++) {
    SD_MAP_IMPL<ConstantDescriptor, ConstantHolder *> map;

    _cache[e] = map;
    _counters[e] = 0L;
  }
}

ConstantHelper::~ConstantHelper() {
  for (const auto &v : _cache) {
    for (const auto &c : v) {
      delete c.second;
    }
  }
}

ConstantHelper &ConstantHelper::getInstance() {
  static ConstantHelper instance;
  return instance;
}

void *ConstantHelper::replicatePointer(void *src, size_t numBytes, memory::Workspace *workspace) {
  if (workspace == nullptr) {
    auto deviceId = getCurrentDevice();
    _counters[deviceId] += numBytes;
  }

  int8_t *ptr = nullptr;
  ALLOCATE(ptr, workspace, numBytes, int8_t);

  std::memcpy(ptr, src, numBytes);
  return ptr;
}

int ConstantHelper::getCurrentDevice() { return AffinityManager::currentDeviceId(); }

int ConstantHelper::getNumberOfDevices() { return AffinityManager::numberOfDevices(); }

ConstantDataBuffer *ConstantHelper::constantBuffer(const ConstantDescriptor &descriptor, sd::DataType dataType) {
  const auto deviceId = getCurrentDevice();

  // we're locking away cache modification
  _mutexHolder.lock();

  if (_cache[deviceId].count(descriptor) == 0) {
    _cache[deviceId][descriptor] = new ConstantHolder();
  }

  auto holder = _cache[deviceId][descriptor];

  // releasing cache lock
  _mutexHolder.unlock();

  ConstantDataBuffer *result;

  // access to this holder instance is synchronous
  holder->mutex()->lock();

  if (holder->hasBuffer(dataType))
    result = holder->getConstantDataBuffer(dataType);
  else {
    auto size = descriptor.length() * DataTypeUtils::sizeOf(dataType);
    auto cbuff = std::make_shared<PointerWrapper>(new int8_t[size], std::make_shared<PrimaryPointerDeallocator>());
    _counters[deviceId] += size;

    // create buffer with this dtype
    if (descriptor.isFloat()) {
      BUILD_DOUBLE_SELECTOR(
          sd::DataType::DOUBLE, dataType, sd::TypeCast::convertGeneric,
          (nullptr, const_cast<double *>(descriptor.floatValues().data()), descriptor.length(), cbuff->pointer()),
          (sd::DataType::DOUBLE, double), SD_COMMON_TYPES_ALL);
    } else if (descriptor.isInteger()) {
      BUILD_DOUBLE_SELECTOR(sd::DataType::INT64, dataType, sd::TypeCast::convertGeneric,
                            (nullptr, const_cast<sd::LongType *>(descriptor.integerValues().data()),
                             descriptor.length(), cbuff->pointer()),
                            (sd::DataType::INT64, sd::LongType), SD_COMMON_TYPES_ALL);
    }

    ConstantDataBuffer dataBuffer(cbuff, descriptor.length(), dataType);
    holder->addBuffer(dataBuffer, dataType);

    result = holder->getConstantDataBuffer(dataType);
  }
  holder->mutex()->unlock();

  return result;
}

sd::LongType ConstantHelper::getCachedAmount(int deviceId) {
  int numDevices = getNumberOfDevices();
  if (deviceId > numDevices || deviceId < 0)
    return 0L;
  else
    return _counters[deviceId];
}
}  // namespace sd

#endif
