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
//  @author raver119@gmail.com
//
#include <array/DataTypeUtils.h>
#include <array/PrimaryPointerDeallocator.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <exceptions/cuda_exception.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <helpers/ConstantHelper.h>
#include <helpers/logger.h>
#include <helpers/shape.h>
#include <ops/specials.h>

#define CONSTANT_LIMIT 49152

__constant__ char deviceConstantMemory[CONSTANT_LIMIT];

namespace sd {
static void *getConstantSpace() {
  Pointer dConstAddr;
  auto dZ = cudaGetSymbolAddress(reinterpret_cast<void **>(&dConstAddr), deviceConstantMemory);

  if (dZ != 0) throw cuda_exception::build("cudaGetSymbolAddress(...) failed", dZ);

  return dConstAddr;
}

int ConstantHelper::getCurrentDevice() { return AffinityManager::currentDeviceId(); }

int ConstantHelper::getNumberOfDevices() { return AffinityManager::numberOfDevices(); }

ConstantHelper::ConstantHelper() {
  auto initialDevice = getCurrentDevice();

  auto numDevices = getNumberOfDevices();
  _devicePointers.resize(numDevices);
  _deviceOffsets.resize(numDevices);
  _cache.resize(numDevices);
  _counters.resize(numDevices);

  // filling all pointers
  for (int e = 0; e < numDevices; e++) {
    auto res = cudaSetDevice(e);
    if (res != 0) throw cuda_exception::build("cudaSetDevice failed", res);
    auto constant = getConstantSpace();

    SD_MAP_IMPL<ConstantDescriptor, ConstantHolder *> devCache;

    _devicePointers[e] = constant;
    _deviceOffsets[e] = 0;
    _cache[e] = devCache;
    _counters[e] = 0L;
  }

  //
  auto res = cudaSetDevice(initialDevice);
  if (res != 0) throw cuda_exception::build("Final cudaSetDevice failed", res);
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
  std::lock_guard<std::mutex> lock(_mutex);

  auto deviceId = getCurrentDevice();
  Pointer constantPtr = nullptr;
  LongType constantOffset = 0L;
  if (_devicePointers[deviceId] == 0) {
    auto constant = getConstantSpace();

    // filling default ptr, which will be 0 probably
    _devicePointers[deviceId] = constant;
    _deviceOffsets[deviceId] = 0;
    constantPtr = constant;
  } else {
    constantPtr = _devicePointers[deviceId];
    constantOffset = _deviceOffsets[deviceId];
  }

  int8_t *ptr = nullptr;
  ALLOCATE_SPECIAL(ptr, workspace, numBytes, int8_t);
  auto res = cudaMemcpy(ptr, src, numBytes, cudaMemcpyHostToDevice);
  if (res != 0) {
    std::string errorMessage = "cudaMemcpy failed with error code " + std::to_string(res);
    auto lastError = cudaGetLastError(); // get last error
    if (lastError != cudaSuccess) {
      errorMessage += "; last error: " + std::string(cudaGetErrorString(lastError));
    }

    THROW_EXCEPTION(errorMessage.c_str());

  }

  constantPtr = ptr;
  return reinterpret_cast<int8_t *>(constantPtr) + constantOffset;
}

ConstantDataBuffer *ConstantHelper::constantBuffer(const ConstantDescriptor &descriptor, DataType dataType) {
  const auto deviceId = getCurrentDevice();

  // all cache modifications are synchronous
  _mutexHolder.lock();

  if (_cache[deviceId].count(descriptor) == 0) {
    _cache[deviceId][descriptor] = new ConstantHolder();
  }
  auto holder = _cache[deviceId][descriptor];

  // release cache lock
  _mutexHolder.unlock();

  ConstantDataBuffer *result;

  // access to this holder instance is synchronous
  std::lock_guard<std::mutex> lock(*holder->mutex());

  if (holder->hasBuffer(dataType)) {
    result = holder->getConstantDataBuffer(dataType);
  } else {
    auto numBytes = descriptor.length() * DataTypeUtils::sizeOf(dataType);
    auto cbuff = std::make_shared<PointerWrapper>(new int8_t[numBytes], std::make_shared<PointerDeallocator>());
    _counters[deviceId] += numBytes;

    // create buffer with this dtype
    if (descriptor.isFloat()) {
      BUILD_DOUBLE_SELECTOR(
          sd::DataType::DOUBLE, dataType, sd::SpecialTypeConverter::convertGeneric,
          (nullptr, const_cast<double *>(descriptor.floatValues().data()), descriptor.length(), cbuff->pointer()),
          (sd::DataType::DOUBLE, double), SD_COMMON_TYPES);
    } else if (descriptor.isInteger()) {
      BUILD_DOUBLE_SELECTOR(sd::DataType::INT64, dataType, sd::SpecialTypeConverter::convertGeneric,
                            (nullptr, const_cast<sd::LongType *>(descriptor.integerValues().data()),
                                descriptor.length(), cbuff->pointer()),
                            (sd::DataType::INT64, sd::LongType), SD_COMMON_TYPES);
    }

    // we don't have deallocator here.
    // TODO: we probably want to make use deallocator here, if we're not using constant memory
    auto dbuff = std::make_shared<PointerWrapper>(
        replicatePointer(cbuff->pointer(), descriptor.length() * DataTypeUtils::sizeOf(dataType)));

    ConstantDataBuffer *dataBuffer = new ConstantDataBuffer(cbuff, dbuff, descriptor.length(), dataType);

    holder->addBuffer(*dataBuffer, dataType);
    result = holder->getConstantDataBuffer(dataType);
  }

  return result;
}

LongType ConstantHelper::getCachedAmount(int deviceId) {
  int numDevices = getNumberOfDevices();
  if (deviceId > numDevices || deviceId < 0)
    return 0L;
  else
    return _counters[deviceId];
}
}  // namespace sd
