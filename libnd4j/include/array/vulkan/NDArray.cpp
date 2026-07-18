/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#if defined(SD_VULKAN)

#if !defined(HAVE_VULKAN) || !HAVE_VULKAN
#error "SD_VULKAN requires HAVE_VULKAN=1"
#endif

#include <array/NDArray.h>
#include <array/NDArray.hXX>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanDeviceManager.h>

#include <string>

namespace sd {
namespace {

void waitForVulkanDevice(int deviceId, const char* operation) {
  if (deviceId < 0) {
    THROW_EXCEPTION("NDArray Vulkan synchronization requires a valid device");
  }

  auto* context = graph::VulkanDeviceContext::getContext(deviceId);
  if (context == nullptr) {
    THROW_EXCEPTION("NDArray Vulkan synchronization could not acquire the device context");
  }
  if (context->isLost()) {
    THROW_EXCEPTION("NDArray Vulkan synchronization attempted on a lost device");
  }

  const VkResult result = context->waitComputeIdle();
  if (result == VK_ERROR_DEVICE_LOST) {
    context->markLost();
  }
  if (result != VK_SUCCESS) {
    std::string message = operation != nullptr ? operation : "NDArray Vulkan synchronization";
    message += ": vkQueueWaitIdle failed with VkResult ";
    message += std::to_string(static_cast<int>(result));
    THROW_EXCEPTION(message.c_str());
  }
}

}  // namespace

void* NDArray::platformBuffer() {
  return specialBuffer();
}

void NDArray::syncToDevice() {
  if (_buffer == nullptr) return;

  const int currentDeviceId = graph::VulkanDeviceManager::currentDeviceId();
  if (currentDeviceId < 0) {
    THROW_EXCEPTION("NDArray::syncToDevice: no current Vulkan device");
  }

  const int bufferDeviceId = _buffer->deviceId();
  if (currentDeviceId != _deviceId || currentDeviceId != bufferDeviceId) {
    const_cast<NDArray*>(this)->setShapeInfo(shapeInfo());
    _buffer->migrate();
    _deviceId = currentDeviceId;
  }

  _buffer->syncToSpecial();
}

void NDArray::syncToHost() {
  if (_buffer != nullptr) _buffer->syncToPrimary(getContext());
}

void NDArray::forceSyncToHost() {
  if (_buffer != nullptr) _buffer->syncToPrimary(getContext(), true);
}

void NDArray::tickWriteHost() {
  if (_buffer != nullptr) _buffer->writePrimary();
}

void NDArray::tickWriteDevice() {
  if (_buffer != nullptr) _buffer->writeSpecial();
}

void NDArray::tickReadHost() {
  if (_buffer != nullptr) _buffer->readPrimary();
}

void NDArray::tickReadDevice() {
  if (_buffer != nullptr) _buffer->readSpecial();
}

void NDArray::tickBothActual() {
  if (_buffer == nullptr) return;
  _buffer->writePrimary();
  _buffer->readSpecial();
}

bool NDArray::isActualOnHostSide() {
  return _buffer == nullptr ? true : _buffer->isPrimaryActual();
}

bool NDArray::isActualOnDeviceSide() {
  return _buffer == nullptr ? true : _buffer->isSpecialActual();
}

void NDArray::makeBothBuffersActual() {
  if (!isActualOnHostSide()) syncToHost();
  if (!isActualOnDeviceSide()) syncToDevice();
}

void NDArray::synchronize(const char* msg) {
  const int deviceId = _buffer != nullptr
                           ? _buffer->deviceId()
                           : graph::VulkanDeviceManager::currentDeviceId();
  waitForVulkanDevice(deviceId, msg);
}

void NDArray::synchronizeExecStream(const char* msg) {
  synchronize(msg);
}

void NDArray::syncShape() {
  // Vulkan lowering consumes shape/stride/offset metadata while recording
  // descriptors and push constants. Shape metadata is not a device pointer.
}

void* NDArray::specialBuffer() {
  if (_buffer == nullptr) return nullptr;

  const int currentDeviceId = graph::VulkanDeviceManager::currentDeviceId();
  void* allocationIdentity = _buffer->special();

  if (allocationIdentity == nullptr || _buffer->deviceId() != currentDeviceId) {
    syncToDevice();
    tickReadHost();
    allocationIdentity = _buffer->special();
  }

  // VulkanMemoryPool keys allocations by this exact identity. NDArray view
  // offsets remain in shape metadata and must never be encoded by arithmetic
  // on a DEVICE_LOCAL registry token.
  return allocationIdentity;
}

void NDArray::prepareSpecialUse(const std::vector<NDArray*>& writeList,
                                const std::vector<NDArray*>& readList,
                                bool synchronizeWritables) {
  for (auto* array : readList) {
    if (array != nullptr) array->syncToDevice();
  }

  for (auto* array : writeList) {
    if (array == nullptr) continue;
    auto* dataBuffer = array->getDataBuffer();
    if (dataBuffer != nullptr) dataBuffer->allocateSpecial();
    if (synchronizeWritables) array->syncToDevice();
  }
}

void NDArray::registerSpecialUse(const std::vector<NDArray*>& writeList,
                                 const std::vector<NDArray*>& readList) {
  for (auto* array : readList) {
    if (array != nullptr) array->tickReadDevice();
  }
  for (auto* array : writeList) {
    if (array != nullptr) array->tickWriteDevice();
  }
}

void NDArray::preparePrimaryUse(const std::vector<NDArray*>& writeList,
                                const std::vector<NDArray*>& readList,
                                bool synchronizeWritables) {
  for (auto* array : readList) {
    if (array != nullptr) array->syncToHost();
  }

  for (auto* array : writeList) {
    if (array == nullptr) continue;
    auto* dataBuffer = array->getDataBuffer();
    if (dataBuffer != nullptr) dataBuffer->allocatePrimary();
    if (synchronizeWritables) array->syncToHost();
  }
}

void NDArray::registerPrimaryUse(const std::vector<NDArray*>& writeList,
                                 const std::vector<NDArray*>& readList) {
  for (auto* array : readList) {
    if (array != nullptr) array->tickReadHost();
  }
  for (auto* array : writeList) {
    if (array != nullptr) array->tickWriteHost();
  }
}

}  // namespace sd

#endif  // SD_VULKAN
