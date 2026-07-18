/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#if !defined(SD_VULKAN)
#error "helpers/vulkan/ConstantHelper.cpp is only valid for SD_VULKAN"
#endif

#if !defined(HAVE_VULKAN) || !HAVE_VULKAN
#error "SD_VULKAN requires HAVE_VULKAN=1"
#endif

#include <array/DataTypeUtils.h>
#include <array/PrimaryPointerDeallocator.h>
#include <array/VulkanPointerDeallocator.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <helpers/ConstantHelper.h>
#include <loops/type_conversions.h>
#include <system/selective_rendering.h>
#include <types/types.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

SD_BACKEND_ABI_NAMESPACE_BEGIN
namespace {

constexpr VkDeviceSize kConstantSpaceBytes = 49152;

int requireCurrentDevice() {
  auto& manager = ::sd::graph::VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    THROW_EXCEPTION("ConstantHelper: Vulkan device initialization failed");
  }

  const int deviceId = ::sd::graph::VulkanDeviceManager::currentDeviceId();
  const int deviceCount = manager.deviceCount();
  if (deviceId < 0 || deviceId >= deviceCount) {
    const std::string message =
        "ConstantHelper: current Vulkan device is out of range: " +
        std::to_string(deviceId);
    THROW_EXCEPTION(message.c_str());
  }
  return deviceId;
}

::sd::graph::VulkanExecutionStream* orderedStream(int deviceId) {
  auto* stream =
      ::sd::graph::VulkanExecutionStream::currentOrDefault(deviceId);
  if (stream == nullptr || stream->deviceId() != deviceId) {
    THROW_EXCEPTION(
        "ConstantHelper: no active Vulkan stream for allocation device");
  }
  return stream;
}

void retireFailedAllocation(::sd::graph::VulkanExecutionStream* stream,
                            void* ptr) {
  if (ptr != nullptr && (stream == nullptr || !stream->retireAllocation(ptr))) {
    THROW_EXCEPTION(
        "ConstantHelper: failed to retire Vulkan allocation after copy failure");
  }
}

}  // namespace

ConstantHelper::ConstantHelper() {
  const int numDevices = getNumberOfDevices();
  if (numDevices <= 0) {
    THROW_EXCEPTION("ConstantHelper: no Vulkan devices are available");
  }

  _devicePointers.resize(numDevices, nullptr);
  _deviceOffsets.resize(numDevices, 0);
  _cache.resize(numDevices);
  _counters.resize(numDevices, 0);
}

ConstantHelper::~ConstantHelper() {
  for (const auto& deviceCache : _cache) {
    for (const auto& entry : deviceCache) {
      delete entry.second;
    }
  }
}

ConstantHelper& ConstantHelper::getInstance() {
  static ConstantHelper* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() { instance = new ConstantHelper(); });
  return *instance;
}

int ConstantHelper::getCurrentDevice() { return requireCurrentDevice(); }

int ConstantHelper::getNumberOfDevices() {
  auto& manager = ::sd::graph::VulkanDeviceManager::getInstance();
  return manager.initialize() ? manager.deviceCount() : 0;
}

void* ConstantHelper::getConstantSpace() {
  const int deviceId = requireCurrentDevice();
  std::lock_guard<std::mutex> lock(_mutex);

  void*& constantSpace = _devicePointers[deviceId];
  if (constantSpace == nullptr) {
    constantSpace = ::sd::graph::VulkanMemoryPool::getInstance().allocate(
        deviceId, kConstantSpaceBytes);
    if (constantSpace == nullptr) {
      THROW_EXCEPTION(
          "ConstantHelper: Vulkan constant-space allocation failed");
    }
    _deviceOffsets[deviceId] = 0;
  }

  return constantSpace;
}

void* ConstantHelper::replicatePointer(void* src, size_t numBytes,
                                       memory::Workspace* workspace) {
  if (numBytes == 0) return nullptr;
  if (src == nullptr) {
    THROW_EXCEPTION("ConstantHelper::replicatePointer: source is null");
  }

  const int deviceId = requireCurrentDevice();
  auto* stream = orderedStream(deviceId);
  auto& pool = ::sd::graph::VulkanMemoryPool::getInstance();
  const VkDeviceSize allocationBytes =
      static_cast<VkDeviceSize>(numBytes + SD_ALLOC_PADDING);

  void* ptr = nullptr;
  bool workspaceOwned = workspace != nullptr;
  if (workspaceOwned) {
    ptr = workspace->allocateBytes(memory::MemoryType::DEVICE,
                                   static_cast<LongType>(allocationBytes));
    if (ptr == nullptr) {
      THROW_EXCEPTION(
          "ConstantHelper::replicatePointer: Vulkan workspace allocation failed");
    }

    ::sd::graph::VulkanAllocRecord record;
    if (!pool.queryRecord(ptr, record) || record.deviceId != deviceId ||
        record.logicalSize < static_cast<VkDeviceSize>(numBytes)) {
      THROW_EXCEPTION(
          "ConstantHelper::replicatePointer: workspace did not provide "
          "Vulkan device storage on the current device");
    }
  } else {
    ptr = pool.allocate(deviceId, allocationBytes);
    if (ptr == nullptr) {
      THROW_EXCEPTION(
          "ConstantHelper::replicatePointer: VulkanMemoryPool allocation failed");
    }
  }

  if (!stream->enqueueCopy(ptr, src, static_cast<VkDeviceSize>(numBytes), 1)) {
    if (!workspaceOwned) retireFailedAllocation(stream, ptr);
    THROW_EXCEPTION(
        "ConstantHelper::replicatePointer: stream-ordered Vulkan H2D copy failed");
  }

  const uint64_t copySequence = stream->lastSequence();
  if (!stream->waitThrough(copySequence)) {
    if (!workspaceOwned) retireFailedAllocation(stream, ptr);
    THROW_EXCEPTION(
        "ConstantHelper::replicatePointer: Vulkan H2D completion failed");
  }

  return ptr;
}

ConstantDataBuffer* ConstantHelper::constantBuffer(
    const ConstantDescriptor& descriptor, DataType dataType) {
  const int deviceId = requireCurrentDevice();

  ConstantHolder* holder = nullptr;
  {
    std::lock_guard<std::mutex> cacheLock(_mutexHolder);
    auto& deviceCache = _cache[deviceId];
    auto found = deviceCache.find(descriptor);
    if (found == deviceCache.end()) {
      found =
          deviceCache.emplace(descriptor, new ConstantHolder()).first;
    }
    holder = found->second;
  }

  std::lock_guard<std::mutex> holderLock(*holder->mutex());
  if (holder->hasBuffer(dataType)) {
    return holder->getConstantDataBuffer(dataType);
  }

  const size_t numBytes =
      static_cast<size_t>(descriptor.length()) *
      ::sd::DataTypeUtils::sizeOf(dataType);
  auto primary = std::make_shared<PointerWrapper>(
      new int8_t[numBytes], std::make_shared<::sd::PrimaryPointerDeallocator>());

  if (descriptor.isFloat()) {
    BUILD_DOUBLE_SELECTOR(
        ::sd::DataType::DOUBLE, dataType, ::sd::TypeCast::convertGeneric,
        (nullptr, const_cast<double*>(descriptor.floatValues().data()),
         descriptor.length(), primary->pointer()),
        (DOUBLE, double), SD_COMMON_TYPES);
  } else if (descriptor.isInteger()) {
    BUILD_DOUBLE_SELECTOR(
        ::sd::DataType::INT64, dataType, ::sd::TypeCast::convertGeneric,
        (nullptr,
         const_cast<::sd::LongType*>(descriptor.integerValues().data()),
         descriptor.length(), primary->pointer()),
        (INT64, LongType), SD_COMMON_TYPES);
  } else {
    THROW_EXCEPTION(
        "ConstantHelper::constantBuffer: descriptor has no numeric values");
  }

  auto special = std::make_shared<PointerWrapper>(
      replicatePointer(primary->pointer(), numBytes),
      std::make_shared<::sd::VulkanPointerDeallocator>());

  ConstantDataBuffer dataBuffer(primary, special, descriptor.length(), dataType);
  holder->addBuffer(dataBuffer, dataType);

  {
    std::lock_guard<std::mutex> counterLock(_mutex);
    _counters[deviceId] += static_cast<LongType>(numBytes);
  }

  return holder->getConstantDataBuffer(dataType);
}

LongType ConstantHelper::getCachedAmount(int deviceId) {
  const int numDevices = getNumberOfDevices();
  if (deviceId < 0 || deviceId >= numDevices) return 0;

  std::lock_guard<std::mutex> lock(_mutex);
  return _counters[deviceId];
}

SD_BACKEND_ABI_NAMESPACE_END
