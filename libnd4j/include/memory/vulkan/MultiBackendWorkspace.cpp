/* ******************************************************************************
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <memory/MultiBackendWorkspace.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <helpers/logger.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace sd {
namespace memory {
namespace {

bool isVulkan(const DeviceDescriptor& device) {
  return device.deviceType == DeviceType::VULKAN_GPU;
}

bool isHost(const DeviceDescriptor& device) {
  return device.deviceType == DeviceType::CPU;
}

void validateDescriptor(const DeviceDescriptor& device) {
  if (isHost(device)) {
    if (device.deviceIndex != 0)
      THROW_EXCEPTION("MultiBackendWorkspace CPU device index must be zero");
    return;
  }
  if (!isVulkan(device))
    THROW_EXCEPTION("Vulkan MultiBackendWorkspace supports CPU and VULKAN_GPU descriptors");

  auto& manager = graph::VulkanDeviceManager::getInstance();
  if (!manager.initialize() || device.deviceIndex < 0 ||
      device.deviceIndex >= manager.deviceCount())
    THROW_EXCEPTION("Vulkan MultiBackendWorkspace device descriptor is unavailable");
  if (graph::VulkanDeviceContext::getContext(device.deviceIndex) == nullptr)
    THROW_EXCEPTION("Vulkan MultiBackendWorkspace could not create device context");
}

class CurrentDeviceGuard {
 public:
  explicit CurrentDeviceGuard(const DeviceDescriptor& device)
      : manager_(graph::VulkanDeviceManager::getInstance()),
        previous_(graph::VulkanDeviceManager::currentDeviceId()), changed_(false) {
    if (isVulkan(device)) {
      changed_ = previous_ != device.deviceIndex;
      if (changed_ && !manager_.setCurrentDevice(device.deviceIndex))
        THROW_EXCEPTION("Vulkan MultiBackendWorkspace could not select device");
    }
  }
  ~CurrentDeviceGuard() {
    if (changed_ && previous_ >= 0 && previous_ < manager_.deviceCount())
      manager_.setCurrentDevice(previous_);
  }
 private:
  graph::VulkanDeviceManager& manager_;
  int previous_;
  bool changed_;
};

graph::VulkanExecutionStream* copyStream(int deviceId) {
  auto* stream = graph::VulkanExecutionStream::defaultCopy(deviceId);
  if (stream == nullptr)
    THROW_EXCEPTION("Vulkan MultiBackendWorkspace could not acquire copy stream");
  return stream;
}

void synchronizeVulkan(int deviceId) {
  if (!graph::VulkanExecutionStream::synchronizeDevice(deviceId))
    THROW_EXCEPTION("Vulkan MultiBackendWorkspace device synchronization failed");
}

sd::LongType allocationBytes(void* token, int expectedDevice) {
  graph::VulkanAllocRecord record;
  if (!graph::VulkanMemoryPool::getInstance().queryRecord(token, record) ||
      record.deviceId != expectedDevice)
    THROW_EXCEPTION("Vulkan MultiBackendWorkspace encountered an invalid allocation token");
  if (record.logicalSize >
      static_cast<VkDeviceSize>(std::numeric_limits<sd::LongType>::max()))
    THROW_EXCEPTION("Vulkan MultiBackendWorkspace allocation size exceeds LongType");
  return static_cast<sd::LongType>(record.logicalSize);
}

sd::LongType checkedAdd(sd::LongType left, sd::LongType right,
                        const char* message) {
  if (left < 0 || right < 0 ||
      left > std::numeric_limits<sd::LongType>::max() - right)
    THROW_EXCEPTION(message);
  return left + right;
}

std::vector<sd::LongType> tokenSizes(const std::vector<void*>& tokens,
                                     int deviceId,
                                     sd::LongType* total = nullptr) {
  std::vector<sd::LongType> sizes;
  sizes.reserve(tokens.size());
  sd::LongType sum = 0;
  for (void* token : tokens) {
    const sd::LongType bytes = allocationBytes(token, deviceId);
    if (bytes <= 0)
      THROW_EXCEPTION("Vulkan MultiBackendWorkspace allocation token has invalid size");
    sum = checkedAdd(sum, bytes,
                     "Vulkan MultiBackendWorkspace allocation size overflow");
    sizes.push_back(bytes);
  }
  if (total != nullptr) *total = sum;
  return sizes;
}

void validateConfiguredSize(const MultiBackendWorkspaceConfig& config,
                            sd::LongType size) {
  if (size < 0)
    THROW_EXCEPTION("MultiBackendWorkspace size must be non-negative");
  if (config.maxSize < 0)
    THROW_EXCEPTION("MultiBackendWorkspace maximum size must be non-negative");
  if (config.maxSize > 0 && size > config.maxSize)
    THROW_EXCEPTION("MultiBackendWorkspace maximum size exceeded");
}

}  // namespace

MultiBackendWorkspace::MultiBackendWorkspace(
    const MultiBackendWorkspaceConfig& config, const std::string& id)
    : _id(id.empty() ? "mbw_" +
                           std::to_string(reinterpret_cast<uintptr_t>(this))
                     : id),
      _config(config),
      _primaryDevice(config.primaryDevice),
      _globalVersion(0),
      _scopeDepth(0),
      _scopeActive(false),
      _totalAllocations(0),
      _totalDeallocations(0),
      _totalTransfers(0) {
  validateConfiguredSize(_config, config.initialSize);
  validateDescriptor(_primaryDevice);
  if (config.initialSize > 0)
    initDeviceWorkspace(_primaryDevice, config.initialSize);
}

MultiBackendWorkspace::MultiBackendWorkspace(
    sd::LongType initialSize, const DeviceDescriptor& primaryDevice)
    : _id("mbw_" + std::to_string(reinterpret_cast<uintptr_t>(this))),
      _primaryDevice(primaryDevice),
      _globalVersion(0),
      _scopeDepth(0),
      _scopeActive(false),
      _totalAllocations(0),
      _totalDeallocations(0),
      _totalTransfers(0) {
  _config.initialSize = initialSize;
  _config.primaryDevice = primaryDevice;
  validateConfiguredSize(_config, initialSize);
  validateDescriptor(_primaryDevice);
  if (initialSize > 0) initDeviceWorkspace(_primaryDevice, initialSize);
}

MultiBackendWorkspace::~MultiBackendWorkspace() {
  try {
    destroy();
  } catch (...) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& entry : _deviceAllocations) {
      if (entry.second.isOwned && entry.second.workspace != nullptr) {
        delete entry.second.workspace;
        entry.second.workspace = nullptr;
      }
    }
    _deviceAllocations.clear();
  }
}

MultiBackendWorkspace::MultiBackendWorkspace(MultiBackendWorkspace&& other) noexcept
    : _id(std::move(other._id)),
      _config(std::move(other._config)),
      _deviceAllocations(std::move(other._deviceAllocations)),
      _primaryDevice(std::move(other._primaryDevice)),
      _globalVersion(other._globalVersion.load()),
      _scopeDepth(other._scopeDepth.load()),
      _scopeActive(other._scopeActive.load()),
      _totalAllocations(other._totalAllocations.load()),
      _totalDeallocations(other._totalDeallocations.load()),
      _totalTransfers(other._totalTransfers.load()) {
  other._deviceAllocations.clear();
}

MultiBackendWorkspace& MultiBackendWorkspace::operator=(
    MultiBackendWorkspace&& other) noexcept {
  if (this == &other) return *this;
  try {
    destroy();
  } catch (...) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& entry : _deviceAllocations) {
      if (entry.second.isOwned && entry.second.workspace != nullptr)
        delete entry.second.workspace;
    }
    _deviceAllocations.clear();
  }
  std::scoped_lock<std::mutex, std::mutex> lock(_mutex, other._mutex);
  _id = std::move(other._id);
  _config = std::move(other._config);
  _deviceAllocations = std::move(other._deviceAllocations);
  _primaryDevice = std::move(other._primaryDevice);
  _globalVersion = other._globalVersion.load();
  _scopeDepth = other._scopeDepth.load();
  _scopeActive = other._scopeActive.load();
  _totalAllocations = other._totalAllocations.load();
  _totalDeallocations = other._totalDeallocations.load();
  _totalTransfers = other._totalTransfers.load();
  other._deviceAllocations.clear();
  return *this;
}

void MultiBackendWorkspace::initDeviceWorkspace(
    const DeviceDescriptor& device, sd::LongType size) {
  validateConfiguredSize(_config, size);
  validateDescriptor(device);
  auto it = _deviceAllocations.find(device);
  if (it != _deviceAllocations.end() && it->second.workspace != nullptr) {
    Workspace* workspace = it->second.workspace;
    sd::LongType current = isHost(device)
                               ? workspace->getCurrentSecondarySize()
                               : workspace->getCurrentSize();
    if (current < size) {
      CurrentDeviceGuard guard(device);
      workspace->expandTo(isVulkan(device) ? size : 0, size);
    }
    return;
  }

  CurrentDeviceGuard guard(device);
  DeviceAllocation allocation;
  allocation.workspace =
      new Workspace(isVulkan(device) ? size : 0, size, isHost(device));
  allocation.coherenceState = CoherenceState::EXCLUSIVE;
  allocation.version = _globalVersion.load();
  allocation.isOwned = true;
  _deviceAllocations[device] = allocation;
}

void MultiBackendWorkspace::freeDeviceWorkspace(
    const DeviceDescriptor& device) {
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _deviceAllocations.find(device);
  if (it == _deviceAllocations.end()) return;
  if (it->second.isOwned && it->second.workspace != nullptr) {
    if (isVulkan(device)) synchronizeVulkan(device.deviceIndex);
    delete it->second.workspace;
    ++_totalDeallocations;
  }
  _deviceAllocations.erase(it);
}

void MultiBackendWorkspace::invalidateOtherDevices(
    const DeviceDescriptor& exceptDevice) {
  for (auto& entry : _deviceAllocations)
    if (entry.first != exceptDevice)
      entry.second.coherenceState = CoherenceState::INVALID;
}

void MultiBackendWorkspace::setPrimaryDevice(
    const DeviceDescriptor& device) {
  validateDescriptor(device);
  std::lock_guard<std::mutex> lock(_mutex);
  _primaryDevice = device;
}

std::vector<DeviceDescriptor> MultiBackendWorkspace::getActiveDevices() const {
  std::lock_guard<std::mutex> lock(_mutex);
  std::vector<DeviceDescriptor> devices;
  devices.reserve(_deviceAllocations.size());
  for (const auto& entry : _deviceAllocations) devices.push_back(entry.first);
  return devices;
}

bool MultiBackendWorkspace::hasDeviceAllocation(
    const DeviceDescriptor& device) const {
  std::lock_guard<std::mutex> lock(_mutex);
  return _deviceAllocations.find(device) != _deviceAllocations.end();
}

void* MultiBackendWorkspace::allocateBytes(sd::LongType numBytes) {
  return allocateBytes(_primaryDevice, numBytes);
}

void* MultiBackendWorkspace::allocateBytes(
    const DeviceDescriptor& device, sd::LongType numBytes) {
  return allocateBytes(device, isVulkan(device) ? MemoryType::DEVICE
                                                : MemoryType::HOST,
                       numBytes);
}

void* MultiBackendWorkspace::allocateBytes(
    const DeviceDescriptor& device, MemoryType type, sd::LongType numBytes) {
  if (numBytes < 0)
    THROW_EXCEPTION("MultiBackendWorkspace allocation size must be non-negative");
  if (numBytes == 0) return nullptr;
  validateDescriptor(device);
  if (isHost(device) && type != MemoryType::HOST)
    THROW_EXCEPTION("CPU workspace cannot allocate Vulkan DEVICE memory");
  if (isVulkan(device) && type != MemoryType::DEVICE &&
      type != MemoryType::HOST)
    THROW_EXCEPTION("Unsupported Vulkan workspace memory type");

  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _deviceAllocations.find(device);
  if (it == _deviceAllocations.end() || it->second.workspace == nullptr) {
    initDeviceWorkspace(device, std::max(numBytes, _config.initialSize));
    it = _deviceAllocations.find(device);
  }

  Workspace* workspace = it->second.workspace;
  const sd::LongType used =
      checkedAdd(workspace->getCurrentOffset(),
                 workspace->getCurrentSecondaryOffset(),
                 "MultiBackendWorkspace used-size overflow");
  validateConfiguredSize(
      _config, checkedAdd(used, numBytes,
                          "MultiBackendWorkspace allocation size overflow"));
  const sd::LongType requiredDevice =
      type == MemoryType::DEVICE
          ? checkedAdd(workspace->getCurrentOffset(), numBytes,
                       "MultiBackendWorkspace DEVICE size overflow")
          : workspace->getCurrentSize();
  const sd::LongType requiredHost =
      type == MemoryType::HOST
          ? checkedAdd(workspace->getCurrentSecondaryOffset(), numBytes,
                       "MultiBackendWorkspace HOST size overflow")
          : workspace->getCurrentSecondarySize();
  CurrentDeviceGuard guard(device);
  if (requiredDevice > workspace->getCurrentSize() ||
      requiredHost > workspace->getCurrentSecondarySize())
    workspace->expandTo(requiredDevice, requiredHost);
  void* ptr = workspace->allocateBytes(type, numBytes);
  ++_totalAllocations;
  it->second.coherenceState = CoherenceState::EXCLUSIVE;
  it->second.version = ++_globalVersion;
  invalidateOtherDevices(device);
  return ptr;
}

Workspace* MultiBackendWorkspace::getDeviceWorkspace(
    const DeviceDescriptor& device) {
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _deviceAllocations.find(device);
  return it == _deviceAllocations.end() ? nullptr : it->second.workspace;
}

void MultiBackendWorkspace::ensureWorkspaceOnDevice(
    const DeviceDescriptor& device, sd::LongType minSize) {
  std::lock_guard<std::mutex> lock(_mutex);
  initDeviceWorkspace(device, std::max(minSize, _config.initialSize));
}

void MultiBackendWorkspace::scopeIn() {
  std::lock_guard<std::mutex> lock(_mutex);
  ++_scopeDepth;
  _scopeActive = true;
  for (auto& entry : _deviceAllocations) {
    CurrentDeviceGuard guard(entry.first);
    if (entry.second.workspace != nullptr) entry.second.workspace->scopeIn();
  }
}

void MultiBackendWorkspace::scopeOut() {
  std::lock_guard<std::mutex> lock(_mutex);
  if (_scopeDepth > 0) --_scopeDepth;
  if (_scopeDepth == 0) _scopeActive = false;
  for (auto& entry : _deviceAllocations) {
    CurrentDeviceGuard guard(entry.first);
    if (entry.second.workspace != nullptr) entry.second.workspace->scopeOut();
  }
}

CoherenceState MultiBackendWorkspace::getCoherenceState(
    const DeviceDescriptor& device) const {
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _deviceAllocations.find(device);
  return it == _deviceAllocations.end() ? CoherenceState::INVALID
                                        : it->second.coherenceState;
}

void MultiBackendWorkspace::setCoherenceState(
    const DeviceDescriptor& device, CoherenceState state) {
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _deviceAllocations.find(device);
  if (it != _deviceAllocations.end()) it->second.coherenceState = state;
}

void MultiBackendWorkspace::markModified(const DeviceDescriptor& device) {
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _deviceAllocations.find(device);
  if (it == _deviceAllocations.end()) return;
  it->second.coherenceState = CoherenceState::MODIFIED;
  it->second.version = ++_globalVersion;
  invalidateOtherDevices(device);
}

void MultiBackendWorkspace::invalidateDevice(
    const DeviceDescriptor& device) {
  setCoherenceState(device, CoherenceState::INVALID);
}

void MultiBackendWorkspace::invalidateAllExcept(
    const DeviceDescriptor& device) {
  std::lock_guard<std::mutex> lock(_mutex);
  invalidateOtherDevices(device);
}

sd::LongType MultiBackendWorkspace::getDeviceVersion(
    const DeviceDescriptor& device) const {
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _deviceAllocations.find(device);
  return it == _deviceAllocations.end() ? 0 : it->second.version;
}

void MultiBackendWorkspace::transferTo(
    const DeviceDescriptor& source, const DeviceDescriptor& target) {
  validateDescriptor(source);
  validateDescriptor(target);
  if (source == target) return;

  std::scoped_lock<std::mutex, std::mutex> lock(_transferMutex, _mutex);
  auto srcIt = _deviceAllocations.find(source);
  if (srcIt == _deviceAllocations.end() || srcIt->second.workspace == nullptr)
    THROW_EXCEPTION("Source device has no workspace allocation");

  Workspace* src = srcIt->second.workspace;
  if (src->getSpilledSize() > 0 ||
      src->getSpilledSecondarySize() > 0)
    THROW_EXCEPTION("MultiBackendWorkspace cannot mirror spilled allocations");

  const sd::LongType hostBytes = src->getCurrentSecondaryOffset();
  if (hostBytes > 0 && src->getHostPointer() == nullptr)
    THROW_EXCEPTION("MultiBackendWorkspace HOST mirror pointer is null");

  const auto srcTokens =
      isVulkan(source) ? src->snapshotPrimaryAllocations()
                       : std::vector<void*>();
  sd::LongType sourceDeviceBytes = 0;
  const auto srcTokenSizes =
      isVulkan(source)
          ? tokenSizes(srcTokens, source.deviceIndex, &sourceDeviceBytes)
          : std::vector<sd::LongType>();
  const sd::LongType mirrorBytes =
      checkedAdd(hostBytes, sourceDeviceBytes,
                 "MultiBackendWorkspace mirror size overflow");
  validateConfiguredSize(_config, mirrorBytes);

  const sd::LongType required =
      isHost(target) ? mirrorBytes
                     : std::max(src->getCurrentSize(),
                                src->getCurrentSecondarySize());
  initDeviceWorkspace(target, required);
  auto dstIt = _deviceAllocations.find(target);
  Workspace* dst = dstIt->second.workspace;

  auto dstTokens =
      isVulkan(target) ? dst->snapshotPrimaryAllocations()
                       : std::vector<void*>();
  if (isVulkan(source) && isVulkan(target)) {
    if (dstTokens.empty()) {
      CurrentDeviceGuard guard(target);
      for (sd::LongType bytes : srcTokenSizes)
        dst->allocateBytes(MemoryType::DEVICE, bytes);
      dstTokens = dst->snapshotPrimaryAllocations();
    }
  } else if (isHost(source) && isVulkan(target) && hostBytes > 0 &&
             dstTokens.empty()) {
    CurrentDeviceGuard guard(target);
    dst->allocateBytes(MemoryType::DEVICE, hostBytes);
    dstTokens = dst->snapshotPrimaryAllocations();
  }

  std::vector<sd::LongType> dstTokenSizes;
  sd::LongType targetDeviceBytes = 0;
  if (isVulkan(target))
    dstTokenSizes =
        tokenSizes(dstTokens, target.deviceIndex, &targetDeviceBytes);

  if (isVulkan(source) && isVulkan(target)) {
    if (dstTokenSizes != srcTokenSizes)
      THROW_EXCEPTION("Vulkan mirror allocation layout does not match source");
  } else if (isHost(source) && isVulkan(target) &&
             targetDeviceBytes != hostBytes) {
    THROW_EXCEPTION("Vulkan mirror allocation layout does not match CPU source");
  }

  unsigned char* targetHost =
      static_cast<unsigned char*>(dst->getHostPointer());
  const sd::LongType targetHostBytes =
      isHost(target) ? mirrorBytes : hostBytes;
  if (targetHostBytes > 0) {
    if (targetHost == nullptr)
      THROW_EXCEPTION("MultiBackendWorkspace HOST mirror pointer is null");
    const sd::LongType targetOffset = dst->getCurrentSecondaryOffset();
    if (targetOffset == 0) {
      void* base = dst->allocateBytes(MemoryType::HOST, targetHostBytes);
      if (base != targetHost)
        THROW_EXCEPTION("HOST mirror requires contiguous workspace storage");
    } else if (targetOffset != targetHostBytes) {
      THROW_EXCEPTION("HOST mirror allocation layout does not match source");
    }
  }

  // All token ownership, exact sizes, aggregate sizes, and host layouts are
  // validated before the first host mutation or Vulkan copy submission.
  if (hostBytes > 0)
    std::memcpy(targetHost, src->getHostPointer(),
                static_cast<size_t>(hostBytes));

  if (isVulkan(source)) synchronizeVulkan(source.deviceIndex);
  if (isVulkan(target)) synchronizeVulkan(target.deviceIndex);

  if (isVulkan(source)) {
    sd::LongType deviceOffset = hostBytes;
    for (size_t i = 0; i < srcTokens.size(); ++i) {
      void* targetPointer =
          isVulkan(target)
              ? dstTokens[i]
              : static_cast<void*>(targetHost + deviceOffset);
      auto* stream = copyStream(isVulkan(target) ? target.deviceIndex
                                                 : source.deviceIndex);
      const int direction = isVulkan(target) ? 3 : 2;
      if (!stream->enqueueCopy(
              targetPointer, srcTokens[i],
              static_cast<VkDeviceSize>(srcTokenSizes[i]), direction))
        THROW_EXCEPTION("Vulkan MultiBackendWorkspace copy submission failed");
      if (!stream->synchronize())
        THROW_EXCEPTION("Vulkan MultiBackendWorkspace copy synchronization failed");
      deviceOffset =
          checkedAdd(deviceOffset, srcTokenSizes[i],
                     "MultiBackendWorkspace mirror offset overflow");
    }
  } else if (isVulkan(target) && hostBytes > 0) {
    sd::LongType hostOffset = 0;
    auto* stream = copyStream(target.deviceIndex);
    for (size_t i = 0; i < dstTokens.size(); ++i) {
      if (!stream->enqueueCopy(
              dstTokens[i],
              static_cast<unsigned char*>(src->getHostPointer()) + hostOffset,
              static_cast<VkDeviceSize>(dstTokenSizes[i]), 1))
        THROW_EXCEPTION("CPU to Vulkan workspace copy submission failed");
      if (!stream->synchronize())
        THROW_EXCEPTION("CPU to Vulkan workspace copy synchronization failed");
      hostOffset =
          checkedAdd(hostOffset, dstTokenSizes[i],
                     "MultiBackendWorkspace mirror offset overflow");
    }
  }

  dstIt->second.coherenceState = CoherenceState::SHARED;
  dstIt->second.version = srcIt->second.version;
  srcIt->second.coherenceState = CoherenceState::SHARED;
  ++_totalTransfers;
}

void MultiBackendWorkspace::ensureValidOn(
    const DeviceDescriptor& device) {
  DeviceDescriptor source;
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _deviceAllocations.find(device);
    if (it != _deviceAllocations.end() &&
        it->second.coherenceState != CoherenceState::INVALID)
      return;
    sd::LongType highestVersion = std::numeric_limits<sd::LongType>::min();
    for (const auto& entry : _deviceAllocations) {
      if (entry.first != device &&
          entry.second.coherenceState != CoherenceState::INVALID &&
          (!found || entry.second.version > highestVersion)) {
        source = entry.first;
        highestVersion = entry.second.version;
        found = true;
      }
    }
  }
  if (found)
    transferTo(source, device);
  else
    ensureWorkspaceOnDevice(device, _config.initialSize);
}

void MultiBackendWorkspace::syncDevice(const DeviceDescriptor& device) {
  validateDescriptor(device);
  if (isVulkan(device)) synchronizeVulkan(device.deviceIndex);
}

void MultiBackendWorkspace::syncAllDevices() {
  const auto devices = getActiveDevices();
  for (const auto& device : devices) syncDevice(device);
}

sd::LongType MultiBackendWorkspace::getTotalAllocatedSize() const {
  std::lock_guard<std::mutex> lock(_mutex);
  sd::LongType total = 0;
  for (const auto& entry : _deviceAllocations) {
    if (entry.second.workspace == nullptr) continue;
    sd::LongType deviceBytes = entry.second.workspace->getAllocatedSize();
    if (isVulkan(entry.first)) {
      const auto tokens = entry.second.workspace->snapshotPrimaryAllocations();
      tokenSizes(tokens, entry.first.deviceIndex, &deviceBytes);
    }
    const sd::LongType allocationBytes =
        checkedAdd(deviceBytes,
                   entry.second.workspace->getAllocatedSecondarySize(),
                   "MultiBackendWorkspace allocated-size overflow");
    total = checkedAdd(total, allocationBytes,
                       "MultiBackendWorkspace total allocated-size overflow");
  }
  return total;
}

sd::LongType MultiBackendWorkspace::getAllocatedSizeOnDevice(
    const DeviceDescriptor& device) const {
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _deviceAllocations.find(device);
  if (it == _deviceAllocations.end() || it->second.workspace == nullptr)
    return 0;
  sd::LongType deviceBytes = it->second.workspace->getAllocatedSize();
  if (isVulkan(device)) {
    const auto tokens = it->second.workspace->snapshotPrimaryAllocations();
    tokenSizes(tokens, device.deviceIndex, &deviceBytes);
  }
  return checkedAdd(deviceBytes,
                    it->second.workspace->getAllocatedSecondarySize(),
                    "MultiBackendWorkspace allocated-size overflow");
}

sd::LongType MultiBackendWorkspace::getCurrentOffset() const {
  return getCurrentOffsetOnDevice(_primaryDevice);
}

sd::LongType MultiBackendWorkspace::getCurrentOffsetOnDevice(
    const DeviceDescriptor& device) const {
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _deviceAllocations.find(device);
  if (it == _deviceAllocations.end() || it->second.workspace == nullptr)
    return 0;
  return isHost(device) ? it->second.workspace->getCurrentSecondaryOffset()
                        : it->second.workspace->getCurrentOffset();
}

sd::LongType MultiBackendWorkspace::getSpilledSize() const {
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _deviceAllocations.find(_primaryDevice);
  return it == _deviceAllocations.end() || it->second.workspace == nullptr
             ? 0
             : it->second.workspace->getSpilledSize() +
                   it->second.workspace->getSpilledSecondarySize();
}

void MultiBackendWorkspace::destroy() {
  std::lock_guard<std::mutex> lock(_mutex);
  for (auto& entry : _deviceAllocations) {
    if (entry.second.isOwned && entry.second.workspace != nullptr) {
      if (isVulkan(entry.first)) synchronizeVulkan(entry.first.deviceIndex);
      delete entry.second.workspace;
      entry.second.workspace = nullptr;
      ++_totalDeallocations;
    }
  }
  _deviceAllocations.clear();
  _scopeActive = false;
  _scopeDepth = 0;
}

void MultiBackendWorkspace::releaseOnDevice(
    const DeviceDescriptor& device) {
  freeDeviceWorkspace(device);
}

void MultiBackendWorkspace::resetStatistics() {
  _totalAllocations = 0;
  _totalDeallocations = 0;
  _totalTransfers = 0;
}

MultiBackendWorkspace* MultiBackendWorkspace::clone() const {
  std::lock_guard<std::mutex> lock(_mutex);
  return new MultiBackendWorkspace(_config, _id + "_clone");
}

extern "C" {

MultiBackendWorkspaceHandle createMultiBackendWorkspace(
    sd::LongType initialSize, int primaryDeviceType, int primaryDeviceIndex) {
  return new MultiBackendWorkspace(
      initialSize,
      DeviceDescriptor(static_cast<DeviceType>(primaryDeviceType),
                       primaryDeviceIndex));
}

MultiBackendWorkspaceHandle createMultiBackendWorkspaceWithConfig(
    sd::LongType initialSize, sd::LongType maxSize,
    bool crossDeviceMirroring, bool asyncTransfers, int primaryDeviceType,
    int primaryDeviceIndex, const char* id) {
  MultiBackendWorkspaceConfig config;
  config.initialSize = initialSize;
  config.maxSize = maxSize;
  config.crossDeviceMirroring = crossDeviceMirroring;
  config.asyncTransfers = asyncTransfers;
  config.primaryDevice =
      DeviceDescriptor(static_cast<DeviceType>(primaryDeviceType),
                       primaryDeviceIndex);
  return new MultiBackendWorkspace(config, id == nullptr ? "" : id);
}

void destroyMultiBackendWorkspace(MultiBackendWorkspaceHandle handle) {
  if (handle != nullptr) delete handle;
}

void* mbwAllocateBytes(MultiBackendWorkspaceHandle h, sd::LongType n) {
  return h == nullptr ? nullptr : h->allocateBytes(n);
}
void* mbwAllocateBytesOnDevice(MultiBackendWorkspaceHandle h, sd::LongType n,
                               int t, int i) {
  return h == nullptr ? nullptr
                      : h->allocateBytes(
                            DeviceDescriptor(static_cast<DeviceType>(t), i), n);
}
void mbwScopeIn(MultiBackendWorkspaceHandle h) { if (h != nullptr) h->scopeIn(); }
void mbwScopeOut(MultiBackendWorkspaceHandle h) { if (h != nullptr) h->scopeOut(); }
bool mbwIsScopeActive(MultiBackendWorkspaceHandle h) {
  return h != nullptr && h->isScopeActive();
}
int mbwGetCoherenceState(MultiBackendWorkspaceHandle h, int t, int i) {
  return h == nullptr
             ? static_cast<int>(CoherenceState::INVALID)
             : static_cast<int>(h->getCoherenceState(
                   DeviceDescriptor(static_cast<DeviceType>(t), i)));
}
void mbwSetCoherenceState(MultiBackendWorkspaceHandle h, int t, int i, int s) {
  if (h != nullptr)
    h->setCoherenceState(DeviceDescriptor(static_cast<DeviceType>(t), i),
                         static_cast<CoherenceState>(s));
}
void mbwMarkModified(MultiBackendWorkspaceHandle h, int t, int i) {
  if (h != nullptr)
    h->markModified(DeviceDescriptor(static_cast<DeviceType>(t), i));
}
void mbwTransferTo(MultiBackendWorkspaceHandle h, int st, int si, int dt, int di) {
  if (h != nullptr)
    h->transferTo(DeviceDescriptor(static_cast<DeviceType>(st), si),
                  DeviceDescriptor(static_cast<DeviceType>(dt), di));
}
void mbwEnsureValidOn(MultiBackendWorkspaceHandle h, int t, int i) {
  if (h != nullptr)
    h->ensureValidOn(DeviceDescriptor(static_cast<DeviceType>(t), i));
}
sd::LongType mbwGetTotalAllocatedSize(MultiBackendWorkspaceHandle h) {
  return h == nullptr ? 0 : h->getTotalAllocatedSize();
}
sd::LongType mbwGetAllocatedSizeOnDevice(MultiBackendWorkspaceHandle h, int t,
                                         int i) {
  return h == nullptr
             ? 0
             : h->getAllocatedSizeOnDevice(
                   DeviceDescriptor(static_cast<DeviceType>(t), i));
}
sd::LongType mbwGetCurrentOffset(MultiBackendWorkspaceHandle h) {
  return h == nullptr ? 0 : h->getCurrentOffset();
}
void mbwReleaseOnDevice(MultiBackendWorkspaceHandle h, int t, int i) {
  if (h != nullptr)
    h->releaseOnDevice(DeviceDescriptor(static_cast<DeviceType>(t), i));
}
void mbwSyncDevice(MultiBackendWorkspaceHandle h, int t, int i) {
  if (h != nullptr)
    h->syncDevice(DeviceDescriptor(static_cast<DeviceType>(t), i));
}
void mbwSyncAllDevices(MultiBackendWorkspaceHandle h) {
  if (h != nullptr) h->syncAllDevices();
}
int mbwGetActiveDeviceCount(MultiBackendWorkspaceHandle h) {
  return h == nullptr ? 0 : static_cast<int>(h->getActiveDevices().size());
}
bool mbwHasDeviceAllocation(MultiBackendWorkspaceHandle h, int t, int i) {
  return h != nullptr &&
         h->hasDeviceAllocation(DeviceDescriptor(static_cast<DeviceType>(t), i));
}
const char* mbwGetId(MultiBackendWorkspaceHandle h) {
  return h == nullptr ? "" : h->getId().c_str();
}

}  // extern "C"
}  // namespace memory
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
