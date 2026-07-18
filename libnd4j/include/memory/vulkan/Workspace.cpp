/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Vulkan workspaces implementation.
//
// VulkanMemoryPool may return opaque registry tokens for DEVICE_LOCAL memory.
// Such tokens are allocation identities, not host addresses.  Consequently a
// Vulkan workspace owns one pool allocation per DEVICE request and uses its
// byte offsets only for capacity/accounting; it never forms token + offset.
//

#include "../Workspace.h"

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <helpers/logger.h>
#include <math/templatemath.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace sd {
namespace memory {

namespace {

void* allocateHost(LongType bytes) {
  void* ptr = std::malloc(static_cast<size_t>(bytes));
  if (ptr == nullptr) {
    std::string message =
        "Can't allocate [HOST] memory; size: [" + std::to_string(bytes) + "]";
    THROW_EXCEPTION(message.c_str());
  }
  return ptr;
}

int currentVulkanDevice() {
  auto& manager = graph::VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    THROW_EXCEPTION("Vulkan Workspace could not initialize VulkanDeviceManager");
  }

  const int deviceId = graph::VulkanDeviceManager::currentDeviceId();
  if (deviceId < 0 || deviceId >= manager.deviceCount()) {
    std::string message =
        "Vulkan Workspace has invalid current device id: [" +
        std::to_string(deviceId) + "]";
    THROW_EXCEPTION(message.c_str());
  }
  return deviceId;
}

}  // namespace

Workspace::Workspace(ExternalWorkspace* external) {
  if (external == nullptr) {
    THROW_EXCEPTION("Vulkan Workspace requires a non-null ExternalWorkspace");
  }

  _ptrHost = reinterpret_cast<char*>(external->pointerHost());
  _ptrDevice = reinterpret_cast<char*>(external->pointerDevice());
  _initialSize = external->sizeDevice();
  _currentSize = external->sizeDevice();
  _initialSizeSecondary = external->sizeHost();
  _currentSizeSecondary = external->sizeHost();
  _offset = 0;
  _offsetSecondary = 0;
  _cycleAllocations = 0;
  _cycleAllocationsSecondary = 0;
  _spillsSize = 0;
  _spillsSizeSecondary = 0;
  _externalized = true;

  if (_ptrDevice != nullptr) {
    _deviceId = graph::VulkanMemoryPool::getInstance().getDeviceId(_ptrDevice);
  }
}

Workspace::Workspace(LongType primarySize, LongType secondarySize,
                     bool secondaryUsePlainMalloc) {
  if (primarySize < 0 || secondarySize < 0) {
    THROW_EXCEPTION("Vulkan Workspace sizes must be non-negative");
  }

  // Vulkan HOST storage is ordinary host memory.  This flag is CUDA-specific;
  // retaining it only preserves the common constructor ABI.
  _secondaryUsePlainMalloc = secondaryUsePlainMalloc;

  if (secondarySize > 0) {
    _ptrHost = reinterpret_cast<char*>(
        allocateHost(secondarySize + CANARY_SIZE));
    std::memset(_ptrHost, 0, static_cast<size_t>(secondarySize));
    std::memset(_ptrHost + secondarySize, CANARY_BYTE, CANARY_SIZE);
    _allocatedHost = true;
    _canaryEnabled = true;
  }

  // Do not allocate a DEVICE base here. Vulkan DEVICE_LOCAL allocations can be
  // opaque tokens, so a base token could not correctly represent subranges.
  // _currentSize remains the workspace's accounting/capacity target.
  _ptrDevice = nullptr;
  _allocatedDevice = false;
  _deviceId = primarySize > 0 ? currentVulkanDevice() : -1;

  _initialSize = primarySize;
  _initialSizeSecondary = secondarySize;
  _currentSize = primarySize;
  _currentSizeSecondary = secondarySize;
  _offset = 0;
  _offsetSecondary = 0;
  _cycleAllocations = 0;
  _cycleAllocationsSecondary = 0;
  _spillsSize = 0;
  _spillsSizeSecondary = 0;
}

void Workspace::init(LongType primaryBytes, LongType secondaryBytes) {
  if (primaryBytes < 0 || secondaryBytes < 0) {
    THROW_EXCEPTION("Vulkan Workspace sizes must be non-negative");
  }

  if (_currentSize < primaryBytes) {
    if (_externalized) {
      std::string message =
          "Cannot expand an external Vulkan DEVICE workspace; fixed capacity: [" +
          std::to_string(_currentSize) + "], requested: [" +
          std::to_string(primaryBytes) + "]";
      THROW_EXCEPTION(message.c_str());
    }

    _currentSize = primaryBytes;
    if (_deviceId < 0) _deviceId = currentVulkanDevice();
  }

  if (_currentSizeSecondary < secondaryBytes) {
    if (_externalized) {
      THROW_EXCEPTION("Cannot expand an external Vulkan HOST workspace");
    }

    char* replacement = reinterpret_cast<char*>(
        allocateHost(secondaryBytes + CANARY_SIZE));
    std::memset(replacement, 0, static_cast<size_t>(secondaryBytes));
    std::memset(replacement + secondaryBytes, CANARY_BYTE, CANARY_SIZE);

    if (_allocatedHost) std::free(_ptrHost);
    _ptrHost = replacement;
    _currentSizeSecondary = secondaryBytes;
    _allocatedHost = true;
    _canaryEnabled = true;
  }
}

void Workspace::expandBy(LongType primaryBytes, LongType secondaryBytes) {
  init(_currentSize + primaryBytes, _currentSizeSecondary + secondaryBytes);
}

void Workspace::expandTo(LongType primaryBytes, LongType secondaryBytes) {
  init(primaryBytes, secondaryBytes);
}

void Workspace::freeSpills() {
  _spillsSize = 0;
  _spillsSizeSecondary = 0;

  // _spills owns every Vulkan DEVICE allocation, including allocations that
  // fit the conceptual workspace capacity. They cannot be represented as
  // subranges of one opaque DEVICE_LOCAL token.
  if (!_spills.empty()) {
    auto& pool = graph::VulkanMemoryPool::getInstance();
    for (void* ptr : _spills) {
      if (!pool.freeSynchronized(ptr)) {
        THROW_EXCEPTION("Vulkan Workspace lost ownership of a DEVICE allocation");
      }
    }
    _spills.clear();
  }

  for (void* ptr : _spillsSecondary) std::free(ptr);
  _spillsSecondary.clear();
}

Workspace::~Workspace() {
  if (_allocatedHost && !_externalized) std::free(_ptrHost);

  // Destructors must not throw. Pool shutdown may already have reclaimed an
  // allocation, so best-effort synchronized release is appropriate here.
  if (!_spills.empty()) {
    auto& pool = graph::VulkanMemoryPool::getInstance();
    for (void* ptr : _spills) pool.freeSynchronized(ptr);
    _spills.clear();
  }
  for (void* ptr : _spillsSecondary) std::free(ptr);
  _spillsSecondary.clear();
}

LongType Workspace::getUsedSize() { return getCurrentOffset(); }

LongType Workspace::getCurrentSize() { return _currentSize; }

LongType Workspace::getCurrentOffset() { return _offset.load(); }

void* Workspace::allocateBytes(LongType numBytes) {
  return allocateBytes(HOST, numBytes);
}

LongType Workspace::getAllocatedSize() {
  return getCurrentSize() + getSpilledSize();
}

void Workspace::scopeIn() {
  freeSpills();
  init(_cycleAllocations.load(), _cycleAllocationsSecondary.load());
  _cycleAllocations = 0;
  _cycleAllocationsSecondary = 0;
}

void Workspace::scopeOut() {
  if (_canaryEnabled && Environment::getInstance().isDebugAndVerbose()) {
    const LongType corruptedAt = checkCanary();
    if (corruptedAt >= 0) {
      sd_printf("Vulkan Workspace HOST canary corrupted at offset %lld "
                "(workspace size %lld, host ptr %p)\n",
                corruptedAt, _currentSizeSecondary, _ptrHost);
      std::memset(_ptrHost + _currentSizeSecondary, CANARY_BYTE, CANARY_SIZE);
    }
  }

  _offset = 0;
  _offsetSecondary = 0;
}

void Workspace::enableCanary() {
  if (_allocatedHost && _ptrHost != nullptr) {
    std::memset(_ptrHost + _currentSizeSecondary, CANARY_BYTE, CANARY_SIZE);
    _canaryEnabled = true;
  }
}

LongType Workspace::checkCanary() const {
  if (!_canaryEnabled || !_allocatedHost || _ptrHost == nullptr) return -1;

  const auto* canary = reinterpret_cast<const unsigned char*>(
      _ptrHost + _currentSizeSecondary);
  for (LongType i = 0; i < CANARY_SIZE; ++i) {
    if (canary[i] != CANARY_BYTE) return i;
  }
  return -1;
}

LongType Workspace::getSpilledSize() { return _spillsSize.load(); }

void* Workspace::allocateBytes(MemoryType type, LongType numBytes) {
  if (numBytes < 1) {
    std::string message =
        "Number of workspace bytes must be positive; requested: [" +
        std::to_string(numBytes) + "]";
    THROW_EXCEPTION(message.c_str());
  }

  switch (type) {
    case HOST: {
      _cycleAllocationsSecondary += numBytes;
      std::lock_guard<std::mutex> allocationLock(_mutexAllocation);

      const LongType oldOffset = _offsetSecondary.load();
      if (oldOffset + numBytes <= _currentSizeSecondary) {
        if (_ptrHost == nullptr) {
          THROW_EXCEPTION("Vulkan HOST workspace has no host allocation");
        }
        _offsetSecondary = oldOffset + numBytes;
        return static_cast<void*>(_ptrHost + oldOffset);
      }

      void* ptr = allocateHost(numBytes + SD_ALLOC_PADDING);
      {
        std::lock_guard<std::mutex> spillsLock(_mutexSpills);
        _spillsSecondary.push_back(ptr);
      }
      _spillsSizeSecondary += numBytes;
      return ptr;
    }

    case DEVICE: {
      if (_externalized && _ptrDevice != nullptr) {
        THROW_EXCEPTION(
            "External Vulkan DEVICE workspace cannot expose pointer subranges; "
            "use VulkanMemoryPool-owned allocations");
      }

      _cycleAllocations += numBytes;
      std::lock_guard<std::mutex> allocationLock(_mutexAllocation);

      const int deviceId =
          _deviceId >= 0 ? _deviceId : currentVulkanDevice();
      if (_deviceId < 0) _deviceId = deviceId;

      void* ptr = graph::VulkanMemoryPool::getInstance().allocate(
          deviceId, static_cast<VkDeviceSize>(numBytes));
      if (ptr == nullptr) {
        std::string message =
            "Can't allocate Vulkan [DEVICE] memory on device [" +
            std::to_string(deviceId) + "]; size: [" +
            std::to_string(numBytes) + "]";
        THROW_EXCEPTION(message.c_str());
      }

      {
        std::lock_guard<std::mutex> spillsLock(_mutexSpills);
        _spills.push_back(ptr);
      }

      const LongType oldOffset = _offset.load();
      if (oldOffset + numBytes <= _currentSize) {
        _offset = oldOffset + numBytes;
      } else {
        _spillsSize += numBytes;
      }
      return ptr;
    }

    default:
      THROW_EXCEPTION("Unknown MemoryType passed to Vulkan Workspace");
  }
}

Workspace* Workspace::clone() {
  return new Workspace(
      math::sd_max<LongType>(getCurrentSize(), _cycleAllocations.load()),
      math::sd_max<LongType>(getCurrentSecondarySize(),
                             _cycleAllocationsSecondary.load()),
      true);
}

LongType Workspace::getAllocatedSecondarySize() {
  return getCurrentSecondarySize() + getSpilledSecondarySize();
}

LongType Workspace::getCurrentSecondarySize() {
  return _currentSizeSecondary;
}

LongType Workspace::getCurrentSecondaryOffset() {
  return _offsetSecondary.load();
}

LongType Workspace::getSpilledSecondarySize() {
  return _spillsSizeSecondary.load();
}

LongType Workspace::getUsedSecondarySize() {
  return getCurrentSecondaryOffset();
}

}  // namespace memory
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
