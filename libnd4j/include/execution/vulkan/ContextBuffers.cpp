/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <execution/ContextBuffers.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>

#include <utility>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

namespace sd {

ContextBuffers::ContextBuffers()
    : _deviceId(graph::VulkanDeviceManager::currentDeviceId()) {}

ContextBuffers::ContextBuffers(const ContextBuffers& other)
    : _reductionPointer(other._reductionPointer),
      _scalarPointer(other._scalarPointer),
      _allocationPointer(other._allocationPointer),
      _execStream(other._execStream),
      _specialStream(other._specialStream),
      _allocated(false),
      _initialized(other._initialized),
      _deviceId(other._deviceId) {}

ContextBuffers::ContextBuffers(void* rPointer, void* sPointer, void* aPointer,
                               bool isOwner)
    : _reductionPointer(rPointer),
      _scalarPointer(sPointer),
      _allocationPointer(aPointer),
      _allocated(isOwner),
      _deviceId(graph::VulkanDeviceManager::currentDeviceId()) {}

ContextBuffers::~ContextBuffers() { release(); }

ContextBuffers& ContextBuffers::operator=(const ContextBuffers& other) {
  if (this == &other) return *this;

  release();
  _reductionPointer = other._reductionPointer;
  _scalarPointer = other._scalarPointer;
  _allocationPointer = other._allocationPointer;
  _execStream = other._execStream;
  _specialStream = other._specialStream;
  _allocated = false;
  _initialized = other._initialized;
  _deviceId = other._deviceId;
  return *this;
}

ContextBuffers& ContextBuffers::operator=(ContextBuffers&& other) {
  if (this == &other) return *this;

  release();
  _reductionPointer = other._reductionPointer;
  _scalarPointer = other._scalarPointer;
  _allocationPointer = other._allocationPointer;
  _execStream = other._execStream;
  _specialStream = other._specialStream;
  _allocated = other._allocated;
  _initialized = other._initialized;
  _deviceId = other._deviceId;

  other._reductionPointer = nullptr;
  other._scalarPointer = nullptr;
  other._allocationPointer = nullptr;
  other._execStream = nullptr;
  other._specialStream = nullptr;
  other._allocated = false;
  other._initialized = false;
  other._deviceId = -1;
  return *this;
}

void ContextBuffers::initialize() {
  auto& manager = graph::VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    THROW_EXCEPTION("Vulkan device initialization failed");
  }

  _deviceId = graph::VulkanDeviceManager::currentDeviceId();
  if (_deviceId < 0 || _deviceId >= manager.deviceCount()) {
    THROW_EXCEPTION("Invalid current Vulkan device");
  }

  _execStream = graph::VulkanExecutionStream::currentOrDefault(_deviceId);
  _specialStream = graph::VulkanExecutionStream::defaultCopy(_deviceId);
  if (_execStream == nullptr || _specialStream == nullptr) {
    THROW_EXCEPTION("Failed to initialize Vulkan context streams");
  }

  _initialized = true;
}

void ContextBuffers::release() {
  if (_allocated) {
    auto& pool = graph::VulkanMemoryPool::getInstance();
    if (_allocationPointer != nullptr) pool.freeSynchronized(_allocationPointer);
    if (_scalarPointer != nullptr) pool.freeSynchronized(_scalarPointer);
    if (_reductionPointer != nullptr) pool.freeSynchronized(_reductionPointer);
  }

  _reductionPointer = nullptr;
  _scalarPointer = nullptr;
  _allocationPointer = nullptr;
  _execStream = nullptr;
  _specialStream = nullptr;
  _allocated = false;
  _initialized = false;
  _deviceId = -1;
}

void* ContextBuffers::reductionBuffer() {
  if (!_initialized) initialize();
  return _reductionPointer;
}

void* ContextBuffers::scalarBuffer() {
  if (!_initialized) initialize();
  return _scalarPointer;
}

void* ContextBuffers::allocationBuffer() {
  if (!_initialized) initialize();
  return _allocationPointer;
}

void* ContextBuffers::execStream() {
  if (!_initialized) initialize();
  return _execStream;
}

void* ContextBuffers::specialStream() {
  if (!_initialized) initialize();
  return _specialStream;
}

void ContextBuffers::setReductionBuffer(void* pointer) {
  _reductionPointer = pointer;
}

void ContextBuffers::setScalarBuffer(void* pointer) { _scalarPointer = pointer; }

void ContextBuffers::setAllocationBuffer(void* pointer) {
  _allocationPointer = pointer;
}

ErrorReference* ContextBuffers::errorReference() { return &_errorReference; }

void ContextBuffers::triggerOwnership(bool isOwner) { _allocated = isOwner; }

int ContextBuffers::deviceId() { return _deviceId; }

bool ContextBuffers::isInitialized() { return _initialized; }

}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
