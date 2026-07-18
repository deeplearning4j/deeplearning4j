/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <execution/LaunchContext.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/vulkan/VulkanDeviceManager.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <utility>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

namespace {

thread_local std::map<int, sd::ContextBuffers> contextBuffersByDevice;

std::mutex& streamBindingsMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::map<const sd::LaunchContext*, sd::Pointer>& streamBindings() {
  static auto* bindings =
      new std::map<const sd::LaunchContext*, sd::Pointer>();
  return *bindings;
}

sd::ContextBuffers& contextBuffersForCurrentDevice() {
  const int deviceId = sd::graph::VulkanDeviceManager::currentDeviceId();
  return contextBuffersByDevice[deviceId];
}

sd::ContextBuffers& initializedContextBuffers() {
  auto& buffers = contextBuffersForCurrentDevice();
  const int deviceId = sd::graph::VulkanDeviceManager::currentDeviceId();
  if (!buffers.isInitialized() || buffers.deviceId() != deviceId) {
    buffers.initialize();
  }
  return buffers;
}

}  // namespace

namespace sd {
namespace graph {

void bindVulkanExecutionStream(LaunchContext* context, Pointer stream) {
  if (context == nullptr) {
    THROW_EXCEPTION("Cannot bind a Vulkan stream to a null LaunchContext");
  }

  std::lock_guard<std::mutex> lock(streamBindingsMutex());
  if (stream == nullptr) {
    streamBindings().erase(context);
  } else {
    streamBindings()[context] = stream;
  }
}

Pointer vulkanExecutionStream(const LaunchContext* context) {
  if (context == nullptr) return nullptr;

  std::lock_guard<std::mutex> lock(streamBindingsMutex());
  const auto it = streamBindings().find(context);
  return it == streamBindings().end() ? nullptr : it->second;
}

void releaseVulkanExecutionStream(const LaunchContext* context) {
  if (context == nullptr) return;
  std::lock_guard<std::mutex> lock(streamBindingsMutex());
  streamBindings().erase(context);
}

}  // namespace graph

std::vector<LaunchContext*>& LaunchContext::contexts() {
  static auto* managedContexts = new std::vector<LaunchContext*>();
  return *managedContexts;
}

std::mutex LaunchContext::_mutex;
SD_MAP_IMPL<int, std::mutex*> LaunchContext::_deviceMutexes;

LaunchContext::LaunchContext()
    : _deviceID(graph::VulkanDeviceManager::currentDeviceId()) {}

LaunchContext::LaunchContext(Pointer stream, Pointer reductionPointer,
                             Pointer scalarPointer, Pointer allocationPointer)
    : _deviceID(graph::VulkanDeviceManager::currentDeviceId()) {
  Pointer validatedStream = nullptr;
  if (stream != nullptr) {
    auto* executionStream =
        graph::VulkanExecutionStream::fromOpaque(stream, false);
    if (executionStream == nullptr) {
      THROW_EXCEPTION("Invalid Vulkan execution stream");
    }
    if (executionStream->deviceId() != _deviceID) {
      THROW_EXCEPTION("Vulkan execution stream belongs to a different device");
    }
    validatedStream = executionStream;
  }

  auto& buffers = initializedContextBuffers();
  buffers.setReductionBuffer(reductionPointer);
  buffers.setScalarBuffer(scalarPointer);
  buffers.setAllocationBuffer(allocationPointer);

  if (validatedStream != nullptr) {
    graph::bindVulkanExecutionStream(this, validatedStream);
  }
}

LaunchContext::~LaunchContext() {
  graph::releaseVulkanExecutionStream(this);
}

bool LaunchContext::isManagedContext(LaunchContext* contextPtr) {
  std::lock_guard<std::mutex> lock(_mutex);
  const auto& managedContexts = contexts();
  return std::find(managedContexts.begin(), managedContexts.end(), contextPtr) !=
         managedContexts.end();
}

void LaunchContext::operator delete(void* ptr) noexcept {
  if (ptr == nullptr) return;
  auto* context = reinterpret_cast<LaunchContext*>(ptr);
  if (isManagedContext(context)) return;
  ::operator delete(ptr);
}

#ifndef __JAVACPP_HACK__
std::mutex* LaunchContext::deviceMutex() {
  const int deviceId = graph::VulkanDeviceManager::currentDeviceId();
  std::lock_guard<std::mutex> lock(_mutex);
  auto& mutex = _deviceMutexes[deviceId];
  if (mutex == nullptr) mutex = new std::mutex();
  return mutex;
}
#endif

LaunchContext* LaunchContext::defaultContext() {
  auto& manager = graph::VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    THROW_EXCEPTION("Vulkan device initialization failed");
  }

  const int deviceId = graph::VulkanDeviceManager::currentDeviceId();
  if (deviceId < 0 || deviceId >= manager.deviceCount()) {
    THROW_EXCEPTION("Invalid current Vulkan device");
  }

  std::lock_guard<std::mutex> lock(_mutex);
  const size_t requiredSize = static_cast<size_t>(deviceId + 1);
  if (contexts().size() < requiredSize) {
    contexts().resize(requiredSize, nullptr);
  }
  if (_deviceMutexes[deviceId] == nullptr) {
    _deviceMutexes[deviceId] = new std::mutex();
  }
  if (contexts()[deviceId] == nullptr) {
    contexts()[deviceId] = new LaunchContext();
  }
  return contexts()[deviceId];
}

bool LaunchContext::isInitialized() {
  auto it = contextBuffersByDevice.find(
      graph::VulkanDeviceManager::currentDeviceId());
  return it != contextBuffersByDevice.end() && it->second.isInitialized();
}

void LaunchContext::releaseBuffers() {
  auto it = contextBuffersByDevice.find(
      graph::VulkanDeviceManager::currentDeviceId());
  if (it != contextBuffersByDevice.end()) it->second.release();
}

void LaunchContext::swapContextBuffers(ContextBuffers& buffers) {
  contextBuffersForCurrentDevice() = std::move(buffers);
}

ErrorReference* LaunchContext::errorReference() {
  return contextBuffersForCurrentDevice().errorReference();
}

void* LaunchContext::engine() { return _engine; }

}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
