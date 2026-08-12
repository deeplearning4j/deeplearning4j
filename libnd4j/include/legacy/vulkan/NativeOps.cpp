/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <array/ArrayOptions.h>
#include <array/DataBuffer.h>
#include <array/DataTypeUtils.h>
#include <array/InteropDataBuffer.h>
#include <array/NDArray.h>
#include <dsp/NativeOpsDsp.h>
#include <execution/LaunchContext.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/Context.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDeviceDispatch.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <helpers/DebugHelper.h>
#include <helpers/shape.h>
#include <legacy/vulkan/NativeOpsVulkan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>


namespace {

bool validDeviceId(int deviceId) {
  auto& manager = sd::graph::VulkanDeviceManager::getInstance();
  return manager.initialize() && deviceId >= 0 &&
         deviceId < manager.deviceCount();
}

void setNativeError(sd::Status status, const std::string& message) {
  safeSetErrorContext(static_cast<int>(status), message.c_str());
}

sd::LongType opaqueHandleValue(VkCommandPool handle) {
  uint64_t value = 0;
  static_assert(sizeof(handle) <= sizeof(value),
                "VkCommandPool does not fit in NativeOps handle type");
  std::memcpy(&value, &handle, sizeof(handle));
  return static_cast<sd::LongType>(value);
}

bool resolveCopyOwner(void* dst, void* src, int direction, int& ownerDevice,
                      std::string& error) {
  auto& pool = sd::graph::VulkanMemoryPool::getInstance();
  sd::graph::VulkanAllocRecord dstRecord;
  sd::graph::VulkanAllocRecord srcRecord;
  switch (direction) {
    case 0:
      ownerDevice = sd::graph::VulkanDeviceManager::currentDeviceId();
      return validDeviceId(ownerDevice);
    case 1:
      if (!pool.queryRecord(dst, dstRecord)) {
        error = "Vulkan H2D destination is not owned by VulkanMemoryPool";
        return false;
      }
      ownerDevice = dstRecord.deviceId;
      return true;
    case 2:
      if (!pool.queryRecord(src, srcRecord)) {
        error = "Vulkan D2H source is not owned by VulkanMemoryPool";
        return false;
      }
      ownerDevice = srcRecord.deviceId;
      return true;
    case 3:
      if (!pool.queryRecord(dst, dstRecord) ||
          !pool.queryRecord(src, srcRecord)) {
        error = "Vulkan D2D endpoints must both be owned by VulkanMemoryPool";
        return false;
      }
      ownerDevice = dstRecord.deviceId;
      if (srcRecord.deviceId != dstRecord.deviceId) {
        std::string reason;
        if (!sd::graph::VulkanExecutionStream::isCrossDeviceCopySupported(
                srcRecord.deviceId, dstRecord.deviceId, &reason)) {
          error = "Vulkan cross-device copy is unsupported: " + reason;
          return false;
        }
      }
      return true;
    default:
      error = "undefined Vulkan copy direction (expected 0..3)";
      return false;
  }
}

sd::graph::VulkanExecutionStream* resolveStream(
    void* opaque, int ownerDevice, std::string& error) {
  auto* stream =
      opaque != nullptr
          ? sd::graph::VulkanExecutionStream::fromOpaque(opaque, false)
          : sd::graph::VulkanExecutionStream::currentOrDefault(ownerDevice);
  if (stream == nullptr || !stream->isActive()) {
    error = "Vulkan execution stream is unavailable";
    return nullptr;
  }
  if (stream->deviceId() != ownerDevice) {
    error = "Vulkan execution stream does not belong to the allocation owner";
    return nullptr;
  }
  return stream;
}

bool resolveLaunchContextBinding(
    sd::LaunchContext* launchContext, int& deviceId,
    sd::graph::VulkanExecutionStream*& contextOwnedStream) {
  deviceId = -1;
  contextOwnedStream = nullptr;
  if (launchContext == nullptr) return false;

  deviceId = launchContext->getDeviceID();
  if (!validDeviceId(deviceId)) return false;

  void* contextOwned = sd::graph::vulkanExecutionStream(launchContext);
  if (contextOwned == nullptr) return true;

  contextOwnedStream =
      sd::graph::VulkanExecutionStream::fromOpaque(contextOwned, false);
  return contextOwnedStream != nullptr && contextOwnedStream->isActive() &&
         contextOwnedStream->deviceId() == deviceId;
}

sd::graph::VulkanExecutionStream* resolveLaunchContextStream(
    sd::LaunchContext* launchContext) {
  int deviceId = -1;
  sd::graph::VulkanExecutionStream* contextOwnedStream = nullptr;
  if (!resolveLaunchContextBinding(
          launchContext, deviceId, contextOwnedStream)) {
    return nullptr;
  }
  if (contextOwnedStream != nullptr) return contextOwnedStream;

  auto* stream =
      sd::graph::VulkanExecutionStream::currentOrDefault(deviceId);
  return stream != nullptr && stream->isActive() &&
                 stream->deviceId() == deviceId
             ? stream
             : nullptr;
}

sd::graph::VulkanExecutionStream* resolveLaunchContextCopyStream(
    sd::LaunchContext* launchContext) {
  int deviceId = -1;
  sd::graph::VulkanExecutionStream* contextOwnedStream = nullptr;
  if (!resolveLaunchContextBinding(
          launchContext, deviceId, contextOwnedStream)) {
    return nullptr;
  }

  auto* stream = sd::graph::VulkanExecutionStream::defaultCopy(deviceId);
  return stream != nullptr && stream->isActive() &&
                 stream->deviceId() == deviceId
             ? stream
             : nullptr;
}

bool enqueueCopy(void* dst, void* src, VkDeviceSize bytes, int direction,
                 void* opaqueStream, bool synchronize, std::string& error) {
  int ownerDevice = -1;
  if (!resolveCopyOwner(dst, src, direction, ownerDevice, error)) return false;
  auto* stream = resolveStream(opaqueStream, ownerDevice, error);
  if (stream == nullptr) return false;

  bool queued = false;
  auto& pool = sd::graph::VulkanMemoryPool::getInstance();
  switch (direction) {
    case 0: {
      const uint64_t sequence = stream->enqueueHostCallback(
          [dst, src, bytes]() {
            std::memmove(dst, src, static_cast<size_t>(bytes));
          });
      queued = sequence != 0;
      break;
    }
    case 1:
      queued = pool.copyHostToDeviceAsync(dst, src, bytes, stream);
      break;
    case 2:
      queued = pool.copyDeviceToHostAsync(dst, src, bytes, stream);
      break;
    case 3:
      queued = pool.copyDeviceToDeviceAsync(dst, src, bytes, stream);
      break;
    default:
      break;
  }
  if (!queued) {
    error = "Vulkan transfer could not be enqueued on the owning stream";
    return false;
  }
  if (synchronize && !stream->synchronize()) {
    error = "Vulkan transfer stream synchronization failed";
    return false;
  }
  return true;
}

bool enqueueFill(void* dst, int value, VkDeviceSize bytes, void* opaqueStream,
                 bool synchronize, std::string& error) {
  auto& pool = sd::graph::VulkanMemoryPool::getInstance();
  sd::graph::VulkanAllocRecord record;
  if (!pool.queryRecord(dst, record)) {
    error = "Vulkan memset destination is not owned by VulkanMemoryPool";
    return false;
  }
  auto* stream = resolveStream(opaqueStream, record.deviceId, error);
  if (stream == nullptr) return false;
  if (!pool.fillAsync(dst, value, bytes, stream)) {
    error = "Vulkan fill could not be enqueued on the owning stream";
    return false;
  }
  if (synchronize && !stream->synchronize()) {
    error = "Vulkan fill stream synchronization failed";
    return false;
  }
  return true;
}

void destroyTemporaryStreams(
    std::vector<sd::graph::VulkanExecutionStream*>& streams) {
  for (auto* stream : streams) {
    if (stream != nullptr) {
      sd::graph::VulkanExecutionStream::destroy(stream);
    }
  }
  streams.clear();
}

}  // namespace


sd::Status execCustomOp2(sd::Pointer* /*extraPointers*/, sd::LongType hash,
                         OpaqueContext* opContext) {
  if (opContext == nullptr) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan execCustomOp2 received a null operation context");
    return sd::Status::BAD_INPUT;
  }

  auto* stream = resolveLaunchContextStream(opContext->launchContext());
  if (stream == nullptr || !stream->isActive()) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   "Vulkan execCustomOp2 has no active execution stream");
    return sd::Status::KERNEL_FAILURE;
  }

  std::string errorMessage;
  const sd::Status status = sd::graph::VulkanEagerExecutor::execute(
      hash, *opContext, *stream, &errorMessage);
  if (status != sd::Status::OK) {
    if (errorMessage.empty()) {
      errorMessage = "Vulkan eager execution failed without a diagnostic";
    }
    setNativeError(status, errorMessage);
  }
  return status;
}

void setGraphContextCudaContext(OpaqueContext* ptr, void* stream,
                                void* reductionPointer,
                                void* allocationPointer) {
  if (ptr == nullptr) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan graph context binding received a null context");
    return;
  }
  if (reductionPointer != nullptr || allocationPointer != nullptr) {
    setNativeError(
        sd::Status::BAD_INPUT,
        "Vulkan graph context binding does not accept CUDA workspace pointers");
    return;
  }

  try {
    // Match CUDA's ownership boundary: Context creates a LaunchContext that
    // retains the validated external stream. Binding must not mutate TLS.
    ptr->setCudaContext(stream, nullptr, nullptr);
  } catch (const std::exception& error) {
    setNativeError(
        sd::Status::BAD_INPUT,
        std::string("Vulkan graph context binding failed: ") + error.what());
  }
}

void inspectArray(sd::Pointer* /*extraPointers*/, sd::Pointer buffer,
                  sd::LongType* shapeInfo, sd::Pointer specialBuffer,
                  sd::LongType* /*specialShapeInfo*/, sd::Pointer debugInfo) {
  if (shapeInfo == nullptr || debugInfo == nullptr) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan inspectArray requires host shape info and debug output");
    return;
  }

  const sd::LongType length = shape::length(shapeInfo);
  const size_t elementSize =
      sd::DataTypeUtils::sizeOfElement(sd::ArrayOptions::dataType(shapeInfo));
  if (length < 0 || elementSize == 0 ||
      static_cast<uint64_t>(length) >
          std::numeric_limits<size_t>::max() / elementSize) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan inspectArray received an invalid array extent");
    return;
  }

  const size_t bytes = static_cast<size_t>(length) * elementSize;
  void* inspectionBuffer = buffer;
  std::vector<std::max_align_t> hostStorage;

  if (specialBuffer != nullptr) {
    auto& pool = sd::graph::VulkanMemoryPool::getInstance();
    const int deviceId = pool.getDeviceId(specialBuffer);
    if (deviceId < 0) {
      setNativeError(
          sd::Status::BAD_INPUT,
          "Vulkan inspectArray received a device pointer outside the Vulkan pool");
      return;
    }

    const size_t units =
        bytes == 0 ? 0 : (bytes + sizeof(std::max_align_t) - 1) /
                              sizeof(std::max_align_t);
    hostStorage.resize(units);
    inspectionBuffer = bytes == 0 ? nullptr : hostStorage.data();

    if (bytes > 0) {
      auto* stream =
          sd::graph::VulkanExecutionStream::currentOrDefault(deviceId);
      if (stream == nullptr || stream->deviceId() != deviceId ||
          !stream->enqueueCopy(inspectionBuffer, specialBuffer,
                               static_cast<VkDeviceSize>(bytes), 2) ||
          !stream->synchronize()) {
        setNativeError(
            sd::Status::KERNEL_FAILURE,
            "Vulkan inspectArray could not copy device data on its owning stream");
        return;
      }
    }
  } else if (bytes > 0 && inspectionBuffer == nullptr) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan inspectArray received no readable array buffer");
    return;
  }

#ifdef __cpp_exceptions
  try {
#endif
    sd::NDArray array(inspectionBuffer, shapeInfo, nullptr, 0, 0);
    sd::DebugHelper::retrieveDebugStatistics(
        reinterpret_cast<sd::DebugInfo*>(debugInfo), &array);
#ifdef __cpp_exceptions
  } catch (const std::exception& e) {
    setNativeError(sd::Status::KERNEL_FAILURE, e.what());
  }
#endif
}

sd::Pointer mallocHost(sd::LongType memorySize, int /*flags*/) {
  if (memorySize < 0 ||
      static_cast<uint64_t>(memorySize) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 8)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan mallocHost received an invalid allocation size");
    return nullptr;
  }
  void* pointer = std::malloc(static_cast<size_t>(memorySize) + 8);
  if (pointer == nullptr) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   "Vulkan mallocHost allocation failed");
  }
  return pointer;
}

sd::Pointer mallocDevice(sd::LongType memorySize, int deviceId, int /*flags*/) {
  if (memorySize < 0 ||
      static_cast<uint64_t>(memorySize) >
          std::numeric_limits<uint64_t>::max() - 8 ||
      !validDeviceId(deviceId)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan mallocDevice received an invalid size or device");
    return nullptr;
  }
  void* pointer = sd::graph::VulkanMemoryPool::getInstance().allocate(
      deviceId, static_cast<VkDeviceSize>(memorySize) + 8);
  if (pointer == nullptr) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   "Vulkan device allocation failed");
  }
  return pointer;
}

int freeHost(sd::Pointer pointer) {
  std::free(pointer);
  return 1;
}

int freeDevice(sd::Pointer pointer, int deviceId) {
  if (pointer == nullptr) return 1;
  auto& pool = sd::graph::VulkanMemoryPool::getInstance();
  sd::graph::VulkanAllocRecord record;
  if (!pool.queryRecord(pointer, record)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan freeDevice received an unknown allocation");
    return 0;
  }
  if (deviceId >= 0 && deviceId != record.deviceId) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan freeDevice device does not own the allocation");
    return 0;
  }
  auto* stream =
      sd::graph::VulkanExecutionStream::currentOrDefault(record.deviceId);
  if (stream == nullptr || stream->deviceId() != record.deviceId ||
      !stream->retireAllocation(pointer)) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   "Vulkan freeDevice could not enqueue the allocation retire");
    return 0;
  }
  return 1;
}

bool isMemoryPoolEnabled() { return true; }

void setMemoryPoolEnabled(bool enabled) {
  if (!enabled) {
    setNativeError(
        sd::Status::BAD_INPUT,
        "The Vulkan device memory pool is mandatory and cannot be disabled");
  }
}

void getMemoryPoolStats(int deviceId, sd::LongType* usedBytes,
                        sd::LongType* reservedBytes) {
  uint64_t used = 0;
  uint64_t reserved = 0;
  if (!validDeviceId(deviceId)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan memory-pool statistics requested for an invalid device");
  } else {
    // Match CUDA's pool-statistics contract: a valid device whose pool has not
    // allocated yet reports zero used/reserved bytes. Vulkan creates pool state
    // lazily, so absence of that state is not an invalid-device condition.
    (void)sd::graph::VulkanMemoryPool::getInstance().getMemoryPoolStats(
        deviceId, used, reserved);
  }
  if (usedBytes != nullptr) *usedBytes = static_cast<sd::LongType>(used);
  if (reservedBytes != nullptr) {
    *reservedBytes = static_cast<sd::LongType>(reserved);
  }
}

void trimMemoryPool(int deviceId) {
  if (!validDeviceId(deviceId)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan memory-pool trim requested for an invalid device");
    return;
  }
  sd::graph::VulkanMemoryPool::getInstance().trim(deviceId);
}

void trimMemoryPoolOnStream(int deviceId, void* opaqueStream) {
  if (!validDeviceId(deviceId)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan stream-ordered trim requested for an invalid device");
    return;
  }
  std::string error;
  auto* stream = resolveStream(opaqueStream, deviceId, error);
  if (stream == nullptr ||
      stream->enqueueHostCallback([deviceId]() {
        sd::graph::VulkanMemoryPool::getInstance().trim(deviceId);
      }) == 0) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   error.empty()
                       ? "Vulkan stream-ordered memory-pool trim enqueue failed"
                       : error);
  }
}

sd::LongType getPinnedHostBytesUsed() { return 0; }

sd::LongType getPinnedHostBytesLimit() { return 0; }

void setPinnedHostBytesLimit(sd::LongType maxBytes) {
  if (maxBytes != 0) {
    setNativeError(
        sd::Status::BAD_INPUT,
        "Vulkan has no pinned-host failover tier; a nonzero limit is invalid");
  }
}

void setMemoryPoolSoftLimitPercent(int percent) {
  if (percent != 100) {
    setNativeError(
        sd::Status::BAD_INPUT,
        "Vulkan memory placement has no host failover soft limit; use 100 percent");
  }
}

int getMemoryPoolSoftLimitPercent() { return 100; }

void addExcludedFailoverDevice(int /*deviceId*/) {
  setNativeError(
      sd::Status::BAD_INPUT,
      "Vulkan has no host failover device list; allocations fail explicitly");
}

void removeExcludedFailoverDevice(int /*deviceId*/) {}

void clearExcludedFailoverDevices() {}

bool isDeviceExcludedFromFailover(int /*deviceId*/) { return false; }

sd::Pointer createContext() { return nullptr; }

sd::Pointer createStream() {
  const int deviceId = sd::graph::VulkanDeviceManager::currentDeviceId();
  if (!validDeviceId(deviceId)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan createStream has no valid current device");
    return nullptr;
  }
  auto* stream = sd::graph::VulkanExecutionStream::create(deviceId);
  if (stream == nullptr) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   "Vulkan execution-stream creation failed");
  }
  return stream;
}

sd::Pointer createEvent() {
  auto* event = sd::graph::VulkanExecutionEvent::create();
  if (event == nullptr) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   "Vulkan execution-event creation failed");
  }
  return event;
}

int registerEvent(sd::Pointer opaqueEvent, sd::Pointer opaqueStream) {
  auto* event = sd::graph::VulkanExecutionEvent::fromOpaque(opaqueEvent);
  auto* stream =
      sd::graph::VulkanExecutionStream::fromOpaque(opaqueStream, false);
  if (event == nullptr || stream == nullptr || !event->record(stream)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan registerEvent received an invalid event or stream");
    return 0;
  }
  return 1;
}

int destroyEvent(sd::Pointer opaqueEvent) {
  auto* event = sd::graph::VulkanExecutionEvent::fromOpaque(opaqueEvent);
  if (event == nullptr || !sd::graph::VulkanExecutionEvent::destroy(event)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan destroyEvent received an invalid event");
    return 0;
  }
  return 1;
}

int streamSynchronize(sd::Pointer opaqueStream) {
  auto* stream =
      sd::graph::VulkanExecutionStream::fromOpaque(opaqueStream, true);
  if (stream == nullptr || !stream->synchronize()) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan streamSynchronize received an invalid stream or failed");
    return 0;
  }
  return 1;
}

int eventSynchronize(sd::Pointer opaqueEvent) {
  auto* event = sd::graph::VulkanExecutionEvent::fromOpaque(opaqueEvent);
  if (event == nullptr || !event->synchronize()) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan eventSynchronize received an invalid event or failed");
    return 0;
  }
  return 1;
}

int setDevice(int deviceId) {
  auto& manager = sd::graph::VulkanDeviceManager::getInstance();
  if (!manager.initialize() || !manager.setCurrentDevice(deviceId)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan setDevice received an invalid device");
    return 0;
  }
  return 1;
}

void setAvailableDevices(int* devices, int size) {
  auto& manager = sd::graph::VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   "Vulkan device enumeration is unavailable");
    return;
  }
  const int count = manager.deviceCount();
  if (size != count || (count > 0 && devices == nullptr)) {
    setNativeError(
        sd::Status::BAD_INPUT,
        "Vulkan setAvailableDevices cannot hide hardware; pass every enumerated device");
    return;
  }
  std::vector<bool> seen(static_cast<size_t>(count), false);
  for (int i = 0; i < size; ++i) {
    if (devices[i] < 0 || devices[i] >= count || seen[devices[i]]) {
      setNativeError(
          sd::Status::BAD_INPUT,
          "Vulkan setAvailableDevices requires each enumerated device exactly once");
      return;
    }
    seen[devices[i]] = true;
  }
}

int getAvailableDevices() {
  auto& manager = sd::graph::VulkanDeviceManager::getInstance();
  return manager.initialize() ? manager.deviceCount() : 0;
}

bool isPeerAccessSupported(int srcDevice, int dstDevice) {
  return sd::graph::VulkanExecutionStream::isCrossDeviceCopySupported(
      srcDevice, dstDevice, nullptr);
}

sd::LongType getDeviceFreeMemoryDefault() {
  return getDeviceFreeMemory(
      sd::graph::VulkanDeviceManager::currentDeviceId());
}

sd::LongType getDeviceFreeMemory(int device) {
  if (!validDeviceId(device)) return 0;
  return static_cast<sd::LongType>(
      sd::graph::VulkanMemoryPool::getInstance().getFreeMemory(device));
}

sd::LongType getDeviceTotalMemory(int device) {
  if (!validDeviceId(device)) return 0;
  const auto* info =
      sd::graph::VulkanDeviceManager::getInstance().getDeviceInfo(device);
  return info == nullptr
             ? 0
             : static_cast<sd::LongType>(info->totalMemoryBytes);
}

int getDevice() {
  return sd::graph::VulkanDeviceManager::currentDeviceId();
}

int getDeviceId(sd::Pointer ptrToDeviceId) {
  return static_cast<int>(reinterpret_cast<intptr_t>(ptrToDeviceId));
}

int getDeviceMajor(int device) {
  if (!validDeviceId(device)) return 0;
  const auto* info =
      sd::graph::VulkanDeviceManager::getInstance().getDeviceInfo(device);
  return info == nullptr ? 0 : info->vkMajor;
}

int getDeviceMinor(int device) {
  if (!validDeviceId(device)) return 0;
  const auto* info =
      sd::graph::VulkanDeviceManager::getInstance().getDeviceInfo(device);
  return info == nullptr ? 0 : info->vkMinor;
}

const char* getDeviceName(int device) {
  if (!validDeviceId(device)) return "";
  const auto* info =
      sd::graph::VulkanDeviceManager::getInstance().getDeviceInfo(device);
  return info == nullptr ? "" : info->name.c_str();
}

int getDeviceBlockThreshold(int device) {
  if (!validDeviceId(device)) return 0;
  VkPhysicalDeviceProperties properties = {};
  vkGetPhysicalDeviceProperties(
      sd::graph::VulkanDeviceManager::getInstance().getPhysicalDevice(device),
      &properties);
  return static_cast<int>(std::min<uint32_t>(
      properties.limits.maxComputeWorkGroupInvocations,
      static_cast<uint32_t>(std::numeric_limits<int>::max())));
}

int getDeviceSharedThreshold(int device) {
  if (!validDeviceId(device)) return 0;
  VkPhysicalDeviceProperties properties = {};
  vkGetPhysicalDeviceProperties(
      sd::graph::VulkanDeviceManager::getInstance().getPhysicalDevice(device),
      &properties);
  return static_cast<int>(std::min<uint32_t>(
      properties.limits.maxComputeSharedMemorySize,
      static_cast<uint32_t>(std::numeric_limits<int>::max())));
}

int memcpySync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags,
               sd::Pointer reserved) {
  if (size < 0 || flags < 0 || flags > 3 ||
      (size > 0 && (dst == nullptr || src == nullptr)) ||
      static_cast<uint64_t>(size) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan memcpySync received invalid arguments");
    return 0;
  }
  if (size == 0) return 1;
  std::string error;
  if (!enqueueCopy(dst, src, static_cast<VkDeviceSize>(size), flags, reserved,
                   true, error)) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   "Vulkan memcpySync failed: " + error);
    return 0;
  }
  return 1;
}

int memcpyAsync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags,
                sd::Pointer reserved) {
  if (size < 0 || flags < 0 || flags > 3 ||
      (size > 0 && (dst == nullptr || src == nullptr)) ||
      static_cast<uint64_t>(size) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan memcpyAsync received invalid arguments");
    return 0;
  }
  if (size == 0) return 1;
  std::string error;
  if (!enqueueCopy(dst, src, static_cast<VkDeviceSize>(size), flags, reserved,
                   false, error)) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   "Vulkan memcpyAsync failed: " + error);
    return 0;
  }
  return 1;
}

int memsetSync(sd::Pointer dst, int value, sd::LongType size, int /*flags*/,
               sd::Pointer reserved) {
  if (size < 0 || (size > 0 && dst == nullptr)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan memsetSync received invalid arguments");
    return 0;
  }
  if (size == 0) return 1;
  std::string error;
  if (!enqueueFill(dst, value, static_cast<VkDeviceSize>(size), reserved, true,
                   error)) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   "Vulkan memsetSync failed: " + error);
    return 0;
  }
  return 1;
}

int memsetAsync(sd::Pointer dst, int value, sd::LongType size, int /*flags*/,
                sd::Pointer reserved) {
  if (size < 0 || (size > 0 && dst == nullptr)) {
    setNativeError(sd::Status::BAD_INPUT,
                   "Vulkan memsetAsync received invalid arguments");
    return 0;
  }
  if (size == 0) return 1;
  std::string error;
  if (!enqueueFill(dst, value, static_cast<VkDeviceSize>(size), reserved, false,
                   error)) {
    setNativeError(sd::Status::KERNEL_FAILURE,
                   "Vulkan memsetAsync failed: " + error);
    return 0;
  }
  return 1;
}

sd::Pointer lcScalarPointer(OpaqueLaunchContext /*lc*/) { return nullptr; }

sd::Pointer lcReductionPointer(OpaqueLaunchContext /*lc*/) { return nullptr; }

sd::Pointer lcAllocationPointer(OpaqueLaunchContext /*lc*/) { return nullptr; }

sd::Pointer lcExecutionStream(OpaqueLaunchContext lc) {
  return resolveLaunchContextStream(lc);
}

sd::Pointer lcCopyStream(OpaqueLaunchContext lc) {
  return resolveLaunchContextCopyStream(lc);
}

sd::Pointer lcBlasHandle(OpaqueLaunchContext /*lc*/) { return nullptr; }

sd::Pointer lcSolverHandle(OpaqueLaunchContext /*lc*/) { return nullptr; }

void batchSyncToSpecialAsync(OpaqueDataBuffer** buffers, int bufferCount,
                             int streamCount) {
  if (bufferCount == 0) return;
  if (buffers == nullptr || bufferCount < 0 || streamCount <= 0 ||
      streamCount > 64) {
    const std::string message =
        "Vulkan batchSyncToSpecialAsync received invalid buffers or stream count";
    setNativeError(sd::Status::BAD_INPUT, message);
    THROW_EXCEPTION(message.c_str());
  }

  const int deviceId = sd::graph::VulkanDeviceManager::currentDeviceId();
  if (!validDeviceId(deviceId)) {
    const std::string message =
        "Vulkan batchSyncToSpecialAsync has no valid current device";
    setNativeError(sd::Status::BAD_INPUT, message);
    THROW_EXCEPTION(message.c_str());
  }

  std::vector<sd::graph::VulkanExecutionStream*> streams;
  streams.reserve(static_cast<size_t>(streamCount));
  try {
    for (int i = 0; i < streamCount; ++i) {
      auto* stream = sd::graph::VulkanExecutionStream::create(deviceId);
      if (stream == nullptr) {
        THROW_EXCEPTION(
            "Vulkan batchSyncToSpecialAsync could not create every requested stream");
      }
      streams.push_back(stream);
    }

    auto& pool = sd::graph::VulkanMemoryPool::getInstance();
    for (int i = 0; i < bufferCount; ++i) {
      if (buffers[i] == nullptr || buffers[i]->dataBuffer() == nullptr) {
        THROW_EXCEPTION(
            "Vulkan batchSyncToSpecialAsync received a null data buffer");
      }
      auto* dataBuffer = buffers[i]->dataBuffer();
      const size_t bytes = dataBuffer->getLenInBytes();
      if (bytes == 0) continue;
      if (dataBuffer->primary() == nullptr) {
        THROW_EXCEPTION(
            "Vulkan batchSyncToSpecialAsync requires an authoritative host buffer");
      }
      if (dataBuffer->special() == nullptr) dataBuffer->allocateSpecial();
      void* special = dataBuffer->special();
      const int owner = pool.getDeviceId(special);
      if (special == nullptr || owner != deviceId) {
        THROW_EXCEPTION(
            "Vulkan batchSyncToSpecialAsync destination has the wrong device owner");
      }

      auto* stream = streams[static_cast<size_t>(i % streamCount)];
      dataBuffer->waitForSpecialWriteEvent(stream);
      if (!pool.copyHostToDeviceAsync(
              special, dataBuffer->primary(),
              static_cast<VkDeviceSize>(bytes), stream)) {
        THROW_EXCEPTION(
            "Vulkan batchSyncToSpecialAsync could not enqueue an H2D transfer");
      }
      dataBuffer->writeSpecial();
      dataBuffer->recordSpecialWriteEvent(stream);
    }

    for (auto* stream : streams) {
      if (!stream->synchronize()) {
        THROW_EXCEPTION(
            "Vulkan batchSyncToSpecialAsync stream synchronization failed");
      }
    }
    destroyTemporaryStreams(streams);
  } catch (const std::exception& error) {
    destroyTemporaryStreams(streams);
    setNativeError(sd::Status::KERNEL_FAILURE, error.what());
    throw;
  }
}

void dbAsyncCrossDeviceCopy(OpaqueDataBuffer* dstBuffer,
                            OpaqueDataBuffer* srcBuffer, void* dstStream) {
  if (dstBuffer == nullptr || srcBuffer == nullptr ||
      dstBuffer->dataBuffer() == nullptr ||
      srcBuffer->dataBuffer() == nullptr) {
    const std::string message =
        "Vulkan dbAsyncCrossDeviceCopy received a null data buffer";
    setNativeError(sd::Status::BAD_INPUT, message);
    THROW_EXCEPTION(message.c_str());
  }

  auto* dst = dstBuffer->dataBuffer();
  auto* src = srcBuffer->dataBuffer();
  if (!src->isSpecialActual()) {
    const std::string message =
        "Vulkan dbAsyncCrossDeviceCopy source device data is not authoritative";
    setNativeError(sd::Status::BAD_INPUT, message);
    THROW_EXCEPTION(message.c_str());
  }

  void* dstPointer = dst->special();
  void* srcPointer = src->special();
  const size_t bytes = src->getLenInBytes();
  if (bytes == 0) return;
  if (dstPointer == nullptr || srcPointer == nullptr ||
      dst->getLenInBytes() < bytes) {
    const std::string message =
        "Vulkan dbAsyncCrossDeviceCopy requires allocated, non-truncating device buffers";
    setNativeError(sd::Status::BAD_INPUT, message);
    THROW_EXCEPTION(message.c_str());
  }

  auto& pool = sd::graph::VulkanMemoryPool::getInstance();
  const int srcDevice = pool.getDeviceId(srcPointer);
  const int dstDevice = pool.getDeviceId(dstPointer);
  if (srcDevice < 0 || dstDevice < 0) {
    const std::string message =
        "Vulkan dbAsyncCrossDeviceCopy endpoints lack exact pool ownership";
    setNativeError(sd::Status::BAD_INPUT, message);
    THROW_EXCEPTION(message.c_str());
  }

  std::string reason;
  if (!sd::graph::VulkanExecutionStream::isCrossDeviceCopySupported(
          srcDevice, dstDevice, &reason)) {
    const std::string message =
        "Vulkan dbAsyncCrossDeviceCopy capability error: " + reason;
    setNativeError(sd::Status::KERNEL_FAILURE, message);
    THROW_EXCEPTION(message.c_str());
  }

  auto* stream =
      dstStream == nullptr
          ? sd::graph::VulkanExecutionStream::currentOrDefault(dstDevice)
          : sd::graph::VulkanExecutionStream::fromOpaque(dstStream, false);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != dstDevice) {
    const std::string message =
        "Vulkan dbAsyncCrossDeviceCopy requires a destination-owned stream";
    setNativeError(sd::Status::BAD_INPUT, message);
    THROW_EXCEPTION(message.c_str());
  }

  if (!pool.copyDeviceToDeviceAsync(
          dstPointer, srcPointer, static_cast<VkDeviceSize>(bytes), stream)) {
    const std::string message =
        "Vulkan dbAsyncCrossDeviceCopy external-memory transfer enqueue failed";
    setNativeError(sd::Status::KERNEL_FAILURE, message);
    THROW_EXCEPTION(message.c_str());
  }

  dst->writeSpecial();
  dst->recordSpecialWriteEvent(stream);
}

// CUDA's device backend has no host-ISA admission tier: once its runtime and
// device initialize, the backend itself satisfies both requirement levels.
// Vulkan has the same device-runtime contract.
int binaryLevel() { return 0; }

int optimalLevel() { return 0; }

bool isMinimalRequirementsMet() {
  auto& manager = sd::graph::VulkanDeviceManager::getInstance();
  return manager.initialize() && manager.deviceCount() > 0;
}

bool isOptimalRequirementsMet() { return isMinimalRequirementsMet(); }

int dspSyncStream(sd::Pointer streamPtr) {
  const int deviceId = sd::graph::VulkanDeviceManager::currentDeviceId();
  auto* source =
      streamPtr != nullptr
          ? sd::graph::VulkanExecutionStream::fromOpaque(streamPtr, false)
          : sd::graph::VulkanExecutionStream::currentOrDefault(deviceId);
  if (source == nullptr || !source->isActive()) return -1;

  // CUDA records an event on the source stream and makes the default stream
  // wait. Vulkan streams for one logical device submit through one canonical
  // VkQueue, so submitting this marker establishes the identical ordering:
  // every later default-stream submission follows all source work queued here.
  return source->enqueueHostCallback([]() {}) != 0 ? 0 : -1;
}

sd::Pointer dspGetExecutionStream(sd::Pointer planHandle) {
  if (planHandle == nullptr) return nullptr;
  auto* plan =
      reinterpret_cast<sd::graph::NativeDynamicShapePlan*>(planHandle);
  return reinterpret_cast<sd::Pointer>(plan->getExecutionStream());
}

sd::Pointer dspGetDefaultStream() {
  return sd::graph::VulkanExecutionStream::defaultExecution(
      sd::graph::VulkanDeviceManager::currentDeviceId());
}

sd::LongType vulkanGetThreadCommandPoolHandle(int deviceId) {
  if (!validDeviceId(deviceId)) return 0;
  auto* context = sd::graph::VulkanDeviceContext::getContext(deviceId);
  if (context == nullptr) return 0;
  return opaqueHandleValue(context->getThreadCommandPool());
}

sd::LongType vulkanGetTimelineValue(int deviceId) {
  if (!validDeviceId(deviceId)) return 0;
  auto* context = sd::graph::VulkanDeviceContext::getContext(deviceId);
  if (context == nullptr || !context->hasTimelineSemaphore()) return 0;
  return static_cast<sd::LongType>(context->queryTimeline());
}

bool vulkanHasDedicatedTransferQueue(int deviceId) {
  if (!validDeviceId(deviceId)) return false;
  auto* context = sd::graph::VulkanDeviceContext::getContext(deviceId);
  if (context == nullptr) return false;
  const auto& caps = context->caps();
  return caps.transferQueueFamily != UINT32_MAX &&
         caps.transferQueueFamily != caps.computeQueueFamily;
}

bool dbIsPrimaryActual(OpaqueDataBuffer* dataBuffer) {
  if (dataBuffer == nullptr || dataBuffer->dataBuffer() == nullptr) return true;
  return dataBuffer->dataBuffer()->isPrimaryActual();
}

bool dbIsSpecialActual(OpaqueDataBuffer* dataBuffer) {
  return dataBuffer != nullptr && dataBuffer->dataBuffer() != nullptr &&
         dataBuffer->dataBuffer()->isSpecialActual();
}

void dbAllocateSpecial(OpaqueDataBuffer* dataBuffer) {
  if (dataBuffer == nullptr || dataBuffer->dataBuffer() == nullptr) {
    setNativeError(sd::Status::BAD_INPUT,
                   "dbAllocateSpecial received a null Vulkan data buffer");
    return;
  }
#ifdef __cpp_exceptions
  try {
#endif
    dataBuffer->dataBuffer()->allocateSpecial();
#ifdef __cpp_exceptions
  } catch (const std::exception& e) {
    setNativeError(sd::Status::KERNEL_FAILURE, e.what());
  }
#endif
}

int vulkanGetPoolBlockId(sd::Pointer ptr, int deviceId) {
  if (ptr == nullptr || !validDeviceId(deviceId)) return -1;
  auto& pool = sd::graph::VulkanMemoryPool::getInstance();
  if (pool.getDeviceId(ptr) != deviceId) return -1;
  return pool.getBlockId(ptr);
}

int vulkanGetAllocationMemoryPropertyFlags(sd::Pointer pointer, int deviceId) {
  if (pointer == nullptr || !validDeviceId(deviceId)) return -1;
  return sd::graph::VulkanMemoryPool::getInstance()
      .getAllocationMemoryPropertyFlags(pointer, deviceId);
}

int vulkanGetRetireListPendingCount(int deviceId) {
  if (!validDeviceId(deviceId)) return 0;
  return sd::graph::VulkanMemoryPool::getInstance()
      .getRetireListPendingCount(deviceId);
}

void vulkanShutdown() {
  sd::graph::VulkanExecutionEvent::destroyAll();
  sd::graph::VulkanExecutionStream::destroyAll();
  sd::graph::VulkanMemoryPool::getInstance().shutdown();
  sd::graph::VulkanDeviceContext::destroyAll();
  sd::graph::VulkanDeviceManager::getInstance().shutdown();
}


#endif  // SD_VULKAN && HAVE_VULKAN
