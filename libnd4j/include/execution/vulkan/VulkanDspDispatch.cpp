/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <graph/DspDeviceDispatch.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <array/DataBuffer.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>

#include <cstdlib>
#include <new>

namespace sd {
namespace graph {
namespace vulkan {
namespace {

constexpr int kDispatchSuccess = 0;
constexpr int kDispatchInvalidArgument = -1;
constexpr int kDispatchInvalidDevice = -2;
constexpr int kDispatchInvalidResource = -3;
constexpr int kDispatchSubmissionFailed = -4;
constexpr int kDispatchAllocationFailed = -5;

thread_local int tlLastDispatchError = kDispatchSuccess;

struct ThreadResources {
  VulkanExecutionStream* gapStream = nullptr;
  VulkanExecutionEvent* completionEvent = nullptr;

  ~ThreadResources() {
    if (completionEvent != nullptr) {
      VulkanExecutionEvent::destroy(completionEvent);
      completionEvent = nullptr;
    }
    if (gapStream != nullptr) {
      VulkanExecutionStream::destroy(gapStream);
      gapStream = nullptr;
    }
  }
};

thread_local ThreadResources tlResources;

int fail(int code) {
  tlLastDispatchError = code;
  return code;
}

void succeed() { tlLastDispatchError = kDispatchSuccess; }

VulkanExecutionStream* resolveStream(void* opaque, int deviceId) {
  if (opaque != nullptr) {
    auto* stream = VulkanExecutionStream::fromOpaque(opaque, false);
    if (stream == nullptr || stream->deviceId() != deviceId) return nullptr;
    return stream;
  }
  return VulkanExecutionStream::defaultExecution(deviceId);
}

const char* dispatchErrorString(int errorCode) {
  switch (errorCode) {
    case kDispatchSuccess:
      return "success";
    case kDispatchInvalidArgument:
      return "invalid Vulkan DSP argument";
    case kDispatchInvalidDevice:
      return "invalid Vulkan device";
    case kDispatchInvalidResource:
      return "unknown or mismatched Vulkan resource";
    case kDispatchSubmissionFailed:
      return "Vulkan queue submission failed";
    case kDispatchAllocationFailed:
      return "Vulkan allocation failed";
    default:
      return "unknown Vulkan DSP error";
  }
}

}  // namespace

void* dspBuffer(NDArray* arr) {
  return arr != nullptr ? arr->specialBuffer() : nullptr;
}

const void* dspBufferConst(const NDArray* arr) {
  return arr != nullptr ? const_cast<NDArray*>(arr)->specialBuffer() : nullptr;
}

int dspClearLastCudaError() {
  const int previous = tlLastDispatchError;
  tlLastDispatchError = kDispatchSuccess;
  return previous;
}

int dspPeekLastCudaError() { return tlLastDispatchError; }

const char* dspCudaErrorString(int errorCode) {
  return dispatchErrorString(errorCode);
}

int dspGetCurrentDevice() {
  auto& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    fail(kDispatchInvalidDevice);
    return -1;
  }
  succeed();
  return VulkanDeviceManager::currentDeviceId();
}

void dspSetCurrentDevice(int deviceId) {
  auto& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize() || !manager.setCurrentDevice(deviceId)) {
    fail(kDispatchInvalidDevice);
    return;
  }
  succeed();
}

int dspGetDeviceCount() {
  auto& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    fail(kDispatchInvalidDevice);
    return 0;
  }
  succeed();
  return manager.deviceCount();
}

bool dspStreamIsCapturing(void* stream) {
  auto* resolved = VulkanExecutionStream::fromOpaque(stream, false);
  return resolved != nullptr && tl_graphExecutionActive &&
         tl_graphCaptureStream == stream;
}

bool dspEndStaleCapture(void* stream, const char* /*label*/) {
  if (!dspStreamIsCapturing(stream)) return false;
  tl_graphExecutionActive = false;
  tl_graphCaptureStream = nullptr;
  tl_dspReplayActive = false;
  succeed();
  return true;
}

void dspDeviceFree(void* ptr) {
  if (ptr == nullptr) return;
  auto& pool = VulkanMemoryPool::getInstance();
  if (pool.getDeviceId(ptr) < 0 || !pool.freeSynchronized(ptr)) {
    fail(kDispatchInvalidResource);
    return;
  }
  succeed();
}

int dspMemcpyH2DAsync(void* dst, const void* src, size_t bytes, void* stream) {
  if (bytes == 0) return kDispatchSuccess;
  if (dst == nullptr || src == nullptr) return fail(kDispatchInvalidArgument);

  auto& pool = VulkanMemoryPool::getInstance();
  const int deviceId = pool.getDeviceId(dst);
  if (deviceId < 0) return fail(kDispatchInvalidResource);

  auto* resolved = resolveStream(stream, deviceId);
  if (resolved == nullptr) return fail(kDispatchInvalidResource);
  if (!resolved->enqueueCopy(dst, src, static_cast<VkDeviceSize>(bytes), 1)) {
    return fail(kDispatchSubmissionFailed);
  }
  succeed();
  return kDispatchSuccess;
}

void dspMemcpyD2DDefaultStream(void* dst, const void* src, size_t bytes) {
  if (bytes == 0) return;
  if (dst == nullptr || src == nullptr) {
    fail(kDispatchInvalidArgument);
    return;
  }

  auto& pool = VulkanMemoryPool::getInstance();
  const int dstDevice = pool.getDeviceId(dst);
  const int srcDevice = pool.getDeviceId(const_cast<void*>(src));
  if (dstDevice < 0 || dstDevice != srcDevice) {
    fail(kDispatchInvalidResource);
    return;
  }

  auto* stream = VulkanExecutionStream::defaultExecution(dstDevice);
  if (stream == nullptr ||
      !stream->enqueueCopy(dst, src, static_cast<VkDeviceSize>(bytes), 3)) {
    fail(kDispatchSubmissionFailed);
    return;
  }
  succeed();
}

bool dspMemPoolTrim(int deviceId, size_t minBytes) {
  auto& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize() || deviceId < 0 ||
      deviceId >= manager.deviceCount()) {
    fail(kDispatchInvalidDevice);
    return false;
  }
  VulkanMemoryPool::getInstance().trim(deviceId, minBytes);
  succeed();
  return true;
}

size_t dspGetDeviceTotalMemory() {
  auto& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    fail(kDispatchInvalidDevice);
    return 0;
  }
  const auto* info =
      manager.getDeviceInfo(VulkanDeviceManager::currentDeviceId());
  if (info == nullptr) {
    fail(kDispatchInvalidDevice);
    return 0;
  }
  succeed();
  return static_cast<size_t>(info->totalMemoryBytes);
}

bool dspHasDeviceMemory() { return true; }

bool dspIsCudaBuild() { return false; }

bool dspIsHostBuild() { return false; }

void dspFreeWorkspaceOnPool(void* ptr) {
  if (ptr == nullptr) return;
  auto& pool = VulkanMemoryPool::getInstance();
  if (pool.getDeviceId(ptr) < 0 || !pool.freeSynchronized(ptr)) {
    fail(kDispatchInvalidResource);
    return;
  }
  succeed();
}

bool dspIsGlobalCaptureWorkspace(void* /*ptr*/) {
  // Vulkan capture workspaces are plan-owned device allocations. There is no
  // process-global workspace identity.
  return false;
}

void* dspCreateEvent() {
  auto* event = VulkanExecutionEvent::create();
  if (event == nullptr) fail(kDispatchAllocationFailed);
  else succeed();
  return event;
}

void dspDestroyEvent(void* event) {
  if (event == nullptr) return;
  auto* resolved = VulkanExecutionEvent::fromOpaque(event);
  if (resolved == nullptr || !VulkanExecutionEvent::destroy(resolved)) {
    fail(kDispatchInvalidResource);
    return;
  }
  succeed();
}

void dspEventRecord(void* event, void* stream) {
  auto* resolvedEvent = VulkanExecutionEvent::fromOpaque(event);
  if (resolvedEvent == nullptr) {
    fail(kDispatchInvalidResource);
    return;
  }
  const int deviceId =
      resolvedEvent->deviceId() >= 0 ? resolvedEvent->deviceId()
                                     : VulkanDeviceManager::currentDeviceId();
  auto* resolvedStream = resolveStream(stream, deviceId);
  if (resolvedStream == nullptr || !resolvedEvent->record(resolvedStream)) {
    fail(kDispatchSubmissionFailed);
    return;
  }
  succeed();
}

void dspStreamWaitEvent(void* stream, void* event) {
  auto* resolvedEvent = VulkanExecutionEvent::fromOpaque(event);
  if (resolvedEvent == nullptr || resolvedEvent->deviceId() < 0) {
    fail(kDispatchInvalidResource);
    return;
  }
  auto* resolvedStream = resolveStream(stream, resolvedEvent->deviceId());
  if (resolvedStream == nullptr || !resolvedStream->waitEvent(*resolvedEvent)) {
    fail(kDispatchSubmissionFailed);
    return;
  }
  succeed();
}

void* dspStreamPtrToValue(void* streamPtr) {
  if (streamPtr == nullptr) return nullptr;
  auto* stream = VulkanExecutionStream::fromOpaque(streamPtr, false);
  if (stream == nullptr) {
    fail(kDispatchInvalidResource);
    return nullptr;
  }
  succeed();
  return stream;
}

void* dspGetExecutionStream() { return tl_dspExecutionStream; }

void dspSetExecutionStream(void* stream) {
  auto* resolved = stream != nullptr
                       ? VulkanExecutionStream::fromOpaque(stream, false)
                       : nullptr;
  if (stream != nullptr && resolved == nullptr) {
    fail(kDispatchInvalidResource);
    return;
  }
  tl_dspExecutionStream = stream;
  VulkanExecutionStream::setCurrent(resolved);
  succeed();
}

void* dspGetGapStream() {
  const int deviceId = VulkanDeviceManager::currentDeviceId();
  if (tlResources.gapStream != nullptr &&
      tlResources.gapStream->deviceId() != deviceId) {
    VulkanExecutionStream::destroy(tlResources.gapStream);
    tlResources.gapStream = nullptr;
  }
  if (tlResources.gapStream == nullptr) {
    tlResources.gapStream = VulkanExecutionStream::create(deviceId);
  }
  if (tlResources.gapStream == nullptr) {
    fail(kDispatchAllocationFailed);
    return nullptr;
  }
  succeed();
  return tlResources.gapStream;
}

void* dspGetGraphCaptureStream() { return tl_graphCaptureStream; }

void dspSetGraphCaptureStream(void* stream) {
  if (stream != nullptr &&
      VulkanExecutionStream::fromOpaque(stream, false) == nullptr) {
    fail(kDispatchInvalidResource);
    return;
  }
  tl_graphCaptureStream = stream;
  succeed();
}

void* dspGetLcDefaultStream() {
  auto* stream = VulkanExecutionStream::defaultExecution(
      VulkanDeviceManager::currentDeviceId());
  if (stream == nullptr) fail(kDispatchInvalidResource);
  else succeed();
  return stream;
}

void dspSyncDefaultStream() {
  auto* stream = VulkanExecutionStream::defaultExecution(
      VulkanDeviceManager::currentDeviceId());
  if (stream == nullptr || !stream->synchronize()) {
    fail(kDispatchSubmissionFailed);
    return;
  }
  succeed();
}

void dspPublishThreadCompletionEvent(void* streamPtr) {
  auto* stream = VulkanExecutionStream::fromOpaque(streamPtr, true);
  if (stream == nullptr) {
    fail(kDispatchInvalidResource);
    return;
  }
  if (tlResources.completionEvent == nullptr) {
    tlResources.completionEvent = VulkanExecutionEvent::create();
  }
  if (tlResources.completionEvent == nullptr ||
      !tlResources.completionEvent->record(stream)) {
    fail(kDispatchSubmissionFailed);
    return;
  }
  auto* defaultStream =
      VulkanExecutionStream::defaultExecution(stream->deviceId());
  if (defaultStream == nullptr ||
      !defaultStream->waitEvent(*tlResources.completionEvent)) {
    fail(kDispatchSubmissionFailed);
    return;
  }
  succeed();
}

bool dspGetReplayActive() { return tl_dspReplayActive; }

void dspSetReplayActive(bool active) { tl_dspReplayActive = active; }

GraphBackend* dspGetNvrtcBackend() { return nullptr; }

GraphBackend* dspGetPtxBackend() { return nullptr; }

void dspCopyCublasHandle(LaunchContext* /*dst*/, LaunchContext* /*src*/) {}

void dspTritonClearFailedCache() {
  // TritonGraphBackend is CUDA-specific. Vulkan replay owns its compiler cache
  // through the Vulkan recorder/catalog boundary.
}

GraphBackend* dspTritonGetBackendIfAvailable() { return nullptr; }

}  // namespace vulkan
}  // namespace graph
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
