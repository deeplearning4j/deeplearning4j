/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Vulkan NativeOps DSP runtime bridge.
 */
#include <dsp/NativeOpsDsp.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/ReplayCacheManager.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <graph/vulkan/VulkanReplayHandle.h>
#if defined(HAVE_MLIR) && HAVE_MLIR
#include <graph/vulkan/VulkanPipelineCache.h>
#endif

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

using namespace sd;
using namespace sd::graph;


namespace {

NativeDynamicShapePlan* planOf(sd::Pointer handle) {
  return reinterpret_cast<NativeDynamicShapePlan*>(handle);
}

bool validSegment(const NativeDynamicShapePlan* plan, int segmentIdx) {
  return plan != nullptr && segmentIdx >= 0 &&
         segmentIdx < static_cast<int>(plan->getSegments().size());
}

void setRuntimeError(int code, const std::string& message) {
  auto* context = sd::LaunchContext::defaultContext();
  if (context == nullptr || context->errorReference() == nullptr) return;
  context->errorReference()->setErrorCode(code);
  context->errorReference()->setErrorMessage(message);
}

template <typename Fn>
void forEachReplayHandle(const GraphSegment& segment, Fn&& fn) {
  if (segment.exec.replayHandle) fn(segment.exec.replayHandle.get());
  for (const auto& handle :
       segment.exec.compositeReplaySchedule.mergedReplayHandles) {
    if (handle) fn(handle.get());
  }
  for (const auto& handle :
       segment.exec.compositeReplaySchedule.compositeReplayHandles) {
    if (handle) fn(handle.get());
  }
}

ReplayCacheDeviceKey cacheKey(int deviceType, int deviceIndex) {
  return ReplayCacheDeviceKey::fromDeviceManager(
      static_cast<sd::modelparallel::DeviceType>(deviceType), deviceIndex);
}

}  // namespace

int getPlanSegmentReplayState(sd::Pointer planHandle, int segmentIdx) {
  auto* plan = planOf(planHandle);
  if (!validSegment(plan, segmentIdx)) return -1;
  const auto& segment = plan->getSegments()[segmentIdx];
  if (plan->hasCompositeHandles(segment)) {
    return static_cast<int>(ReplayState::READY);
  }
  return segment.exec.replayHandle
             ? static_cast<int>(segment.exec.replayHandle->getState())
             : -1;
}

int getPlanSegmentReplayCount(sd::Pointer planHandle, int segmentIdx) {
  auto* plan = planOf(planHandle);
  if (!validSegment(plan, segmentIdx)) return 0;
  const auto& segment = plan->getSegments()[segmentIdx];

  int compositeReplayCount = 0;
  for (const auto& handle :
       segment.exec.compositeReplaySchedule.mergedReplayHandles) {
    if (handle && handle->isReady()) {
      compositeReplayCount += handle->getStatistics().replayCount;
    }
  }
  for (const auto& handle :
       segment.exec.compositeReplaySchedule.compositeReplayHandles) {
    if (handle && handle->isReady()) {
      compositeReplayCount += handle->getStatistics().replayCount;
    }
  }
  if (compositeReplayCount > 0) return compositeReplayCount;
  return segment.exec.replayHandle
             ? segment.exec.replayHandle->getStatistics().replayCount
             : 0;
}

const char* getPlanSegmentTrackedPointers(sd::Pointer planHandle,
                                          int segmentIdx) {
  thread_local std::string result;
  auto* plan = planOf(planHandle);
  if (!validSegment(plan, segmentIdx)) {
    result = "[]";
    return result.c_str();
  }

  const auto& segment = plan->getSegments()[segmentIdx];
  if (!segment.exec.replayHandle) {
    result = "[]";
    return result.c_str();
  }

  const auto& captured =
      segment.exec.replayHandle->getCapturedExternalAddresses();
  std::ostringstream json;
  json << '[';
  for (size_t i = 0; i < captured.size(); ++i) {
    if (i != 0) json << ',';
    void* current = nullptr;
    if (i < static_cast<size_t>(plan->getNumExternalInputs())) {
      auto* input = plan->getLastExternalInput(static_cast<int>(i));
      if (input != nullptr) current = input->specialBuffer();
    }
    json << "{\"inputIdx\":" << i
         << ",\"capturedAddr\":\"0x" << std::hex
         << reinterpret_cast<std::uintptr_t>(captured[i])
         << "\",\"currentAddr\":\"0x"
         << reinterpret_cast<std::uintptr_t>(current) << std::dec
         << "\",\"match\":"
         << (captured[i] == current ? "true" : "false") << '}';
  }
  json << ']';
  result = json.str();
  return result.c_str();
}

int getPlanSegmentNumCaptureBuffers(sd::Pointer planHandle, int segmentIdx) {
  auto* plan = planOf(planHandle);
  if (!validSegment(plan, segmentIdx)) return 0;
  int count = 0;
  forEachReplayHandle(plan->getSegments()[segmentIdx],
                      [&](const GraphReplayHandle* handle) {
                        if (handle->getWorkspacePtr() != nullptr &&
                            handle->getWorkspaceBytes() > 0) {
                          ++count;
                        }
                      });
  return count;
}

const char* getPlanSegmentCaptureBuffersJson(sd::Pointer planHandle,
                                             int segmentIdx) {
  thread_local std::string result;
  auto* plan = planOf(planHandle);
  if (!validSegment(plan, segmentIdx)) {
    result = "[]";
    return result.c_str();
  }

  std::ostringstream json;
  json << '[';
  bool first = true;
  forEachReplayHandle(plan->getSegments()[segmentIdx],
                      [&](const GraphReplayHandle* handle) {
                        if (handle->getWorkspacePtr() == nullptr ||
                            handle->getWorkspaceBytes() == 0) {
                          return;
                        }
                        if (!first) json << ',';
                        first = false;
                        json << "{\"kind\":\"workspace\",\"deviceId\":"
                             << handle->getDeviceId()
                             << ",\"bytes\":" << handle->getWorkspaceBytes()
                             << ",\"address\":\"0x" << std::hex
                             << reinterpret_cast<std::uintptr_t>(
                                    handle->getWorkspacePtr())
                             << std::dec << "\",\"backend\":\""
                             << handle->backendName() << "\"}";
                      });
  json << ']';
  result = json.str();
  return result.c_str();
}

int getPlanSegmentNumHostPointers(sd::Pointer planHandle, int segmentIdx) {
  auto* plan = planOf(planHandle);
  if (!validSegment(plan, segmentIdx)) return 0;
  int count = 0;
  forEachReplayHandle(plan->getSegments()[segmentIdx],
                      [&](const GraphReplayHandle* handle) {
                        count += static_cast<int>(
                            handle->getCapturedHostPtrs().size());
                      });
  return count;
}

bool isReplayCacheEnabled() {
  return ReplayCacheManager::getInstance().isEnabled();
}

int getReplayCacheHits() {
  return ReplayCacheManager::getInstance().getCacheHits();
}

int getReplayCacheMisses() {
  return ReplayCacheManager::getInstance().getCacheMisses();
}

void clearReplayCache() {
  ReplayCacheManager::getInstance().clearAll();
}

const char* getReplayCacheDir() {
  thread_local std::string result;
  result = ReplayCacheManager::getInstance().getCacheDir();
  return result.c_str();
}

const char* getReplayCacheDeviceStatsJson() {
  thread_local std::string result;
  result = ReplayCacheManager::getInstance().getDeviceCacheStatsJson();
  return result.c_str();
}

int getReplayCacheDeviceEntryCount(int deviceType, int deviceIndex) {
  return ReplayCacheManager::getInstance().getDeviceCacheEntryCount(
      cacheKey(deviceType, deviceIndex));
}

void clearReplayCacheForDevice(int deviceType, int deviceIndex) {
  ReplayCacheManager::getInstance().clearDevice(
      cacheKey(deviceType, deviceIndex));
}

bool migrateReplayCache(int fromType, int fromIdx, int toType, int toIdx) {
  return ReplayCacheManager::getInstance().migrateDeviceCache(
      cacheKey(fromType, fromIdx), cacheKey(toType, toIdx));
}

int pruneStaleReplayCacheDevices() {
  return ReplayCacheManager::getInstance().pruneStaleDevices();
}

int loadReplayCacheForDevice(sd::Pointer planHandle, int deviceType,
                             int deviceIndex) {
  if (planHandle == nullptr) return -1;
  return ReplayCacheManager::getInstance().loadAllForDevice(
      cacheKey(deviceType, deviceIndex));
}

const char* getReplayCachedDevicesJson() {
  thread_local std::string result;
  result = ReplayCacheManager::getInstance().getCachedDevicesJson();
  return result.c_str();
}

sd::Pointer dspCreateTestStream() {
  auto& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    setRuntimeError(-1, "dspCreateTestStream: Vulkan is unavailable");
    return nullptr;
  }
  auto* stream =
      VulkanExecutionStream::create(VulkanDeviceManager::currentDeviceId());
  if (stream == nullptr) {
    setRuntimeError(-1, "dspCreateTestStream: stream creation failed");
  }
  return reinterpret_cast<sd::Pointer>(stream);
}

void dspDestroyTestStream(sd::Pointer streamPtr) {
  if (streamPtr == nullptr) return;
  auto* stream = VulkanExecutionStream::fromOpaque(streamPtr, false);
  if (stream == nullptr || !VulkanExecutionStream::destroy(stream)) {
    setRuntimeError(-1, "dspDestroyTestStream: invalid or active stream");
  }
}

int dspWriteDeviceBufferOnDefaultStream(sd::Pointer planHandle, int extIdx,
                                        sd::Pointer srcHost,
                                        long long numBytes) {
  if (planHandle == nullptr || srcHost == nullptr || numBytes < 0) return -1;
  return planOf(planHandle)->writeDeviceBufferOnDefaultStream(
      extIdx, reinterpret_cast<void*>(srcHost), numBytes);
}

int dspWriteDeviceBufferOnExplicitStream(sd::Pointer planHandle, int extIdx,
                                         sd::Pointer srcHost,
                                         long long numBytes,
                                         sd::Pointer streamPtr) {
  if (planHandle == nullptr || srcHost == nullptr || streamPtr == nullptr ||
      numBytes < 0) {
    return -1;
  }
  auto* stream = VulkanExecutionStream::fromOpaque(streamPtr, false);
  if (stream == nullptr) return -1;
  return planOf(planHandle)->writeDeviceBufferOnExplicitStream(
      extIdx, reinterpret_cast<void*>(srcHost), numBytes, stream);
}

int dspIsExtInputDeviceAuthoritative(sd::Pointer planHandle, int extIdx) {
  return planHandle != nullptr
             ? planOf(planHandle)->isExtInputDeviceAuthoritative(extIdx)
             : 0;
}

bool isTritonAvailable() {
#if defined(HAVE_TRITON) && HAVE_TRITON && defined(HAVE_MLIR) && HAVE_MLIR
  auto& manager = VulkanDeviceManager::getInstance();
  return manager.initialize() && manager.deviceCount() > 0;
#else
  return false;
#endif
}

sd::LongType getTritonKernelLaunchCount() {
#if defined(HAVE_MLIR) && HAVE_MLIR
  return static_cast<sd::LongType>(
      VulkanPipelineCache::totalKernelLaunches());
#else
  return 0;
#endif
}

sd::LongType getTritonCacheHitCount() {
#if defined(HAVE_MLIR) && HAVE_MLIR
  return static_cast<sd::LongType>(VulkanPipelineCache::totalCacheHits());
#else
  return 0;
#endif
}

void resetTritonCounters() {
#if defined(HAVE_MLIR) && HAVE_MLIR
  VulkanPipelineCache::resetCounters();
#endif
}

void invalidateTritonCache() {
#if defined(HAVE_MLIR) && HAVE_MLIR
  VulkanPipelineCache::invalidateAll();
#endif
}

int exportTritonCacheBundle(const char* outputPath) {
  if (outputPath == nullptr || outputPath[0] == '\0') return -1;
  setRuntimeError(
      -1,
      "Vulkan SPIR-V cache bundle export is not implemented for this backend");
  return -1;
}

int importTritonCacheBundle(const char* bundlePath, bool /*validateArch*/) {
  if (bundlePath == nullptr || bundlePath[0] == '\0') return -1;
  setRuntimeError(
      -1,
      "Vulkan SPIR-V cache bundle import is not implemented for this backend");
  return -1;
}

const char* inspectTritonCacheBundle(const char* bundlePath) {
  thread_local std::string result;
  if (bundlePath == nullptr || bundlePath[0] == '\0') {
    result = "{\"error\":\"bundle path is empty\",\"backend\":\"Vulkan\"}";
  } else {
    result =
        "{\"error\":\"Vulkan SPIR-V cache bundle inspection is not "
        "implemented\",\"backend\":\"Vulkan\"}";
  }
  setRuntimeError(-1, result);
  return result.c_str();
}

sd::LongType getBufferPoolPooledBytes(int deviceId) {
  uint64_t bytes = 0;
  uint64_t spans = 0;
  if (!VulkanMemoryPool::getInstance().getReusablePoolStats(
          deviceId, bytes, spans)) {
    return 0;
  }
  return static_cast<sd::LongType>(bytes);
}

int getBufferPoolPooledCount(int deviceId) {
  uint64_t bytes = 0;
  uint64_t spans = 0;
  if (!VulkanMemoryPool::getInstance().getReusablePoolStats(
          deviceId, bytes, spans)) {
    return 0;
  }
  return spans > static_cast<uint64_t>(std::numeric_limits<int>::max())
             ? std::numeric_limits<int>::max()
             : static_cast<int>(spans);
}

sd::LongType getBufferPoolTotalAcquired(int deviceId) {
  uint64_t acquired = 0;
  uint64_t reused = 0;
  if (!VulkanMemoryPool::getInstance().getLifetimeAllocationStats(
          deviceId, acquired, reused)) {
    return 0;
  }
  return static_cast<sd::LongType>(acquired);
}

sd::LongType getBufferPoolTotalReused(int deviceId) {
  uint64_t acquired = 0;
  uint64_t reused = 0;
  if (!VulkanMemoryPool::getInstance().getLifetimeAllocationStats(
          deviceId, acquired, reused)) {
    return 0;
  }
  return static_cast<sd::LongType>(reused);
}

void drainPlanFingerprintRing(sd::Pointer planHandle) {
  if (planHandle != nullptr) planOf(planHandle)->drainFingerprintRingPublic();
}

const char* getPlanFingerprintJson(sd::Pointer planHandle) {
  return planHandle != nullptr ? planOf(planHandle)->getFingerprintJson()
                               : "null";
}


#endif  // SD_VULKAN && HAVE_VULKAN
