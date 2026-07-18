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
#include <array/DataBufferLifecycleTracker.h>
#include <array/NDArray.h>
#include <array/NDArrayLifecycleTracker.h>
#include <execution/LaunchContext.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/Context.h>
#include <graph/OpContextLifecycleTracker.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <helpers/BlasHelper.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/shape.h>
#include <legacy/NativeOps.h>
#include <system/Environment.h>
#include <system/env_functions.h>

#include <atomic>
#include <cstring>
#include <vector>


namespace {
std::atomic<bool> p2pEnabled{true};

bool allPeerPathsSupported() {
  auto& manager = sd::graph::VulkanDeviceManager::getInstance();
  if (!manager.initialize()) return false;
  const int count = manager.deviceCount();
  for (int src = 0; src < count; ++src) {
    for (int dst = 0; dst < count; ++dst) {
      if (!sd::graph::VulkanExecutionStream::isCrossDeviceCopySupported(
              src, dst, nullptr)) {
        return false;
      }
    }
  }
  return count > 0;
}
}  // namespace

void enableDebugMode(bool reallyEnable) {
  sd::Environment::getInstance().setDebug(reallyEnable);
}

void enableVerboseMode(bool reallyEnable) {
  sd::Environment::getInstance().setVerbose(reallyEnable);
}

void setGridLimit(int gridSize) {
  if (gridSize > 0) sd::Environment::getInstance().setMaxMasterThreads(gridSize);
}

int ompGetMaxThreads() {
  return sd::Environment::getInstance().maxThreads();
}

int ompGetNumThreads() {
  return sd::Environment::getInstance().maxMasterThreads();
}

void setOmpNumThreads(int threads) {
  if (threads > 0) {
    auto& environment = sd::Environment::getInstance();
    environment.setMaxThreads(threads);
    environment.setMaxMasterThreads(threads);
  }
}

void setOmpMinThreads(int threads) {
  if (threads > 0) sd::Environment::getInstance().setMaxThreads(threads);
}

bool isExperimentalEnabled() {
  return sd::Environment::getInstance().isExperimentalBuild();
}

void initializeDevicesAndFunctions() {
  auto& manager = sd::graph::VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    THROW_EXCEPTION("initializeDevicesAndFunctions: Vulkan initialization failed");
  }
  for (int device = 0; device < manager.deviceCount(); ++device) {
    if (sd::graph::VulkanDeviceContext::getContext(device) == nullptr) {
      THROW_EXCEPTION("initializeDevicesAndFunctions: Vulkan device context initialization failed");
    }
  }
}

void initializeShapeCache() {
  sd::ConstantShapeHelper::initializeEarly();
}

void initializeTadCache() {
  sd::ConstantTadHelper::getInstance();
}

void initializeFunctions(sd::Pointer* functions) {
  sd::BlasHelper::getInstance().initializeDeviceFunctions(functions);
}

void checkP2P() {
  p2pEnabled.store(allPeerPathsSupported(), std::memory_order_release);
}

void enableP2P(bool enable) {
  p2pEnabled.store(enable && allPeerPathsSupported(), std::memory_order_release);
}

bool isP2PAvailable() {
  return p2pEnabled.load(std::memory_order_acquire) && allPeerPathsSupported();
}

sd::Pointer getConstantSpace() {
  return sd::ConstantHelper::getInstance().getConstantSpace();
}

void tryPointer(sd::Pointer extra, sd::Pointer p, int len) {
  static_cast<void>(extra);
  try {
    if (p == nullptr) {
      THROW_EXCEPTION("tryPointer: pointer is null");
    }
    if (len <= 0) {
      THROW_EXCEPTION("tryPointer: length must be positive");
    }

    auto& pool = sd::graph::VulkanMemoryPool::getInstance();
    sd::graph::VulkanAllocRecord record;
    if (!pool.queryRecord(p, record)) {
      THROW_EXCEPTION("tryPointer: pointer is not owned by the Vulkan memory pool");
    }

    const auto bytes = static_cast<VkDeviceSize>(len);
    if (bytes > record.logicalSize) {
      THROW_EXCEPTION("tryPointer: requested range exceeds the logical allocation size");
    }

    auto* stream =
        sd::graph::VulkanExecutionStream::defaultCopy(record.deviceId);
    if (stream == nullptr || !stream->isActive()) {
      THROW_EXCEPTION("tryPointer: Vulkan diagnostic stream is unavailable");
    }

    std::vector<uint8_t> diagnostic(static_cast<size_t>(bytes));
    if (!stream->enqueueCopy(diagnostic.data(), p, bytes, 2) ||
        !stream->synchronize()) {
      THROW_EXCEPTION("tryPointer: Vulkan device read or synchronization failed");
    }
  } catch (const std::exception& e) {
    auto* context = sd::LaunchContext::defaultContext();
    if (context != nullptr && context->errorReference() != nullptr) {
      context->errorReference()->setErrorCode(1);
      context->errorReference()->setErrorMessage(e.what());
    }
  }
}

int memcpyConstantAsync(sd::LongType dst, sd::Pointer src, sd::LongType size,
                        int flags, sd::Pointer reserved) {
  if (dst < 0 || size < 0 || src == nullptr) return 0;
  const int device = sd::graph::VulkanDeviceManager::currentDeviceId();
  auto* stream = sd::graph::VulkanExecutionStream::fromOpaque(reserved);
  if (stream == nullptr || stream->deviceId() != device) return 0;
  auto* constant = static_cast<unsigned char*>(getConstantSpace());
  const int direction = flags == 0 ? 1 : flags;
  return stream->enqueueCopy(constant, src, static_cast<VkDeviceSize>(size),
                             direction, static_cast<VkDeviceSize>(dst), 0)
             ? 1
             : 0;
}

void setShapeBuffer(sd::LongType* inputShapeData, sd::DataType dt,
                    sd::LongType* bufferToSet, char order,
                    int elementWiseStride, bool isEmpty, bool isView) {
  if (inputShapeData == nullptr)
    THROW_EXCEPTION("setShapeBuffer: inputShapeData is null");
  if (bufferToSet == nullptr)
    THROW_EXCEPTION("setShapeBuffer: bufferToSet is null");

  const sd::LongType rank = inputShapeData[0];
  if (rank > SD_MAX_RANK || rank < 0)
    THROW_EXCEPTION("Invalid rank for shape buffer.");

  std::memcpy(bufferToSet, inputShapeData,
              static_cast<size_t>(shape::shapeInfoByteLength(rank)));
  shape::setOrder(bufferToSet, order);
  bufferToSet[shape::shapeInfoLength(rank) - 2] = elementWiseStride;
  sd::ArrayOptions::setDataType(bufferToSet, dt);
  if (isView != sd::ArrayOptions::isView(bufferToSet))
    sd::ArrayOptions::toggleIsView(bufferToSet);
  if (isEmpty != sd::ArrayOptions::isEmpty(bufferToSet))
    sd::ArrayOptions::toggleIsEmpty(bufferToSet);
}

void clearLastError() {
  auto* context = sd::LaunchContext::defaultContext();
  if (context != nullptr && context->errorReference() != nullptr) {
    context->errorReference()->setErrorCode(0);
    context->errorReference()->setErrorMessage("");
  }
}

OpaqueWorkspace createNativeWorkspace(sd::LongType initialSize) {
  return new sd::memory::Workspace(initialSize, 0);
}

void destroyNativeWorkspace(OpaqueWorkspace workspace) { delete workspace; }

void workspaceScopeIn(OpaqueWorkspace workspace) {
  if (workspace != nullptr) workspace->scopeIn();
}

void workspaceScopeOut(OpaqueWorkspace workspace) {
  if (workspace != nullptr) workspace->scopeOut();
}

void attachWorkspaceToContext(OpaqueContext* ctx, OpaqueWorkspace workspace) {
  if (ctx != nullptr) ctx->attachWorkspace(workspace);
}

void detachWorkspaceFromContext(OpaqueContext* ctx) {
  if (ctx != nullptr) ctx->forgetWorkspace();
}

sd::LongType getWorkspaceCurrentOffset(OpaqueWorkspace workspace) {
  return workspace == nullptr ? 0 : workspace->getCurrentOffset();
}

sd::LongType getWorkspaceAllocatedSize(OpaqueWorkspace workspace) {
  return workspace == nullptr ? 0 : workspace->getAllocatedSize();
}

OpaqueMultiBackendWorkspace createNativeMultiBackendWorkspace(
    sd::LongType initialSize, int primaryDeviceType, int primaryDeviceIndex) {
  return sd::memory::createMultiBackendWorkspace(
      initialSize, primaryDeviceType, primaryDeviceIndex);
}

void destroyNativeMultiBackendWorkspace(OpaqueMultiBackendWorkspace handle) {
  sd::memory::destroyMultiBackendWorkspace(handle);
}

void* nativeMbwAllocateBytes(OpaqueMultiBackendWorkspace handle,
                             sd::LongType numBytes) {
  return sd::memory::mbwAllocateBytes(handle, numBytes);
}

void nativeMbwScopeIn(OpaqueMultiBackendWorkspace handle) {
  sd::memory::mbwScopeIn(handle);
}

void nativeMbwScopeOut(OpaqueMultiBackendWorkspace handle) {
  sd::memory::mbwScopeOut(handle);
}

void nativeMbwTransferTo(OpaqueMultiBackendWorkspace handle,
                         int srcDeviceType, int srcDeviceIndex,
                         int dstDeviceType, int dstDeviceIndex) {
  sd::memory::mbwTransferTo(handle, srcDeviceType, srcDeviceIndex,
                            dstDeviceType, dstDeviceIndex);
}

int nativeMbwGetCoherenceState(OpaqueMultiBackendWorkspace handle,
                               int deviceType, int deviceIndex) {
  return sd::memory::mbwGetCoherenceState(handle, deviceType, deviceIndex);
}

sd::LongType nativeMbwGetTotalAllocatedSize(
    OpaqueMultiBackendWorkspace handle) {
  return sd::memory::mbwGetTotalAllocatedSize(handle);
}

void recordJavaNDArrayAllocation(OpaqueNDArray array, long size, int dataType,
                                 bool isView) {
#if defined(SD_GCC_FUNCTRACE)
  std::vector<sd::LongType> shapeVector;
  if (array != nullptr && array->shapeInfo() != nullptr) {
    shapeVector.assign(array->shapeOf(), array->shapeOf() + array->rankOf());
  }
  sd::array::NDArrayLifecycleTracker::getInstance().recordAllocation(
      array, size, static_cast<sd::DataType>(dataType), shapeVector, isView,
      sd::array::NDArraySegment::JAVA);
#endif
}

void recordJavaNDArrayDeallocation(OpaqueNDArray array) {
#if defined(SD_GCC_FUNCTRACE)
  sd::array::NDArrayLifecycleTracker::getInstance().recordDeallocation(array);
#endif
}

void recordJavaDataBufferAllocation(OpaqueDataBuffer* buffer, long size,
                                    int dataType, bool isWorkspace) {
#if defined(SD_GCC_FUNCTRACE)
  if (buffer != nullptr) {
    sd::array::DataBufferLifecycleTracker::getInstance().recordAllocation(
        buffer->primary(), size, static_cast<sd::DataType>(dataType),
        sd::array::BufferType::PRIMARY, buffer, isWorkspace,
        sd::array::DataBufferSegment::JAVA);
  }
#endif
}

void recordJavaDataBufferDeallocation(OpaqueDataBuffer* buffer) {
#if defined(SD_GCC_FUNCTRACE)
  if (buffer != nullptr) {
    sd::array::DataBufferLifecycleTracker::getInstance().recordDeallocation(
        buffer->primary(), sd::array::BufferType::PRIMARY);
  }
#endif
}

void recordJavaOpContextAllocation(
    OpaqueContext* context, int nodeId, long fastpathInSize,
    long fastpathOutSize, long intermediateResultsSize, long handlesSize,
    bool hasWorkspace, bool isFastPath) {
#if defined(SD_GCC_FUNCTRACE)
  if (context != nullptr) {
    sd::graph::OpContextLifecycleTracker::getInstance().recordAllocation(
        context, nodeId, fastpathInSize, fastpathOutSize,
        intermediateResultsSize, handlesSize, hasWorkspace, isFastPath,
        sd::graph::OpContextSegment::JAVA);
  }
#endif
}

void recordJavaOpContextDeallocation(OpaqueContext* context) {
#if defined(SD_GCC_FUNCTRACE)
  if (context != nullptr) {
    sd::graph::OpContextLifecycleTracker::getInstance().recordDeallocation(
        context);
  }
#endif
}

extern std::atomic<size_t> g_opaqueArrayCount;

void deleteNDArray(OpaqueNDArray array) {
  if (array == nullptr) return;
  int device = sd::graph::VulkanDeviceManager::currentDeviceId();
  if (array->specialBuffer() != nullptr) {
    sd::graph::VulkanAllocRecord record;
    if (sd::graph::VulkanMemoryPool::getInstance().queryRecord(
            array->specialBuffer(), record)) {
      device = record.deviceId;
    }
  }
  if (!sd::graph::VulkanExecutionStream::synchronizeDevice(device)) {
    THROW_EXCEPTION("deleteNDArray: Vulkan device synchronization failed");
  }
  g_opaqueArrayCount.fetch_sub(1, std::memory_order_relaxed);
  delete array;
}

sd::LongType getConstantCacheBytes(int deviceId) {
  return sd::ConstantHelper::getInstance().getCachedAmount(deviceId);
}

sd::LongType getTadCacheEntries() {
  return sd::ConstantTadHelper::getInstance().getCachedEntries();
}

sd::LongType getTadCacheBytes() {
  return sd::ConstantTadHelper::getInstance().getCachedBytes();
}

void clearConstantCache() {
  // ConstantHelper owns per-device allocations for the backend lifetime.
}

void clearTadCache() {
  sd::ConstantTadHelper::getInstance().clearCache();
}


#endif  // SD_VULKAN && HAVE_VULKAN
