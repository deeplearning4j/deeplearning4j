/* ******************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <execution/DataTransferManager.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>

#include <chrono>
#include <cstring>

namespace sd {
namespace modelparallel {

namespace {
graph::VulkanExecutionStream* transferStream(int deviceId) {
  return graph::VulkanExecutionStream::defaultCopy(deviceId);
}
}  // namespace

RingBuffer::~RingBuffer() {
  auto& pool = graph::VulkanMemoryPool::getInstance();
  for (void* buffer : _buffers) {
    if (buffer != nullptr && pool.getDeviceId(buffer) >= 0) {
      pool.freeSynchronized(buffer);
    }
  }
}

void RingBuffer::synchronize() {
  for (int deviceId : _deviceIds) {
    if (deviceId >= 0) {
      graph::VulkanExecutionStream::synchronizeDevice(deviceId);
    }
  }
}

bool DataTransferManager::initialize(const ModelParallelConfig& config) {
  std::lock_guard<std::mutex> queueLock(_queueMutex);
  if (_initialized.load()) return true;

  auto& devices = graph::VulkanDeviceManager::getInstance();
  if (!devices.initialize() || devices.deviceCount() <= 0) return false;

  _config = config;
  _shutdownRequested.store(false);

  {
    std::lock_guard<std::mutex> peerLock(_p2pMutex);
    _p2pEnabled.clear();
    _p2pBandwidth.clear();
  }
  if (config.enableP2P) initializeP2PConnections();

  std::vector<void*> newStaging;
  newStaging.reserve(static_cast<size_t>(_maxConcurrentTransfers));
  for (int i = 0; i < _maxConcurrentTransfers; ++i) {
    void* buffer = allocatePinnedMemory(_stagingBufferSize);
    if (buffer == nullptr) {
      auto& pool = graph::VulkanMemoryPool::getInstance();
      for (void* allocated : newStaging) {
        pool.freeSynchronized(allocated);
      }
      return false;
    }
    newStaging.push_back(buffer);
  }

  {
    std::lock_guard<std::mutex> stagingLock(_stagingMutex);
    for (void* buffer : newStaging) {
      _stagingBuffers.emplace_back(buffer, _stagingBufferSize);
      _stagingInUse.push_back(false);
    }
  }
  _initialized.store(true);
  return true;
}

bool DataTransferManager::setupP2PAccess(int device1, int device2) {
  std::string reason;
  const bool supported =
      graph::VulkanExecutionStream::isCrossDeviceCopySupported(
          device1, device2, &reason);
  std::lock_guard<std::mutex> lock(_p2pMutex);
  _p2pEnabled[{device1, device2}] = supported;
  _p2pEnabled[{device2, device1}] = supported;
  return supported;
}

void DataTransferManager::initializeP2PConnections() {
  auto& devices = graph::VulkanDeviceManager::getInstance();
  if (!devices.initialize()) return;

  for (int device = 0; device < devices.deviceCount(); ++device) {
    std::lock_guard<std::mutex> lock(_p2pMutex);
    _p2pEnabled[{device, device}] = true;
  }
  for (int first = 0; first < devices.deviceCount(); ++first) {
    for (int second = first + 1; second < devices.deviceCount(); ++second) {
      setupP2PAccess(first, second);
    }
  }
}

void DataTransferManager::shutdown() {
  _shutdownRequested.store(true);
  waitAll();

  {
    std::lock_guard<std::mutex> stagingLock(_stagingMutex);
    auto& pool = graph::VulkanMemoryPool::getInstance();
    for (auto& entry : _stagingBuffers) {
      if (entry.first != nullptr && pool.getDeviceId(entry.first) >= 0) {
        pool.freeSynchronized(entry.first);
      }
    }
    _stagingBuffers.clear();
    _stagingInUse.clear();
  }
  {
    std::lock_guard<std::mutex> peerLock(_p2pMutex);
    _p2pEnabled.clear();
    _p2pBandwidth.clear();
  }
  {
    std::lock_guard<std::mutex> queueLock(_queueMutex);
    _activeTransfers.clear();
  }
  _initialized.store(false);
}

TransferResult DataTransferManager::doSyncTransfer(TransferRequest& request) {
  TransferResult result;
  result.transferId = request.transferId;
  const auto start = std::chrono::high_resolution_clock::now();

  bool copied = false;
  graph::VulkanExecutionStream* stream = nullptr;
  switch (request.direction) {
    case TransferDirection::HOST_TO_DEVICE:
      stream = transferStream(request.dstDevice);
      copied = stream != nullptr &&
               stream->enqueueCopy(request.dstPtr, request.srcPtr, request.bytes, 1);
      break;
    case TransferDirection::DEVICE_TO_HOST:
      stream = transferStream(request.srcDevice);
      copied = stream != nullptr &&
               stream->enqueueCopy(request.dstPtr, request.srcPtr, request.bytes, 2);
      break;
    case TransferDirection::DEVICE_TO_DEVICE: {
      stream = transferStream(request.dstDevice);
      if (request.srcDevice != request.dstDevice) {
        std::string reason;
        if (!graph::VulkanExecutionStream::isCrossDeviceCopySupported(
                request.srcDevice, request.dstDevice, &reason)) {
          result.errorMessage = "Vulkan cross-physical-device transfer is unavailable: " + reason;
          return result;
        }
      }
      copied = stream != nullptr &&
               stream->enqueueCopy(request.dstPtr, request.srcPtr, request.bytes, 3);
      break;
    }
    case TransferDirection::HOST_TO_HOST:
      std::memmove(request.dstPtr, request.srcPtr, request.bytes);
      copied = true;
      break;
  }

  if (!copied) {
    result.errorMessage = "Vulkan transfer submission failed";
    return result;
  }
  if (!request.async && stream != nullptr && !stream->synchronize()) {
    result.errorMessage = "Vulkan transfer synchronization failed";
    return result;
  }

  const auto end = std::chrono::high_resolution_clock::now();
  result.durationMs = std::chrono::duration<double, std::milli>(end - start).count();
  result.bandwidthGBps = result.durationMs > 0.0
                             ? (request.bytes / 1.0e9) / (result.durationMs / 1000.0)
                             : 0.0;
  result.success = true;
  updateStats(request, TransferStatus::COMPLETED);
  return result;
}

TransferResult DataTransferManager::p2pTransfer(void* srcPtr, void* dstPtr, size_t bytes,
                                                int srcDevice, int dstDevice, bool async) {
  TransferResult result;
  result.transferId = _nextTransferId.fetch_add(1);

  if (srcDevice != dstDevice) {
    std::string reason;
    if (!graph::VulkanExecutionStream::isCrossDeviceCopySupported(srcDevice, dstDevice, &reason)) {
      result.errorMessage = "Vulkan cross-physical-device transfer is unavailable: " + reason;
      return result;
    }
  }

  const auto start = std::chrono::high_resolution_clock::now();
  auto* stream = transferStream(dstDevice);
  if (stream == nullptr || !stream->enqueueCopy(dstPtr, srcPtr, bytes, 3)) {
    result.errorMessage = "Vulkan direct device transfer submission failed";
    return result;
  }
  if (!async && !stream->synchronize()) {
    result.errorMessage = "Vulkan direct device transfer synchronization failed";
    return result;
  }

  const auto end = std::chrono::high_resolution_clock::now();
  result.durationMs = std::chrono::duration<double, std::milli>(end - start).count();
  result.bandwidthGBps = result.durationMs > 0.0
                             ? (bytes / 1.0e9) / (result.durationMs / 1000.0)
                             : 0.0;
  result.success = true;
  return result;
}

void DataTransferManager::waitAll() {
  auto& devices = graph::VulkanDeviceManager::getInstance();
  if (!devices.initialize()) return;
  for (int device = 0; device < devices.deviceCount(); ++device) {
    graph::VulkanExecutionStream::synchronizeDevice(device);
  }
}

void DataTransferManager::synchronizeDevice(int deviceId) {
  graph::VulkanExecutionStream::synchronizeDevice(deviceId);
}

void DataTransferManager::barrier(const std::vector<int>& devices) {
  for (int deviceId : devices) {
    if (!graph::VulkanExecutionStream::synchronizeDevice(deviceId)) {
      THROW_EXCEPTION(
          "DataTransferManager: Vulkan device barrier synchronization failed");
    }
  }
}

void* DataTransferManager::allocatePinnedMemory(size_t bytes) {
  if (bytes == 0) return nullptr;

  auto& devices = graph::VulkanDeviceManager::getInstance();
  if (!devices.initialize()) return nullptr;
  const int deviceId = graph::VulkanDeviceManager::currentDeviceId();
  if (deviceId < 0 || deviceId >= devices.deviceCount()) return nullptr;

  return graph::VulkanMemoryPool::getInstance().allocateHostVisible(
      deviceId, static_cast<VkDeviceSize>(bytes));
}

void DataTransferManager::freePinnedMemory(void* ptr) {
  if (ptr == nullptr) return;

  auto& pool = graph::VulkanMemoryPool::getInstance();
  graph::VulkanAllocRecord record;
  if (!pool.queryRecord(ptr, record) || !record.hostVisible) {
    THROW_EXCEPTION(
        "DataTransferManager: pointer is not a Vulkan host-visible allocation");
  }
  if (!pool.freeSynchronized(ptr)) {
    THROW_EXCEPTION(
        "DataTransferManager: failed to release Vulkan host-visible allocation");
  }
}

void* DataTransferManager::allocateDeviceMemory(int deviceId, size_t bytes) {
  if (bytes == 0) return nullptr;

  auto& devices = graph::VulkanDeviceManager::getInstance();
  if (!devices.initialize() || deviceId < 0 ||
      deviceId >= devices.deviceCount()) {
    return nullptr;
  }
  return graph::VulkanMemoryPool::getInstance().allocate(
      deviceId, static_cast<VkDeviceSize>(bytes));
}

void DataTransferManager::freeDeviceMemory(int deviceId, void* ptr) {
  if (ptr == nullptr) return;

  auto& pool = graph::VulkanMemoryPool::getInstance();
  graph::VulkanAllocRecord record;
  if (!pool.queryRecord(ptr, record) || record.deviceId != deviceId) {
    THROW_EXCEPTION(
        "DataTransferManager: Vulkan allocation/device ownership mismatch");
  }
  if (!pool.freeSynchronized(ptr)) {
    THROW_EXCEPTION(
        "DataTransferManager: failed to release Vulkan device allocation");
  }
}

float DataTransferManager::benchmarkBandwidth(int srcDevice, int dstDevice,
                                              size_t bytes) {
  if (bytes == 0) return 0.0f;

  void* src = srcDevice >= 0 ? allocateDeviceMemory(srcDevice, bytes)
                             : allocatePinnedMemory(bytes);
  void* dst = dstDevice >= 0 ? allocateDeviceMemory(dstDevice, bytes)
                             : allocatePinnedMemory(bytes);
  auto release = [this](void* ptr, int deviceId) {
    if (ptr == nullptr) return;
    if (deviceId >= 0) {
      freeDeviceMemory(deviceId, ptr);
    } else {
      freePinnedMemory(ptr);
    }
  };

  if (src == nullptr || dst == nullptr) {
    release(src, srcDevice);
    release(dst, dstDevice);
    return 0.0f;
  }

  TransferResult warmup =
      transfer(src, dst, bytes, srcDevice, dstDevice, false);
  if (!warmup.success) {
    release(src, srcDevice);
    release(dst, dstDevice);
    return 0.0f;
  }

  constexpr int iterations = 10;
  const auto start = std::chrono::high_resolution_clock::now();
  bool success = true;
  for (int i = 0; i < iterations; ++i) {
    if (!transfer(src, dst, bytes, srcDevice, dstDevice, false).success) {
      success = false;
      break;
    }
  }
  const auto end = std::chrono::high_resolution_clock::now();

  release(src, srcDevice);
  release(dst, dstDevice);
  if (!success) return 0.0f;

  const double milliseconds =
      std::chrono::duration<double, std::milli>(end - start).count();
  const float bandwidth =
      milliseconds > 0.0
          ? static_cast<float>(
                (static_cast<double>(bytes) * iterations / 1.0e9) /
                (milliseconds / 1000.0))
          : 0.0f;
  {
    std::lock_guard<std::mutex> lock(_p2pMutex);
    _p2pBandwidth[{srcDevice, dstDevice}] = bandwidth;
  }
  return bandwidth;
}

}  // namespace modelparallel
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
