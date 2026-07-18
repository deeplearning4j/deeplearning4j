/* ******************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <helpers/PointersManager.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <execution/LaunchContext.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/vulkan/VulkanMemoryPool.h>

namespace sd {

namespace {
graph::VulkanExecutionStream* contextStream(const LaunchContext* context) {
  if (context != nullptr) {
    auto* stream = graph::VulkanExecutionStream::fromOpaque(
        graph::vulkanExecutionStream(context), true);
    if (stream != nullptr) return stream;
  }
  return graph::VulkanExecutionStream::currentOrDefault();
}
}  // namespace

PointersManager::PointersManager(const LaunchContext* context, const std::string& funcName) {
  _context = const_cast<LaunchContext*>(context);
  _funcName = funcName;
  _workspaceWasActive = (_context != nullptr && _context->getWorkspace() != nullptr);
}

void* PointersManager::allocateDevMem(const size_t sizeInBytes) {
  if (sizeInBytes == 0) return nullptr;

  auto* stream = contextStream(_context);
  if (stream == nullptr) {
    const std::string message =
        _funcName + ": no active Vulkan execution stream for device allocation";
    THROW_EXCEPTION(message.c_str());
  }

  auto& pool = graph::VulkanMemoryPool::getInstance();
  void* allocation = nullptr;
  bool poolOwned = false;

  if (_context != nullptr && _context->getWorkspace() != nullptr) {
    allocation = _context->getWorkspace()->allocateBytes(
        memory::MemoryType::DEVICE, static_cast<LongType>(sizeInBytes));
    graph::VulkanAllocRecord record;
    if (allocation == nullptr || !pool.queryRecord(allocation, record) ||
        record.deviceId != stream->deviceId() ||
        record.logicalSize < static_cast<VkDeviceSize>(sizeInBytes)) {
      const std::string message =
          _funcName +
          ": workspace did not provide Vulkan device storage on the execution stream device";
      THROW_EXCEPTION(message.c_str());
    }
  } else {
    allocation =
        pool.allocate(stream->deviceId(), static_cast<VkDeviceSize>(sizeInBytes));
    if (allocation == nullptr) {
      const std::string message =
          _funcName + ": Vulkan device allocation failed";
      THROW_EXCEPTION(message.c_str());
    }
    poolOwned = true;
  }

  _allocatedPointers.emplace_back(allocation, poolOwned);
  return allocation;
}

void* PointersManager::replicatePointer(const void* src,
                                        const size_t numberOfBytes) {
  void* dst = allocateDevMem(numberOfBytes);
  if (src != nullptr && numberOfBytes != 0) {
    auto* stream = contextStream(_context);
    if (stream == nullptr ||
        !stream->enqueueCopy(dst, src,
                             static_cast<VkDeviceSize>(numberOfBytes), 1)) {
      const std::string message =
          _funcName + ": Vulkan stream-ordered H2D copy failed";
      THROW_EXCEPTION(message.c_str());
    }
  }
  return dst;
}

void PointersManager::synchronize() const {
  auto* stream = contextStream(_context);
  if (stream == nullptr || !stream->synchronize()) {
    const std::string message = _funcName + ": Vulkan stream synchronization failed";
    THROW_EXCEPTION(message.c_str());
  }
}

PointersManager::~PointersManager() {
  if (_allocatedPointers.empty()) return;

  auto& pool = graph::VulkanMemoryPool::getInstance();
  auto* stream = contextStream(_context);
  for (const auto& allocation : _allocatedPointers) {
    if (!allocation.fromCudaMalloc || allocation.ptr == nullptr) continue;
    if (stream == nullptr || !stream->retireAllocation(allocation.ptr)) {
      pool.freeSynchronized(allocation.ptr);
    }
  }
}

}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
