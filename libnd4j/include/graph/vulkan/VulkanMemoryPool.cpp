/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <graph/vulkan/VulkanMemoryPool.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>

namespace sd {
namespace graph {

// ── Singleton ──────────────────────────────────────────────────────────────

VulkanMemoryPool& VulkanMemoryPool::getInstance() {
  static VulkanMemoryPool instance;
  return instance;
}

VulkanMemoryPool::VulkanMemoryPool(VkDeviceSize blockSize,
                                   VkDeviceSize dedicatedThreshold)
    : blockSize_(blockSize), dedicatedThreshold_(dedicatedThreshold) {}

VulkanMemoryPool::~VulkanMemoryPool() { shutdown(); }

void VulkanMemoryPool::shutdown() {
  std::call_once(shutdownFlag_, [this]() {
    // Reclaim both retired and still-live records before freeing their backing
    // blocks. This runs from the backend lifecycle while the Vulkan loader and
    // ICD are still live; the destructor later becomes an idempotent no-op.
    std::vector<void*> livePointers;
    std::vector<int> deviceIds;
    {
      std::lock_guard<std::mutex> registryLock(registryMtx_);
      livePointers.reserve(registry_.size());
      for (const auto& entry : registry_) {
        livePointers.push_back(entry.first);
        if (entry.second.deviceId >= 0 &&
            std::find(deviceIds.begin(), deviceIds.end(),
                      entry.second.deviceId) == deviceIds.end()) {
          deviceIds.push_back(entry.second.deviceId);
        }
      }
    }
    for (int deviceId : deviceIds) {
      auto* context = VulkanDeviceContext::getContext(deviceId);
      if (context != nullptr) {
        VkResult waitResult = context->waitComputeIdle();
        if (waitResult != VK_SUCCESS) {
          sd_printf("VulkanMemoryPool::shutdown: compute-queue wait failed "
                    "device=%d res=%d\n",
                    deviceId, static_cast<int>(waitResult));
          if (waitResult == VK_ERROR_DEVICE_LOST) context->markLost();
        }
      }
    }

    std::lock_guard<std::mutex> devicesLock(devicesMtx_);
    for (void* ptr : livePointers) {
      doReclaim(ptr, false);
    }

    for (auto& statePtr : deviceStates_) {
      if (!statePtr) continue;
      auto& state = *statePtr;
      std::lock_guard<std::mutex> stateLock(state.mtx);
      state.retireList.clear();

      for (auto& poolEntry : state.pools) {
        for (auto& block : poolEntry.second.blocks) {
          if (!block) continue;
          if (block->mappedBase != nullptr &&
              block->memory != VK_NULL_HANDLE) {
            vkUnmapMemory(block->logicalDevice, block->memory);
            block->mappedBase = nullptr;
          }
          if (block->memory != VK_NULL_HANDLE) {
            vkFreeMemory(block->logicalDevice, block->memory, nullptr);
            block->memory = VK_NULL_HANDLE;
          }
        }
      }
      state.pools.clear();
      state.dedicatedMemMap.clear();
    }
    deviceStates_.clear();
  });
}

// ── Block::tryAllocate ─────────────────────────────────────────────────────

VkDeviceSize VulkanMemoryPool::Block::tryAllocate(VkDeviceSize size, VkDeviceSize align) {
  if (align == 0) align = 1;
  for (size_t i = 0; i < freeList.size(); ++i) {
    FreeSpan& span = freeList[i];
    // Align up without assuming the caller supplied a power of two.
    VkDeviceSize remainder = span.offset % align;
    VkDeviceSize padding = remainder == 0 ? 0 : align - remainder;
    if (padding > span.size || size > span.size - padding) continue;
    VkDeviceSize aligned = span.offset + padding;

    VkDeviceSize remaining = span.size - padding - size;
    // If there's a non-trivial padding prefix, keep it as a free span.
    if (padding > 0) {
      FreeSpan prefix{span.offset, padding};
      freeList.erase(freeList.begin() + static_cast<ptrdiff_t>(i));
      // Re-insert prefix and suffix.
      auto ins = std::lower_bound(freeList.begin(), freeList.end(), prefix,
                                  [](const FreeSpan& a, const FreeSpan& b){
                                    return a.offset < b.offset;
                                  });
      ins = freeList.insert(ins, prefix);
      if (remaining > 0) {
        FreeSpan suffix{aligned + size, remaining};
        ins = std::lower_bound(freeList.begin(), freeList.end(), suffix,
                               [](const FreeSpan& a, const FreeSpan& b){
                                 return a.offset < b.offset;
                               });
        freeList.insert(ins, suffix);
      }
    } else {
      // No padding — resize or remove the span.
      if (remaining > 0) {
        span.offset = aligned + size;
        span.size   = remaining;
      } else {
        freeList.erase(freeList.begin() + static_cast<ptrdiff_t>(i));
      }
    }
    ++activeAllocs;
    return aligned;
  }
  return std::numeric_limits<VkDeviceSize>::max();  // no fit
}

// ── Block::reclaim ─────────────────────────────────────────────────────────

void VulkanMemoryPool::Block::reclaim(VkDeviceSize offset, VkDeviceSize size) {
  if (activeAllocs > 0) --activeAllocs;
  FreeSpan span{offset, size};
  auto it = std::lower_bound(freeList.begin(), freeList.end(), span,
                             [](const FreeSpan& a, const FreeSpan& b){
                               return a.offset < b.offset;
                             });
  it = freeList.insert(it, span);

  // Merge with next span.
  auto next = std::next(it);
  if (next != freeList.end() && it->offset + it->size == next->offset) {
    it->size += next->size;
    freeList.erase(next);
  }
  // Merge with previous span.
  if (it != freeList.begin()) {
    auto prev = std::prev(it);
    if (prev->offset + prev->size == it->offset) {
      prev->size += it->size;
      freeList.erase(it);
    }
  }
}

// ── Static helpers ─────────────────────────────────────────────────────────

uint32_t VulkanMemoryPool::findMemoryType(VkPhysicalDevice physDev,
                                          uint32_t typeFilter,
                                          VkMemoryPropertyFlags required,
                                          VkMemoryPropertyFlags preferred) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

  uint32_t bestIdx = UINT32_MAX;
  // First pass: prefer compatible types that also satisfy `preferred`.
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((typeFilter & (1u << i)) == 0) continue;
    VkMemoryPropertyFlags flags = memProps.memoryTypes[i].propertyFlags;
    if ((flags & required) == required) {
      if (preferred == 0 || (flags & preferred) == preferred) {
        bestIdx = i;
        break;
      }
      if (bestIdx == UINT32_MAX) bestIdx = i;
    }
  }
  return bestIdx;
}

VkDeviceSize VulkanMemoryPool::defaultAlignment(VkPhysicalDevice physDev) {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physDev, &props);
  // Buffer binding additionally enforces VkMemoryRequirements::alignment.
  // bufferImageGranularity applies to mixed image/buffer aliasing; this pool
  // contains buffers only and must not impose that unrelated spacing.
  VkDeviceSize align = props.limits.minStorageBufferOffsetAlignment;
  return align == 0 ? 1 : align;
}

void VulkanMemoryPool::diagPlacement(int deviceId, VkDeviceSize bytes,
                                     const char* memoryType) {
  DSP_DIAG(MEMORY,
    "VulkanMemoryPool: alloc device=%d bytes=%llu memoryType=%s",
    deviceId, (unsigned long long)bytes, memoryType);
}

// ── DeviceState access ─────────────────────────────────────────────────────
// NOTE: caller must hold devicesMtx before calling this.

VulkanMemoryPool::DeviceState* VulkanMemoryPool::ensureDeviceState(int deviceId) {
  if (deviceId < 0) return nullptr;
  size_t idx = static_cast<size_t>(deviceId);
  if (idx >= deviceStates_.size()) deviceStates_.resize(idx + 1);
  if (!deviceStates_[idx]) deviceStates_[idx] = std::make_unique<DeviceState>();
  return deviceStates_[idx].get();
}

// ── makeBlock ──────────────────────────────────────────────────────────────

std::unique_ptr<VulkanMemoryPool::Block> VulkanMemoryPool::makeBlock(
    int deviceId, VkDevice logDev, VkPhysicalDevice phys,
    uint32_t memTypeIdx, VkDeviceSize blockSize)
{
  // Driver-reported memory availability is a hard constraint.
  const uint64_t freeVk = VulkanDeviceManager::getInstance().getFreeMemory(deviceId);
  if (freeVk < static_cast<uint64_t>(blockSize)) {
    sd_printf("VulkanMemoryPool::makeBlock: insufficient device memory device=%d "
              "freeVk=%llu blockSize=%llu\n",
              deviceId, (unsigned long long)freeVk,
              (unsigned long long)blockSize);
    return nullptr;
  }

  auto* context = VulkanDeviceContext::getContext(deviceId);
  const bool externalShareable =
      context != nullptr && context->hasExternalDeviceSharing() &&
      !context->caps().externalMemoryDedicatedOnly;

  VkExportMemoryAllocateInfo exportInfo = {};
  exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.pNext           = externalShareable ? &exportInfo : nullptr;
  allocInfo.allocationSize  = blockSize;
  allocInfo.memoryTypeIndex = memTypeIdx;

  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkResult res = vkAllocateMemory(logDev, &allocInfo, nullptr, &mem);
  if (res != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::makeBlock: vkAllocateMemory failed device=%d memType=%u size=%llu res=%d\n",
              deviceId, memTypeIdx, (unsigned long long)blockSize, static_cast<int>(res));
    return nullptr;
  }

  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
  bool hostVisible = (memProps.memoryTypes[memTypeIdx].propertyFlags &
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

  auto blk = std::make_unique<Block>();
  blk->logicalDevice = logDev;
  blk->memory        = mem;
  blk->blockSize     = blockSize;
  blk->memTypeIdx    = memTypeIdx;
  blk->deviceId      = deviceId;
  blk->hostVisible   = hostVisible;
  blk->externalShareable = externalShareable;
  blk->activeAllocs  = 0;
  blk->freeList.push_back({0, blockSize});

  if (hostVisible) {
    void* mapped = nullptr;
    res = vkMapMemory(logDev, mem, 0, VK_WHOLE_SIZE, 0, &mapped);
    if (res != VK_SUCCESS) {
      sd_printf("VulkanMemoryPool::makeBlock: vkMapMemory failed device=%d res=%d\n",
                deviceId, static_cast<int>(res));
      vkFreeMemory(logDev, mem, nullptr);
      return nullptr;
    }
    blk->mappedBase = mapped;
  }

  sd_printf("VulkanMemoryPool::makeBlock: new block device=%d memType=%u size=%lluMB host=%s\n",
            deviceId, memTypeIdx, (unsigned long long)(blockSize >> 20),
            hostVisible ? "yes" : "no");
  return blk;
}

// ── findOrCreateBlock ──────────────────────────────────────────────────────

std::pair<VulkanMemoryPool::Block*, VkDeviceSize>
VulkanMemoryPool::findOrCreateBlock(DeviceState& state, int deviceId,
                                    uint32_t memTypeIdx,
                                    VkDeviceSize size, VkDeviceSize alignment)
{
  // state.mtx must be held by the caller.
  MemTypePool& mtp = state.pools[memTypeIdx];

  // Try existing blocks first (LRU: scan from the end — most recently used).
  for (auto it = mtp.blocks.rbegin(); it != mtp.blocks.rend(); ++it) {
    Block* blk = it->get();
    if (!blk || blk->memTypeIdx != memTypeIdx) continue;
    VkDeviceSize off = blk->tryAllocate(size, alignment);
    if (off != std::numeric_limits<VkDeviceSize>::max())
      return {blk, off};
  }

  // No existing block fits — create a new one.
  auto* context = VulkanDeviceContext::getContext(deviceId);
  VkDevice logDev = context != nullptr ? context->device() : VK_NULL_HANDLE;
  VkPhysicalDevice phys = context != nullptr ? context->physicalDevice() : VK_NULL_HANDLE;

  if (logDev == VK_NULL_HANDLE || phys == VK_NULL_HANDLE) {
    sd_printf("VulkanMemoryPool::findOrCreateBlock: no Vulkan context for id=%d\n", deviceId);
    return {nullptr, 0};
  }

  // Grow the block if necessary to fit oversized-but-under-threshold allocs.
  VkDeviceSize newBlockSize = blockSize_;
  if (size > newBlockSize) newBlockSize = size + alignment;

  const int64_t configuredBudget = sd::Environment::getInstance().maxDeviceMemory();
  if (configuredBudget > 0 &&
      state.reservedBytes.load(std::memory_order_relaxed) +
              static_cast<uint64_t>(newBlockSize) >
          static_cast<uint64_t>(configuredBudget)) {
    sd_printf("VulkanMemoryPool::findOrCreateBlock: configured budget exceeded "
              "device=%d reserved=%llu block=%llu budget=%lld\n",
              deviceId,
              (unsigned long long)state.reservedBytes.load(std::memory_order_relaxed),
              (unsigned long long)newBlockSize, (long long)configuredBudget);
    return {nullptr, 0};
  }

  auto newBlk = makeBlock(deviceId, logDev, phys, memTypeIdx, newBlockSize);
  if (!newBlk) return {nullptr, 0};

  Block* rawBlk = newBlk.get();
  VkDeviceSize off = rawBlk->tryAllocate(size, alignment);
  mtp.blocks.push_back(std::move(newBlk));
  state.reservedBytes.fetch_add(
      static_cast<uint64_t>(newBlockSize), std::memory_order_relaxed);

  if (off == std::numeric_limits<VkDeviceSize>::max()) {
    // Should not happen with a fresh block sized >= size + alignment.
    sd_printf("VulkanMemoryPool::findOrCreateBlock: fresh block tryAllocate failed\n", 0);
    return {nullptr, 0};
  }
  return {rawBlk, off};
}

// ── allocateDedicated ──────────────────────────────────────────────────────

void* VulkanMemoryPool::allocateDedicated(
    int deviceId, VkDevice logDev, VkBuffer buffer, uint32_t memTypeIdx,
    VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize logicalSize,
    VkDeviceSize reservedSize, bool hostVisible, bool externalShareable) {
  {
    std::lock_guard<std::mutex> dlk(devicesMtx_);
    if (ensureDeviceState(deviceId) == nullptr) return nullptr;
  }

  VkMemoryDedicatedAllocateInfo dedicatedInfo = {};
  dedicatedInfo.sType =
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
  dedicatedInfo.buffer = buffer;

  VkExportMemoryAllocateInfo exportInfo = {};
  exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  exportInfo.pNext = &dedicatedInfo;
  exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.pNext = externalShareable
                        ? static_cast<const void*>(&exportInfo)
                        : static_cast<const void*>(&dedicatedInfo);
  allocInfo.allocationSize  = reservedSize;
  allocInfo.memoryTypeIndex = memTypeIdx;

  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkResult res = vkAllocateMemory(logDev, &allocInfo, nullptr, &mem);
  if (res != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::allocateDedicated: vkAllocateMemory failed device=%d size=%llu res=%d\n",
              deviceId, (unsigned long long)reservedSize, static_cast<int>(res));
    return nullptr;
  }

  void* mapped = nullptr;
  if (hostVisible) {
    res = vkMapMemory(logDev, mem, 0, VK_WHOLE_SIZE, 0, &mapped);
    if (res != VK_SUCCESS || mapped == nullptr) {
      sd_printf("VulkanMemoryPool::allocateDedicated: vkMapMemory failed device=%d res=%d\n",
                deviceId, static_cast<int>(res));
      vkFreeMemory(logDev, mem, nullptr);
      return nullptr;
    }
  }

  res = vkBindBufferMemory(logDev, buffer, mem, 0);
  if (res != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::allocateDedicated: vkBindBufferMemory failed device=%d res=%d\n",
              deviceId, static_cast<int>(res));
    if (mapped != nullptr) vkUnmapMemory(logDev, mem);
    vkFreeMemory(logDev, mem, nullptr);
    return nullptr;
  }

  void* retPtr = mapped;
  if (!hostVisible) {
    // DEVICE_LOCAL only: issue a unique sentinel pointer (never dereferenced).
    std::lock_guard<std::mutex> slk(sentinelMtx_);
    auto sentinel = std::make_unique<uint8_t>(0xDE);
    retPtr = sentinel.get();
    sentinelStore_.push_back(std::move(sentinel));
  }

  {
    std::lock_guard<std::mutex> dlk(devicesMtx_);
    auto* statePtr = ensureDeviceState(deviceId);
    std::lock_guard<std::mutex> slk(statePtr->mtx);
    statePtr->dedicatedMemMap[retPtr] = mem;
    statePtr->trackedBytes.fetch_add(
        static_cast<uint64_t>(reservedSize), std::memory_order_relaxed);
    statePtr->reservedBytes.fetch_add(
        static_cast<uint64_t>(reservedSize), std::memory_order_relaxed);
    statePtr->totalAcquired.fetch_add(1, std::memory_order_relaxed);
  }

  VulkanAllocRecord rec;
  rec.deviceId      = deviceId;
  rec.memTypeIdx    = memTypeIdx;
  rec.memoryPropertyFlags = memoryPropertyFlags;
  rec.dedicated     = true;
  rec.hostVisible   = hostVisible;
  rec.mappedPtr     = mapped;
  rec.blockKey      = nullptr;
  rec.offsetInBlock = 0;
  rec.buffer        = buffer;
  rec.logicalDevice = logDev;
  rec.logicalSize   = logicalSize;
  rec.reservedSize  = reservedSize;
  rec.backingMemory = mem;
  rec.backingSize = reservedSize;
  rec.externalShareable = externalShareable;

  {
    std::lock_guard<std::mutex> rlk(registryMtx_);
    registry_[retPtr] = rec;
  }

  VulkanDeviceManager::getInstance().trackAllocation(
      deviceId, static_cast<size_t>(reservedSize));
  return retPtr;
}

// ── allocate (main entry point) ────────────────────────────────────────────

void* VulkanMemoryPool::allocate(int deviceId, VkDeviceSize bytes, VkDeviceSize alignment) {
  VulkanDeviceManager& mgr = VulkanDeviceManager::getInstance();
  if (!mgr.initialize()) {
    sd_printf("VulkanMemoryPool::allocate: VulkanDeviceManager initialization failed\n", 0);
    return nullptr;
  }

  if (deviceId < 0 || deviceId >= mgr.deviceCount()) {
    sd_printf("VulkanMemoryPool::allocate: invalid deviceId=%d\n", deviceId);
    return nullptr;
  }

  auto* context = VulkanDeviceContext::getContext(deviceId);
  VkPhysicalDevice phys = context != nullptr ? context->physicalDevice() : VK_NULL_HANDLE;
  VkDevice logDev = context != nullptr ? context->device() : VK_NULL_HANDLE;
  if (phys == VK_NULL_HANDLE || logDev == VK_NULL_HANDLE) {
    sd_printf("VulkanMemoryPool::allocate: no Vulkan context for deviceId=%d\n", deviceId);
    return nullptr;
  }

  const VkDeviceSize logicalSize = bytes == 0 ? 1 : bytes;
  // Transfer commands operate on four-byte units. The public logical size remains
  // exact; padding is private to the VkBuffer and is never exposed as array data.
  const VkDeviceSize bufferSize = (logicalSize + 3u) & ~VkDeviceSize(3u);

  VkExternalMemoryBufferCreateInfo externalBufferInfo = {};
  externalBufferInfo.sType =
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  externalBufferInfo.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.pNext = context->hasExternalDeviceSharing()
                         ? &externalBufferInfo
                         : nullptr;
  bufferInfo.size = bufferSize;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer buffer = VK_NULL_HANDLE;
  VkResult result = vkCreateBuffer(logDev, &bufferInfo, nullptr, &buffer);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::allocate: vkCreateBuffer failed device=%d bytes=%llu res=%d\n",
              deviceId, (unsigned long long)logicalSize, static_cast<int>(result));
    return nullptr;
  }

  VkMemoryRequirements requirements = {};
  vkGetBufferMemoryRequirements(logDev, buffer, &requirements);
  if (requirements.size == 0 || requirements.memoryTypeBits == 0) {
    sd_printf("VulkanMemoryPool::allocate: invalid buffer memory requirements device=%d\n",
              deviceId);
    vkDestroyBuffer(logDev, buffer, nullptr);
    return nullptr;
  }

  if (alignment == 0) alignment = defaultAlignment(phys);
  alignment = std::max(alignment, requirements.alignment);
  if (alignment == 0) alignment = 1;
  const VkDeviceSize reservedSize = requirements.size;

  // The configured budget is a hard device-memory limit. Crossing it is OOM;
  // it must never change the memory placement selected for the allocation.
  int64_t envBudget = sd::Environment::getInstance().maxDeviceMemory();
  if (envBudget > 0) {
    uint64_t trackedUsage = 0;
    {
      std::lock_guard<std::mutex> dlk(devicesMtx_);
      if (static_cast<size_t>(deviceId) < deviceStates_.size() &&
          deviceStates_[static_cast<size_t>(deviceId)]) {
        trackedUsage = deviceStates_[static_cast<size_t>(deviceId)]->trackedBytes.load(
            std::memory_order_relaxed);
      }
    }
    if (trackedUsage + static_cast<uint64_t>(reservedSize) >
        static_cast<uint64_t>(envBudget)) {
      sd_printf("VulkanMemoryPool::allocate: device budget exceeded device=%d "
                "trackedBytes=%llu + req=%llu > budget=%lld\n",
                deviceId, (unsigned long long)trackedUsage,
                (unsigned long long)reservedSize, (long long)envBudget);
      DSP_DIAG(MEMORY,
        "VulkanMemoryPool: DEVICE_BUDGET_EXCEEDED device=%d trackedBytes=%llu req=%llu budget=%lld",
        deviceId, (unsigned long long)trackedUsage,
        (unsigned long long)reservedSize, (long long)envBudget);
      vkDestroyBuffer(logDev, buffer, nullptr);
      return nullptr;
    }
  }

  const bool isDedicated =
      reservedSize >= dedicatedThreshold_ ||
      (context->hasExternalDeviceSharing() &&
       context->caps().externalMemoryDedicatedOnly);

  VkPhysicalDeviceMemoryProperties memoryProperties = {};
  vkGetPhysicalDeviceMemoryProperties(phys, &memoryProperties);

  const VkMemoryPropertyFlags hostPreference =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const uint32_t memTypeIdx = findMemoryType(
      phys, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      hostPreference);
  if (memTypeIdx == UINT32_MAX) {
    sd_printf("VulkanMemoryPool::allocate: no compatible DEVICE_LOCAL memory type device=%d\n",
              deviceId);
    vkDestroyBuffer(logDev, buffer, nullptr);
    return nullptr;
  }

  const VkMemoryPropertyFlags selectedFlags =
      memoryProperties.memoryTypes[memTypeIdx].propertyFlags;
  const bool hostVisible = (selectedFlags & hostPreference) == hostPreference;
  const char* memoryTypeName =
      hostVisible ? "DEVICE_LOCAL|HOST_VISIBLE|COHERENT" : "DEVICE_LOCAL";

  if (isDedicated) {
    void* ptr = allocateDedicated(
        deviceId, logDev, buffer, memTypeIdx, selectedFlags, logicalSize,
        reservedSize, hostVisible, context->hasExternalDeviceSharing());
    if (ptr == nullptr) {
      vkDestroyBuffer(logDev, buffer, nullptr);
      return nullptr;
    }
    diagPlacement(deviceId, logicalSize, memoryTypeName);
    return ptr;
  }

  std::lock_guard<std::mutex> dlk(devicesMtx_);
  auto* statePtr = ensureDeviceState(deviceId);
  if (statePtr == nullptr) {
    vkDestroyBuffer(logDev, buffer, nullptr);
    return nullptr;
  }
  DeviceState& state = *statePtr;
  std::lock_guard<std::mutex> slk(state.mtx);

  auto [block, offset] = findOrCreateBlock(
      state, deviceId, memTypeIdx, reservedSize, alignment);
  if (block == nullptr) {
    sd_printf("VulkanMemoryPool::allocate: DEVICE_LOCAL block allocation failed device=%d\n",
              deviceId);
    vkDestroyBuffer(logDev, buffer, nullptr);
    return nullptr;
  }

  result = vkBindBufferMemory(logDev, buffer, block->memory, offset);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::allocate: vkBindBufferMemory failed "
              "device=%d offset=%llu res=%d\n",
              deviceId, (unsigned long long)offset, static_cast<int>(result));
    block->reclaim(offset, reservedSize);
    vkDestroyBuffer(logDev, buffer, nullptr);
    return nullptr;
  }

  void* blockKey = static_cast<void*>(block);
  const bool reusedExistingBlock =
      state.blockIds.find(blockKey) != state.blockIds.end();
  if (!reusedExistingBlock) {
    state.blockIds[blockKey] = state.nextBlockId++;
  }

  void* ptr = nullptr;
  if (hostVisible && block->mappedBase != nullptr) {
    ptr = static_cast<uint8_t*>(block->mappedBase) + offset;
  } else {
    std::lock_guard<std::mutex> sentLk(sentinelMtx_);
    auto sentinel = std::make_unique<uint8_t>(0xAB);
    ptr = sentinel.get();
    sentinelStore_.push_back(std::move(sentinel));
  }

  VulkanAllocRecord record;
  record.deviceId = deviceId;
  record.memTypeIdx = memTypeIdx;
  record.memoryPropertyFlags = selectedFlags;
  record.dedicated = false;
  record.hostVisible = hostVisible;
  record.mappedPtr = hostVisible ? ptr : nullptr;
  record.blockKey = blockKey;
  record.offsetInBlock = offset;
  record.buffer = buffer;
  record.logicalDevice = logDev;
  record.logicalSize = logicalSize;
  record.reservedSize = reservedSize;
  record.backingMemory = block->memory;
  record.backingSize = block->blockSize;
  record.externalShareable = block->externalShareable;

  {
    std::lock_guard<std::mutex> rlk(registryMtx_);
    registry_[ptr] = record;
  }

  state.trackedBytes.fetch_add(
      static_cast<uint64_t>(reservedSize), std::memory_order_relaxed);
  state.totalAcquired.fetch_add(1, std::memory_order_relaxed);
  if (reusedExistingBlock) {
    state.totalReused.fetch_add(1, std::memory_order_relaxed);
  }
  mgr.trackAllocation(deviceId, static_cast<size_t>(reservedSize));

  diagPlacement(deviceId, logicalSize, memoryTypeName);
  return ptr;
}

void* VulkanMemoryPool::allocateHostVisible(int deviceId, VkDeviceSize bytes) {
  VulkanDeviceManager& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize() || deviceId < 0 ||
      deviceId >= manager.deviceCount() || bytes == 0) {
    return nullptr;
  }

  auto* context = VulkanDeviceContext::getContext(deviceId);
  const VkPhysicalDevice physicalDevice =
      context != nullptr ? context->physicalDevice() : VK_NULL_HANDLE;
  const VkDevice logicalDevice =
      context != nullptr ? context->device() : VK_NULL_HANDLE;
  if (physicalDevice == VK_NULL_HANDLE || logicalDevice == VK_NULL_HANDLE) {
    return nullptr;
  }

  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bytes;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer buffer = VK_NULL_HANDLE;
  VkResult result =
      vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer);
  if (result != VK_SUCCESS) {
    sd_printf(
        "VulkanMemoryPool::allocateHostVisible: vkCreateBuffer failed "
        "device=%d bytes=%llu res=%d\n",
        deviceId, static_cast<unsigned long long>(bytes),
        static_cast<int>(result));
    return nullptr;
  }

  VkMemoryRequirements requirements = {};
  vkGetBufferMemoryRequirements(logicalDevice, buffer, &requirements);
  const VkMemoryPropertyFlags required =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const uint32_t memoryType = findMemoryType(
      physicalDevice, requirements.memoryTypeBits, required,
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
  if (memoryType == UINT32_MAX) {
    sd_printf(
        "VulkanMemoryPool::allocateHostVisible: no HOST_VISIBLE|HOST_COHERENT "
        "memory type for device=%d\n",
        deviceId);
    vkDestroyBuffer(logicalDevice, buffer, nullptr);
    return nullptr;
  }

  VkPhysicalDeviceMemoryProperties properties = {};
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);
  const VkMemoryPropertyFlags selectedFlags =
      properties.memoryTypes[memoryType].propertyFlags;
  void* ptr = allocateDedicated(
      deviceId, logicalDevice, buffer, memoryType, selectedFlags, bytes,
      requirements.size, true, false);
  if (ptr == nullptr) {
    vkDestroyBuffer(logicalDevice, buffer, nullptr);
    return nullptr;
  }

  diagPlacement(deviceId, bytes, "HOST_VISIBLE|HOST_COHERENT");
  return ptr;
}

// ── free ──────────────────────────────────────────────────────────────────

bool VulkanMemoryPool::free(void* ptr, uint64_t timelineValue) {
  if (!ptr) return true;

  VulkanAllocRecord rec;
  bool found = false;
  {
    std::lock_guard<std::mutex> rlk(registryMtx_);
    auto it = registry_.find(ptr);
    if (it == registry_.end()) return false;
    rec   = it->second;
    found = true;
  }
  if (!found) return false;

  // Enqueue on the device's retire list.
  std::lock_guard<std::mutex> dlk(devicesMtx_);
  auto* statePtr = ensureDeviceState(rec.deviceId);
  if (!statePtr) {
    // Shouldn't happen — free allocation without a state?  Reclaim immediately.
    doReclaim(ptr);
    return true;
  }
  {
    std::lock_guard<std::mutex> slk(statePtr->mtx);
    statePtr->retireList.push_back({ptr, timelineValue});
  }
  return true;
}

// ── sweep ──────────────────────────────────────────────────────────────────

void VulkanMemoryPool::sweep(int deviceId, uint64_t completedTimeline) {
  std::lock_guard<std::mutex> dlk(devicesMtx_);
  if (deviceId < 0 || static_cast<size_t>(deviceId) >= deviceStates_.size()) return;
  auto* statePtr = deviceStates_[static_cast<size_t>(deviceId)].get();
  if (!statePtr) return;

  std::lock_guard<std::mutex> slk(statePtr->mtx);
  auto& rl = statePtr->retireList;
  while (!rl.empty() && rl.front().timelineStamp <= completedTimeline) {
    void* ptr = rl.front().ptr;
    rl.pop_front();
    doReclaim(ptr);
  }
}

// ── freeImmediate ──────────────────────────────────────────────────────────

void VulkanMemoryPool::freeImmediate(void* ptr) {
  if (!ptr) return;
  // Hold devicesMtx_ for the duration: doReclaim() requires it,
  // and retire-list removal also needs it.
  std::lock_guard<std::mutex> dlk(devicesMtx_);
  doReclaim(ptr);
  // Also remove from any retire list (if queued before this immediate free).
  for (auto& statePtr : deviceStates_) {
    if (!statePtr) continue;
    std::lock_guard<std::mutex> slk(statePtr->mtx);
    auto& rl = statePtr->retireList;
    rl.erase(std::remove_if(rl.begin(), rl.end(),
                            [ptr](const DeviceState::RetireEntry& e){ return e.ptr == ptr; }),
             rl.end());
  }
}

bool VulkanMemoryPool::freeSynchronized(void* ptr) {
  if (ptr == nullptr) return true;

  VulkanAllocRecord record;
  if (!queryRecord(ptr, record)) return false;
  if (!VulkanExecutionStream::synchronizeDevice(record.deviceId)) return false;

  freeImmediate(ptr);
  return true;
}

// ── doReclaim (internal) ───────────────────────────────────────────────────
//
// NOTE on locking: doReclaim() must NOT acquire devicesMtx_.
// Callers from sweep() and the destructor already hold devicesMtx_.
// Callers from freeImmediate() must acquire it before calling.
// Block suallocation reclaim uses state.mtx (already held by sweep callers)
// or acquired here for freeImmediate.
// The dedicated memory map access needs devicesMtx_ to be held by the caller.
//
// Bottom line: the caller must hold devicesMtx_ before calling doReclaim().
// freeImmediate() acquires devicesMtx_ for this purpose.

void VulkanMemoryPool::doReclaim(void* ptr, bool updateManagerAccounting) {
  // devicesMtx_ must be held by caller.

  VulkanAllocRecord rec;
  {
    std::lock_guard<std::mutex> rlk(registryMtx_);
    auto it = registry_.find(ptr);
    if (it == registry_.end()) return;
    rec = it->second;
    registry_.erase(it);
  }

  // VkBuffer lifetime ends only after the caller's retirement proof and
  // always before its bound memory is reclaimed or freed.
  if (rec.buffer != VK_NULL_HANDLE && rec.logicalDevice != VK_NULL_HANDLE) {
    vkDestroyBuffer(rec.logicalDevice, rec.buffer, nullptr);
  }

  if (updateManagerAccounting) {
    VulkanDeviceManager::getInstance().untrackAllocation(
        rec.deviceId, static_cast<size_t>(rec.reservedSize));
  }

  DeviceState* owningState = nullptr;
  if (rec.deviceId >= 0 &&
      static_cast<size_t>(rec.deviceId) < deviceStates_.size()) {
    owningState = deviceStates_[static_cast<size_t>(rec.deviceId)].get();
  }
  auto subtractAtomic = [](std::atomic<uint64_t>& counter, uint64_t bytes) {
    uint64_t current = counter.load(std::memory_order_relaxed);
    uint64_t desired;
    do {
      desired = current >= bytes ? current - bytes : 0u;
    } while (!counter.compare_exchange_weak(
        current, desired, std::memory_order_relaxed, std::memory_order_relaxed));
  };
  if (owningState != nullptr) {
    subtractAtomic(owningState->trackedBytes,
                   static_cast<uint64_t>(rec.reservedSize));
    if (rec.dedicated) {
      subtractAtomic(owningState->reservedBytes,
                     static_cast<uint64_t>(rec.reservedSize));
    }
  }

  if (rec.dedicated) {
    // Recover the VkDeviceMemory from the per-device dedicated map.
    // (devicesMtx_ is held by caller — DeviceState can be accessed directly.)
    VkDeviceMemory dedMem = VK_NULL_HANDLE;
    if (rec.deviceId >= 0 && static_cast<size_t>(rec.deviceId) < deviceStates_.size()
        && deviceStates_[static_cast<size_t>(rec.deviceId)]) {
      // state.mtx is NOT held here; we only access dedicatedMemMap which is
      // protected by devicesMtx_ (the caller's lock).
      auto& dedMap = deviceStates_[static_cast<size_t>(rec.deviceId)]->dedicatedMemMap;
      auto it2 = dedMap.find(ptr);
      if (it2 != dedMap.end()) {
        dedMem = it2->second;
        dedMap.erase(it2);
      }
    }
    if (dedMem != VK_NULL_HANDLE && rec.logicalDevice != VK_NULL_HANDLE) {
      if (rec.hostVisible) vkUnmapMemory(rec.logicalDevice, dedMem);
      vkFreeMemory(rec.logicalDevice, dedMem, nullptr);
    }
    // Remove sentinel if device-local.
    if (!rec.hostVisible) {
      std::lock_guard<std::mutex> slk(sentinelMtx_);
      sentinelStore_.erase(
        std::remove_if(sentinelStore_.begin(), sentinelStore_.end(),
                       [ptr](const std::unique_ptr<uint8_t>& p){ return p.get() == ptr; }),
        sentinelStore_.end());
    }
    return;
  }

  // Suballocated: return span to the block.
  // The Block lives inside a DeviceState which is accessed under devicesMtx_
  // (held by caller).  The state.mtx is NOT needed here because the block
  // pointer is stable and devicesMtx_ provides the serialization.
  Block* blk = static_cast<Block*>(rec.blockKey);
  if (blk) {
    blk->reclaim(rec.offsetInBlock, rec.reservedSize);
  }

  // Remove sentinel for device-local suballocations.
  if (!rec.hostVisible) {
    std::lock_guard<std::mutex> slk(sentinelMtx_);
    sentinelStore_.erase(
      std::remove_if(sentinelStore_.begin(), sentinelStore_.end(),
                     [ptr](const std::unique_ptr<uint8_t>& p){ return p.get() == ptr; }),
      sentinelStore_.end());
  }

}

// ── queryRecord ───────────────────────────────────────────────────────────

bool VulkanMemoryPool::queryRecord(void* ptr, VulkanAllocRecord& record) const {
  std::lock_guard<std::mutex> rlk(registryMtx_);
  auto it = registry_.find(ptr);
  if (it == registry_.end()) return false;
  record = it->second;
  return true;
}

// ── checked synchronous transfers ─────────────────────────────────────────

bool VulkanMemoryPool::createStagingBuffer(int deviceId, VkDeviceSize bytes,
                                           VkBufferUsageFlags usage,
                                           StagingBuffer& staging) {
  if (bytes == 0) return false;

  auto* context = VulkanDeviceContext::getContext(deviceId);
  if (context == nullptr || context->device() == VK_NULL_HANDLE ||
      context->physicalDevice() == VK_NULL_HANDLE) {
    sd_printf("VulkanMemoryPool::createStagingBuffer: no context for device=%d\n",
              deviceId);
    return false;
  }

  staging.device = context->device();

  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bytes;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result =
      vkCreateBuffer(staging.device, &bufferInfo, nullptr, &staging.buffer);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::createStagingBuffer: vkCreateBuffer failed "
              "device=%d res=%d\n", deviceId, static_cast<int>(result));
    destroyStagingBuffer(staging);
    return false;
  }

  VkMemoryRequirements requirements = {};
  vkGetBufferMemoryRequirements(staging.device, staging.buffer, &requirements);
  uint32_t memoryType = findMemoryType(
      context->physicalDevice(), requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (memoryType == UINT32_MAX) {
    sd_printf("VulkanMemoryPool::createStagingBuffer: no compatible "
              "HOST_VISIBLE|COHERENT memory type device=%d\n", deviceId);
    destroyStagingBuffer(staging);
    return false;
  }

  VkMemoryAllocateInfo allocationInfo = {};
  allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocationInfo.allocationSize = requirements.size;
  allocationInfo.memoryTypeIndex = memoryType;
  result =
      vkAllocateMemory(staging.device, &allocationInfo, nullptr, &staging.memory);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::createStagingBuffer: vkAllocateMemory failed "
              "device=%d res=%d\n", deviceId, static_cast<int>(result));
    destroyStagingBuffer(staging);
    return false;
  }

  result = vkBindBufferMemory(staging.device, staging.buffer, staging.memory, 0);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::createStagingBuffer: vkBindBufferMemory failed "
              "device=%d res=%d\n", deviceId, static_cast<int>(result));
    destroyStagingBuffer(staging);
    return false;
  }

  result = vkMapMemory(staging.device, staging.memory, 0, bytes, 0,
                       &staging.mapped);
  if (result != VK_SUCCESS || staging.mapped == nullptr) {
    sd_printf("VulkanMemoryPool::createStagingBuffer: vkMapMemory failed "
              "device=%d res=%d\n", deviceId, static_cast<int>(result));
    destroyStagingBuffer(staging);
    return false;
  }

  return true;
}

void VulkanMemoryPool::destroyStagingBuffer(StagingBuffer& staging) {
  if (staging.device != VK_NULL_HANDLE && staging.mapped != nullptr &&
      staging.memory != VK_NULL_HANDLE) {
    vkUnmapMemory(staging.device, staging.memory);
  }
  staging.mapped = nullptr;

  if (staging.device != VK_NULL_HANDLE &&
      staging.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(staging.device, staging.buffer, nullptr);
  }
  staging.buffer = VK_NULL_HANDLE;

  if (staging.device != VK_NULL_HANDLE &&
      staging.memory != VK_NULL_HANDLE) {
    vkFreeMemory(staging.device, staging.memory, nullptr);
  }
  staging.memory = VK_NULL_HANDLE;
  staging.device = VK_NULL_HANDLE;
}

bool VulkanMemoryPool::waitForDevice(int deviceId) {
  auto* context = VulkanDeviceContext::getContext(deviceId);
  if (context == nullptr || context->device() == VK_NULL_HANDLE) {
    sd_printf("VulkanMemoryPool::waitForDevice: no context for device=%d\n",
              deviceId);
    return false;
  }

  VkResult result = context->waitComputeIdle();
  if (result != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::waitForDevice: compute-queue wait failed "
              "device=%d res=%d\n", deviceId, static_cast<int>(result));
    if (result == VK_ERROR_DEVICE_LOST) context->markLost();
    return false;
  }
  return true;
}

bool VulkanMemoryPool::submitCopy(int deviceId, VkBuffer src, VkBuffer dst,
                                  VkDeviceSize bytes) {
  if (src == VK_NULL_HANDLE || dst == VK_NULL_HANDLE || bytes == 0) return false;
  if (!waitForDevice(deviceId)) return false;

  auto* context = VulkanDeviceContext::getContext(deviceId);
  VkDevice device = context != nullptr ? context->device() : VK_NULL_HANDLE;
  VkCommandPool commandPool =
      context != nullptr ? context->getThreadCommandPool() : VK_NULL_HANDLE;
  if (device == VK_NULL_HANDLE || commandPool == VK_NULL_HANDLE) {
    sd_printf("VulkanMemoryPool::submitCopy: Vulkan objects unavailable "
              "device=%d\n", deviceId);
    return false;
  }

  VkCommandBufferAllocateInfo allocationInfo = {};
  allocationInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocationInfo.commandPool = commandPool;
  allocationInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocationInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkResult result =
      vkAllocateCommandBuffers(device, &allocationInfo, &commandBuffer);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::submitCopy: vkAllocateCommandBuffers failed "
              "device=%d res=%d\n", deviceId, static_cast<int>(result));
    return false;
  }

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  if (result == VK_SUCCESS) {
    VkBufferCopy region = {};
    region.size = bytes;
    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &region);
    result = vkEndCommandBuffer(commandBuffer);
  }

  if (result == VK_SUCCESS) {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    result = context->submitCompute(1, &submitInfo, VK_NULL_HANDLE);
    if (result == VK_SUCCESS) result = context->waitComputeIdle();
  }

  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::submitCopy: command submission failed "
              "device=%d res=%d\n", deviceId, static_cast<int>(result));
    if (result == VK_ERROR_DEVICE_LOST && context != nullptr) context->markLost();
    return false;
  }
  return true;
}

bool VulkanMemoryPool::submitFill(int deviceId, VkBuffer dst,
                                  VkDeviceSize bytes, uint32_t pattern) {
  if (dst == VK_NULL_HANDLE || bytes == 0 || (bytes & 3u) != 0) return false;
  if (!waitForDevice(deviceId)) return false;

  auto* context = VulkanDeviceContext::getContext(deviceId);
  VkDevice device = context != nullptr ? context->device() : VK_NULL_HANDLE;
  VkCommandPool commandPool =
      context != nullptr ? context->getThreadCommandPool() : VK_NULL_HANDLE;
  if (device == VK_NULL_HANDLE || commandPool == VK_NULL_HANDLE) {
    sd_printf("VulkanMemoryPool::submitFill: Vulkan objects unavailable "
              "device=%d\n", deviceId);
    return false;
  }

  VkCommandBufferAllocateInfo allocationInfo = {};
  allocationInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocationInfo.commandPool = commandPool;
  allocationInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocationInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkResult result =
      vkAllocateCommandBuffers(device, &allocationInfo, &commandBuffer);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::submitFill: vkAllocateCommandBuffers failed "
              "device=%d res=%d\n", deviceId, static_cast<int>(result));
    return false;
  }

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  if (result == VK_SUCCESS) {
    vkCmdFillBuffer(commandBuffer, dst, 0, bytes, pattern);
    result = vkEndCommandBuffer(commandBuffer);
  }

  if (result == VK_SUCCESS) {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    result = context->submitCompute(1, &submitInfo, VK_NULL_HANDLE);
    if (result == VK_SUCCESS) result = context->waitComputeIdle();
  }

  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanMemoryPool::submitFill: command submission failed "
              "device=%d res=%d\n", deviceId, static_cast<int>(result));
    if (result == VK_ERROR_DEVICE_LOST && context != nullptr) context->markLost();
    return false;
  }
  return true;
}

bool VulkanMemoryPool::copyHostToDeviceAsync(
    void* dstDevice, const void* srcHost, VkDeviceSize bytes,
    VulkanExecutionStream* stream, VkDeviceSize dstOffset,
    VkDeviceSize srcOffset) {
  if (bytes == 0) return true;
  VulkanAllocRecord record;
  return dstDevice != nullptr && srcHost != nullptr && stream != nullptr &&
         queryRecord(dstDevice, record) &&
         dstOffset <= record.logicalSize &&
         bytes <= record.logicalSize - dstOffset &&
         stream->deviceId() == record.deviceId &&
         stream->enqueueCopy(dstDevice, srcHost, bytes, 1, dstOffset, srcOffset);
}

bool VulkanMemoryPool::copyDeviceToHostAsync(
    void* dstHost, void* srcDevice, VkDeviceSize bytes,
    VulkanExecutionStream* stream, VkDeviceSize dstOffset,
    VkDeviceSize srcOffset) {
  if (bytes == 0) return true;
  VulkanAllocRecord record;
  return dstHost != nullptr && srcDevice != nullptr && stream != nullptr &&
         queryRecord(srcDevice, record) &&
         srcOffset <= record.logicalSize &&
         bytes <= record.logicalSize - srcOffset &&
         stream->deviceId() == record.deviceId &&
         stream->enqueueCopy(dstHost, srcDevice, bytes, 2, dstOffset, srcOffset);
}

bool VulkanMemoryPool::copyDeviceToDeviceAsync(
    void* dstDevice, void* srcDevice, VkDeviceSize bytes,
    VulkanExecutionStream* stream, VkDeviceSize dstOffset,
    VkDeviceSize srcOffset) {
  if (bytes == 0) return true;
  if (dstDevice == nullptr || srcDevice == nullptr || stream == nullptr) return false;

  VulkanAllocRecord dstRecord;
  VulkanAllocRecord srcRecord;
  if (!queryRecord(dstDevice, dstRecord) ||
      !queryRecord(srcDevice, srcRecord) ||
      dstOffset > dstRecord.logicalSize ||
      bytes > dstRecord.logicalSize - dstOffset ||
      srcOffset > srcRecord.logicalSize ||
      bytes > srcRecord.logicalSize - srcOffset ||
      stream->deviceId() != dstRecord.deviceId) {
    return false;
  }

  if (dstDevice == srcDevice) {
    if (dstOffset == srcOffset) return true;
    const bool overlaps =
        dstOffset < srcOffset ? bytes > srcOffset - dstOffset
                              : bytes > dstOffset - srcOffset;
    if (overlaps) return false;
  }

  return stream->enqueueCopy(
      dstDevice, srcDevice, bytes, 3, dstOffset, srcOffset);
}

bool VulkanMemoryPool::fillAsync(void* dstDevice, int value,
                                 VkDeviceSize bytes,
                                 VulkanExecutionStream* stream) {
  if (bytes == 0) return true;
  VulkanAllocRecord record;
  return dstDevice != nullptr && stream != nullptr &&
         queryRecord(dstDevice, record) && bytes <= record.logicalSize &&
         stream->deviceId() == record.deviceId &&
         stream->enqueueFill(dstDevice, value, bytes);
}

bool VulkanMemoryPool::copyHostToDevice(void* dstDevice, const void* srcHost,
                                        VkDeviceSize bytes) {
  if (bytes == 0) return true;
  VulkanAllocRecord record;
  if (!queryRecord(dstDevice, record)) return false;
  auto* stream = VulkanExecutionStream::defaultCopy(record.deviceId);
  return copyHostToDeviceAsync(dstDevice, srcHost, bytes, stream) &&
         stream->synchronize();
}

bool VulkanMemoryPool::copyDeviceToHost(void* dstHost, void* srcDevice,
                                        VkDeviceSize bytes) {
  if (bytes == 0) return true;
  VulkanAllocRecord record;
  if (!queryRecord(srcDevice, record)) return false;
  auto* stream = VulkanExecutionStream::defaultCopy(record.deviceId);
  return copyDeviceToHostAsync(dstHost, srcDevice, bytes, stream) &&
         stream->synchronize();
}

bool VulkanMemoryPool::copyDeviceToDevice(void* dstDevice, void* srcDevice,
                                          VkDeviceSize bytes) {
  if (bytes == 0 || dstDevice == srcDevice) return true;
  VulkanAllocRecord dstRecord;
  if (!queryRecord(dstDevice, dstRecord)) return false;
  auto* stream = VulkanExecutionStream::defaultCopy(dstRecord.deviceId);
  return copyDeviceToDeviceAsync(dstDevice, srcDevice, bytes, stream) &&
         stream->synchronize();
}

bool VulkanMemoryPool::fill(void* dstDevice, int value, VkDeviceSize bytes) {
  if (bytes == 0) return true;
  VulkanAllocRecord record;
  if (!queryRecord(dstDevice, record)) return false;
  auto* stream = VulkanExecutionStream::defaultCopy(record.deviceId);
  return fillAsync(dstDevice, value, bytes, stream) &&
         stream->synchronize();
}

bool VulkanMemoryPool::getMemoryPoolStats(
    int deviceId, uint64_t& usedBytes, uint64_t& reservedBytes) const {
  usedBytes = 0;
  reservedBytes = 0;
  std::lock_guard<std::mutex> deviceGuard(devicesMtx_);
  if (deviceId < 0 || static_cast<size_t>(deviceId) >= deviceStates_.size() ||
      !deviceStates_[static_cast<size_t>(deviceId)]) {
    return false;
  }
  const DeviceState& state = *deviceStates_[static_cast<size_t>(deviceId)];
  usedBytes = state.trackedBytes.load(std::memory_order_relaxed);
  reservedBytes = state.reservedBytes.load(std::memory_order_relaxed);
  return true;
}

bool VulkanMemoryPool::getReusablePoolStats(
    int deviceId, uint64_t& reusableBytes, uint64_t& reusableSpanCount) const {
  reusableBytes = 0;
  reusableSpanCount = 0;
  std::lock_guard<std::mutex> deviceGuard(devicesMtx_);
  if (deviceId < 0 || static_cast<size_t>(deviceId) >= deviceStates_.size() ||
      !deviceStates_[static_cast<size_t>(deviceId)]) {
    return false;
  }

  const DeviceState& state = *deviceStates_[static_cast<size_t>(deviceId)];
  std::lock_guard<std::mutex> stateGuard(state.mtx);
  for (const auto& poolEntry : state.pools) {
    for (const auto& blockEntry : poolEntry.second.blocks) {
      if (!blockEntry) continue;
      reusableSpanCount += static_cast<uint64_t>(blockEntry->freeList.size());
      for (const auto& span : blockEntry->freeList) {
        reusableBytes += static_cast<uint64_t>(span.size);
      }
    }
  }
  return true;
}

bool VulkanMemoryPool::getLifetimeAllocationStats(
    int deviceId, uint64_t& totalAcquired, uint64_t& totalReused) const {
  totalAcquired = 0;
  totalReused = 0;
  std::lock_guard<std::mutex> deviceGuard(devicesMtx_);
  if (deviceId < 0 || static_cast<size_t>(deviceId) >= deviceStates_.size() ||
      !deviceStates_[static_cast<size_t>(deviceId)]) {
    return false;
  }

  const DeviceState& state = *deviceStates_[static_cast<size_t>(deviceId)];
  totalAcquired = state.totalAcquired.load(std::memory_order_relaxed);
  totalReused = state.totalReused.load(std::memory_order_relaxed);
  return true;
}

uint64_t VulkanMemoryPool::trim(int deviceId, uint64_t minimumBytesToKeep) {
  std::lock_guard<std::mutex> deviceGuard(devicesMtx_);
  if (deviceId < 0 || static_cast<size_t>(deviceId) >= deviceStates_.size() ||
      !deviceStates_[static_cast<size_t>(deviceId)]) {
    return 0;
  }

  DeviceState& state = *deviceStates_[static_cast<size_t>(deviceId)];
  std::lock_guard<std::mutex> stateGuard(state.mtx);
  uint64_t released = 0;
  for (auto& poolEntry : state.pools) {
    auto& blocks = poolEntry.second.blocks;
    for (auto it = blocks.begin(); it != blocks.end();) {
      Block* block = it->get();
      const uint64_t reserved =
          state.reservedBytes.load(std::memory_order_relaxed);
      if (block == nullptr || block->activeAllocs != 0 ||
          reserved <= minimumBytesToKeep) {
        ++it;
        continue;
      }
      if (block->mappedBase != nullptr) {
        vkUnmapMemory(block->logicalDevice, block->memory);
        block->mappedBase = nullptr;
      }
      if (block->memory != VK_NULL_HANDLE) {
        vkFreeMemory(block->logicalDevice, block->memory, nullptr);
      }
      const uint64_t blockBytes = static_cast<uint64_t>(block->blockSize);
      released += blockBytes;
      const uint64_t nextReserved =
          reserved >= blockBytes ? reserved - blockBytes : 0u;
      state.reservedBytes.store(nextReserved, std::memory_order_relaxed);
      state.blockIds.erase(static_cast<void*>(block));
      it = blocks.erase(it);
    }
  }
  return released;
}

// ── getFreeMemory ──────────────────────────────────────────────────────────

uint64_t VulkanMemoryPool::getFreeMemory(int deviceId) const {
  return VulkanDeviceManager::getInstance().getFreeMemory(deviceId);
}

// ── getDeviceId ────────────────────────────────────────────────────────────

int VulkanMemoryPool::getDeviceId(void* ptr) const {
  std::lock_guard<std::mutex> rlk(registryMtx_);
  auto it = registry_.find(ptr);
  if (it == registry_.end()) return -1;
  return it->second.deviceId;
}

int VulkanMemoryPool::getAllocationMemoryPropertyFlags(void* ptr,
                                                       int deviceId) const {
  std::lock_guard<std::mutex> rlk(registryMtx_);
  auto it = registry_.find(ptr);
  if (it == registry_.end() || it->second.deviceId != deviceId) return -1;
  return static_cast<int>(it->second.memoryPropertyFlags);
}

// ── getBuffer ─────────────────────────────────────────────────────────────

VkBuffer VulkanMemoryPool::getBuffer(void* ptr) const {
  std::lock_guard<std::mutex> rlk(registryMtx_);
  auto it = registry_.find(ptr);
  if (it == registry_.end()) return VK_NULL_HANDLE;
  return it->second.buffer;
}

// ── getBlockId (Gap M1) ───────────────────────────────────────────────────

int VulkanMemoryPool::getBlockId(void* ptr) const {
  // Look up the allocation record.
  VulkanAllocRecord rec;
  {
    std::lock_guard<std::mutex> rlk(registryMtx_);
    auto it = registry_.find(ptr);
    if (it == registry_.end()) return -1;
    rec = it->second;
  }
  // Dedicated allocations have no block.
  if (rec.dedicated || rec.blockKey == nullptr) return -1;

  // Look up the block id in the per-device state.
  std::lock_guard<std::mutex> dlk(devicesMtx_);
  if (rec.deviceId < 0 || static_cast<size_t>(rec.deviceId) >= deviceStates_.size()
      || !deviceStates_[static_cast<size_t>(rec.deviceId)]) {
    return -1;
  }
  const DeviceState& state = *deviceStates_[static_cast<size_t>(rec.deviceId)];
  auto it = state.blockIds.find(rec.blockKey);
  if (it == state.blockIds.end()) return -1;
  return it->second;
}

// ── getRetireListPendingCount (Gap M4) ────────────────────────────────────

int VulkanMemoryPool::getRetireListPendingCount(int deviceId) const {
  if (deviceId < 0) return 0;
  std::lock_guard<std::mutex> dlk(devicesMtx_);
  if (static_cast<size_t>(deviceId) >= deviceStates_.size()
      || !deviceStates_[static_cast<size_t>(deviceId)]) {
    return 0;
  }
  const DeviceState& state = *deviceStates_[static_cast<size_t>(deviceId)];
  std::lock_guard<std::mutex> slk(state.mtx);
  return static_cast<int>(state.retireList.size());
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
