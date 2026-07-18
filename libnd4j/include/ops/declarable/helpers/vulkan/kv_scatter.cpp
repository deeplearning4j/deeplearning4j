/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <array/DataTypeUtils.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <ops/declarable/helpers/kv_scatter.h>

#include <limits>
#include <sstream>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {
namespace {

graph::VulkanExecutionStream* resolveExecutionStream(LaunchContext* context,
                                                     int deviceId) {
  if (context == nullptr || context->getDeviceID() != deviceId) {
    THROW_EXCEPTION(
        "Vulkan KV scatter requires an exact-device launch context");
  }

  const auto contextStream = graph::vulkanExecutionStream(context);
  auto* stream = contextStream == nullptr
                     ? graph::VulkanExecutionStream::currentOrDefault(deviceId)
                     : graph::VulkanExecutionStream::fromOpaque(contextStream,
                                                                false);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != deviceId) {
    THROW_EXCEPTION(
        "Vulkan KV scatter could not resolve the exact-device execution stream");
  }
  return stream;
}

VkDeviceSize checkedProduct(LongType left, LongType right,
                            const char* description) {
  if (left < 0 || right < 0) {
    std::ostringstream message;
    message << "Vulkan KV scatter received a negative " << description;
    THROW_EXCEPTION(message.str().c_str());
  }

  const auto unsignedLeft = static_cast<uint64_t>(left);
  const auto unsignedRight = static_cast<uint64_t>(right);
  if (unsignedRight != 0 &&
      unsignedLeft > std::numeric_limits<uint64_t>::max() / unsignedRight) {
    std::ostringstream message;
    message << "Vulkan KV scatter overflowed " << description;
    THROW_EXCEPTION(message.str().c_str());
  }
  return static_cast<VkDeviceSize>(unsignedLeft * unsignedRight);
}

VkDeviceSize checkedAdd(VkDeviceSize left, VkDeviceSize right,
                        const char* description) {
  if (left > std::numeric_limits<VkDeviceSize>::max() - right) {
    std::ostringstream message;
    message << "Vulkan KV scatter overflowed " << description;
    THROW_EXCEPTION(message.str().c_str());
  }
  return left + right;
}

void validateRange(const graph::VulkanAllocRecord& record,
                   VkDeviceSize offset, VkDeviceSize bytes,
                   const char* role) {
  if (offset > record.logicalSize || bytes > record.logicalSize - offset) {
    std::ostringstream message;
    message << "Vulkan KV scatter " << role
            << " range exceeds its device allocation";
    THROW_EXCEPTION(message.str().c_str());
  }
}

LongType readPosition(const LongType* position,
                      graph::VulkanExecutionStream* stream,
                      int deviceId) {
  if (position == nullptr) {
    THROW_EXCEPTION("Vulkan KV scatter received a null cache-position pointer");
  }

  auto& pool = graph::VulkanMemoryPool::getInstance();
  graph::VulkanAllocRecord positionRecord;
  if (!pool.queryRecord(const_cast<LongType*>(position), positionRecord)) {
    // The public DSP ABI also permits caller-owned host scalar metadata.
    return *position;
  }

  if (positionRecord.deviceId != deviceId ||
      positionRecord.logicalSize < sizeof(LongType)) {
    THROW_EXCEPTION(
        "Vulkan KV scatter received an invalid device cache-position scalar");
  }

  LongType value = 0;
  if (!stream->enqueueCopy(&value, position, sizeof(value), 2) ||
      !stream->synchronize()) {
    THROW_EXCEPTION(
        "Vulkan KV scatter could not read its device cache-position scalar");
  }
  return value;
}

}  // namespace

void kvScatterBatched(const KvScatterEntry* entries, int numEntries,
                      DataType dtype, LaunchContext* context) {
  if (numEntries < 0) {
    THROW_EXCEPTION("Vulkan KV scatter received a negative entry count");
  }
  if (numEntries == 0) return;
  if (entries == nullptr) {
    THROW_EXCEPTION("Vulkan KV scatter received a null entry array");
  }

  // Reuse the dynamic-position Vulkan implementation. Autoregressive decode
  // submits one cache position for every K/V entry, so the common path remains
  // one batched device submission. Preserve the general helper contract by
  // falling back to one submission per entry only when callers provide mixed
  // positions.
  const LongType sharedPosition = entries[0].cachePos;
  bool positionsMatch = true;
  for (int i = 1; i < numEntries; ++i) {
    positionsMatch = positionsMatch && entries[i].cachePos == sharedPosition;
  }

  if (positionsMatch) {
    std::vector<KvScatterDynEntry> dynamicEntries;
    dynamicEntries.reserve(static_cast<size_t>(numEntries));
    for (int i = 0; i < numEntries; ++i) {
      const auto& entry = entries[i];
      dynamicEntries.push_back(KvScatterDynEntry{
          entry.srcPtr, entry.dstPtr, &sharedPosition, entry.heads,
          entry.srcSeqLen, entry.dstSeqLen, entry.dim, entry.lastPos});
    }
    kvScatterDynBatched(dynamicEntries.data(), numEntries, dtype, context);
    return;
  }

  for (int i = 0; i < numEntries; ++i) {
    const auto& entry = entries[i];
    const LongType position = entry.cachePos;
    const KvScatterDynEntry dynamicEntry{
        entry.srcPtr, entry.dstPtr, &position, entry.heads, entry.srcSeqLen,
        entry.dstSeqLen, entry.dim, entry.lastPos};
    kvScatterDynBatched(&dynamicEntry, 1, dtype, context);
  }
}

void kvScatterDynBatched(const KvScatterDynEntry* entries, int numEntries,
                         DataType dtype, LaunchContext* context) {
  if (numEntries < 0) {
    THROW_EXCEPTION("Vulkan KV scatter received a negative entry count");
  }
  if (numEntries == 0) return;
  if (entries == nullptr) {
    THROW_EXCEPTION("Vulkan KV scatter received a null entry array");
  }

  const size_t elementSize = DataTypeUtils::sizeOfElement(dtype);
  if (elementSize == 0) {
    THROW_EXCEPTION("Vulkan KV scatter received an unsupported data type");
  }

  const int deviceId = context == nullptr ? -1 : context->getDeviceID();
  auto* stream = resolveExecutionStream(context, deviceId);
  const LongType* const position = entries[0].kvPosPtr;
  const LongType cachePosition =
      readPosition(position, stream, deviceId);

  auto& pool = graph::VulkanMemoryPool::getInstance();
  for (int entryIndex = 0; entryIndex < numEntries; ++entryIndex) {
    const auto& entry = entries[entryIndex];
    if (entry.kvPosPtr != position) {
      THROW_EXCEPTION(
          "Vulkan KV scatter entries must share one cache-position pointer");
    }
    if (entry.srcPtr == nullptr || entry.dstPtr == nullptr) {
      THROW_EXCEPTION("Vulkan KV scatter received a null device buffer");
    }
    if (entry.heads < 0 || entry.srcSeqLen < 0 || entry.dstSeqLen < 0 ||
        entry.dim < 0 || entry.lastPos < 0 ||
        entry.lastPos >= entry.srcSeqLen ||
        cachePosition < 0 || cachePosition >= entry.dstSeqLen) {
      THROW_EXCEPTION("Vulkan KV scatter received invalid tensor metadata");
    }
    if (entry.heads == 0 || entry.dim == 0) continue;

    graph::VulkanAllocRecord sourceRecord;
    graph::VulkanAllocRecord destinationRecord;
    if (!pool.queryRecord(const_cast<void*>(entry.srcPtr), sourceRecord) ||
        !pool.queryRecord(entry.dstPtr, destinationRecord)) {
      THROW_EXCEPTION(
          "Vulkan KV scatter requires Vulkan-owned device allocations");
    }
    if (sourceRecord.deviceId != deviceId ||
        destinationRecord.deviceId != deviceId) {
      THROW_EXCEPTION(
          "Vulkan KV scatter buffers must reside on the execution device");
    }

    const VkDeviceSize rowBytes =
        checkedProduct(entry.dim, static_cast<LongType>(elementSize),
                       "row byte count");
    const VkDeviceSize sourceHeadStride =
        checkedProduct(entry.srcSeqLen, entry.dim,
                       "source head stride");
    const VkDeviceSize destinationHeadStride =
        checkedProduct(entry.dstSeqLen, entry.dim,
                       "destination head stride");
    const VkDeviceSize sourcePosition =
        checkedProduct(entry.lastPos, entry.dim,
                       "source position offset");
    const VkDeviceSize destinationPosition =
        checkedProduct(cachePosition, entry.dim,
                       "destination position offset");

    for (LongType head = 0; head < entry.heads; ++head) {
      const VkDeviceSize sourceElements =
          checkedAdd(checkedProduct(head,
                                    static_cast<LongType>(sourceHeadStride),
                                    "source head offset"),
                     sourcePosition, "source element offset");
      const VkDeviceSize destinationElements =
          checkedAdd(checkedProduct(head,
                                    static_cast<LongType>(destinationHeadStride),
                                    "destination head offset"),
                     destinationPosition, "destination element offset");
      const VkDeviceSize sourceOffset =
          checkedProduct(static_cast<LongType>(sourceElements),
                         static_cast<LongType>(elementSize),
                         "source byte offset");
      const VkDeviceSize destinationOffset =
          checkedProduct(static_cast<LongType>(destinationElements),
                         static_cast<LongType>(elementSize),
                         "destination byte offset");

      validateRange(sourceRecord, sourceOffset, rowBytes, "source");
      validateRange(destinationRecord, destinationOffset, rowBytes,
                    "destination");
      if (!stream->enqueueCopy(entry.dstPtr, entry.srcPtr, rowBytes, 3,
                               destinationOffset, sourceOffset)) {
        THROW_EXCEPTION(
            "Vulkan KV scatter device-to-device copy submission failed");
      }
    }
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
