/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/DspHashUtils.h>
#include <graph/DspPhaseUtils.h>
#include <graph/DspSegmentHelpers.h>
#include <graph/DspSegmentLifecycle.h>
#include <graph/GraphReplayHandle.h>
#include <graph/LegacyOpTypeCodes.h>
#include <graph/ModeContract.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/PlanExecutionContext.h>
#include <graph/DspDeviceDispatch.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <graph/vulkan/VulkanReplayHandle.h>

#include <graph/vulkan/VulkanPipelineCache.h>
#include <graph/vulkan/VulkanSegmentRecorder.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

namespace {

struct ExecutionBinding {
  PlanExecutionContext* context = nullptr;
  void* previousDspStream = nullptr;
  int previousDevice = -1;
};

struct SegmentDeviceBinding {
  bool active = false;
  int previousDevice = -1;
  void* previousDspStream = nullptr;
};

thread_local std::vector<ExecutionBinding> executionBindings;
thread_local SegmentDeviceBinding segmentDeviceBinding;

VulkanExecutionStream* resolveExecutionStream(void* stream, int deviceId = -1) {
  if (stream != nullptr) {
    auto* resolved = VulkanExecutionStream::fromOpaque(stream, false);
    if (resolved == nullptr || (deviceId >= 0 && resolved->deviceId() != deviceId)) {
      return nullptr;
    }
    return resolved;
  }
  return VulkanExecutionStream::currentOrDefault(deviceId);
}

int targetDeviceFor(const GraphSegment& segment, const NativeSlot* slots,
                    int numSlots) {
  if (segment.def.startSlot >= 0 && segment.def.startSlot < numSlots) {
    const int target = slots[segment.def.startSlot].targetDeviceId;
    if (target >= 0) return target;
  }
  if (segment.exec.replayHandle && segment.exec.replayHandle->getDeviceId() >= 0) {
    return segment.exec.replayHandle->getDeviceId();
  }
  return VulkanDeviceManager::currentDeviceId();
}

std::set<int> replayOwningDevices(const std::vector<GraphSegment>& segments) {
  std::set<int> result;
  auto remember = [&](const std::unique_ptr<GraphReplayHandle>& handle) {
    if (handle && handle->getDeviceId() >= 0) result.insert(handle->getDeviceId());
  };
  for (const auto& segment : segments) {
    remember(segment.exec.replayHandle);
    for (const auto& handle :
         segment.exec.compositeReplaySchedule.mergedReplayHandles) {
      remember(handle);
    }
    for (const auto& handle :
         segment.exec.compositeReplaySchedule.compositeReplayHandles) {
      remember(handle);
    }
  }
  return result;
}

void resetSegmentReplay(GraphSegment& segment) {
  segment.exec.replayHandle.reset();
  SegmentLifecycle::resetForResourceRelease(segment.exec);
  segment.exec.compositeReplaySchedule.mergedReplayHandles.clear();
  segment.exec.compositeReplaySchedule.compositeReplayHandles.clear();
  segment.exec.compositeReplaySchedule.units.clear();
  segment.exec.markArgsStale();
  segment.exec.gapOpsCapturedInGraph = false;
  segment.resolvedCpuBackend = nullptr;
}

}  // namespace

void* NativeDynamicShapePlan::platformGetExecutionStream() const {
  if (ownedVulkanStream_ != nullptr && ownedVulkanStream_->isActive()) {
    return ownedVulkanStream_;
  }

  auto& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize()) return nullptr;
  ownedVulkanStream_ =
      VulkanExecutionStream::create(VulkanDeviceManager::currentDeviceId());
  return ownedVulkanStream_;
}

Status NativeDynamicShapePlan::platformTryFrozenFastPath(
    NDArray** externalInputs, int numExternalInputs,
    NDArray** requestedOutputs, int numRequestedOutputs, void* stream) {
  const auto& contract = ModeContract::forMode(graphExecutionMode_);
  if (contract.isSlotBySlot || !contract.allowsFrozenFastPath ||
      !planLifecycle_.isInFrozenOrReplayState() || !allSegmentsReplayReady()) {
    return Status::MAYBE;
  }

  // Match CUDA's frozen fast-path contract: mutable external inputs are copied
  // into the stable, capture-owned staging buffers before any replay submission.
  // Vulkan command buffers bake descriptor addresses just as CUDA graphs bake
  // kernel arguments, so replaying without this refresh reuses capture-time data.
  if (auto* execCtx = static_cast<PlanExecutionContext*>(activeExecCtx_);
      execCtx != nullptr) {
    execCtx->execTarget = ExecTarget::GRAPH_REPLAY;
  }
  externalInputs = performPreReplaySync(
      externalInputs, numExternalInputs, stream, "vulkan_frozen_fast_path");
  if (externalInputs == nullptr && numExternalInputs > 0) {
    return Status::KERNEL_FAILURE;
  }

  // CUDA refreshes zero-copy view wrappers again in the frozen fast path,
  // because requested-output materialization replaces the slot wrapper after
  // each call. Re-mint the aliases over the stable staging buffers before the
  // Vulkan command buffer is replayed for the next input set.
  for (auto& segment : segments_) {
    refreshStaleViewWrappersInSegment(
        segment, externalInputs, numExternalInputs);
  }

  for (auto& segment : segments_) {
    if (segment.def.allFrozenConstants) {
      segment.exec.executionCount++;
      continue;
    }
    if (segment.def.selectedBackend != SelectedBackend::VULKAN_REPLAY) {
      SegmentLifecycle::markFailed(segment.exec,
                                   "vulkan_frozen_non_vulkan_segment",
                                   segment.def.startSlot, segment.def.endSlot);
      return Status::KERNEL_FAILURE;
    }
    bool usedGraph = false;
    const Status status = platformExecuteSegmentWithBackends(
        segment, externalInputs, numExternalInputs, stream, usedGraph);
    if (status != Status::OK || !usedGraph) return Status::KERNEL_FAILURE;
  }

  // Match CUDA's requested-output boundary contract. A replayed view still
  // aliases an internal/external buffer; materialize it before returning so the
  // next replay cannot mutate a previously returned Java result.
  for (int i = 0; i < numRequestedOutputs; ++i) {
    const int slot = requestedOutputSlotIndices_[i];
    if (slot >= 0 && slot < totalOutputSlots_) {
      NDArray* slotArray = outputSlots_[slot];
      if (slotArray != nullptr && slotArray->isView()) {
        materializeViewSlot(slot, "plan-output-view-boundary-frozen");
      }
    }
  }

  for (int i = 0; i < numRequestedOutputs; ++i) {
    const int slot = requestedOutputSlotIndices_[i];
    requestedOutputs[i] =
        slot >= 0 && slot < totalOutputSlots_ ? outputSlots_[slot] : nullptr;
  }
  incrementExecuteCount("native_replay");
  return Status::OK;
}

void NativeDynamicShapePlan::platformPreExecuteSetup(
    NDArray** /*externalInputs*/, int /*numExternalInputs*/, void* stream) {
  auto* resolved = resolveExecutionStream(stream);
  if (resolved == nullptr) {
    throw std::runtime_error("Vulkan DSP execution stream is unavailable");
  }
}

bool NativeDynamicShapePlan::platformShouldKeepSegmentCache(
    const GraphSegment& segment) const {
  return segment.def.selectedBackend == SelectedBackend::VULKAN_REPLAY &&
         ((segment.exec.replayHandle && segment.exec.replayHandle->isReady()) ||
          (segment.def.isCapturable && !segment.exec.compilationFailed));
}

void NativeDynamicShapePlan::platformPrecompileSegments(
    NDArray** /*externalInputs*/, int /*numExternalInputs*/) {
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SHAPES_FROZEN,
                                 "platformPrecompileSegments");
  // Vulkan pipelines are compiled by VulkanSegmentRecorder while recording the
  // exact segment command buffer. There is no separate CPU or CUDA compiler path.
}

bool NativeDynamicShapePlan::platformBindSegmentDevice(
    const GraphSegment& segment) {
  auto& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize()) return false;

  const int target = targetDeviceFor(segment, slots_, numSlots_);
  const int previous = VulkanDeviceManager::currentDeviceId();
  if (target < 0 || target >= manager.deviceCount()) return false;
  if (target == previous) return true;
  if (segmentDeviceBinding.active) return false;

  segmentDeviceBinding.active = true;
  segmentDeviceBinding.previousDevice = previous;
  segmentDeviceBinding.previousDspStream = dspGetExecutionStream();
  if (!manager.setCurrentDevice(target)) {
    segmentDeviceBinding = {};
    return false;
  }

  auto* targetStream = VulkanExecutionStream::currentOrDefault(target);
  if (targetStream == nullptr) {
    manager.setCurrentDevice(previous);
    segmentDeviceBinding = {};
    return false;
  }
  dspSetExecutionStream(targetStream);
  return true;
}

void NativeDynamicShapePlan::platformRestoreSegmentDevice() {
  if (!segmentDeviceBinding.active) return;
  auto saved = segmentDeviceBinding;
  segmentDeviceBinding = {};
  VulkanDeviceManager::getInstance().setCurrentDevice(saved.previousDevice);
  dspSetExecutionStream(saved.previousDspStream);
}

void NativeDynamicShapePlan::platformMigrateSegmentInputs(
    const GraphSegment& segment, NDArray** externalInputs,
    int numExternalInputs) {
  const int target = targetDeviceFor(segment, slots_, numSlots_);
  if (target < 0) return;

  auto& pool = VulkanMemoryPool::getInstance();
  auto* targetStream = resolveExecutionStream(dspGetExecutionStream(), target);
  if (targetStream == nullptr || targetStream->deviceId() != target) {
    throw std::runtime_error(
        "Vulkan segment migration requires the target device execution stream");
  }

  // External inputs are owned by the caller/Java affinity layer. Upload a
  // host-authoritative input directly to its target allocation when necessary,
  // but never silently move an allocation that belongs to another device.
  std::unordered_set<DataBuffer*> checkedExternalBuffers;
  auto requireExternalOnTarget = [&](NDArray* array) {
    if (array == nullptr || array->isEmpty() || array->dataBuffer() == nullptr ||
        !checkedExternalBuffers.insert(array->dataBuffer()).second) {
      return;
    }

    auto* buffer = array->dataBuffer();
    void* special = buffer->special();
    if (special == nullptr) {
      buffer->allocateSpecial();
      special = buffer->special();
    }
    const int owner = pool.getDeviceId(special);
    if (owner != target) {
      std::string reason;
      VulkanExecutionStream::isCrossDeviceCopySupported(owner, target, &reason);
      throw std::runtime_error(
          "Vulkan external segment input belongs to device " +
          std::to_string(owner) + ", not target device " +
          std::to_string(target) +
          "; caller affinity replication is required (" + reason + ")");
    }

    buffer->waitForSpecialWriteEvent(targetStream);
    if (!buffer->isSpecialActual()) {
      if (buffer->primary() == nullptr || !buffer->isPrimaryActual()) {
        throw std::runtime_error(
            "Vulkan external segment input has no authoritative source");
      }
      if (!pool.copyHostToDeviceAsync(
              special, buffer->primary(),
              static_cast<VkDeviceSize>(buffer->getLenInBytes()),
              targetStream)) {
        throw std::runtime_error(
            "Vulkan external segment input H2D enqueue failed");
      }
      buffer->writeSpecial();
      buffer->recordSpecialWriteEvent(targetStream);
    }
  };

  std::unordered_set<int> neededInputSlots;
  for (int slotIndex = segment.def.startSlot;
       slotIndex <= segment.def.endSlot && slotIndex < numSlots_; ++slotIndex) {
    const NativeSlot& slot = slots_[slotIndex];
    for (int input = 0; input < slot.wiring.numInputs; ++input) {
      const int source = slot.wiring.inputSourceIndices[input];
      if (source >= 0 && source < totalOutputSlots_) {
        if (outputSlots_[source] != nullptr) neededInputSlots.insert(source);
      } else {
        const int external = -(source + 1);
        if (externalInputs != nullptr && external >= 0 &&
            external < numExternalInputs) {
          requireExternalOnTarget(externalInputs[external]);
        }
      }
    }
  }

  if (!migratedInputs_.empty()) {
    throw std::runtime_error(
        "Vulkan segment migration state was not cleaned up by the prior segment");
  }

  try {
    for (int slotIndex : neededInputSlots) {
      NDArray* sourceArray = outputSlots_[slotIndex];
      if (sourceArray == nullptr || sourceArray->isEmpty() ||
          sourceArray->dataBuffer() == nullptr) {
        continue;
      }

      auto* sourceBuffer = sourceArray->dataBuffer();
      if (!sourceBuffer->isSpecialActual()) {
        throw std::runtime_error(
            "Vulkan internal segment input is not device-authoritative");
      }
      void* sourcePointer = sourceBuffer->special();
      VulkanAllocRecord sourceRecord;
      if (sourcePointer == nullptr ||
          !pool.queryRecord(sourcePointer, sourceRecord)) {
        throw std::runtime_error(
            "Vulkan internal segment input lacks exact allocation ownership");
      }

      if (sourceRecord.deviceId == target) {
        sourceBuffer->waitForSpecialWriteEvent(targetStream);
        continue;
      }

      std::string reason;
      if (!VulkanExecutionStream::isCrossDeviceCopySupported(
              sourceRecord.deviceId, target, &reason)) {
        throw std::runtime_error(
            "Vulkan internal segment peer migration is unsupported: " + reason);
      }
      if (sourceRecord.logicalSize == 0) continue;

      // Clone the complete backing allocation, not merely lengthOf() bytes. This
      // preserves the exact offset/stride semantics of views without a host
      // materialization pass. The temporary NDArray keeps the original shape,
      // strides and offset while owning the target-device allocation.
      void* targetPointer = pool.allocate(target, sourceRecord.logicalSize);
      if (targetPointer == nullptr) {
        throw std::runtime_error(
            "Vulkan internal segment peer allocation failed");
      }
      if (!pool.copyDeviceToDeviceAsync(
              targetPointer, sourcePointer, sourceRecord.logicalSize,
              targetStream)) {
        pool.freeImmediate(targetPointer);
        throw std::runtime_error(
            "Vulkan internal segment peer transfer enqueue failed");
      }

      NDArray* migrated = nullptr;
      try {
        migrated = new NDArray(
            nullptr, targetPointer,
            const_cast<LongType*>(sourceArray->shapeInfo()),
            LaunchContext::defaultContext(), false, true,
            sourceArray->offset());
        // The raw constructor transfers device-buffer ownership to the
        // DataBuffer. Keep the local pointer only until construction succeeds
        // so the submitted allocation can still be retired on failure.
        targetPointer = nullptr;
        migrated->dataBuffer()->writeSpecial();
        migrated->dataBuffer()->recordSpecialWriteEvent(targetStream);
      } catch (...) {
        if (migrated != nullptr) {
          if (migrated->dataBuffer() != nullptr &&
              migrated->dataBuffer()->special() != nullptr) {
            migrated->dataBuffer()->freeGpuOnStream(targetStream);
          }
          delete migrated;
        } else if (targetPointer != nullptr) {
          targetStream->retireAllocation(targetPointer);
        }
        throw;
      }

      sourceBuffer->readSpecial();
      migratedInputs_.push_back({slotIndex, sourceArray, migrated});
      outputSlots_[slotIndex] = migrated;
    }
  } catch (...) {
    platformCleanupMigratedInputs();
    throw;
  }
}

void NativeDynamicShapePlan::platformCleanupMigratedInputs() {
  for (auto& migratedInput : migratedInputs_) {
    if (outputSlots_ != nullptr && migratedInput.outputSlotIdx >= 0 &&
        migratedInput.outputSlotIdx < totalOutputSlots_) {
      outputSlots_[migratedInput.outputSlotIdx] = migratedInput.original;
    }

    if (migratedInput.migrated != nullptr) {
      auto* buffer = migratedInput.migrated->dataBuffer();
      if (buffer != nullptr && buffer->special() != nullptr) {
        const int owner =
            VulkanMemoryPool::getInstance().getDeviceId(buffer->special());
        auto* stream = VulkanExecutionStream::currentOrDefault(owner);
        if (stream == nullptr || stream->deviceId() != owner) {
          throw std::runtime_error(
              "Vulkan migrated input cleanup lacks its owning stream");
        }
        buffer->freeGpuOnStream(stream);
      }
      delete migratedInput.migrated;
      migratedInput.migrated = nullptr;
    }
  }
  migratedInputs_.clear();
}

NDArray* NativeDynamicShapePlan::platformGetOutputForDevice0(
    NDArray* array, int /*slotIdx*/, int /*outputIdx*/) {
  if (array == nullptr || array->isEmpty() || array->dataBuffer() == nullptr) {
    return array;
  }
  if (array->isView()) {
    throw std::runtime_error(
        "Vulkan requested output view reached the peer boundary without "
        "device-side materialization");
  }

  auto& pool = VulkanMemoryPool::getInstance();
  auto* sourceBuffer = array->dataBuffer();
  if (!sourceBuffer->isSpecialActual()) {
    throw std::runtime_error(
        "Vulkan requested output is not device-authoritative");
  }

  void* sourcePointer = sourceBuffer->special();
  VulkanAllocRecord sourceRecord;
  if (sourcePointer == nullptr ||
      !pool.queryRecord(sourcePointer, sourceRecord)) {
    throw std::runtime_error(
        "Vulkan requested output lacks exact allocation ownership");
  }
  if (sourceRecord.deviceId == 0) return array;

  std::string reason;
  if (!VulkanExecutionStream::isCrossDeviceCopySupported(
          sourceRecord.deviceId, 0, &reason)) {
    throw std::runtime_error(
        "Vulkan requested output peer migration is unsupported: " + reason);
  }

  auto& manager = VulkanDeviceManager::getInstance();
  const int previousDevice = VulkanDeviceManager::currentDeviceId();
  if (!manager.setCurrentDevice(0)) {
    throw std::runtime_error(
        "Vulkan requested output could not bind primary device");
  }

  void* targetPointer = nullptr;
  NDArray* migrated = nullptr;
  VulkanExecutionStream* targetStream = nullptr;
  try {
    targetStream = VulkanExecutionStream::currentOrDefault(0);
    if (targetStream == nullptr || targetStream->deviceId() != 0) {
      throw std::runtime_error(
          "Vulkan requested output lacks the primary-device stream");
    }

    targetPointer = pool.allocate(0, sourceRecord.logicalSize);
    if (targetPointer == nullptr) {
      throw std::runtime_error(
          "Vulkan requested output peer allocation failed");
    }
    if (!pool.copyDeviceToDeviceAsync(
            targetPointer, sourcePointer, sourceRecord.logicalSize,
            targetStream)) {
      pool.freeImmediate(targetPointer);
      targetPointer = nullptr;
      throw std::runtime_error(
          "Vulkan requested output peer transfer enqueue failed");
    }

    migrated = new NDArray(
        nullptr, targetPointer, const_cast<LongType*>(array->shapeInfo()),
        LaunchContext::defaultContext(), false, true, array->offset());
    targetPointer = nullptr;
    migrated->dataBuffer()->writeSpecial();
    migrated->dataBuffer()->recordSpecialWriteEvent(targetStream);
    sourceBuffer->readSpecial();
  } catch (...) {
    if (migrated != nullptr) {
      auto* buffer = migrated->dataBuffer();
      if (buffer != nullptr && buffer->special() != nullptr &&
          targetStream != nullptr) {
        buffer->freeGpuOnStream(targetStream);
      }
      delete migrated;
    } else if (targetPointer != nullptr && targetStream != nullptr) {
      targetStream->retireAllocation(targetPointer);
    }
    if (previousDevice >= 0) manager.setCurrentDevice(previousDevice);
    throw;
  }

  if (previousDevice >= 0 && !manager.setCurrentDevice(previousDevice)) {
    if (migrated != nullptr && migrated->dataBuffer() != nullptr &&
        targetStream != nullptr) {
      migrated->dataBuffer()->freeGpuOnStream(targetStream);
    }
    delete migrated;
    throw std::runtime_error(
        "Vulkan requested output could not restore the prior device");
  }
  return migrated;
}

NDArray** NativeDynamicShapePlan::ensureAndSyncStagingBuffers(
    NDArray** externalArrays, int numExt, void* stream) {
  if (externalArrays == nullptr || numExt <= 0 ||
      externalInputIsVariable_.empty()) {
    return externalArrays;
  }

  auto* executionStream = resolveExecutionStream(stream);
  if (executionStream == nullptr) return nullptr;

  if (placeholderStagingBuffers_ == nullptr) {
    placeholderStagingBuffers_ = new NDArray*[numExt]();
    effectiveExternals_ = new NDArray*[numExt]();
  }
  std::memcpy(effectiveExternals_, externalArrays,
              sizeof(NDArray*) * static_cast<size_t>(numExt));

  if (cachedVariableExtIndices_.empty()) {
    for (int i = 0; i < numExt &&
                    i < static_cast<int>(externalInputIsVariable_.size()); ++i) {
      if (externalInputIsVariable_[i]) cachedVariableExtIndices_.push_back(i);
    }
  }

  for (int index : cachedVariableExtIndices_) {
    if (index < 0 || index >= numExt) continue;
    NDArray* source = externalArrays[index];
    if (source == nullptr || source->isEmpty() ||
        source->dataBuffer() == nullptr) {
      continue;
    }

    // A strided view must keep its descriptor offset/strides. Raw byte staging
    // would change its semantics, so bind it directly and require address
    // stability from the replay recorder.
    if (!shape::strideDescendingCAscendingF(source->shapeInfo())) {
      effectiveExternals_[index] = source;
      continue;
    }

    NDArray*& staging = placeholderStagingBuffers_[index];
    if (staging == nullptr) {
      staging = new NDArray(source->ordering(), *source->getShapeAsVector(),
                            source->dataType(), LaunchContext::defaultContext());
      staging->dataBuffer()->syncToSpecial(false);
    }
    if (staging->lengthOf() != source->lengthOf() ||
        staging->dataType() != source->dataType()) {
      return nullptr;
    }

    const bool directWrite =
        index < static_cast<int>(deviceWritePending_.size()) &&
        deviceWritePending_[index];
    if (directWrite) {
      deviceWritePending_[index] = false;
    } else {
      const size_t bytes = static_cast<size_t>(source->lengthOf()) *
                           static_cast<size_t>(source->sizeOfT());
      if (bytes > 0 &&
          !executionStream->enqueueCopy(staging->specialBuffer(),
                                        source->specialBuffer(), bytes, 3)) {
        return nullptr;
      }
    }
    staging->dataBuffer()->writeSpecial();
    effectiveExternals_[index] = staging;
  }

  stagingMaintainedThisExec_ = true;
  return effectiveExternals_;
}

NDArray** NativeDynamicShapePlan::performPreReplaySync(
    NDArray** externalArrays, int numExt, void* stream, const char* diagTag) {
  if (numExt <= 0) return externalArrays;
  if (externalArrays == nullptr) return nullptr;

  auto* executionStream = resolveExecutionStream(stream);
  if (executionStream == nullptr) return nullptr;
  VulkanExecutionStreamGuard guard(executionStream);

  std::vector<NDArray*> reads;
  reads.reserve(static_cast<size_t>(std::max(0, numExt)));
  for (int i = 0; i < numExt; ++i) {
    if (externalArrays != nullptr && externalArrays[i] != nullptr &&
        !externalArrays[i]->isEmpty()) {
      reads.push_back(externalArrays[i]);
    }
  }
  if (!reads.empty()) {
    NDArray::prepareSpecialUse({}, reads);
    NDArray::registerSpecialUse({}, reads);
  }

  NDArray** effective =
      ensureAndSyncStagingBuffers(externalArrays, numExt, executionStream);
  if (effective == nullptr) {
    DSP_DIAG(EXECUTE, "%s: Vulkan pre-replay staging failed", diagTag);
    return nullptr;
  }
  verifyStagingNotStale(externalArrays, effective, numExt, executionStream,
                        diagTag);
  return effective;
}

void NativeDynamicShapePlan::verifyStagingNotStale(
    NDArray** /*externalArrays*/, NDArray** /*effectiveArrays*/, int numExt,
    void* /*stream*/, const char* diagTag) {
  if (placeholderStagingBuffers_ == nullptr) return;
  for (int index : cachedVariableExtIndices_) {
    if (index < 0 || index >= numExt ||
        placeholderStagingBuffers_[index] == nullptr) {
      continue;
    }
    void* address = placeholderStagingBuffers_[index]->specialBuffer();
    auto previous = prevStagingAddresses_.find(index);
    if (previous != prevStagingAddresses_.end() &&
        previous->second != address) {
      DSP_DIAG(VERIFY,
               "%s: Vulkan staging address changed for external input %d",
               diagTag, index);
      throw std::runtime_error(
          "Vulkan replay staging address changed after capture");
    }
    prevStagingAddresses_[index] = address;
  }
}

bool NativeDynamicShapePlan::platformShouldUseGraph(
    const GraphSegment& segment) {
  return segment.def.isCapturable &&
         segment.def.selectedBackend == SelectedBackend::VULKAN_REPLAY;
}

Status NativeDynamicShapePlan::platformExecuteSegmentWithBackends(
    GraphSegment& segment, NDArray** externalInputs, int numExternalInputs,
    void* stream, bool& usedGraph) {
  usedGraph = false;
  auto fail = [&](const char* reason) {
    SegmentLifecycle::markFailed(segment.exec, reason, segment.def.startSlot,
                                 segment.def.endSlot);
    return Status::KERNEL_FAILURE;
  };

  if (segment.def.selectedBackend != SelectedBackend::VULKAN_REPLAY) {
    return fail("vulkan_backend_mismatch");
  }

  VulkanPipelineCache::ScopedCompilationPolicy compilationPolicy(
      runtimeCompilationAllowed_, runtimeArtifactDirectory_);
  const int deviceId = VulkanDeviceManager::currentDeviceId();
  // platformBindSegmentDevice routes the DSP stream to the segment's target
  // device. Keep warmup, capture, and replay on that same bound stream.
  auto* executionStream =
      resolveExecutionStream(dspGetExecutionStream(), deviceId);
  auto* deviceContext = VulkanDeviceContext::getContext(deviceId);
  if (executionStream == nullptr || deviceContext == nullptr ||
      deviceContext->isLost()) {
    return fail("vulkan_execution_context_unavailable");
  }
  VulkanExecutionStreamGuard streamGuard(executionStream);

  auto collectOperands = [&](const NativeSlot& slot, NDArray** inputs,
                             int& numInputs, NDArray** outputs,
                             int& numOutputs) {
    if (slot.wiring.numInputs < 0 || slot.wiring.numInputs > 16 ||
        slot.wiring.numOutputs < 0 || slot.wiring.numOutputs > 16) {
      return false;
    }
    numInputs = slot.wiring.numInputs;
    numOutputs = slot.wiring.numOutputs;
    for (int input = 0; input < numInputs; ++input) {
      const int source = slot.wiring.inputSourceIndices[input];
      if (source >= 0) {
        inputs[input] = getSlotOutputArray(source);
      } else {
        const int external = -(source + 1);
        inputs[input] =
            externalInputs != nullptr && external >= 0 &&
                    external < numExternalInputs
                ? externalInputs[external]
                : nullptr;
      }
      if (inputs[input] == nullptr) return false;
    }
    for (int output = 0; output < numOutputs; ++output) {
      const int slotIndex = slot.wiring.outputSlotIndices[output];
      outputs[output] =
          slotIndex >= 0 ? getSlotOutputArray(slotIndex) : nullptr;
      if (outputs[output] == nullptr) return false;
    }
    return true;
  };

  // Match CUDA graph capture: resolved views and forward identity slots
  // are zero-compute metadata nodes. VIEW_PRODUCING only expresses capability;
  // warmup may resolve a concrete invocation to an owned output (for example, a
  // non-unit-stride slice). BACKWARD identity descriptors also carry real copy
  // semantics: their op-local trait selects the trailing gradient payload.
  auto slotRecordsDeviceWork = [](const NativeSlot& slot) {
    const bool resolvedIdentityAlias =
        slot.isIdentityOp() &&
        !slot.hasOpTrait(sd::ops::OP_TRAIT_BACKWARD);
    return !slot.slotPhase.isViewProducer && !resolvedIdentityAlias;
  };
  bool segmentRecordsDeviceWork = false;
  for (int slotIndex = segment.def.startSlot;
       slotIndex <= segment.def.endSlot; ++slotIndex) {
    if (slotRecordsDeviceWork(slots_[slotIndex])) {
      segmentRecordsDeviceWork = true;
      break;
    }
  }

  if (segment.exec.replayHandle) {
    auto* handle =
        dynamic_cast<VulkanReplayHandle*>(segment.exec.replayHandle.get());
    if (handle == nullptr || !handle->isReady() ||
        handle->getDeviceId() != deviceId ||
        (segmentRecordsDeviceWork && !handle->replayDoesCompute()) ||
        handle->getRecorder() == nullptr) {
      return fail("vulkan_replay_handle_invalid");
    }
    if (!handle->getRecorder()->prepareReplayInputs(executionStream) ||
        !handle->replay(executionStream)) {
      return fail("vulkan_replay_submission_failed");
    }
    handle->getRecorder()->markReplayOutputs();
    segment.exec.lastReplayExecCount = executeCount_;
    totalGraphReplays_++;
    dspSegIncrementExecCount(segment, "vulkan-command-buffer-replay");
    usedGraph = true;
    return Status::OK;
  }

  // Match the shared GPU lifecycle: the first execution is the normal
  // slot-by-slot warmup that materializes output arrays and value-dependent
  // shape metadata. Vulkan recordability and pipeline construction operate on
  // those resolved arrays on the following NEEDS_COMPILE execution.
  if (segment.exec.segPhase.needsWarmup()) {
    return segDispatchWarmup(segment, externalInputs, numExternalInputs,
                             executionStream);
  }

  const LongType segmentShapeKey =
      computeSegmentShapeKey(segment, externalInputs, numExternalInputs);
  segment.def.shapeKeyState.recordComputed(segmentShapeKey);

  for (int slotIndex = segment.def.startSlot;
       slotIndex <= segment.def.endSlot; ++slotIndex) {
    const NativeSlot& slot = slots_[slotIndex];
    if (!slotRecordsDeviceWork(slot)) continue;
    NDArray* inputs[16] = {};
    NDArray* outputs[16] = {};
    int numInputs = 0;
    int numOutputs = 0;
    if (!collectOperands(slot, inputs, numInputs, outputs, numOutputs) ||
        !VulkanSegmentRecorder::opIsRecordable(
            slot, inputs, numInputs, outputs, numOutputs,
            deviceContext->caps())) {
      return fail("vulkan_segment_contains_unrecordable_op");
    }
  }

  if (!segment.exec.segPhase.needsCompile() &&
      !segment.exec.segPhase.needsCapture()) {
    return fail("vulkan_invalid_compile_or_capture_phase");
  }

  segment.exec.replayHandle =
      GraphReplayFactory::create(ReplayBackend::VULKAN, deviceId);
  auto* handle =
      dynamic_cast<VulkanReplayHandle*>(segment.exec.replayHandle.get());
  if (handle == nullptr || !handle->beginCapture(executionStream)) {
    return fail("vulkan_begin_capture_failed");
  }

  auto recorder = std::make_unique<VulkanSegmentRecorder>(handle);
  for (int slotIndex = segment.def.startSlot;
       slotIndex <= segment.def.endSlot; ++slotIndex) {
    const NativeSlot& slot = slots_[slotIndex];
    if (!slotRecordsDeviceWork(slot)) continue;
    NDArray* inputs[16] = {};
    NDArray* outputs[16] = {};
    int numInputs = 0;
    int numOutputs = 0;
    RandomGenerator* randomState =
        contextPool_ != nullptr && contextPool_[slotIndex] != nullptr
            ? &contextPool_[slotIndex]->randomGenerator()
            : nullptr;
    if (!collectOperands(slot, inputs, numInputs, outputs, numOutputs) ||
        !recorder->recordOp(slot, inputs, numInputs, outputs, numOutputs,
                            randomState)) {
      return fail("vulkan_record_op_failed");
    }
  }

  // VulkanSegmentRecorder resolves bundle-owned SPIR-V pipelines (or, in a
  // development build, compiles them) while recording the command buffer. Once
  // every slot has been resolved and recorded, advance through the same
  // NEEDS_COMPILE -> CAPTURE_PENDING transition used by CUDA.
  if (segment.exec.segPhase.needsCompile()) {
    SegmentLifecycle::markCompiled(segment.exec, "vulkan-native",
                                   segmentShapeKey);
    segment.def.shapeKeyState.markCompiled(segmentShapeKey);
    if (!planLifecycle_.isSlotBySlot()) {
      segment.exec.cachedShapeKey = segmentShapeKey;
    }
  }
  if (!segment.exec.segPhase.needsCapture()) {
    return fail("vulkan_invalid_capture_phase");
  }

  if ((segmentRecordsDeviceWork && handle->getNumDispatches() <= 0) ||
      !handle->endCapture(executionStream) || !handle->finalize() ||
      !recorder->prepareReplayInputs(executionStream) ||
      !handle->replay(executionStream)) {
    return fail("vulkan_capture_finalize_or_submit_failed");
  }
  recorder->markReplayOutputs();
  handle->setRecorder(std::move(recorder));
  handle->snapshotExternalAddresses(externalInputs, numExternalInputs);

  const LongType inputAddressKey = computeSegmentInputAddrKeyPortable(
      segment, externalInputs, numExternalInputs);
  const LongType slotAddressHash = static_cast<LongType>(
      dsp::computeSlotAddrHash(
          outputSlots_, segment.def.startSlot, segment.def.endSlot,
          totalOutputSlots_, [](NDArray* array) -> void* {
            return array != nullptr && array->dataBuffer() != nullptr
                       ? array->dataBuffer()->special()
                       : nullptr;
          }));
  SegmentLifecycle::markCaptured(segment.exec, inputAddressKey, 0,
                                 slotAddressHash, "vulkan-native");
  segment.exec.lastReplayExecCount = executeCount_;
  totalGraphReplays_++;
  dspSegIncrementExecCount(segment,
                           "vulkan-command-buffer-capture-replay");
  usedGraph = true;
  return Status::OK;
}

Status NativeDynamicShapePlan::platformCheckPostSegment(
    GraphSegment& /*segment*/) {
  return dspPeekLastCudaError() == 0 ? Status::OK : Status::KERNEL_FAILURE;
}

Status NativeDynamicShapePlan::postGraphReplayFixup(
    GraphSegment& segment, NDArray** /*externalArrays*/, int /*numExt*/,
    void* /*stream*/, const char* /*diagTag*/) {
  auto* handle =
      dynamic_cast<VulkanReplayHandle*>(segment.exec.replayHandle.get());
  if (handle == nullptr || handle->getRecorder() == nullptr) {
    return Status::KERNEL_FAILURE;
  }
  handle->getRecorder()->markReplayOutputs();
  return Status::OK;
}

Status NativeDynamicShapePlan::replayMonolithicGraph(
    GraphSegment& segment, NDArray** externalArrays, int numExt, void* stream,
    const char* /*diagTag*/) {
  bool usedGraph = false;
  const Status status = platformExecuteSegmentWithBackends(
      segment, externalArrays, numExt, stream, usedGraph);
  return status == Status::OK && usedGraph ? Status::OK
                                           : Status::KERNEL_FAILURE;
}

void NativeDynamicShapePlan::platformCleanupSegmentForRebuild(
    GraphSegment& segment) {
  if (segment.exec.replayHandle) {
    auto* handle =
        dynamic_cast<VulkanReplayHandle*>(segment.exec.replayHandle.get());
    if (handle != nullptr && handle->waitForReplayIdle() != VK_SUCCESS) {
      throw std::runtime_error(
          "Vulkan segment replay did not complete before rebuild");
    }
  }
  resetSegmentReplay(segment);
}

void NativeDynamicShapePlan::platformFreePlanResources() {
  const auto devices = replayOwningDevices(segments_);
  for (int device : devices) {
    if (!VulkanExecutionStream::synchronizeDevice(device)) {
      throw std::runtime_error(
          "Vulkan plan stream synchronization failed during teardown");
    }
  }
  for (auto& segment : segments_) resetSegmentReplay(segment);
  graphPinnedAddrs_.clear();
  if (steadyStateExecCtx_ != nullptr) {
    delete static_cast<PlanExecutionContext*>(steadyStateExecCtx_);
    steadyStateExecCtx_ = nullptr;
  }
  if (ownedVulkanStream_ != nullptr) {
    auto* stream = ownedVulkanStream_;
    ownedVulkanStream_ = nullptr;
    if (!VulkanExecutionStream::destroy(stream)) {
      throw std::runtime_error(
          "Vulkan plan-owned stream destruction failed during teardown");
    }
  }
}

void NativeDynamicShapePlan::platformFreeCaptureWorkspace() {
  // Vulkan replay resources are owned by each VulkanReplayHandle. Stream-ordered
  // staging allocations are reclaimed by their submission-completion callbacks.
}

int NativeDynamicShapePlan::platformCountCapturedGraphSegments() const {
  int count = 0;
  for (const auto& segment : segments_) {
    if ((segment.exec.replayHandle &&
         segment.exec.replayHandle->isReady()) ||
        hasCompositeHandles(segment)) {
      ++count;
    }
  }
  return count;
}

void NativeDynamicShapePlan::platformMaybeSplitIfEnabled() {
  // Common segmentation already uses the Vulkan emitter traits and memory
  // budget. Vulkan intentionally keeps fusible ranges whole.
}

void* NativeDynamicShapePlan::platformBeginExecution(
    void* stream, bool frozen, int execCount) {
  const int device = VulkanDeviceManager::currentDeviceId();
  void* requestedStream =
      stream != nullptr ? stream : platformGetExecutionStream();
  auto* executionStream = resolveExecutionStream(requestedStream, device);
  if (executionStream == nullptr) return nullptr;

  PlanExecutionContext* context = nullptr;
  if (frozen && execCount > 2 && steadyStateExecCtx_ != nullptr) {
    context = static_cast<PlanExecutionContext*>(steadyStateExecCtx_);
    context->resetSyncPhase();
    context->flowEventCount = 0;
    context->streamSyncCount = 0;
    context->eventSyncCount = 0;
  } else {
    context = new PlanExecutionContext();
    if (frozen && execCount > 1 && steadyStateExecCtx_ == nullptr) {
      steadyStateExecCtx_ = context;
    }
  }

  context->execCount = execCount;
  context->frozen = frozen;
  context->deviceId = device;
  context->dspStream = executionStream;
  context->lcDefaultStream = VulkanExecutionStream::defaultExecution(device);
  const bool warmup = anySegmentNeedsWarmup();
  context->needsFullSync = !frozen || execCount <= 1 || warmup;
  context->isFrozenSteadyState = frozen && execCount > 1 && !warmup;

  executionBindings.push_back(
      {context, dspGetExecutionStream(), device});
  dspSetExecutionStream(executionStream);
  return context;
}

void NativeDynamicShapePlan::platformEndExecution(
    void* executionState, void* /*stream*/, bool /*frozen*/,
    int /*execCount*/) {
  auto* context = static_cast<PlanExecutionContext*>(executionState);
  if (context == nullptr) return;

  auto* executionStream =
      static_cast<VulkanExecutionStream*>(context->dspStream);
  dspPublishThreadCompletionEvent(executionStream);
  const int completionError = dspPeekLastCudaError();
  const std::string completionErrorText =
      completionError == 0 ? std::string() :
      std::string(dspCudaErrorString(completionError));

  for (auto it = executionBindings.rbegin();
       it != executionBindings.rend(); ++it) {
    if (it->context != context) continue;
    const void* previous = it->previousDspStream;
    executionBindings.erase(std::next(it).base());
    dspSetExecutionStream(const_cast<void*>(previous));
    break;
  }

  if (executionState != steadyStateExecCtx_) delete context;

  if (completionError != 0) {
    throw std::runtime_error(
        "Vulkan execution completion event could not be published: " +
        completionErrorText);
  }
}

void NativeDynamicShapePlan::platformDumpExternalInputDiagnostics(
    NDArray** /*ext*/, int /*numExt*/, int /*execCount*/) {}

void NativeDynamicShapePlan::platformDumpExtInputGpuValues(
    NDArray* /*array*/, int /*extIdx*/, int /*execCount*/, void* /*stream*/) {}

void NativeDynamicShapePlan::platformClearCastCache() {}

void NativeDynamicShapePlan::platformSetDeterministicCublas(bool /*enable*/) {
  // The inherited hook is vendor-specific. Vulkan pipeline determinism is
  // controlled by SPIR-V generation and pipeline creation, not a BLAS handle.
}

void NativeDynamicShapePlan::platformSetupSteadyStateCuda(
    void* execCtxVoid, void* stream) {
  auto* context = static_cast<PlanExecutionContext*>(execCtxVoid);
  auto* executionStream = resolveExecutionStream(stream);
  if (context != nullptr && executionStream != nullptr) {
    context->dspStream = executionStream;
    dspSetExecutionStream(executionStream);
  }
}

void NativeDynamicShapePlan::platformTeardownSteadyStateCuda(
    void* /*execCtxVoid*/, void* /*stream*/, void* prevDspStream) {
  dspSetExecutionStream(prevDspStream);
}

void NativeDynamicShapePlan::platformResetGapCaches() {}

void NativeDynamicShapePlan::platformResetBatchD2D() {}

void NativeDynamicShapePlan::platformPostSegmentPoolManagement(
    bool /*frozen*/, int /*execCount*/) {
  const int device = VulkanDeviceManager::currentDeviceId();
  auto* context = VulkanDeviceContext::getContext(device);
  const uint64_t completed =
      context != nullptr && context->hasTimelineSemaphore()
          ? context->queryTimeline()
          : 0;
  VulkanMemoryPool::getInstance().sweep(device, completed);
}

void NativeDynamicShapePlan::platformDumpLogitsArgmax(
    int /*execCount*/, void* /*stream*/) {}

void NativeDynamicShapePlan::platformDetectAndPrepareBatchedGemm(
    NDArray** /*ext*/, int /*numExt*/, void* /*stream*/) {
  // Matmul remains part of the fused Vulkan segment; there is no vendor BLAS
  // side channel or slot-by-slot substitute.
}

const LongType* NativeDynamicShapePlan::resolveInputShapeInfo(
    int /*srcIdx*/, NDArray** /*externalArrays*/, int /*numExt*/) const {
  return nullptr;
}

void NativeDynamicShapePlan::detectBatchedGemmGroups(
    NDArray** /*externalArrays*/, int /*numExt*/) {}

void NativeDynamicShapePlan::reconcileSlotDispatchAfterMerge(
    const ReplaySchedule& /*schedule*/) {}

void NativeDynamicShapePlan::prepareBatchedGemmDevice(void* /*stream*/) {}

Status NativeDynamicShapePlan::executeBatchedGemmGroup(
    int /*groupIdx*/, NDArray** /*externalArrays*/, int /*numExt*/,
    void* /*stream*/) {
  return Status::KERNEL_FAILURE;
}

void NativeDynamicShapePlan::freeBatchedGemmResources() {}

void NativeDynamicShapePlan::platformPreReplayPoolStats(
    size_t& used, size_t& reserved) {
  uint64_t used64 = 0;
  uint64_t reserved64 = 0;
  VulkanMemoryPool::getInstance().getMemoryPoolStats(
      VulkanDeviceManager::currentDeviceId(), used64, reserved64);
  used = static_cast<size_t>(used64);
  reserved = static_cast<size_t>(reserved64);
}

void NativeDynamicShapePlan::platformPostReplayPoolManagement(
    size_t /*poolUsedPre*/, bool frozen, int execCount) {
  platformPostSegmentPoolManagement(frozen, execCount);
}

void NativeDynamicShapePlan::platformTraceSlotValues(
    const GraphSegment& /*segment*/, void* /*stream*/, int /*execCount*/) {}

SelectedBackend NativeDynamicShapePlan::platformResolveBackend(
    bool /*isGraphCapture*/) const {
  if (graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT) {
    return SelectedBackend::SLOT_BY_SLOT;
  }
  if (graphExecutionMode_ == GraphExecutionMode::GEM_VULKAN ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    return SelectedBackend::VULKAN_REPLAY;
  }
  return SelectedBackend::SLOT_BY_SLOT;
}

SelectedBackend
NativeDynamicShapePlan::platformResolvePortableReplayBackend() const {
  // The Vulkan build is itself the portable replay implementation. Device
  // capability validation and artifact compatibility are enforced when each
  // segment creates its Vulkan pipeline.
  return SelectedBackend::VULKAN_REPLAY;
}

bool NativeDynamicShapePlan::platformShouldBreakSegmentAtTraitBoundary(
    int /*currIdx*/, int /*prevIdx*/) const {
  return false;
}

size_t NativeDynamicShapePlan::platformEstimateCaptureBudget() const {
  const uint64_t free = VulkanMemoryPool::getInstance().getFreeMemory(
      VulkanDeviceManager::currentDeviceId());
  return free == 0 ? std::numeric_limits<size_t>::max()
                   : static_cast<size_t>(free);
}

size_t NativeDynamicShapePlan::platformEstimateSegmentCaptureBytes(
    int startSlot, int endSlot) const {
  size_t total = 0;
  for (int slotIndex = startSlot;
       slotIndex <= endSlot && slotIndex < numSlots_; ++slotIndex) {
    const NativeSlot& slot = slots_[slotIndex];
    for (int output = 0; output < slot.wiring.numOutputs; ++output) {
      const int outputIndex = slot.wiring.outputSlotIndices[output];
      if (outputIndex < 0 || outputIndex >= totalOutputSlots_ ||
          outputSlots_[outputIndex] == nullptr) {
        continue;
      }
      const size_t bytes =
          static_cast<size_t>(outputSlots_[outputIndex]->lengthOf()) *
          static_cast<size_t>(outputSlots_[outputIndex]->sizeOfT());
      total += (bytes + 255u) & ~size_t(255u);
    }
  }
  return total + total / 2;
}

void NativeDynamicShapePlan::platformReleaseSegmentGpuResources() {
  const auto devices = replayOwningDevices(segments_);
  for (int device : devices) {
    if (!VulkanExecutionStream::synchronizeDevice(device)) {
      throw std::runtime_error(
          "Vulkan segment resources are still in flight");
    }
  }
  for (auto& segment : segments_) {
    resetSegmentReplay(segment);
    segment.def.shapeKeyState.reset();
  }
}

void NativeDynamicShapePlan::platformMigrateWeightsAndClearCaches() {
  // DataBuffer owns Vulkan device migration. Replay handles are invalidated by
  // the normal shape/address lifecycle before this hook is reached.
}

Status NativeDynamicShapePlan::platformExecuteSlot(const NativeSlot& slot,
                                                   Context& context) {
  if (slot.legacy.legacyOpType == LEGACY_NOT_SET) {
    return slot.ident.op->execute(&context);
  }

  const auto family =
      vulkanLegacyFamilyFromTypeCode(slot.legacy.legacyOpType);
  auto* stream = resolveExecutionStream(nullptr, slot.targetDeviceId);
  if (!family.has_value() || stream == nullptr) return Status::VALIDATION;

  return VulkanEagerExecutor::execute(
      *family, slot.legacy.legacyOpNum, context, *stream,
      &context.randomGenerator());
}

void NativeDynamicShapePlan::platformPrezeroSegmentOutputs(
    const GraphSegment& segment, void* stream) {
  auto* executionStream = resolveExecutionStream(stream);
  if (executionStream == nullptr) {
    throw std::runtime_error("Vulkan prezero stream is unavailable");
  }

  for (int slotIndex = segment.def.startSlot;
       slotIndex <= segment.def.endSlot && slotIndex < numSlots_; ++slotIndex) {
    NativeSlot& slot = slots_[slotIndex];
    if (!slot.needsPrezero()) continue;
    bool zeroed = false;
    for (int output = 0; output < slot.wiring.numOutputs; ++output) {
      const int outputIndex = slot.wiring.outputSlotIndices[output];
      if (outputIndex < 0 || outputIndex >= totalOutputSlots_) continue;
      NDArray* array = outputSlots_[outputIndex];
      if (array == nullptr || array->isEmpty() || array->isView() ||
          array->dataBuffer() == nullptr) {
        continue;
      }
      auto* buffer = array->dataBuffer();
      if (buffer->special() == nullptr) buffer->syncToSpecial(false);
      const size_t bytes = static_cast<size_t>(buffer->getLenInBytes());
      if (bytes > 0 &&
          !executionStream->enqueueFill(buffer->special(), 0, bytes)) {
        throw std::runtime_error("Vulkan segment prezero failed");
      }
      buffer->writeSpecial();
      zeroed = true;
    }
    if (zeroed) slot.bumpGeneration();
  }
}

void NativeDynamicShapePlan::platformReconcileOutputActuality(
    const char* stage, int stepIdx, const NativeSlot& slot, NDArray* output) {
  if (output == nullptr) return;
  auto* db = output->dataBuffer();
  if (db == nullptr || db->isClosed()) return;

  const bool primaryActual = db->isPrimaryActual();
  const bool specialActual = db->isSpecialActual();
  const bool needsDeviceVisibleControl =
      slot.isDataDependent() || slot.flags.outputShapeDependsOnInputValues ||
      ((output->dataType() == INT32 || output->dataType() == INT64 ||
        output->dataType() == BOOL) &&
       output->lengthOf() > 0 && output->lengthOf() <= 32);

  if (primaryActual && !specialActual && needsDeviceVisibleControl) {
    std::vector<NDArray*> reads{output};
    NDArray::prepareSpecialUse({}, reads);
    NDArray::registerSpecialUse({}, reads);
    DSP_DIAG(SHAPE,
             "CONTROL_OUTPUT_SYNC: stage=%s slot=%d (%s) "
             "prepared host-current output for device after native execution",
             stage, stepIdx, slot.ident.opName.c_str());
  }
}

bool NativeDynamicShapePlan::platformValidateSlotInputBuffer(
    int /*stepIdx*/, const NativeSlot& /*slot*/, int /*inputIdx*/,
    NDArray* input) {
  if (input == nullptr || input->isEmpty()) return true;
  auto* buffer = input->dataBuffer();
  if (buffer == nullptr || buffer->special() == nullptr) return false;
  return VulkanMemoryPool::getInstance().getDeviceId(buffer->special()) ==
         VulkanDeviceManager::currentDeviceId();
}

bool NativeDynamicShapePlan::platformValidateReusableSlotBuffer(
    NDArray* cached) {
  if (cached == nullptr || cached->isEmpty()) return true;
  auto* buffer = cached->dataBuffer();
  return buffer != nullptr && buffer->special() != nullptr &&
         VulkanMemoryPool::getInstance().getDeviceId(buffer->special()) ==
             VulkanDeviceManager::currentDeviceId();
}

void NativeDynamicShapePlan::platformSetLtEpilogue(
    const NativeSlot& /*slot*/, NDArray* /*biasArray*/) {}

void NativeDynamicShapePlan::platformClearLtEpilogue() {}

void NativeDynamicShapePlan::platformLogSlotOutput(
    int /*stepIdx*/, const char* /*opName*/, const char* /*tag*/,
    const int* /*outputSlotIndices*/, int /*numOutputs*/) {}

int NativeDynamicShapePlan::copyStagingToBuffer(
    int extIdx, sd::DataBuffer* destination) {
  if (placeholderStagingBuffers_ == nullptr || extIdx < 0 ||
      extIdx >= numExternalInputs_) {
    return -1;
  }
  NDArray* staging = placeholderStagingBuffers_[extIdx];
  if (staging == nullptr || staging->dataBuffer() == nullptr ||
      staging->dataBuffer()->isClosed()) {
    return -2;
  }
  if (destination == nullptr || destination->special() == nullptr) return -3;

  auto* stream = VulkanExecutionStream::currentOrDefault(
      VulkanDeviceManager::currentDeviceId());
  const size_t bytes = static_cast<size_t>(
      std::min(staging->dataBuffer()->getLenInBytes(),
               destination->getLenInBytes()));
  if (stream == nullptr ||
      !stream->enqueueCopy(destination->special(),
                           staging->dataBuffer()->special(), bytes, 3)) {
    return -3;
  }
  destination->writeSpecial();
  return 0;
}

void NativeDynamicShapePlan::platformPinGraphBakedAddress(
    void* ptr, int deviceId) {
  if (ptr == nullptr ||
      VulkanMemoryPool::getInstance().getDeviceId(ptr) != deviceId) {
    throw std::runtime_error(
        "Vulkan graph attempted to pin an invalid device allocation");
  }
}

void NativeDynamicShapePlan::platformFlushGraphBakedPins(void* stream) {
  auto* executionStream = resolveExecutionStream(stream);
  if (executionStream != nullptr && !executionStream->synchronize()) {
    throw std::runtime_error(
        "Vulkan graph-baked address flush synchronization failed");
  }
  graphPinnedAddrs_.clear();
}

void NativeDynamicShapePlan::drainFingerprintRingPublic() {}

const char* NativeDynamicShapePlan::getFingerprintJson() {
  return "null";
}

}  // namespace graph
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
