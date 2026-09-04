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

#ifdef SD_TPU

#include <graph/tpu/TpuGraphBackend.h>

#include <graph/DspDiagnostics.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/tpu/PjrtClientManager.h>
#include <graph/tpu/StableHloGraphLowering.h>
#include <graph/tpu/TpuReplayHandle.h>

#include <mutex>
#include <string>
#include <vector>

namespace sd {
namespace graph {
namespace {

NDArray* resolveBoundaryArray(int sourceIndex,
                              NDArray** externalInputs, int numExternalInputs,
                              NDArray** outputSlots, int totalOutputSlots) {
  if (sourceIndex < 0) {
    const int externalIndex = -(sourceIndex + 1);
    return externalInputs != nullptr && externalIndex >= 0 &&
                   externalIndex < numExternalInputs
               ? externalInputs[externalIndex]
               : nullptr;
  }
  return outputSlots != nullptr && sourceIndex < totalOutputSlots
             ? outputSlots[sourceIndex]
             : nullptr;
}

int resolveSegmentDevice(NativeSlot* slots, int start, int end,
                         int defaultDevice) {
  int selected = -1;
  for (int i = start; i <= end; ++i) {
    const int requested = slots[i].targetDeviceId;
    if (requested < 0) continue;
    if (selected >= 0 && selected != requested) return -2;
    selected = requested;
  }
  return selected < 0 ? defaultDevice : selected;
}

}  // namespace

TpuGraphBackend::TpuGraphBackend() = default;

TpuGraphBackend& TpuGraphBackend::getInstance() {
  static TpuGraphBackend* instance = nullptr;
  static std::once_flag once;
  std::call_once(once, []() { instance = new TpuGraphBackend(); });
  return *instance;
}

bool TpuGraphBackend::isAvailable() const {
  auto& manager = PjrtClientManager::getInstance();
  if (!manager.isAvailable()) {
    DSP_DIAG(BACKEND, "TpuGraphBackend: PJRT unavailable: %s",
             manager.getLastError().c_str());
    return false;
  }
  if (!manager.isTpuPlatform()) {
    DSP_DIAG(BACKEND, "TpuGraphBackend: rejecting non-TPU PJRT platform '%s'",
             manager.getPlatformName().c_str());
    return false;
  }
  return manager.getDeviceCount() > 0;
}

bool TpuGraphBackend::isResolvable(const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_TPU ||
         request.executionMode == GraphExecutionMode::GEM_AUTO;
}

int TpuGraphBackend::resolutionPriority(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_TPU ? 1000 : 400;
}

GraphBackendPlanningPolicy TpuGraphBackend::planningPolicy(
    const GraphBackendRequest& request) const {
  auto policy = GraphBackend::planningPolicy(request);
  policy.artifactKind = GraphBackendArtifactKind::BACKEND_REPLAY_HANDLE;
  policy.requiresShapePrePass = true;
  policy.requiresSuccessfulShapePrePass = true;
  policy.precompileBeforeFirstExecution = true;
  policy.allowsShapeOnlyWarmup = true;
  policy.requiresCapabilityPartitioning = true;
  policy.requiresCompleteLowering =
      request.executionMode == GraphExecutionMode::GEM_TPU;
  return policy;
}

bool TpuGraphBackend::canResolveSlot(const GraphBackendRequest& request,
                                     NativeSlot* slots, int slotIndex) {
  (void)request;
  return slots != nullptr && slotIndex >= 0 &&
         StableHloGraphLowering::canLowerSlot(slots[slotIndex]);
}

bool TpuGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (slots == nullptr || start < 0 || end < start) return false;
  for (int i = start; i <= end; ++i) {
    if (!StableHloGraphLowering::canLowerSlot(slots[i])) return false;
  }
  return true;
}

void TpuGraphBackend::auditRange(NativeSlot* slots, int start, int end,
                                 bool compiled, const std::string& reason) {
  lastAudit_.clear();
  if (slots == nullptr || end < start) return;
  lastAudit_.reserve(static_cast<size_t>(end - start + 1));
  for (int i = start; i <= end; ++i) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].ident.opName;
    entry.wasCompiled = compiled;
    entry.reason = reason;
    lastAudit_.push_back(entry);
  }
}

bool TpuGraphBackend::compileSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    LongType shapeKey, int totalSlots,
    int* requestedOutputSlotIndices, int numRequestedOutputs) {
  return compileInternal(true, seg, slots, externalInputs, numExternalInputs,
                         outputSlots, totalOutputSlots, shapeKey, totalSlots,
                         requestedOutputSlotIndices, numRequestedOutputs);
}

bool TpuGraphBackend::compileSegment(
    const GraphBackendRequest& request, GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    LongType shapeKey, int totalSlots,
    int* requestedOutputSlotIndices, int numRequestedOutputs) {
  return compileInternal(request.runtimeCompilationAllowed, seg, slots,
                         externalInputs, numExternalInputs, outputSlots,
                         totalOutputSlots, shapeKey, totalSlots,
                         requestedOutputSlotIndices, numRequestedOutputs);
}

bool TpuGraphBackend::compileInternal(
    bool runtimeCompilationAllowed, GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    LongType shapeKey, int totalSlots,
    int* requestedOutputSlotIndices, int numRequestedOutputs) {
  std::lock_guard<std::mutex> lock(mutex_);
  const int start = seg.def.startSlot;
  const int end = seg.def.endSlot;

  if (!runtimeCompilationAllowed) {
    auditRange(slots, start, end, false,
               "runtime compilation disabled and no TPU AOT artifact was supplied");
    return false;
  }
  if (!isAvailable() || !canFuseSegment(slots, start, end)) {
    auditRange(slots, start, end, false,
               "TPU unavailable or segment is not completely lowerable");
    return false;
  }

  auto& manager = PjrtClientManager::getInstance();
  const int deviceId = resolveSegmentDevice(
      slots, start, end, manager.getCurrentDevice());
  if (deviceId == -2 || deviceId < 0 || deviceId >= manager.getDeviceCount()) {
    auditRange(slots, start, end, false,
               "segment requests conflicting or unavailable TPU devices");
    return false;
  }

  StableHloLoweringResult program = StableHloGraphLowering::lower(
      slots, start, end, externalInputs, numExternalInputs,
      outputSlots, totalOutputSlots, totalSlots,
      requestedOutputSlotIndices, numRequestedOutputs);
  if (!program.success) {
    auditRange(slots, start, end, false, program.error);
    return false;
  }

  // A shape-keyed plan owns this handle. Replacing it atomically avoids stale
  // READY-state reuse when shape drift recompiles the same segment.
  seg.exec.replayHandle.reset(new TpuReplayHandle(deviceId));
  auto* handle = static_cast<TpuReplayHandle*>(seg.exec.replayHandle.get());
  handle->setProgram(program.program, program.format,
                     program.boundary.inputSourceIndices,
                     program.boundary.outputSlotIndices,
                     end - start + 1);
  if (!handle->beginCapture(nullptr) || !handle->endCapture(nullptr) ||
      !handle->finalize()) {
    auditRange(slots, start, end, false,
               "PJRT StableHLO compilation failed: " + manager.getLastError());
    seg.exec.replayHandle.reset();
    return false;
  }

  seg.exec.cachedShapeKey = shapeKey;
  auditRange(slots, start, end, true, "");
  DSP_DIAG(COMPILE,
           "TpuGraphBackend: compiled inclusive segment [%d,%d] device=%d "
           "shapeKey=0x%llx inputs=%d outputs=%d",
           start, end, deviceId, static_cast<unsigned long long>(shapeKey),
           static_cast<int>(program.boundary.inputSourceIndices.size()),
           static_cast<int>(program.boundary.outputSlotIndices.size()));
  return true;
}

Status TpuGraphBackend::executeSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* stream) {
  (void)slots;
  auto fail = [&](const std::string& reason) {
    const std::string message =
        reason + " [TPU segment " + std::to_string(seg.def.startSlot) + "-" +
        std::to_string(seg.def.endSlot) + ", status=KERNEL_FAILURE (50)]";
    safeSetErrorContext(static_cast<int>(Status::KERNEL_FAILURE), message.c_str());
    return Status::KERNEL_FAILURE;
  };
  if (!seg.exec.replayHandle || !seg.exec.replayHandle->isReady() ||
      std::string(seg.exec.replayHandle->backendName()) != "TPU (PJRT)") {
    return fail("TPU replay handle is absent, not ready, or owned by another backend");
  }
  auto* handle = static_cast<TpuReplayHandle*>(seg.exec.replayHandle.get());

  std::vector<NDArray*> inputs;
  inputs.reserve(handle->inputSourceIndices().size());
  for (int source : handle->inputSourceIndices()) {
    NDArray* array = resolveBoundaryArray(source, externalInputs,
                                          numExternalInputs, outputSlots,
                                          totalOutputSlots);
    if (array == nullptr) {
      return fail("TPU boundary input resolved to null: source=" +
                  std::to_string(source));
    }
    inputs.push_back(array);
  }

  std::vector<NDArray*> outputs;
  outputs.reserve(handle->outputSlotIndices().size());
  for (int outputIndex : handle->outputSlotIndices()) {
    if (outputIndex < 0 || outputIndex >= totalOutputSlots ||
        outputSlots == nullptr || outputSlots[outputIndex] == nullptr) {
      return fail("TPU boundary output is unavailable: outputSlot=" +
                  std::to_string(outputIndex) + ", totalOutputSlots=" +
                  std::to_string(totalOutputSlots));
    }
    outputs.push_back(outputSlots[outputIndex]);
  }

  handle->bindArrays(inputs.empty() ? nullptr : inputs.data(),
                     static_cast<int>(inputs.size()),
                     outputs.empty() ? nullptr : outputs.data(),
                     static_cast<int>(outputs.size()));
  if (!handle->replay(stream)) {
    const std::string runtimeError =
        PjrtClientManager::getInstance().getLastError();
    return fail(runtimeError.empty()
                    ? "TPU PJRT replay returned false without runtime detail"
                    : "TPU PJRT replay failed: " + runtimeError);
  }
  return Status::OK;
}

void TpuGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  PjrtClientManager::getInstance().invalidateCompilationCache();
  lastAudit_.clear();
}

std::vector<CompilationAuditEntry>
TpuGraphBackend::getLastCompilationAudit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return lastAudit_;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
