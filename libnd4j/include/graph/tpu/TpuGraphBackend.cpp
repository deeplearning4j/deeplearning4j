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
#include <graph/tpu/HloIRBuilder.h>
#include <graph/tpu/PjrtClientManager.h>
#include <graph/tpu/TpuReplayHandle.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>

#include <cstring>
#include <memory>

namespace sd {
namespace graph {

// ── Singleton ───────────────────────────────────────────────────────────────

TpuGraphBackend::TpuGraphBackend() {
  DSP_DIAG(BACKEND, "TpuGraphBackend: instance created");
}

TpuGraphBackend* TpuGraphBackend::getInstance() {
  static TpuGraphBackend instance;
  return &instance;
}

// ── Availability ────────────────────────────────────────────────────────────

bool TpuGraphBackend::isAvailable() const {
  auto& mgr = PjrtClientManager::getInstance();
  if (!mgr.isAvailable()) {
    DSP_DIAG(BACKEND, "TpuGraphBackend::isAvailable: PJRT client not available");
    return false;
  }

  int deviceCount = mgr.getDeviceCount();
  if (deviceCount <= 0) {
    DSP_DIAG(BACKEND, "TpuGraphBackend::isAvailable: no TPU devices found");
    return false;
  }

  DSP_DIAG(BACKEND, "TpuGraphBackend::isAvailable: %d TPU device(s) available",
           deviceCount);
  return true;
}

// ── Segment Fusion Check ────────────────────────────────────────────────────

bool TpuGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (slots == nullptr || start >= end) return false;

  for (int i = start; i < end; ++i) {
    const char* opName = slots[i].ident.opName.c_str();
    if (!HloIRBuilder::isHloMappable(opName)) {
      DSP_DIAG(SEGMENT, "TpuGraphBackend::canFuseSegment: slot %d op '%s' "
               "not HLO-mappable", i, opName);
      return false;
    }
  }

  DSP_DIAG(SEGMENT, "TpuGraphBackend::canFuseSegment: range [%d, %d) "
           "all %d ops are HLO-mappable", start, end, end - start);
  return true;
}

// ── Segment Compilation ─────────────────────────────────────────────────────

bool TpuGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                      NDArray** externalInputs,
                                      int numExternalInputs,
                                      NDArray** outputSlots,
                                      int totalOutputSlots,
                                      LongType shapeKey,
                                      int totalSlots,
                                      int* requestedOutputSlotIndices,
                                      int numRequestedOutputs) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Clear previous audit
  lastAudit_.clear();

  int start = seg.def.startSlot;
  int end = seg.def.endSlot;

  DSP_DIAG(COMPILE, "TpuGraphBackend::compileSegment: compiling segment "
           "[%d, %d), shapeKey=0x%llx",
           start, end, static_cast<unsigned long long>(shapeKey));

  // Build HLO module text from slot sequence
  std::string hloText = HloIRBuilder::buildModule(
      slots, start, end, externalInputs, numExternalInputs);

  if (hloText.empty()) {
    DSP_DIAG(COMPILE, "TpuGraphBackend::compileSegment: HLO generation failed "
             "for segment [%d, %d)", start, end);

    // Record audit entries for all slots as uncompiled
    for (int i = start; i < end; ++i) {
      CompilationAuditEntry entry;
      entry.slotIndex = i;
      entry.opName = slots[i].ident.opName;
      entry.wasCompiled = false;
      entry.reason = "HLO module generation failed";
      lastAudit_.push_back(entry);
    }
    return false;
  }

  // Create or reuse TpuReplayHandle for this segment
  if (!seg.exec.replayHandle || std::string(seg.exec.replayHandle->backendName()) != "TPU (PJRT)") {
    seg.exec.replayHandle = std::make_unique<TpuReplayHandle>(0);
  }

  auto* tpuHandle = static_cast<TpuReplayHandle*>(seg.exec.replayHandle.get());

  // Set the HLO module on the replay handle
  tpuHandle->setHloModule(hloText);

  // Run through the capture lifecycle (state transitions for TPU)
  if (!tpuHandle->beginCapture(nullptr)) {
    DSP_DIAG(COMPILE, "TpuGraphBackend::compileSegment: beginCapture failed");
    return false;
  }

  if (!tpuHandle->endCapture(nullptr)) {
    DSP_DIAG(COMPILE, "TpuGraphBackend::compileSegment: endCapture failed");
    return false;
  }

  // Finalize: compile HLO -> XLA executable
  if (!tpuHandle->finalize()) {
    DSP_DIAG(COMPILE, "TpuGraphBackend::compileSegment: finalize (HLO "
             "compilation) failed");

    for (int i = start; i < end; ++i) {
      CompilationAuditEntry entry;
      entry.slotIndex = i;
      entry.opName = slots[i].ident.opName;
      entry.wasCompiled = false;
      entry.reason = "XLA compilation via PJRT failed";
      lastAudit_.push_back(entry);
    }
    return false;
  }

  // Update segment metadata
  seg.exec.cachedShapeKey = shapeKey;

  // Record successful audit entries
  for (int i = start; i < end; ++i) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].ident.opName;
    entry.wasCompiled = true;
    lastAudit_.push_back(entry);
  }

  DSP_DIAG(COMPILE, "TpuGraphBackend::compileSegment: segment [%d, %d) "
           "compiled successfully (%zu bytes HLO)", start, end, hloText.size());
  return true;
}

// ── Segment Execution ───────────────────────────────────────────────────────

Status TpuGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs,
                                        int numExternalInputs,
                                        NDArray** outputSlots,
                                        int totalOutputSlots,
                                        void* stream) {
  if (!seg.exec.replayHandle || !seg.exec.replayHandle->isReady()) {
    DSP_DIAG(EXECUTE, "TpuGraphBackend::executeSegment: segment [%d, %d) "
             "not compiled/ready", seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }

  auto* tpuHandle = static_cast<TpuReplayHandle*>(seg.exec.replayHandle.get());

  int start = seg.def.startSlot;
  int end = seg.def.endSlot;

  DSP_DIAG(EXECUTE, "TpuGraphBackend::executeSegment: executing segment "
           "[%d, %d), replay #%d", start, end, tpuHandle->getReplayCount() + 1);

  // Collect input PJRT buffer pointers from external inputs and prior slot outputs
  // For now, we would need to create PJRT buffers from NDArray device pointers.
  // This requires the full PJRT buffer creation path to be implemented.

  // Collect input arrays for this segment
  std::vector<void*> inputBuffers;
  for (int i = start; i < end; ++i) {
    for (int j = 0; j < slots[i].wiring.numInputs; ++j) {
      int srcIdx = slots[i].wiring.inputSourceIndices[j];
      if (srcIdx < 0) {
        // External input
        int extIdx = -(srcIdx + 1);
        if (extIdx >= 0 && extIdx < numExternalInputs &&
            externalInputs[extIdx] != nullptr) {
          // In a full implementation, we would create or look up the
          // PJRT buffer for this NDArray's device data.
          inputBuffers.push_back(externalInputs[extIdx]->specialBuffer());
        }
      } else if (srcIdx >= 0 && srcIdx < totalOutputSlots) {
        // Prior slot output
        if (outputSlots[srcIdx] != nullptr) {
          inputBuffers.push_back(outputSlots[srcIdx]->specialBuffer());
        }
      }
    }
  }

  // Collect output buffer pointers
  std::vector<void*> outputBufferPtrs;
  for (int i = start; i < end; ++i) {
    for (int k = 0; k < slots[i].wiring.numOutputs; ++k) {
      int outIdx = slots[i].wiring.outputSlotIndices[k];
      if (outIdx >= 0 && outIdx < totalOutputSlots &&
          outputSlots[outIdx] != nullptr) {
        outputBufferPtrs.push_back(outputSlots[outIdx]->specialBuffer());
      }
    }
  }

  // Bind buffers and replay
  tpuHandle->bindBuffers(
      inputBuffers.empty() ? nullptr : inputBuffers.data(),
      static_cast<int>(inputBuffers.size()),
      outputBufferPtrs.empty() ? nullptr : outputBufferPtrs.data(),
      static_cast<int>(outputBufferPtrs.size()));

  if (!tpuHandle->replay(stream)) {
    DSP_DIAG(EXECUTE, "TpuGraphBackend::executeSegment: replay failed for "
             "segment [%d, %d)", start, end);
    return Status::KERNEL_FAILURE;
  }

  DSP_DIAG(EXECUTE, "TpuGraphBackend::executeSegment: segment [%d, %d) "
           "executed successfully", start, end);
  return Status::OK;
}

// ── Cache Invalidation ──────────────────────────────────────────────────────

void TpuGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(mutex_);

  DSP_DIAG(COMPILE, "TpuGraphBackend::invalidateCache: clearing compilation "
           "caches");

  // Individual segment replay handles own their compiled executables.
  // This method is called when global state changes require re-compilation.
  // Segments will re-compile on next compileSegment() call.
  lastAudit_.clear();
}

// ── Compilation Audit ───────────────────────────────────────────────────────

std::vector<CompilationAuditEntry> TpuGraphBackend::getLastCompilationAudit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return lastAudit_;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
