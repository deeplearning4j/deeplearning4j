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

/**
 * NativeDynamicShapePlan — Slot Execution CUDA Support
 *
 * Contains CUDA-specific implementations for slot execution:
 *   - platformPrezeroSegmentOutputs: batched cudaMemsetAsync for output zeroing
 *   - platformReconcileOutputActuality: device sync for control arrays post-exec
 *   - platformValidateSlotInputBuffer: null GPU buffer detection
 *   - platformSetLtEpilogue / platformClearLtEpilogue: cublasLt epilogue wiring
 *   - platformLogSlotOutput: triton verify kernel logging
 *
 * CPU stubs for these methods are in NativeDynamicShapePlan_cuda_stubs.cpp.
 */

#ifdef SD_CUDA

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspVerifyUtils.h>
#include <helpers/MmulHelper.h>
#include <system/Environment.h>
#include <cuda_runtime.h>

namespace sd {
namespace graph {

// ── Platform prezero: batched cudaMemsetAsync ─────────────────────────────────
void NativeDynamicShapePlan::platformPrezeroSegmentOutputs(const GraphSegment& seg, void* stream) {
  auto cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Collect qualifying buffers first, then batch-launch a single memset kernel
  // instead of issuing N individual cudaMemsetAsync driver calls.
  struct PrezeroTarget { void* buf; size_t bytes; int slotIdx; };
  constexpr int kStackCapacity = 128;
  PrezeroTarget stackBuf[kStackCapacity];
  std::vector<PrezeroTarget> heapBuf;
  PrezeroTarget* targets = stackBuf;
  int targetCount = 0;
  bool useHeap = false;

  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    if (s < 0 || s >= numSlots_) continue;
    NativeSlot& slot = slots_[s];

    if (!slot.needsPrezero()) continue;

    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      int outIdx = slot.wiring.outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots_) continue;
      if (slotIsViewProducer_ != nullptr && slotIsViewProducer_[outIdx]) continue;
      NDArray* arr = outputSlots_[outIdx];
      if (arr == nullptr) continue;
      if (arr->isView()) continue;
      auto* db = arr->dataBuffer();
      if (db == nullptr) continue;
      size_t bytes = db->getLenInBytes();
      if (bytes == 0) continue;
      void* buf = arr->specialBuffer();
      if (buf == nullptr) continue;

      DSP_DIAG_SEG(MEMORY, s, "prezeroSegmentOutputs: seg[%d-%d] slot=%d outIdx=%d op=%s bytes=%lld stream=%p",
                   seg.def.startSlot, seg.def.endSlot, s, outIdx,
                   slot.ident.opName.c_str(), (long long)bytes, (void*)cudaStr);
      DSP_DIAG_SLOT_ZERO(outIdx, "prezero", cudaStr, "segment-prezero");

      if (targetCount >= kStackCapacity && !useHeap) {
        heapBuf.assign(stackBuf, stackBuf + targetCount);
        useHeap = true;
        targets = nullptr;
      }
      PrezeroTarget t{buf, bytes, s};
      if (useHeap) {
        heapBuf.push_back(t);
      } else {
        stackBuf[targetCount] = t;
      }
      targetCount++;
    }
  }

  if (useHeap) targets = heapBuf.data();

  // Dispatch: single buffer → direct memset, multiple → batched kernel
  if (targetCount == 1) {
    auto res = cudaMemsetAsync(targets[0].buf, 0, targets[0].bytes, cudaStr);
    if (res == 901 || res == 906) {
      // cudaErrorStreamCaptureImplicit / cudaErrorStreamCaptureInvalidated
      // Pool-allocated memory retains capture-era associations from a prior
      // Triton CUDA graph capture. Fall back to synchronous memset to bypass
      // the pool ordering check. This is slot-by-slot execution only.
      cudaGetLastError();
      res = cudaMemset(targets[0].buf, 0, targets[0].bytes);
    }
    if (res != cudaSuccess) {
      DSP_THROW_CUDA(MEMORY, res, "prezeroSegmentOutputs: cudaMemsetAsync failed for slot %d (bytes=%zu)", targets[0].slotIdx, targets[0].bytes);
    }
  } else if (targetCount > 1) {
    std::vector<void*> dstPtrs(targetCount);
    std::vector<size_t> sizes(targetCount);
    for (int i = 0; i < targetCount; i++) {
      dstPtrs[i] = targets[i].buf;
      sizes[i] = targets[i].bytes;
    }
    launchBatchMemset(cudaStr, dstPtrs.data(), sizes.data(), targetCount);
    DSP_DIAG(MEMORY, "prezeroSegmentOutputs: batched %d buffers into 1 kernel launch", targetCount);
  }

  // Log prezero summary (STREAM_SYNC so it's visible alongside sync decisions)
  {
    size_t totalBytes = 0;
    for (int i = 0; i < targetCount; i++) {
      totalBytes += (useHeap ? heapBuf[i].bytes : stackBuf[i].bytes);
    }
    DSP_DIAG(STREAM_SYNC,
             "prezeroSegmentOutputs: seg[%d-%d] zeroed %d buffers (%zuKB) execCount=%d",
             seg.def.startSlot, seg.def.endSlot, targetCount,
             totalBytes / 1024, executeCount_);
  }

  // Bump generation for all slots that were zeroed
  if (targetCount > 0) {
    int prevSlot = -1;
    for (int i = 0; i < targetCount; i++) {
      int s = targets[i].slotIdx;
      if (s != prevSlot) {
        slots_[s].bumpGeneration();
        prevSlot = s;
      }
    }
  }
}

// ── Platform reconcile output actuality ───────────────────────────────────────
void NativeDynamicShapePlan::platformReconcileOutputActuality(
    const char* stage, int stepIdx, const NativeSlot& slot, NDArray* output) {
  if (output == nullptr) return;
  auto* db = output->dataBuffer();
  if (db == nullptr || db->isClosed()) return;

  const bool primaryActual = db->isPrimaryActual();
  const bool specialActual = db->isSpecialActual();
  const bool needsDeviceVisibleControl =
      slot.isDataDependent() || slot.flags.outputShapeDependsOnInputValues ||
      ((output->dataType() == INT32 || output->dataType() == INT64 || output->dataType() == BOOL) &&
       output->lengthOf() > 0 && output->lengthOf() <= 32);

  if (primaryActual && !specialActual) {
    if (needsDeviceVisibleControl) {
      output->syncToDevice();
      DSP_DIAG(SHAPE,
               "CONTROL_OUTPUT_SYNC: stage=%s slot=%d (%s) "
               "synced host-current output to device after native execution",
               stage, stepIdx, slot.ident.opName.c_str());
    }
  }
}

// ── Platform validate GPU buffer ──────────────────────────────────────────────
bool NativeDynamicShapePlan::platformValidateSlotInputBuffer(
    int stepIdx, const NativeSlot& slot, int inputIdx, NDArray* input) {
  if (input == nullptr || input->isEmpty()) return true;
  auto* db = input->dataBuffer();
  if (db == nullptr) return true;

  if (db->special() == nullptr) {
    DSP_DIAG_SLOT(EXECUTE, stepIdx,
        "NULL GPU buffer for slot %d (%s) input %d, srcIdx=%d "
        "len=%lld isClosed=%d isConst=%d exec=%d",
        stepIdx, slot.ident.opName.c_str(), inputIdx,
        slot.wiring.inputSourceIndices[inputIdx],
        (long long)input->lengthOf(), db->isClosed() ? 1 : 0,
        db->isConstant ? 1 : 0, executeCount_);
    return false;
  }
  return true;
}

// ── Platform reusable slot array validation (GPU-specific check) ──────────────
bool NativeDynamicShapePlan::platformValidateReusableSlotBuffer(NDArray* cached) {
  if (cached == nullptr) return true;
  auto* db = cached->dataBuffer();
  if (db == nullptr) return true;
  // On CUDA, a non-empty array with null special (device) buffer is invalid
  if (db->special() == nullptr && !cached->isEmpty()) {
    return false;
  }
  return true;
}

// ── Platform set/clear cublasLt epilogue ──────────────────────────────────────
void NativeDynamicShapePlan::platformSetLtEpilogue(const NativeSlot& slot, NDArray* biasArray) {
  if (biasArray == nullptr) return;
  biasArray->syncToDevice();
  MmulHelper::setLtEpilogue(slot.flags.ltEpilogueType, biasArray->specialBuffer(),
                             biasArray->lengthOf() * biasArray->sizeOfT());
}

void NativeDynamicShapePlan::platformClearLtEpilogue() {
  MmulHelper::clearLtEpilogue();
}

// ── Platform log slot output (triton verify) ──────────────────────────────────
void NativeDynamicShapePlan::platformLogSlotOutput(
    int stepIdx, const char* opName, const char* tag,
    const int* outputSlotIndices, int numOutputs) {
  if (!Environment::getInstance().tritonVerifyKernels()) return;
  dspLogSlotOutput(stepIdx, opName, tag,
                   outputSlots_, outputSlotIndices, numOutputs, totalOutputSlots_);
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
