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
#include <graph/DspThreadState.h>
#include <graph/DspVerifyUtils.h>
#include <graph/gpu/DspCudaDispatch.h>
#include <helpers/MmulHelper.h>
#include <system/Environment.h>
#include <cuda_runtime.h>

namespace sd {
namespace graph {

Status NativeDynamicShapePlan::platformExecuteSlot(const NativeSlot& slot,
                                                   Context& context) {
  // During warmup, record the CUDA provenance of every pointer handed to the op.
  // This is intentionally diagnostic-only: it identifies cross-device slot
  // allocation before the asynchronous kernel can report an opaque error 700.
  if (DSP_DIAG_ENABLED(EXECUTE)) {
    int currentDevice = -1;
    const auto deviceStatus = cudaGetDevice(&currentDevice);
    if (deviceStatus != cudaSuccess) {
      DSP_DIAG(EXECUTE, "SLOT_POINTER_PROVENANCE: cudaGetDevice failed op=%s err=%s",
               slot.ident.opName.c_str(), cudaGetErrorString(deviceStatus));
      cudaGetLastError();
    }

    auto logPointer = [&](const char* kind, int index, NDArray* array) {
      if (array == nullptr || array->isEmpty()) return;
      auto* buffer = array->dataBuffer();
      void* pointer = array->specialBuffer();
      if (buffer == nullptr || pointer == nullptr) {
        DSP_DIAG(EXECUTE,
                 "SLOT_POINTER_PROVENANCE op=%s kind=%s[%d] ptr=%p currentDevice=%d "
                 "bufferDevice=%d attrs=unavailable",
                 slot.ident.opName.c_str(), kind, index, pointer, currentDevice,
                 buffer == nullptr ? -1 : buffer->deviceId());
        return;
      }

      cudaPointerAttributes attributes{};
      const auto pointerStatus = cudaPointerGetAttributes(&attributes, pointer);
      if (pointerStatus == cudaSuccess) {
        DSP_DIAG(EXECUTE,
                 "SLOT_POINTER_PROVENANCE op=%s kind=%s[%d] ptr=%p currentDevice=%d "
                 "bufferDevice=%d pointerDevice=%d pointerType=%d bytes=%lld",
                 slot.ident.opName.c_str(), kind, index, pointer, currentDevice,
                 buffer->deviceId(), attributes.device, static_cast<int>(attributes.type),
                 static_cast<long long>(buffer->getLenInBytes()));
      } else {
        DSP_DIAG(EXECUTE,
                 "SLOT_POINTER_PROVENANCE op=%s kind=%s[%d] ptr=%p currentDevice=%d "
                 "bufferDevice=%d attrsError=%s bytes=%lld",
                 slot.ident.opName.c_str(), kind, index, pointer, currentDevice,
                 buffer->deviceId(), cudaGetErrorString(pointerStatus),
                 static_cast<long long>(buffer->getLenInBytes()));
        cudaGetLastError();
      }
    };

    auto& inputs = context.fastpath_in();
    for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
      logPointer("input", i, inputs[i]);
    }
    auto& outputs = context.fastpath_out();
    for (int i = 0; i < static_cast<int>(outputs.size()); i++) {
      logPointer("output", i, outputs[i]);
    }
  }

  // Runtime-controlled slice-family ops materialize begin/size tensors on the
  // host inside execute(). Composite replay normally suppresses D2H synchronization,
  // which leaves these controls at the previous plan invocation's host values. Static
  // argument slices do not perform that host read and remain eligible for capture.
  if (slot.hasOpTrait(sd::ops::OP_TRAIT_SLICE) && slot.hasValueDependentShape()) {
    DspReplayGuard valueDependentHostReadGuard(false);
    return slot.ident.op->execute(&context);
  }

  return slot.ident.op->execute(&context);
}

// ── Platform prezero: batched cudaMemsetAsync ─────────────────────────────────
void NativeDynamicShapePlan::platformPrezeroSegmentOutputs(const GraphSegment& seg, void* stream) {
  // The stream parameter is the Java/JNI stream pointer. On first execution it
  // can still point at the LaunchContext fallback while platformBeginExecution
  // has already installed a plan-owned DSP stream for the slot op. Prezero must
  // run on the same live DSP stream as op->execute(); otherwise the async memset
  // can race after the sparse kernel and erase freshly-written outputs.
  void* liveStream = dspGetExecutionStream();
  auto cudaStr = liveStream != nullptr
      ? reinterpret_cast<cudaStream_t>(liveStream)
      : ((stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr);

  // Collect qualifying buffers first, then batch-launch a single memset kernel
  // instead of issuing N individual cudaMemsetAsync driver calls.
  struct PrezeroTarget { void* buf; size_t bytes; int slotIdx; };
  constexpr int kStackCapacity = 128;
  PrezeroTarget stackBuf[kStackCapacity];
  std::vector<PrezeroTarget> heapBuf;
  PrezeroTarget* targets = stackBuf;
  int targetCount = 0;
  bool useHeap = false;

  // Build a per-slot GAP bitmap from the composite replay schedule (if any).
  // GAP slots execute live — their outputs must NOT be prezeroed because
  // islands captured after them read the live GAP results. Zeroing them
  // would cause e.g. tanh/sigmoid to see 0 inputs and produce wrong values.
  bool hasComposite = !seg.exec.compositeReplaySchedule.units.empty();
  std::vector<bool> isGapSlot;
  if (hasComposite) {
    isGapSlot.resize(numSlots_, false);
    for (const auto& unit : seg.exec.compositeReplaySchedule.units) {
      if (unit.kind == REPLAY_UNIT_GAP) {
        for (int gs = unit.startSlot; gs <= unit.endSlot && gs < numSlots_; gs++) {
          if (gs >= 0) isGapSlot[gs] = true;
        }
      }
    }
  }

  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    if (s < 0 || s >= numSlots_) continue;
    NativeSlot& slot = slots_[s];

    if (!slot.needsPrezero()) continue;

    // GAP slots in a composite replay schedule execute live; their outputs
    // are inputs to subsequently captured islands. Never zero them here.
    if (hasComposite && isGapSlot[s]) {
      DSP_DIAG(MEMORY, "prezeroSegmentOutputs: SKIP slot=%d op=%s (GAP slot in composite schedule)",
               s, slot.ident.opName.c_str());
      continue;
    }

    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      int outIdx = slot.wiring.outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots_) continue;
      if (slot.slotPhase.isViewProducer) continue;
      NDArray* arr = outputSlots_[outIdx];
      if (arr == nullptr) continue;
      if (arr->isView()) continue;
      // Skip prezero for outputs that share their DataBuffer with an input
      // (e.g., KvScatter with ARRAY_COPY_OFFSET_INPUT_0). Zeroing the shared
      // buffer would destroy the input data, which includes previously computed
      // KV cache entries accumulated across decode steps.
      if (sd::ArrayOptions::hasAnyCopyOffset(arr->shapeInfo())) {
        DSP_DIAG(MEMORY, "prezeroSegmentOutputs: SKIP slot=%d outIdx=%d op=%s (copy-offset, shared buffer)",
                 s, outIdx, slot.ident.opName.c_str());
        continue;
      }
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
    // No silent "context recovery" for error 201/200/InvalidResourceHandle: a dead-stream
    // 201 means a stale/cross-thread execution stream — fixed at the root in
    // platformBeginExecution / DynamicShapePlanExecutor — so fail loud here instead of
    // masking it with a cudaSetDevice + synchronous-memset retry.
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
      std::vector<NDArray*> reads{output};
      NDArray::prepareSpecialUse({}, reads);
      NDArray::registerSpecialUse({}, reads);
      DSP_DIAG(SHAPE,
               "CONTROL_OUTPUT_SYNC: stage=%s slot=%d (%s) "
               "prepared host-current output for device after native execution",
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
  // On CUDA, a non-empty array with null special (device) buffer is invalid.
  if (db->special() == nullptr && !cached->isEmpty()) {
    return false;
  }

  // A plan may allocate a slot while device 0 is current and later execute
  // that slot in a secondary-device segment. The NDArray remains structurally
  // valid, but using its device-0 pointer from device 1 causes a deferred
  // illegal access in the consumer kernel. Treat this as a warmup cache miss;
  // the caller will discard the wrapper and allocate it after the segment
  // device has been bound.
  int currentDevice = -1;
  const auto deviceStatus = cudaGetDevice(&currentDevice);
  if (deviceStatus != cudaSuccess) {
    DSP_DIAG(MEMORY,
             "REUSABLE_SLOT_DEVICE_CHECK_FAILED: ptr=%p bufferDevice=%d err=%s",
             db->special(), db->deviceId(), cudaGetErrorString(deviceStatus));
    cudaGetLastError();
    return false;
  }
  if (db->deviceId() != currentDevice) {
    DSP_DIAG(MEMORY,
             "REUSABLE_SLOT_DEVICE_MISMATCH: ptr=%p bufferDevice=%d currentDevice=%d "
             "bytes=%lld — forcing device-local reallocation",
             db->special(), db->deviceId(), currentDevice,
             static_cast<long long>(db->getLenInBytes()));
    return false;
  }

  return true;
}

// ── Platform set/clear cublasLt epilogue ──────────────────────────────────────
void NativeDynamicShapePlan::platformSetLtEpilogue(const NativeSlot& slot, NDArray* biasArray) {
  if (biasArray == nullptr) return;
  std::vector<NDArray*> reads{biasArray};
  NDArray::prepareSpecialUse({}, reads);
  MmulHelper::setLtEpilogue(slot.flags.ltEpilogueType,
                             biasArray->dataBuffer() != nullptr ? biasArray->dataBuffer()->special() : nullptr,
                             biasArray->lengthOf() * biasArray->sizeOfT());
  NDArray::registerSpecialUse({}, reads);
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
