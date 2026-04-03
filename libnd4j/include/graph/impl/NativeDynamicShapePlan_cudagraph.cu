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
 * NativeDynamicShapePlan — CUDA Graph Capture/Replay
 *
 * Contains executeSegmentWithGraph() (CUDA graph warmup/capture/replay
 * state machine), computeSegmentInputAddrKey() (GPU address hashing for
 * graph invalidation), and executeSegmentWithJit() (NVRTC JIT compilation).
 *
 * This file is compiled as .cu (CUDA source). All code is CUDA-only.
 */

#ifdef SD_CUDA

#include <graph/NativeDynamicShapePlan.h>
#include <graph/NativePlanCompiler.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspVerifyUtils.h>
#include <graph/cuda/CudaGraphReplayHandle.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/MmulHelper.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <helpers/AttentionWorkspace.h>
#include <graph/gpu/NvrtcKernelBuilder.h>
#include <graph/gpu/NvrtcKernelCache.h>
#include <graph/gpu/CaptureBufferRegistry.h>
#if HAVE_TRITON
#include <graph/gpu/TritonGraphBackend.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace sd {
namespace graph {

// Default capture host workspace size (32MB, configurable via env var)
static size_t CAPTURE_HOST_WORKSPACE_SIZE = []() -> size_t {
  const char* envVal = std::getenv("ND4J_DSP_CAPTURE_HOST_WORKSPACE_MB");
  size_t mb = 32;
  if (envVal != nullptr) {
    int parsed = std::atoi(envVal);
    if (parsed > 0 && parsed <= 1024) {
      mb = static_cast<size_t>(parsed);
    }
  }
  return mb * 1024ULL * 1024ULL;
}();

// ─── Segment input address key computation ──────────────────────────────────

LongType NativeDynamicShapePlan::computeSegmentInputAddrKey(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  LongType key = 0xcbf29ce484222325ULL;
  auto mix = [&key](LongType val) {
    key ^= val;
    key *= 0x100000001b3ULL;
  };

  std::unordered_set<int> segOutputSlots;
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numOutputs; i++) {
      segOutputSlots.insert(slot.outputSlotIndices[i]);
    }
  }

  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalInputs[extIdx] != nullptr) {
          mix(reinterpret_cast<LongType>(externalInputs[extIdx]->specialBuffer()));
        }
      } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
        if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
          mix(reinterpret_cast<LongType>(outputSlots_[srcIdx]->specialBuffer()));
        }
      }
    }
  }

  return key;
}

// ─── Create (ConstantOfShape) op value key ──────────────────────────────────
// Hashes the input DATA values of all 'create' ops in a segment, PLUS the data
// values of all VARIABLE external inputs (those marked isVariable).
//
// Create ops have value-dependent output shapes: their single input is a shape
// tensor whose *values* determine the output dimensions.  If these values change
// between capture and replay, the baked-in CUDA memset produces wrong-sized output.
//
// Variable external inputs (e.g., ConstantOfShape outputs computed by Java SameDiff)
// may also contain data that changes between steps.  Gap ops within the captured
// graph read from these external addresses — if the data changes but the graph
// isn't re-captured, replay produces stale results.  Hashing their data values
// detects these changes and forces re-capture.
//
// Returns 0 only if the segment has no create ops AND no variable external inputs.

LongType NativeDynamicShapePlan::computeCreateOpValueKey(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  LongType key = 0;
  auto mix = [&key](LongType val) {
    if (key == 0) key = 0xcbf29ce484222325ULL;
    key ^= val;
    key *= 0x100000001b3ULL;
  };

  // Part 1: Hash create op inputs (original logic)
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    if (slot.opName != "create" && slot.opName != "Create") continue;

    // Create op has 1 input: the shape tensor (INT64, small)
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      NDArray* inputArr = nullptr;
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt) inputArr = externalInputs[extIdx];
      } else if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        inputArr = outputSlots_[srcIdx];
      }
      if (inputArr == nullptr || inputArr->lengthOf() == 0) continue;

      // Read values from device (shape tensors are small, typically 4-5 elements)
      int n = (int)inputArr->lengthOf();
      if (n > 16) n = 16;  // Cap for safety
      int elemSize = DataTypeUtils::sizeOf(inputArr->dataType());
      std::vector<uint8_t> buf(n * elemSize);
      if (inputArr->specialBuffer()) {
        cudaMemcpy(buf.data(), inputArr->specialBuffer(), n * elemSize, cudaMemcpyDeviceToHost);
      } else if (inputArr->buffer()) {
        std::memcpy(buf.data(), inputArr->buffer(), n * elemSize);
      }
      // Hash each element as LongType
      for (int j = 0; j < n; j++) {
        LongType val = 0;
        if (elemSize == 8) {
          std::memcpy(&val, buf.data() + j * 8, 8);
        } else if (elemSize == 4) {
          int32_t v32; std::memcpy(&v32, buf.data() + j * 4, 4);
          val = (LongType)v32;
        }
        mix(val);
      }
    }
  }

  // Part 2: Placeholder external inputs (position_ids, attention_mask, etc.)
  // are handled via capture buffers + D2D copy before replay, so we do NOT
  // hash them here. This allows graph replay when only placeholder data changes.
  // The create op hash (Part 1) still catches ConstantOfShape shape changes.

  return key;
}

// ─── External address snapshot/compare ─────────────────────────────────────

void NativeDynamicShapePlan::snapshotExternalAddrs(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  if (seg.exec.replayHandle) {
    seg.exec.replayHandle->snapshotExternalAddresses(externalInputs, numExt);
  }
}

bool NativeDynamicShapePlan::externalAddrsMatch(
    const GraphSegment& seg, NDArray** externalInputs, int numExt) const {
  if (!seg.exec.replayHandle) return false;
  return seg.exec.replayHandle->externalAddressesMatch(externalInputs, numExt);
}

// ─── Segment execution: NVRTC JIT compilation ────────────────────────────────

Status NativeDynamicShapePlan::executeSegmentWithJit(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  LongType segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);

  // ── 1. If cached JIT kernel exists and shape matches, launch directly ──
  if (seg.exec.jitKernel != nullptr && seg.exec.jitKernel->valid && seg.exec.jitShapeKey == segShapeKey) {
    // Phase 2: slotArrayCache_ == outputSlots_ (unified), no restore needed

    int64_t elementCount = 0;
    for (int s = seg.endSlot; s >= seg.startSlot; s--) {
      if (slots_[s].frozenConstantSlot() || slots_[s].isIdentityOp || slots_[s].isFusedChainTail) continue;
      for (int o = 0; o < slots_[s].numOutputs; o++) {
        int si = slots_[s].outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
          elementCount = outputSlots_[si]->lengthOf();
          break;
        }
      }
      if (elementCount > 0) break;
    }

    if (elementCount <= 0) {
      DSP_DIAG(JIT, "NVRTC JIT: zero element count for seg[%d-%d]", seg.startSlot, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }

    auto status = launchKernel(seg.exec.jitKernel, elementCount,
                               externalArrays, numExt,
                               outputSlots_, totalOutputSlots_,
                               stream);
    if (status == Status::OK) {
      seg.exec.executionCount++;
      return Status::OK;
    }
    delete seg.exec.jitKernel;
    seg.exec.jitKernel = nullptr;
    seg.exec.jitCompileFailed = true;
    return Status::KERNEL_FAILURE;
  }

  // ── 2. Warmup pass (executionCount == 0): slot-by-slot ──
  if (seg.exec.executionCount == 0) {
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // ── 3. Compile pass ──
  if (seg.exec.jitKernel != nullptr && seg.exec.jitShapeKey != segShapeKey) {
    delete seg.exec.jitKernel;
    seg.exec.jitKernel = nullptr;
  }

  if (!canJitSegment(slots_, seg.startSlot, seg.endSlot)) {
    seg.exec.jitCompileFailed = true;
    DSP_DIAG(FALLBACK, "NVRTC JIT: seg[%d-%d] not fusible, falling back",
             seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  // Phase 2: slotArrayCache_ == outputSlots_ (unified), no restore needed

  int segIdx = 0;
  for (size_t i = 0; i < segments_.size(); i++) {
    if (&segments_[i] == &seg) { segIdx = static_cast<int>(i); break; }
  }

  auto source = buildKernelSource(slots_, seg.startSlot, seg.endSlot,
                                  outputSlots_, totalOutputSlots_, segIdx);
  if (!source.valid) {
    seg.exec.jitCompileFailed = true;
    DSP_DIAG(JIT, "NVRTC JIT: source generation failed for seg[%d-%d]",
             seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  DSP_DIAG_SEG(JIT, segIdx, "NVRTC: compiling kernel '%s' for seg[%d-%d] (%zu bytes source)",
               source.kernelName.c_str(), seg.startSlot, seg.endSlot,
               source.sourceCode.size());

  int deviceId = 0;
  cudaGetDevice(&deviceId);
  seg.exec.jitKernel = compileKernel(source, deviceId);
  if (seg.exec.jitKernel == nullptr || !seg.exec.jitKernel->valid) {
    seg.exec.jitCompileFailed = true;
    delete seg.exec.jitKernel;
    seg.exec.jitKernel = nullptr;
    DSP_DIAG(JIT, "NVRTC JIT: compilation failed for seg[%d-%d]",
             seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  seg.exec.jitShapeKey = segShapeKey;

  int64_t elementCount = 0;
  for (int s = seg.endSlot; s >= seg.startSlot; s--) {
    if (slots_[s].frozenConstantSlot() || slots_[s].isIdentityOp || slots_[s].isFusedChainTail) continue;
    for (int o = 0; o < slots_[s].numOutputs; o++) {
      int si = slots_[s].outputSlotIndices[o];
      if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
        elementCount = outputSlots_[si]->lengthOf();
        break;
      }
    }
    if (elementCount > 0) break;
  }

  if (elementCount <= 0) {
    DSP_DIAG(JIT, "NVRTC JIT: zero element count for seg[%d-%d]", seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  auto status = launchKernel(seg.exec.jitKernel, elementCount,
                             externalArrays, numExt,
                             outputSlots_, totalOutputSlots_,
                             stream);
  if (status == Status::OK) {
    seg.exec.executionCount++;
    DSP_DIAG_SEG(JIT, segIdx, "NVRTC: launched '%s' for seg[%d-%d] (%lld elements)",
                 seg.exec.jitKernel->kernelName.c_str(), seg.startSlot, seg.endSlot,
                 static_cast<long long>(elementCount));
    return Status::OK;
  }

  delete seg.exec.jitKernel;
  seg.exec.jitKernel = nullptr;
  seg.exec.jitCompileFailed = true;
  return Status::KERNEL_FAILURE;
}

// ─── Segment execution: CUDA Graph capture/replay ────────────────────────────

// cuBLAS workspace functions (ensureCublasWorkspace, setCublasWorkspaceForCapture,
// restoreCublasWorkspaceAfterCapture) are in NativeDynamicShapePlan_cublas.cu
// because cublas_v2.h includes cuda_fp16.h which conflicts with our float16.h
// when compiled by g++.

Status NativeDynamicShapePlan::executeSegmentWithGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  int segIdx = 0;
  for (size_t i = 0; i < segments_.size(); ++i) {
    if (&segments_[i] == &seg) { segIdx = static_cast<int>(i); break; }
  }

  // Compute shape key for this segment's inputs
  LongType segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);

  {
    bool hasGraph = (seg.exec.replayHandle != nullptr);
    bool shapeMatch = hasGraph && (seg.exec.cachedShapeKey == segShapeKey);
    DSP_DIAG_SEG(EXECUTE, segIdx, "seg[%d-%d] execCount=%d hasGraph=%d shapeMatch=%d compilationFailed=%d",
                 seg.startSlot, seg.endSlot, executeCount_,
                 static_cast<int>(hasGraph), static_cast<int>(shapeMatch),
                 static_cast<int>(seg.exec.compilationFailed));
  }

  auto needsHostMirror = [](NDArray* arr) -> bool {
    if (arr == nullptr) return false;
    auto dt = arr->dataType();
    return (dt == INT32 || dt == INT64 || dt == BOOL) && arr->lengthOf() > 0 && arr->lengthOf() <= 32;
  };

  auto mirrorHostAndDevice = [&](NDArray* src, NDArray* dst, size_t bytes) -> bool {
    if (src == nullptr || dst == nullptr || bytes == 0) return true;
    src->syncToHost();
    void* srcHost = src->buffer();
    void* dstHost = dst->buffer();
    if (srcHost == nullptr || dstHost == nullptr) return false;
    std::memcpy(dstHost, srcHost, bytes);
    dst->tickWriteHost();
    dst->syncToDevice();
    return true;
  };

  auto invalidateSegmentShapeState = [&](GraphSegment& segRef) {
    for (int stepIdx = segRef.startSlot; stepIdx <= segRef.endSlot; stepIdx++) {
      auto& slot = slots_[stepIdx];
      slot.state_ = NativeSlot::SlotState::WARMUP;
      slot.cachedShapeKey = 0;
      slot.cachedOutputShapes.clear();
    }
  };

  auto clearGraphStreamError = [&](cudaStream_t cudaStrm) {
    cudaGetLastError();
    if (cudaStrm != nullptr) {
      cudaStreamSynchronize(cudaStrm);
      cudaGetLastError();
    }
  };

  // ── REPLAY: cached graph with matching shapes ──
  if (seg.exec.replayHandle && seg.exec.cachedShapeKey == segShapeKey &&
      seg.exec.replayHandle->isReady()) {

    cudaStream_t cudaStr = (stream != nullptr)
        ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Update capture buffers
  bool captureBuffersOk = true;
  for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
    if (cb.directReference) continue;

    NDArray* src = nullptr;
      if (cb.externalInputIndex >= 0 && cb.externalInputIndex < numExt) {
        src = externalArrays[cb.externalInputIndex];
      } else if (cb.crossSegmentSlotIdx >= 0 && cb.crossSegmentSlotIdx < totalOutputSlots_) {
        src = outputSlots_[cb.crossSegmentSlotIdx];
      }

      if (src == nullptr || cb.buffer == nullptr) {
        captureBuffersOk = false;
        break;
      }

      size_t srcBytes = src->lengthOf() * src->sizeOfT();
      if (srcBytes != cb.capturedSize) {
        captureBuffersOk = false;
        break;
      }

      if (srcBytes > 0) {
        if (needsHostMirror(src)) {
          if (!mirrorHostAndDevice(src, cb.buffer, srcBytes)) {
            captureBuffersOk = false;
            break;
          }
          if (Environment::getInstance().tritonVerifyKernels()) {
            DSP_DIAG(VERIFY, "SLOT_WRITE tag=CAPTURE_BUF_COPY(host_mirror) ext=%d cross=%d "
                      "dtype=%s len=%lld bytes=%zu",
                      cb.externalInputIndex, cb.crossSegmentSlotIdx,
                      DataTypeUtils::asString(src->dataType()).c_str(),
                      (long long)src->lengthOf(), srcBytes);
          }
        } else {
          src->syncToDevice();
          void* srcPtr = src->specialBuffer();
          void* dstPtr = cb.buffer->specialBuffer();
          if (srcPtr == nullptr || dstPtr == nullptr) {
            captureBuffersOk = false;
            break;
          }

          cudaError_t copyErr = cudaMemcpyAsync(dstPtr, srcPtr,
                                                srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
          if (copyErr != cudaSuccess) {
            captureBuffersOk = false;
            DSP_DIAG(FALLBACK, "capture buffer replay copy failed for seg[%d-%d] "
                     "(ext=%d cross=%d): %d (%s)",
                     seg.startSlot, seg.endSlot,
                     cb.externalInputIndex, cb.crossSegmentSlotIdx,
                     static_cast<int>(copyErr), cudaGetErrorString(copyErr));
            cudaGetLastError();
            break;
          }
          // Keep actuality in sync after D2D copy — prevents spurious H2D sync.
          cb.buffer->dataBuffer()->writeSpecial();
          if (Environment::getInstance().tritonVerifyKernels()) {
            DSP_DIAG(VERIFY, "SLOT_WRITE tag=CAPTURE_BUF_COPY ext=%d cross=%d "
                      "dtype=%s len=%lld srcAddr=%p dstAddr=%p bytes=%zu",
                      cb.externalInputIndex, cb.crossSegmentSlotIdx,
                      DataTypeUtils::asString(src->dataType()).c_str(),
                      (long long)src->lengthOf(), srcPtr, dstPtr, srcBytes);
          }
        }
      }
    }

    // Refresh Triton sub-kernel arg tables before replay so that external input
    // buffer pointers (attention_mask, position_ids, etc.) reflect the current
    // step's values, not the stale values baked during graph capture.
#if HAVE_TRITON
    if (captureBuffersOk) {
      auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
      if (tritonBackend != nullptr) {
        tritonBackend->refreshArgTablesForReplay(seg, externalArrays, numExt,
                                                 outputSlots_, totalOutputSlots_,
                                                 stream);
      }
    }
#endif

    // ── DIAGNOSTIC: dump capture buffers and final output to verify replay correctness ──
    {
      int cbDumpCount = 0;
      for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
        if (cb.directReference || cb.buffer == nullptr || cbDumpCount >= 3) continue;
        if (cb.externalInputIndex >= 0 && cb.externalInputIndex < numExt) {
          NDArray* orig = externalArrays[cb.externalInputIndex];
          if (orig != nullptr && orig->lengthOf() > 0 && orig->lengthOf() <= 2048) {
            printf("CAPTURE_BUF ext=%d execCount=%d capBuf: ", cb.externalInputIndex, seg.exec.executionCount);
            fflush(stdout);
            cb.buffer->printIndexedBuffer("capBuf");
            fflush(stdout);
            printf("CAPTURE_BUF ext=%d execCount=%d orig:   ", cb.externalInputIndex, seg.exec.executionCount);
            fflush(stdout);
            orig->printIndexedBuffer("orig");
            fflush(stdout);
            cbDumpCount++;
          }
        }
      }
    }

    if (captureBuffersOk && seg.exec.replayHandle->replay(stream)) {
      {
        int finalSlot = seg.endSlot;
        if (finalSlot >= 0 && finalSlot < totalOutputSlots_ && slotArrayCache_[finalSlot] != nullptr) {
          NDArray* finalOut = slotArrayCache_[finalSlot];
          printf("REPLAY_OUTPUT seg[%d-%d] slot=%d execCount=%d: ",
                  seg.startSlot, seg.endSlot, finalSlot, seg.exec.executionCount);
          fflush(stdout);
          finalOut->printIndexedBuffer("logits", 10);
          fflush(stdout);
        }
      }

      // Phase 2: slotArrayCache_ == outputSlots_ (unified), no restore needed
      totalGraphReplays_++;
      seg.exec.executionCount++;
      return Status::OK;
    }

    if (!captureBuffersOk) {
      DSP_DIAG(FALLBACK, "capture buffer shape mismatch for seg[%d-%d], "
               "invalidating for re-capture", seg.startSlot, seg.endSlot);
      clearGraphStreamError(cudaStr);
      platformCleanupSegmentForRebuild(seg);
    } else {
      DSP_DIAG(FALLBACK, "graph replay failed for seg[%d-%d], "
               "falling back to slot-by-slot", seg.startSlot, seg.endSlot);
      clearGraphStreamError(cudaStr);
      platformCleanupSegmentForRebuild(seg);
    }
  }

  // ── WARM-UP ──
  bool shapeChanged = (seg.exec.cachedShapeKey != segShapeKey);

  if (seg.exec.executionCount == 0 || (shapeChanged && !seg.exec.compilationFailed)) {
    if (shapeChanged && seg.exec.replayHandle) {
      platformCleanupSegmentForRebuild(seg);
    }
    seg.exec.cachedShapeKey = segShapeKey;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // ── CAPTURE ──
  if (seg.exec.replayHandle && seg.exec.cachedShapeKey != segShapeKey) {
    platformCleanupSegmentForRebuild(seg);
  }

  if (seg.exec.captureOomRetries > 0 && seg.exec.executionCount < seg.exec.captureRetryAfterExec) {
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  bool hasValueDependentShapeOps = false;
  for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
    if (slots_[stepIdx].outputShapeDependsOnInputValues) {
      hasValueDependentShapeOps = true;
      break;
    }
  }
  if (hasValueDependentShapeOps) {
    std::vector<NDArray*> preWarmupOutputSlots(outputSlots_, outputSlots_ + totalOutputSlots_);

    auto warmStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    if (warmStatus != Status::OK) {
      seg.exec.compilationFailed = true;
      return warmStatus;
    }

    std::memcpy(outputSlots_, preWarmupOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);

    if (seg.exec.executionCount > 0) {
      seg.exec.executionCount--;
    }

    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    seg.exec.cachedShapeKey = segShapeKey;
  }

  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  auto& scheduler = cuda::CudaGraphScheduler::getInstance();

  int currentDevice = 0;
  cudaError_t currentDeviceErr = cudaGetDevice(&currentDevice);
  if (currentDeviceErr != cudaSuccess) {
    DSP_DIAG(FALLBACK, "cudaGetDevice failed during graph capture setup "
             "for seg[%d-%d]: %s",
             seg.startSlot, seg.endSlot, cudaGetErrorString(currentDeviceErr));
    cudaGetLastError();
    seg.exec.compilationFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }
  if (!scheduler.deviceSupportsGraphs(currentDevice)) {
    seg.exec.compilationFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // ── PRE-CAPTURE MEMORY CHECK ──
  bool isOomRetry = (seg.exec.captureOomRetries > 0);
  if (!isOomRetry) {
    size_t estimatedCaptureBytes = 0;
    for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
      NativeSlot& slot = slots_[stepIdx];
      for (int i = 0; i < slot.numOutputs; i++) {
        int slotIdx = slot.outputSlotIndices[i];
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_ && slotArrayCache_[slotIdx] != nullptr) {
          estimatedCaptureBytes += slotArrayCache_[slotIdx]->lengthOf() *
                                   slotArrayCache_[slotIdx]->sizeOfT();
        }
      }
    }

    size_t gpuFree = 0, gpuTotal = 0;
    cudaMemGetInfo(&gpuFree, &gpuTotal);

    size_t requiredFree = estimatedCaptureBytes * 2;
    if (requiredFree > gpuFree) {
      DSP_DIAG_SEG(MEMORY, segIdx, "skipping graph capture for seg[%d-%d] (%d ops): "
                   "estimated %zuMB (2x %zuMB) > free %zuMB (total %zuMB)",
                   seg.startSlot, seg.endSlot, seg.endSlot - seg.startSlot + 1,
                   requiredFree / (1024 * 1024),
                   estimatedCaptureBytes / (1024 * 1024),
                   gpuFree / (1024 * 1024),
                   gpuTotal / (1024 * 1024));
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  }

  seg.exec.replayHandle = GraphReplayFactory::create(currentDevice);
  auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
  auto handle = cudaReplay->getNativeHandle();

  cudaGetLastError();
  if (cudaStr != nullptr) {
    auto syncErr = cudaStreamSynchronize(cudaStr);
    if (syncErr != cudaSuccess) {
      cudaGetLastError();
      seg.exec.compilationFailed = true;
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  }

  static const size_t CUBLAS_WORKSPACE_SIZE = 256 * 1024 * 1024;
  ensureCublasWorkspace(CUBLAS_WORKSPACE_SIZE);
  setCublasWorkspaceForCapture(stream);

  MmulHelper::resetCastCacheIndices();

  // ── CAPTURE BUFFER CREATION ──
  for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
    if (!cb.directReference) delete cb.buffer;
  }
  seg.exec.replayHandle->getCaptureBuffers().clear();

  std::unordered_set<int> segOutputSlots;
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numOutputs; i++) {
      segOutputSlots.insert(slot.outputSlotIndices[i]);
    }
  }

  std::unordered_map<int, int> extInputToCaptureIdx;
  std::unordered_map<int, int> crossSlotToCaptureIdx;
  bool captureBufferInitFailed = false;

  for (int s = seg.startSlot; s <= seg.endSlot && !captureBufferInitFailed; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numInputs && !captureBufferInitFailed; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalArrays[extIdx] != nullptr &&
            extInputToCaptureIdx.find(extIdx) == extInputToCaptureIdx.end()) {
          NDArray* src = externalArrays[extIdx];
          size_t srcBytes = src->lengthOf() * src->sizeOfT();

          bool isKvCacheInput = false;
          if (kvCacheRetentionEnabled_) {
            for (int km = 0; km < kvCacheNumMappings_; km++) {
              if (kvCacheMappings_[km].pastInputExternalIdx == extIdx) {
                isKvCacheInput = true;
                break;
              }
            }
          }

          if (isKvCacheInput) {
            src->syncToDevice();
            ReplayCaptureBuffer cb;
            cb.buffer = src;
            cb.externalInputIndex = extIdx;
            cb.crossSegmentSlotIdx = -1;
            cb.capturedSize = srcBytes;
            cb.directReference = true;
            cb.initialCopyDone = true;
            cb.lastSourcePtr = src->specialBuffer();
            extInputToCaptureIdx[extIdx] = static_cast<int>(seg.exec.replayHandle->getCaptureBuffers().size());
            seg.exec.replayHandle->addCaptureBuffer(std::move(cb));
          } else {
            // Detect weight tensors: large inputs (> 1MB) that are NOT the dynamic
            // decode inputs (position_ids, attention_mask, input_ids, inputs_embeds).
            // Weights never change between decode steps — using directReference
            // avoids duplicating ~10GB of model weights in capture buffers.
            constexpr size_t WEIGHT_THRESHOLD = 1 * 1024 * 1024;  // 1MB
            
            // During capture, hasPendingDecodeUpdate_ is false. Check external input
            // indices directly to identify dynamic decode inputs.
            // Dynamic decode inputs are small (scalars or 1D/2D tensors).
            // Weights are typically large (>= 1MB) and rank 2 or 4.
            bool isDynamicDecodeInput = false;
            if (isDecodeInputsConfigured()) {
              isDynamicDecodeInput = (extIdx == decodeInputIdsExtIdx_ ||
                                      extIdx == decodePositionIdsExtIdx_ ||
                                      extIdx == decodeAttentionMaskExtIdx_);
            }
            
            // inputs_embeds is typically a rank-3 tensor with shape [batch, seq, hidden]
            // where seq is the prompt length (varies per inference). Check shape to distinguish.
            bool isInputsEmbeds = false;
            if (src->rankOf() == 3) {
              auto* shape = src->shapeOf();
              // inputs_embeds has shape [batch, seq, hidden] where seq >= 1
              // Weight tensors with rank 3 are rare (usually rank 2 or 4)
              // If seq dim (shape[1]) is large (> 100), likely inputs_embeds
              if (shape[1] > 100) {
                isInputsEmbeds = true;
              }
            }

            bool isWeight = !isDynamicDecodeInput && !isInputsEmbeds && srcBytes >= WEIGHT_THRESHOLD;

            if (isWeight) {
              // Weight tensor: use directReference to avoid duplicating GPU memory.
              // The graph reads directly from the original weight buffer, which
              // never moves (protected by frozen ref count).
              src->syncToDevice();
              ReplayCaptureBuffer cb;
              cb.buffer = src;
              cb.externalInputIndex = extIdx;
              cb.crossSegmentSlotIdx = -1;
              cb.capturedSize = srcBytes;
              cb.directReference = true;
              cb.initialCopyDone = true;
              cb.lastSourcePtr = src->specialBuffer();
              extInputToCaptureIdx[extIdx] = static_cast<int>(seg.exec.replayHandle->getCaptureBuffers().size());
              seg.exec.replayHandle->addCaptureBuffer(std::move(cb));
              DSP_DIAG(MEMORY, "capture buffer init: extIdx=%d is weight (%zu MB), using directReference",
                       extIdx, srcBytes / (1024 * 1024));
            } else {
              // Regular placeholder (dynamic decode input or small tensor):
              // create a fixed-address capture buffer
              auto srcShapeVec = *src->getShapeAsVector();
              auto* capBuf = new NDArray(src->ordering(), srcShapeVec, src->dataType(),
                                         sd::LaunchContext::defaultContext());
            if (srcBytes > 0) {
              if (needsHostMirror(src)) {
                if (!mirrorHostAndDevice(src, capBuf, srcBytes)) {
                  DSP_DIAG(MEMORY, "capture buffer init host mirror failed for seg[%d-%d] "
                           "(ext input %d)", seg.startSlot, seg.endSlot, extIdx);
                  delete capBuf;
                  captureBufferInitFailed = true;
                  break;
                }
              } else {
                src->syncToDevice();
                void* srcPtr = src->specialBuffer();
                void* dstPtr = capBuf->specialBuffer();
                if (srcPtr == nullptr || dstPtr == nullptr) {
                  DSP_DIAG(MEMORY, "capture buffer init got null ptr for seg[%d-%d] "
                           "(ext input %d)", seg.startSlot, seg.endSlot, extIdx);
                  delete capBuf;
                  captureBufferInitFailed = true;
                  break;
                }

                cudaError_t copyErr = cudaMemcpyAsync(dstPtr, srcPtr,
                                                      srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
                if (copyErr != cudaSuccess) {
                  DSP_DIAG(MEMORY, "capture buffer init copy failed for seg[%d-%d] "
                           "(ext input %d): %d (%s)",
                           seg.startSlot, seg.endSlot, extIdx,
                           static_cast<int>(copyErr), cudaGetErrorString(copyErr));
                  delete capBuf;
                  captureBufferInitFailed = true;
                  break;
                }
                // Mark device as actual so syncToDevice() is a no-op during capture.
                // Without this, the graph captures H2D memcpy nodes from stale host
                // buffers that overwrite fresh D2D data on every replay.
                capBuf->dataBuffer()->writeSpecial();
              }
              // Also mirror host buffer for ops that read from host during capture.
              if (srcBytes > 0 && src->buffer() && capBuf->buffer()) {
                std::memcpy(capBuf->buffer(), src->buffer(), srcBytes);
                capBuf->dataBuffer()->readPrimary();
              }
            }

            ReplayCaptureBuffer cb;
            cb.buffer = capBuf;
            cb.externalInputIndex = extIdx;
            cb.crossSegmentSlotIdx = -1;
            cb.capturedSize = srcBytes;

            auto srcType = static_cast<NativeSourceType>(slot.inputSourceTypes[i]);
            if (srcType == SOURCE_PLACEHOLDER) {
              cb.neverSkipCopy = true;
            }

            if (kvCacheRetentionEnabled_) {
              for (int km = 0; km < kvCacheNumMappings_; km++) {
                if (extIdx == kvCacheMappings_[km].pastInputExternalIdx) {
                  cb.neverSkipCopy = true;
                  break;
                }
              }
            }

            extInputToCaptureIdx[extIdx] = static_cast<int>(seg.exec.replayHandle->getCaptureBuffers().size());
            seg.exec.replayHandle->addCaptureBuffer(std::move(cb));
            }  // end else (non-weight: create capture buffer)
          }  // end else (non-KV-cache)
        }
      } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
        if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr &&
            crossSlotToCaptureIdx.find(srcIdx) == crossSlotToCaptureIdx.end()) {
          NDArray* src = outputSlots_[srcIdx];
          auto crossShapeVec = *src->getShapeAsVector();
          auto* capBuf = new NDArray(src->ordering(), crossShapeVec, src->dataType(),
                                     sd::LaunchContext::defaultContext());
          size_t srcBytes = src->lengthOf() * src->sizeOfT();
          if (srcBytes > 0) {
            if (needsHostMirror(src)) {
              if (!mirrorHostAndDevice(src, capBuf, srcBytes)) {
                DSP_DIAG(MEMORY, "capture buffer init host mirror failed for seg[%d-%d] "
                         "(cross slot %d)", seg.startSlot, seg.endSlot, srcIdx);
                delete capBuf;
                captureBufferInitFailed = true;
                break;
              }
            } else {
              src->syncToDevice();
              void* srcPtr = src->specialBuffer();
              void* dstPtr = capBuf->specialBuffer();
              if (srcPtr == nullptr || dstPtr == nullptr) {
                DSP_DIAG(MEMORY, "capture buffer init got null ptr for seg[%d-%d] "
                         "(cross slot %d)", seg.startSlot, seg.endSlot, srcIdx);
                delete capBuf;
                captureBufferInitFailed = true;
                break;
              }

              cudaError_t copyErr = cudaMemcpyAsync(dstPtr, srcPtr,
                                                    srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
              if (copyErr != cudaSuccess) {
                DSP_DIAG(MEMORY, "capture buffer init copy failed for seg[%d-%d] "
                         "(cross slot %d): %d (%s)",
                         seg.startSlot, seg.endSlot, srcIdx,
                         static_cast<int>(copyErr), cudaGetErrorString(copyErr));
                delete capBuf;
                captureBufferInitFailed = true;
                break;
              }
              // Mark device as actual — same fix as external input capture buffers.
              capBuf->dataBuffer()->writeSpecial();
            }
            if (srcBytes > 0 && src->buffer() && capBuf->buffer()) {
              std::memcpy(capBuf->buffer(), src->buffer(), srcBytes);
              capBuf->dataBuffer()->readPrimary();
            }
          }

          ReplayCaptureBuffer cb;
          cb.buffer = capBuf;
          cb.externalInputIndex = -1;
          cb.crossSegmentSlotIdx = srcIdx;
          cb.capturedSize = srcBytes;

          crossSlotToCaptureIdx[srcIdx] = static_cast<int>(seg.exec.replayHandle->getCaptureBuffers().size());
          seg.exec.replayHandle->addCaptureBuffer(std::move(cb));
        }
      }
    }
  }

  if (captureBufferInitFailed) {
    platformCleanupSegmentForRebuild(seg);
    cudaGetLastError();
    if (cudaStr != nullptr) {
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();
    }
    seg.exec.compilationFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Wire external/cross-segment inputs to capture buffers
  std::vector<std::pair<int, NDArray*>> savedExternalInputs;
  std::vector<std::pair<int, NDArray*>> savedOutputSlots;
  std::vector<NDArray*> preCapOutputSlots(outputSlots_, outputSlots_ + totalOutputSlots_);
  // pendingClose_ removed: arrays persist (one array per slot)

  std::vector<NativeSlot::SlotState> savedFrozenContextReady(seg.endSlot - seg.startSlot + 1);
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    savedFrozenContextReady[s - seg.startSlot] = slots_[s].state_;
    if (slots_[s].state_ >= NativeSlot::SlotState::FROZEN) slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
  }

  for (auto& [extIdx, cbIdx] : extInputToCaptureIdx) {
    savedExternalInputs.push_back({extIdx, externalArrays[extIdx]});
    externalArrays[extIdx] = seg.exec.replayHandle->getCaptureBuffers()[cbIdx].buffer;
  }
  for (auto& [slotIdx, cbIdx] : crossSlotToCaptureIdx) {
    savedOutputSlots.push_back({slotIdx, outputSlots_[slotIdx]});
    outputSlots_[slotIdx] = seg.exec.replayHandle->getCaptureBuffers()[cbIdx].buffer;
  }

  if (cudaStr != nullptr) {
    cudaStreamSynchronize(cudaStr);
  }

  const cudaStream_t prevCaptureStream = tl_graphCaptureStream;
  cudaStream_t resolvedCaptureStream = cudaStr;
  if (resolvedCaptureStream == nullptr) {
    auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
    if (defaultStreamPtr != nullptr) {
      resolvedCaptureStream = *defaultStreamPtr;
    }
  }
  tl_graphCaptureStream = resolvedCaptureStream;

  // Allocate capture workspace
  static size_t CAPTURE_WORKSPACE_SIZE = []() -> size_t {
    const char* envVal = std::getenv("ND4J_DSP_CAPTURE_WORKSPACE_MB");
    size_t mb = 512;
    if (envVal != nullptr) {
      int parsed = std::atoi(envVal);
      if (parsed > 0 && parsed <= 4096) {
        mb = static_cast<size_t>(parsed);
      }
    }
    return mb * 1024ULL * 1024ULL;
  }();
  DSP_DIAG_SEG(MEMORY, segIdx, "capture workspace check seg[%d-%d]: ptr=%p bytes=%zu",
               seg.startSlot, seg.endSlot, seg.exec.replayHandle->getWorkspacePtr(), seg.exec.replayHandle->getWorkspaceBytes());
  if (seg.exec.replayHandle->getWorkspacePtr() == nullptr) {
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    void* registryPtr = (Environment::getInstance().dspCapturePoolEnabled() && captureBufferRegistry_ != nullptr)
                        ? captureBufferRegistry_ : nullptr;
    if (!seg.exec.replayHandle->allocateWorkspace(CAPTURE_WORKSPACE_SIZE, deviceId, registryPtr, seg.startSlot)) {
      DSP_DIAG_SEG(FALLBACK, segIdx, "capture workspace alloc failed for seg[%d-%d], graph will contain cudaMallocAsync nodes",
                   seg.startSlot, seg.endSlot);
    }
  }
  tl_captureWorkspace = seg.exec.replayHandle->getWorkspacePtr();
  tl_captureWorkspaceSize = seg.exec.replayHandle->getWorkspaceBytes();
  tl_captureWorkspaceOffset = 0;
  DSP_DIAG_SEG(MEMORY, segIdx, "tl_captureWorkspace=%p size=%zu for capture",
               tl_captureWorkspace, tl_captureWorkspaceSize);

  // Allocate pinned host workspace for H2D source copies during capture.
  // This eliminates cudaMallocHost calls during capture — all host data for
  // graph H2D memcpy nodes is bump-allocated from this pre-allocated buffer.
  void* captureHostWs = nullptr;
  auto hostWsErr = cudaMallocHost(&captureHostWs, CAPTURE_HOST_WORKSPACE_SIZE);
  if (hostWsErr != cudaSuccess) {
    cudaGetLastError();
    captureHostWs = nullptr;
    DSP_DIAG_SEG(FALLBACK, segIdx, "capture host workspace alloc failed (%zu bytes), H2D copies may use non-pinned sources",
                 CAPTURE_HOST_WORKSPACE_SIZE);
  }
  tl_captureHostWorkspace = captureHostWs;
  tl_captureHostWorkspaceSize = (captureHostWs != nullptr) ? CAPTURE_HOST_WORKSPACE_SIZE : 0;
  tl_captureHostWorkspaceOffset = 0;

  tl_graphExecutionActive = true;
  tl_capturedHostPtrs.clear();
  tl_captureReplicateCache.clear();

  // Track the host workspace as a single captured host pointer for lifetime management
  if (captureHostWs != nullptr) {
    tl_capturedHostPtrs.push_back(captureHostWs);
  }

  if (!handle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed)) {
    DSP_DIAG(FALLBACK, "graph capture begin failed for seg[%d-%d]",
             seg.startSlot, seg.endSlot);
    tl_graphExecutionActive = false;
    tl_graphCaptureStream = prevCaptureStream;
    tl_captureWorkspace = nullptr;
    tl_captureWorkspaceSize = 0;
    tl_captureWorkspaceOffset = 0;
    // Clean up host workspace on capture begin failure
    if (tl_captureHostWorkspace != nullptr) {
      cudaFreeHost(tl_captureHostWorkspace);
      tl_captureHostWorkspace = nullptr;
    }
    tl_captureHostWorkspaceSize = 0;
    tl_captureHostWorkspaceOffset = 0;
    restoreCublasWorkspaceAfterCapture(stream);
    clearGraphStreamError(cudaStr);
    seg.exec.compilationFailed = true;
    platformCleanupSegmentForRebuild(seg);
    for (auto& [extIdx, origPtr] : savedExternalInputs) {
      externalArrays[extIdx] = origPtr;
    }
    for (auto& [slotIdx, origPtr] : savedOutputSlots) {
      outputSlots_[slotIdx] = origPtr;
    }
    invalidateSegmentShapeState(seg);
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].state_ = savedFrozenContextReady[s - seg.startSlot];
    }
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  bool captureOk = true;
  bool captureOomFailure = false;
  int lastCaptureSlot = seg.startSlot;
  lastCaptureAudit_.clear();

  try {
    for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
      lastCaptureSlot = stepIdx;
      {
        cudaStreamCaptureStatus capStatus;
        cudaError_t capErr = cudaStreamGetCaptureInfo(cudaStr, &capStatus, nullptr);
        if (capErr != cudaSuccess || capStatus != cudaStreamCaptureStatusActive) {
          DSP_DIAG_SLOT(COMPILE, stepIdx, "CAPTURE BROKEN before slot %d (%s): "
                       "capErr=%d capStatus=%d",
                       stepIdx, slots_[stepIdx].opName.c_str(),
                       static_cast<int>(capErr), static_cast<int>(capStatus));
          captureOk = false;
          break;
        }
      }

      size_t nodesBefore = handle->getNumNodesDuringCapture(cudaStr);

      auto status = executeSlot(stepIdx, externalArrays, numExt, stream);
      if (status != Status::OK) {
        DSP_DIAG_SLOT(COMPILE, stepIdx, "op execution during capture failed at slot %d", stepIdx);
        captureOk = false;
        captureOomFailure = true;
        break;
      }

      {
        cudaStreamCaptureStatus capStatus;
        cudaError_t capErr = cudaStreamGetCaptureInfo(cudaStr, &capStatus, nullptr);
        if (capErr != cudaSuccess || capStatus != cudaStreamCaptureStatusActive) {
          DSP_DIAG_SLOT(COMPILE, stepIdx, "CAPTURE INVALIDATED by slot %d (%s)! "
                       "capErr=%d capStatus=%d",
                       stepIdx, slots_[stepIdx].opName.c_str(),
                       static_cast<int>(capErr), static_cast<int>(capStatus));
          captureOk = false;
          break;
        }
      }

      size_t nodesAfter = handle->getNumNodesDuringCapture(cudaStr);
      {
        cuda::CaptureAuditEntry entry;
        entry.slotIndex = stepIdx;
        entry.opName = slots_[stepIdx].opName;
        entry.nodesBefore = nodesBefore;
        entry.nodesAfter = nodesAfter;
        entry.nodesContributed = (nodesAfter > nodesBefore) ? (nodesAfter - nodesBefore) : 0;
        lastCaptureAudit_.push_back(std::move(entry));
      }

      // Release schedule removed: arrays persist (one array per slot)
    }
  } catch (const std::exception& e) {
    DSP_DIAG(FALLBACK, "exception during graph capture at slot %d (%s): %s",
             lastCaptureSlot, slots_[lastCaptureSlot].opName.c_str(), e.what());
    captureOk = false;
    std::string msg(e.what());
    if (msg.find("allocation failed") != std::string::npos) {
      captureOomFailure = true;
    }
  } catch (...) {
    DSP_DIAG(FALLBACK, "unknown exception during graph capture");
    captureOk = false;
    captureOomFailure = true;
  }

  // Capture phase complete — reset the flag
  size_t captureWorkspaceUsed = tl_captureWorkspaceOffset;
  tl_graphExecutionActive = false;
  tl_graphCaptureStream = prevCaptureStream;
  tl_captureWorkspace = nullptr;
  tl_captureWorkspaceSize = 0;
  tl_captureWorkspaceOffset = 0;
  // Reset host workspace thread-locals (ownership moves to tl_capturedHostPtrs → replay handle)
  tl_captureHostWorkspace = nullptr;
  tl_captureHostWorkspaceSize = 0;
  tl_captureHostWorkspaceOffset = 0;
  restoreCublasWorkspaceAfterCapture(stream);

  for (auto& [extIdx, origPtr] : savedExternalInputs) {
    externalArrays[extIdx] = origPtr;
  }
  for (auto& [slotIdx, origPtr] : savedOutputSlots) {
    outputSlots_[slotIdx] = origPtr;
  }

  if (!captureOk) {
    handle->endCapture(cudaStr);

    for (auto* ptr : tl_capturedHostPtrs) {
      if (ptr != nullptr) cudaFreeHost(ptr);
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();

    cudaGetLastError();

    if (cudaStr != nullptr) {
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();
    }

    if (captureOomFailure && seg.exec.captureOomRetries < GraphSegment::MAX_OOM_RETRIES) {
      seg.exec.captureOomRetries++;
      seg.exec.captureRetryAfterExec = seg.exec.executionCount + GraphSegment::RETRY_INTERVAL;
      DSP_DIAG_SEG(MEMORY, segIdx, "graph capture OOM for seg[%d-%d], retry %d/%d after exec %d",
                   seg.startSlot, seg.endSlot,
                   seg.exec.captureOomRetries, GraphSegment::MAX_OOM_RETRIES,
                   seg.exec.captureRetryAfterExec);
    } else {
      seg.exec.compilationFailed = true;
      DSP_DIAG_SEG(FALLBACK, segIdx, "graph capture permanently failed for seg[%d-%d] (oom=%s, retries=%d)",
                   seg.startSlot, seg.endSlot,
                   captureOomFailure ? "true" : "false",
                   seg.exec.captureOomRetries);
    }

    // Arrays persist — no pendingClose_ cleanup needed on capture failure

    platformCleanupSegmentForRebuild(seg);

    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].state_ = savedFrozenContextReady[s - seg.startSlot];
    }
    invalidateSegmentShapeState(seg);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Helper lambda to clean up capture buffers on failure
  auto cleanupCaptureBuffersOnFailure = [&seg, &savedFrozenContextReady, this]() {
    // Arrays persist — no pendingClose_ cleanup needed on capture failure
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].state_ = savedFrozenContextReady[s - seg.startSlot];
    }
  };

  if (!handle->endCapture(cudaStr)) {
    cudaGetLastError();
    DSP_DIAG(FALLBACK, "graph capture end failed for seg[%d-%d]",
             seg.startSlot, seg.endSlot);
    for (auto* ptr : tl_capturedHostPtrs) {
      if (ptr != nullptr) cudaFreeHost(ptr);
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();
    cudaGetLastError();
    cudaStreamSynchronize(cudaStr);
    cudaGetLastError();
    seg.exec.compilationFailed = true;
    cleanupCaptureBuffersOnFailure();
    platformCleanupSegmentForRebuild(seg);
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  if (!handle->instantiate()) {
    // ── OOM eviction: free older/smaller graphs to reclaim GPU memory ──────
    // When instantiation fails with OOM (cudaErrorMemoryAllocation = 2), we
    // evict up to MAX_OOM_RETRIES existing captured graphs, starting with the
    // smallest (fewest CUDA graph nodes). This frees the GPU memory consumed
    // by their cudaGraphExec_t + cudaGraph_t + capture buffers + workspace.
    //
    // Since instantiate() destroys _graph on failure, we cannot retry the
    // instantiation directly. Instead, after eviction we use the deferred
    // OOM retry mechanism (captureRetryAfterExec) to re-capture on the very
    // next execution — by which time the evicted memory is available.
    //
    // Evicted segments are fully reset so they can re-capture later when
    // memory pressure decreases (other segments may also be evicted by then,
    // or the evicted segment may no longer be needed).
    int numEvicted = 0;
    if (handle->wasLastInstantiateOom()) {
      DSP_DIAG(MEMORY, "graph instantiate OOM for seg[%d-%d], attempting eviction (up to %d segments)",
               seg.startSlot, seg.endSlot, GraphSegment::MAX_OOM_RETRIES);

      bool usePool = Environment::getInstance().dspCapturePoolEnabled() &&
                     captureBufferRegistry_ != nullptr;

      for (int evictAttempt = 0; evictAttempt < GraphSegment::MAX_OOM_RETRIES; evictAttempt++) {
        // Find the segment with the smallest captured graph (fewest nodes) to evict.
        // Skip the current segment being instantiated.
        int evictIdx = -1;
        size_t smallestNodes = SIZE_MAX;
        for (size_t si = 0; si < segments_.size(); si++) {
          if (static_cast<int>(si) == segIdx) continue;
          auto& candidate = segments_[si];
          if (!candidate.exec.replayHandle || !candidate.exec.replayHandle->isReady()) continue;
          // Get node count from the CUDA replay handle
          auto* candidateCudaReplay = dynamic_cast<CudaGraphReplayHandle*>(candidate.exec.replayHandle.get());
          size_t nodeCount = candidateCudaReplay ? candidateCudaReplay->getNumNodes() : 0;
          if (nodeCount == 0) nodeCount = 1;  // Treat unknown as minimal
          if (nodeCount < smallestNodes) {
            smallestNodes = nodeCount;
            evictIdx = static_cast<int>(si);
          }
        }

        if (evictIdx < 0) {
          DSP_DIAG(MEMORY, "no more evictable graph segments found for OOM recovery (evicted %d so far)",
                   numEvicted);
          break;
        }

        // Evict the selected segment's graph with full cleanup
        auto& evictSeg = segments_[evictIdx];
        DSP_DIAG(MEMORY, "evicting graph for seg[%d-%d] (%zu nodes) to free memory for seg[%d-%d] (attempt %d/%d)",
                 evictSeg.startSlot, evictSeg.endSlot, smallestNodes,
                 seg.startSlot, seg.endSlot, evictAttempt + 1, GraphSegment::MAX_OOM_RETRIES);

        // Free capture buffer NDArrays before destroying the handle.
        // Owned buffers (directReference=false) must be deleted to free their
        // GPU memory; direct references are not owned by us.
        for (auto& cb : evictSeg.exec.replayHandle->getCaptureBuffers()) {
          if (!cb.directReference) delete cb.buffer;
        }
        evictSeg.exec.replayHandle->getCaptureBuffers().clear();

        // Release capture workspace (pool-aware: returns to pool if available,
        // otherwise cudaFree)
        evictSeg.exec.replayHandle->releaseWorkspace(
            usePool ? captureBufferRegistry_ : nullptr,
            evictSeg.startSlot);

        // Free pinned host pointers allocated during capture
        evictSeg.exec.replayHandle->freeHostPointers();
        evictSeg.exec.replayHandle->clearExternalAddresses();

        // Destroy the replay handle (frees cudaGraphExec + cudaGraph via
        // CudaGraphHandle::cleanup())
        evictSeg.exec.replayHandle.reset();

        // Reset the evicted segment so it can re-capture on a future execution
        evictSeg.exec.cachedShapeKey = 0;
        evictSeg.exec.capturedInputAddrKey = 0;
        evictSeg.exec.capturedCreateValueKey = 0;
        evictSeg.exec.compilationFailed = false;
        evictSeg.exec.gapOpsCapturedInGraph = false;
        evictSeg.exec.argTableStable = false;
        evictSeg.exec.compiledByBackend.clear();
        // Reset execution count so evicted segment goes through warmup -> capture again
        evictSeg.exec.executionCount = 0;

        numEvicted++;

        // Synchronize to ensure GPU memory is actually freed before trying more
        if (cudaStr != nullptr) {
          cudaStreamSynchronize(cudaStr);
        }
        cudaGetLastError();

        DSP_DIAG(MEMORY, "evicted seg[%d-%d] (%zu nodes), total evicted: %d",
                 evictSeg.startSlot, evictSeg.endSlot, smallestNodes, numEvicted);
      }
    }

    // The current segment's _graph was destroyed by instantiate() on failure,
    // so we cannot retry instantiation. Fall through to OOM retry or permanent
    // failure.
    cudaGetLastError();

    if (handle->wasLastInstantiateOom() && seg.exec.captureOomRetries < GraphSegment::MAX_OOM_RETRIES) {
      // Use the OOM retry mechanism: defer re-capture to a future execution.
      // If we evicted segments above, retry on the very next execution (interval=1)
      // since the freed memory should be immediately available. Otherwise use
      // the standard retry interval to wait for memory pressure to decrease.
      seg.exec.captureOomRetries++;
      int retryInterval = (numEvicted > 0) ? 1 : GraphSegment::RETRY_INTERVAL;
      seg.exec.captureRetryAfterExec = seg.exec.executionCount + retryInterval;
      DSP_DIAG(MEMORY, "graph instantiate OOM for seg[%d-%d], will retry %d/%d after exec %d (evicted %d segments)",
               seg.startSlot, seg.endSlot,
               seg.exec.captureOomRetries, GraphSegment::MAX_OOM_RETRIES,
               seg.exec.captureRetryAfterExec, numEvicted);
    } else {
      DSP_DIAG(FALLBACK, "graph instantiate failed for seg[%d-%d] (oom=%s, retries=%d, evicted=%d)",
               seg.startSlot, seg.endSlot,
               handle->wasLastInstantiateOom() ? "true" : "false",
               seg.exec.captureOomRetries, numEvicted);
      seg.exec.compilationFailed = true;
    }

    for (auto* ptr : tl_capturedHostPtrs) {
      if (ptr != nullptr) cudaFreeHost(ptr);
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();
    clearGraphStreamError(cudaStr);
    cleanupCaptureBuffersOnFailure();
    platformCleanupSegmentForRebuild(seg);
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  cudaGetLastError();

  {
    auto stats = handle->getStatistics();
    DSP_DIAG_SEG(COMPILE, segIdx, "graph captured for seg[%d-%d]: "
                 "%zu nodes, %zu edges, %d kernels, %d memcpys, %d memsets, "
                 "%d memAllocs, %d memFrees, %d hostCallbacks, %d events, %d empty",
                 seg.startSlot, seg.endSlot,
                 handle->getNumNodes(), handle->getNumEdges(),
                 stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
                 stats.numMemAllocs, stats.numMemFrees,
                 stats.numHostCallbacks, stats.numEvents, stats.numEmpty);
    if (stats.numMemAllocs != stats.numMemFrees) {
      DSP_DIAG_SEG(COMPILE, segIdx, "WARNING: Unbalanced memory nodes: %d allocs vs %d frees. "
                   "This WILL cause graph launch failure",
                   stats.numMemAllocs, stats.numMemFrees);
    }
  }

  if (!handle->launchAsync(cudaStr)) {
    cudaGetLastError();
    DSP_DIAG(FALLBACK, "graph launch failed for seg[%d-%d]",
             seg.startSlot, seg.endSlot);
    for (auto* ptr : tl_capturedHostPtrs) {
      if (ptr != nullptr) cudaFreeHost(ptr);
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();
    clearGraphStreamError(cudaStr);
    seg.exec.compilationFailed = true;
    cleanupCaptureBuffersOnFailure();
    platformCleanupSegmentForRebuild(seg);
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  for (auto* ptr : tl_capturedHostPtrs) {
    seg.exec.replayHandle->addCapturedHostPtr(ptr);
  }
  tl_capturedHostPtrs.clear();
  tl_captureReplicateCache.clear();

  // replayHandle is already set (created before capture began)
  seg.exec.cachedShapeKey = segShapeKey;
  seg.exec.executionCount++;
  totalGraphReplays_++;

  // Mark as captured by raw CUDA graph path (not Triton).
  // This prevents the Triton replay path in NativeDynamicShapePlan_gpubackend.cpp
  // from incorrectly handling this segment — Triton replay has incompatible
  // D2D copy and arg table logic that can corrupt cross-segment data.
  if (seg.exec.compiledByBackend.empty()) {
    seg.exec.compiledByBackend = "CUDA";
  }

  // Clear compilationFailed — the CUDA graph path succeeded even if the Triton path
  // failed earlier. Without this, cleanup treats this segment as non-graph-managed
  // and frees its output/cross-segment slots, causing stale data on replay.
  if (seg.exec.compilationFailed) {
    DSP_DIAG(COMPILE, "clearing compilationFailed for seg[%d-%d] after successful CUDA graph capture",
             seg.startSlot, seg.endSlot);
    seg.exec.compilationFailed = false;
  }

  // Phase 2: slotArrayCache_ == outputSlots_ (unified), no restore needed

  if (seg.exec.captureOomRetries > 0) {
    DSP_DIAG_SEG(MEMORY, segIdx, "graph capture SUCCEEDED on OOM retry %d for seg[%d-%d]",
                 seg.exec.captureOomRetries, seg.startSlot, seg.endSlot);
    seg.exec.captureOomRetries = 0;
    seg.exec.captureRetryAfterExec = 0;
  }

  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    slots_[s].state_ = savedFrozenContextReady[s - seg.startSlot];
  }

  if (executionTimingEnabled_) {
    auto stats = handle->getStatistics();
    double wsUtilPct = seg.exec.replayHandle->getWorkspaceBytes() > 0
        ? (100.0 * captureWorkspaceUsed / seg.exec.replayHandle->getWorkspaceBytes()) : 0.0;
    DSP_DIAG_SEG(TIMING, segIdx, "captured CUDA graph seg[%d-%d] (%zu nodes, %zu edges) "
                 "[%d kern, %d memcpy, %d memset, %d alloc, %d free] ws=%zuKB/%zuKB (%.1f%%)",
                 seg.startSlot, seg.endSlot,
                 handle->getNumNodes(), handle->getNumEdges(),
                 stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
                 stats.numMemAllocs, stats.numMemFrees,
                 seg.exec.replayHandle->getWorkspacePtr() ? (captureWorkspaceUsed / 1024) : 0,
                 seg.exec.replayHandle->getWorkspaceBytes() / 1024, wsUtilPct);

    if (!lastCaptureAudit_.empty()) {
      printCaptureAudit();
    }
  }

  return Status::OK;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Replay Verification — reusable by all paths (Triton, CUDA_GRAPHS, etc.)
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::performReplayVerify(
    GraphSegment& seg, NDArray** externalArrays, int numExt,
    void* stream, const char* pathLabel) {
  DSP_DIAG(VERIFY, "performReplayVerify ENTERED path=%s execCount=%d",
           pathLabel, seg.exec.executionCount);
  fflush(stderr);

  // Ensure VERIFY diagnostics are enabled (may have been set after DspDiagnostics construction)
  DspDiagnostics::getInstance().enableCategories(DSP_DIAG_VERIFY);
  DspDiagnostics::getInstance().setLevel(DSP_LEVEL_FULL);

  cudaStream_t cudaStr = stream ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Find final output slot for argmax
  int finalOutputSlot = -1;
  if (seg.endSlot >= 0 && seg.endSlot < numSlots_ && slots_[seg.endSlot].numOutputs > 0) {
    finalOutputSlot = slots_[seg.endSlot].outputSlotIndices[0];
  }
  if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_) {
    finalOutputSlot = seg.endSlot;
  }

  // 1. Compute argmax from REPLAY output
  int replayArgmax = -1;
  if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
      outputSlots_[finalOutputSlot] != nullptr) {
    auto* replayFinal = outputSlots_[finalOutputSlot];
    if (replayFinal->lengthOf() > 0 && replayFinal->specialBuffer() != nullptr) {
      replayArgmax = dspArgmax(replayFinal->specialBuffer(), replayFinal->dataType(),
                                replayFinal->lengthOf());
      DSP_DIAG(VERIFY, "REPLAY_VERIFY ARGMAX(replay): slot=%d argmax=%d (of %lld elements) path=%s",
                finalOutputSlot, replayArgmax, (long long)replayFinal->lengthOf(), pathLabel);
    }
  }

  // 2. Snapshot all output slots from replay
  struct SlotSnap {
    int slotIdx, stepIdx;
    DataType dtype;
    LongType length;
    void* bufAddr;
    std::vector<uint8_t> data;
  };
  std::vector<SlotSnap> snaps;

  std::unordered_map<int, int> slotToStep;
  for (int s = seg.startSlot; s <= seg.endSlot && s < numSlots_; s++) {
    for (int oi = 0; oi < slots_[s].numOutputs; oi++) {
      int si = slots_[s].outputSlotIndices[oi];
      if (si >= 0 && si < totalOutputSlots_) slotToStep[si] = s;
    }
  }

  for (int si = 0; si < totalOutputSlots_; si++) {
    NDArray* arr = outputSlots_[si];
    if (!arr || arr->lengthOf() <= 0 || !arr->specialBuffer()) continue;
    if (slotToStep.find(si) == slotToStep.end()) continue;
    DataType dt = arr->dataType();
    int elemSize = DataTypeUtils::sizeOf(dt);
    if (elemSize <= 0) continue;
    int snapCount = std::min(static_cast<int>(arr->lengthOf()), 16);
    SlotSnap snap;
    snap.slotIdx = si;
    snap.stepIdx = slotToStep[si];
    snap.dtype = dt;
    snap.length = arr->lengthOf();
    snap.bufAddr = arr->specialBuffer();
    snap.data.resize(snapCount * elemSize);
    cudaMemcpy(snap.data.data(), arr->specialBuffer(), snapCount * elemSize, cudaMemcpyDeviceToHost);
    snaps.push_back(std::move(snap));
  }
  DSP_DIAG(VERIFY, "REPLAY_VERIFY: saved %zu snapshots from replay (%s path)", snaps.size(), pathLabel);

  // 3. Re-execute slot-by-slot for ground truth
  // Save segment state
  int savedSegExecCount = seg.exec.executionCount;
  bool savedCaptureFailed = seg.exec.compilationFailed;
  seg.exec.compilationFailed = true;
  seg.exec.executionCount = 999;

  // Disable releaseAtStep (prevents nullifying outputs before comparison)
  int** savedReleaseAtStep = releaseAtStep_;
  int* savedReleaseAtStepCounts = releaseAtStepCounts_;
  int* zeroedCounts = new int[numSlots_]();
  int** dummyRelease = new int*[numSlots_]();
  releaseAtStep_ = dummyRelease;
  releaseAtStepCounts_ = zeroedCounts;

  // Reset frozenContextReady to force normal execution path.
  // The frozen path only refreshes external/view-producer inputs — it does NOT
  // refresh regular slot-to-slot inputs, so downstream ops would read stale
  // warmup-era data instead of freshly computed outputs.
  std::vector<NativeSlot::SlotState> savedFrozenCtx(seg.endSlot - seg.startSlot + 1);
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    savedFrozenCtx[s - seg.startSlot] = slots_[s].state_;
    if (slots_[s].state_ >= NativeSlot::SlotState::FROZEN) slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
  }
  // Set executeCount_ to 0 so shape inference runs fresh
  int savedExecCountGlobal = executeCount_;
  executeCount_ = 0;

  // Dump VARIABLE externals before fresh re-execution
  dspDumpVariableExternals(externalArrays, numExt, externalInputIsVariable_,
                           externalInputNames_, "before-fresh");

  // DIAGNOSTIC: Dump small VARIABLE externals with both host AND device values
  {
    cudaDeviceSynchronize();
    for (int ei = 0; ei < numExt; ei++) {
      if (ei < (int)externalInputIsVariable_.size() && externalInputIsVariable_[ei] &&
          externalArrays[ei] != nullptr && externalArrays[ei]->lengthOf() <= 8) {
        auto* arr = externalArrays[ei];
        auto* db = arr->dataBuffer();
        int n = std::min((int)arr->lengthOf(), 8);
        int elemSize = DataTypeUtils::sizeOf(arr->dataType());
        std::vector<uint8_t> hostBytes(n * elemSize), devBytes(n * elemSize);
        if (db && db->primary()) std::memcpy(hostBytes.data(), static_cast<char*>(arr->buffer()), n * elemSize);
        if (arr->specialBuffer()) cudaMemcpy(devBytes.data(), arr->specialBuffer(), n * elemSize, cudaMemcpyDeviceToHost);
        float hv[8]={0}, dv[8]={0};
        dspBytesToFloat(hostBytes.data(), arr->dataType(), hv, n);
        dspBytesToFloat(devBytes.data(), arr->dataType(), dv, n);
        std::string name = (ei < (int)externalInputNames_.size()) ? externalInputNames_[ei] : "?";
        DSP_DIAG(VERIFY, "PRE_FRESH ext#%d:\"%s\" len=%d pAct=%d sAct=%d host=[%.0f,%.0f,%.0f,%.0f] dev=[%.0f,%.0f,%.0f,%.0f]",
                  ei, name.c_str(), n,
                  db ? (db->isPrimaryActual()?1:0) : -1,
                  db ? (db->isSpecialActual()?1:0) : -1,
                  hv[0],hv[1],hv[2],hv[3], dv[0],dv[1],dv[2],dv[3]);
      }
    }
  }

  auto freshStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);

  // Restore all state
  releaseAtStep_ = savedReleaseAtStep;
  releaseAtStepCounts_ = savedReleaseAtStepCounts;
  delete[] zeroedCounts;
  delete[] dummyRelease;
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    slots_[s].state_ = savedFrozenCtx[s - seg.startSlot];
  }
  executeCount_ = savedExecCountGlobal;
  seg.exec.executionCount = savedSegExecCount;
  seg.exec.compilationFailed = savedCaptureFailed;

  if (freshStatus != Status::OK) {
    DSP_DIAG(VERIFY, "REPLAY_VERIFY: slot-by-slot re-execution FAILED (%s path)", pathLabel);
    return;
  }

  cudaStreamSynchronize(cudaStr);

  // 4. Compute argmax from FRESH execution
  int freshArgmax = -1;
  if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
      outputSlots_[finalOutputSlot] != nullptr) {
    auto* freshFinal = outputSlots_[finalOutputSlot];
    if (freshFinal->lengthOf() > 0 && freshFinal->specialBuffer() != nullptr) {
      freshArgmax = dspArgmax(freshFinal->specialBuffer(), freshFinal->dataType(),
                               freshFinal->lengthOf());
      DSP_DIAG(VERIFY, "REPLAY_VERIFY ARGMAX(fresh): slot=%d argmax=%d (of %lld elements)",
                finalOutputSlot, freshArgmax, (long long)freshFinal->lengthOf());
    }
  }

  // 5. Compare snapshots vs fresh
  int mismatchCount = 0;
  int firstMismatchSlot = -1;
  float worstMaxDiff = 0.0f;
  for (auto& snap : snaps) {
    NDArray* fresh = outputSlots_[snap.slotIdx];
    if (!fresh || !fresh->specialBuffer()) continue;
    int elemSize = DataTypeUtils::sizeOf(snap.dtype);
    if (elemSize <= 0) continue;
    int compareCount = std::min(static_cast<int>(snap.data.size()) / elemSize,
                                 std::min(static_cast<int>(fresh->lengthOf()), 16));
    std::vector<uint8_t> freshData(compareCount * elemSize);
    cudaMemcpy(freshData.data(), fresh->specialBuffer(), compareCount * elemSize, cudaMemcpyDeviceToHost);
    float maxDiff = dspMaxDiff(snap.data.data(), freshData.data(), snap.dtype, compareCount);
    if (maxDiff > worstMaxDiff) worstMaxDiff = maxDiff;
    if (maxDiff > 1e-3f) {
      mismatchCount++;
      if (firstMismatchSlot < 0) firstMismatchSlot = snap.slotIdx;
      const char* opName = (snap.stepIdx < numSlots_) ? slots_[snap.stepIdx].opName.c_str() : "?";
      int nShow = std::min(compareCount, 4);
      float rv[4]={0}, fv[4]={0};
      dspBytesToFloat(snap.data.data(), snap.dtype, rv, nShow);
      dspBytesToFloat(freshData.data(), snap.dtype, fv, nShow);
      // Build input info
      std::string inputInfo;
      if (snap.stepIdx < numSlots_) {
        auto& slot = slots_[snap.stepIdx];
        for (int ii = 0; ii < slot.numInputs; ii++) {
          if (ii > 0) inputInfo += " ";
          int srcIdx = slot.inputSourceIndices[ii];
          if (srcIdx >= 0) {
            inputInfo += "slot#" + std::to_string(srcIdx);
            if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx])
              inputInfo += "(len=" + std::to_string(outputSlots_[srcIdx]->lengthOf()) + ")";
          } else {
            int extIdx = -(srcIdx+1);
            inputInfo += "ext#" + std::to_string(extIdx);
            if (extIdx < (int)externalInputNames_.size())
              inputInfo += ":\"" + externalInputNames_[extIdx] + "\"";
          }
        }
      }
      DSP_DIAG(VERIFY, "REPLAY_VERIFY MISMATCH slot=%d step=%d op=%s maxDiff=%.6f "
                "replay=[%.4f,%.4f,%.4f,%.4f] fresh=[%.4f,%.4f,%.4f,%.4f] inputs=[%s]",
                snap.slotIdx, snap.stepIdx, opName, maxDiff,
                rv[0], rv[1], rv[2], rv[3], fv[0], fv[1], fv[2], fv[3],
                inputInfo.c_str());
    }
  }

  if (mismatchCount > 0) {
    DSP_DIAG(VERIFY, "REPLAY_VERIFY SUMMARY: %d/%zu slots exceed 1e-3 tolerance "
              "(first mismatch slot=%d, worst maxDiff=%.6g) path=%s execCount=%d",
              mismatchCount, snaps.size(), firstMismatchSlot, worstMaxDiff,
              pathLabel, executeCount_);
    if (replayArgmax == freshArgmax) {
      DSP_DIAG(VERIFY, "REPLAY_VERIFY ARGMAX: MATCH (replay=%d fresh=%d)", replayArgmax, freshArgmax);
    } else {
      DSP_DIAG(VERIFY, "REPLAY_VERIFY ARGMAX: *** MISMATCH *** (replay=%d fresh=%d)", replayArgmax, freshArgmax);
    }
  } else {
    DSP_DIAG(VERIFY, "REPLAY_VERIFY SUMMARY: ALL MATCH (%zu slots, maxDiff=%.6g) path=%s execCount=%d",
              snaps.size(), worstMaxDiff, pathLabel, executeCount_);
  }
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
