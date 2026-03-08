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
#include <helpers/ConstantShapeHelper.h>
#include <helpers/MmulHelper.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <helpers/AttentionWorkspace.h>
#include <graph/gpu/NvrtcKernelBuilder.h>
#include <graph/gpu/NvrtcKernelCache.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace sd {
namespace graph {

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

// ─── External address snapshot/compare ─────────────────────────────────────

void NativeDynamicShapePlan::snapshotExternalAddrs(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  seg.capturedExternalAddrs.resize(numExt);
  for (int i = 0; i < numExt; i++) {
    seg.capturedExternalAddrs[i] =
        (externalInputs[i] != nullptr) ? externalInputs[i]->specialBuffer() : nullptr;
  }
}

bool NativeDynamicShapePlan::externalAddrsMatch(
    const GraphSegment& seg, NDArray** externalInputs, int numExt) const {
  if (seg.capturedExternalAddrs.empty()) return false;  // no snapshot taken
  if (numExt != static_cast<int>(seg.capturedExternalAddrs.size())) return false;
  for (int i = 0; i < numExt; i++) {
    void* current = (externalInputs[i] != nullptr) ? externalInputs[i]->specialBuffer() : nullptr;
    if (current != seg.capturedExternalAddrs[i]) return false;
  }
  return true;
}

// ─── Segment execution: NVRTC JIT compilation ────────────────────────────────

Status NativeDynamicShapePlan::executeSegmentWithJit(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  LongType segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);

  // ── 1. If cached JIT kernel exists and shape matches, launch directly ──
  if (seg.jitKernel != nullptr && seg.jitKernel->valid && seg.jitShapeKey == segShapeKey) {
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      for (int o = 0; o < slots_[s].numOutputs; o++) {
        int si = slots_[s].outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_ && slotArrayCache_[si] != nullptr) {
          outputSlots_[si] = slotArrayCache_[si];
        }
      }
    }

    int64_t elementCount = 0;
    for (int s = seg.endSlot; s >= seg.startSlot; s--) {
      if (slots_[s].frozenConstantSlot || slots_[s].isIdentityOp || slots_[s].isFusedChainTail) continue;
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

    auto status = launchKernel(seg.jitKernel, elementCount,
                               externalArrays, numExt,
                               outputSlots_, totalOutputSlots_,
                               stream);
    if (status == Status::OK) {
      seg.executionCount++;
      return Status::OK;
    }
    delete seg.jitKernel;
    seg.jitKernel = nullptr;
    seg.jitCompileFailed = true;
    return Status::KERNEL_FAILURE;
  }

  // ── 2. Warmup pass (executionCount == 0): slot-by-slot ──
  if (seg.executionCount == 0) {
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // ── 3. Compile pass ──
  if (seg.jitKernel != nullptr && seg.jitShapeKey != segShapeKey) {
    delete seg.jitKernel;
    seg.jitKernel = nullptr;
  }

  if (!canJitSegment(slots_, seg.startSlot, seg.endSlot)) {
    seg.jitCompileFailed = true;
    DSP_DIAG(FALLBACK, "NVRTC JIT: seg[%d-%d] not fusible, falling back",
             seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    for (int o = 0; o < slots_[s].numOutputs; o++) {
      int si = slots_[s].outputSlotIndices[o];
      if (si >= 0 && si < totalOutputSlots_ && slotArrayCache_[si] != nullptr) {
        outputSlots_[si] = slotArrayCache_[si];
      }
    }
  }

  int segIdx = 0;
  for (size_t i = 0; i < segments_.size(); i++) {
    if (&segments_[i] == &seg) { segIdx = static_cast<int>(i); break; }
  }

  auto source = buildKernelSource(slots_, seg.startSlot, seg.endSlot,
                                  outputSlots_, totalOutputSlots_, segIdx);
  if (!source.valid) {
    seg.jitCompileFailed = true;
    DSP_DIAG(JIT, "NVRTC JIT: source generation failed for seg[%d-%d]",
             seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  DSP_DIAG_SEG(JIT, segIdx, "NVRTC: compiling kernel '%s' for seg[%d-%d] (%zu bytes source)",
               source.kernelName.c_str(), seg.startSlot, seg.endSlot,
               source.sourceCode.size());

  int deviceId = 0;
  cudaGetDevice(&deviceId);
  seg.jitKernel = compileKernel(source, deviceId);
  if (seg.jitKernel == nullptr || !seg.jitKernel->valid) {
    seg.jitCompileFailed = true;
    delete seg.jitKernel;
    seg.jitKernel = nullptr;
    DSP_DIAG(JIT, "NVRTC JIT: compilation failed for seg[%d-%d]",
             seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  seg.jitShapeKey = segShapeKey;

  int64_t elementCount = 0;
  for (int s = seg.endSlot; s >= seg.startSlot; s--) {
    if (slots_[s].frozenConstantSlot || slots_[s].isIdentityOp || slots_[s].isFusedChainTail) continue;
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

  auto status = launchKernel(seg.jitKernel, elementCount,
                             externalArrays, numExt,
                             outputSlots_, totalOutputSlots_,
                             stream);
  if (status == Status::OK) {
    seg.executionCount++;
    DSP_DIAG_SEG(JIT, segIdx, "NVRTC: launched '%s' for seg[%d-%d] (%lld elements)",
                 seg.jitKernel->kernelName.c_str(), seg.startSlot, seg.endSlot,
                 static_cast<long long>(elementCount));
    return Status::OK;
  }

  delete seg.jitKernel;
  seg.jitKernel = nullptr;
  seg.jitCompileFailed = true;
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
    bool hasGraph = (seg.cachedGraph != nullptr);
    bool shapeMatch = hasGraph && (seg.cachedShapeKey == segShapeKey);
    DSP_DIAG_SEG(EXECUTE, segIdx, "seg[%d-%d] execCount=%d hasGraph=%d shapeMatch=%d captureFailed=%d",
                 seg.startSlot, seg.endSlot, executeCount_,
                 static_cast<int>(hasGraph), static_cast<int>(shapeMatch),
                 static_cast<int>(seg.captureFailed));
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
      slot.shapeCacheValid = false;
      slot.cachedShapeKey = 0;
      slot.cachedOutputShapes.clear();
      slot.frozenContextReady = false;
      slot.frozenConstantSlot = false;
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
  if (seg.cachedGraph && seg.cachedShapeKey == segShapeKey &&
      seg.cachedGraph->getState() == cuda::GraphState::INSTANTIATED) {

    cudaStream_t cudaStr = (stream != nullptr)
        ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Update capture buffers
  bool captureBuffersOk = true;
  for (auto& cb : seg.captureBuffers) {
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

    if (captureBuffersOk && seg.cachedGraph->launchAsync(cudaStr)) {
      for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
        NativeSlot& slot = slots_[stepIdx];
        for (int i = 0; i < slot.numOutputs; i++) {
          int slotIdx = slot.outputSlotIndices[i];
          if (slotIdx >= 0 && slotIdx < totalOutputSlots_ && slotArrayCache_[slotIdx] != nullptr) {
            outputSlots_[slotIdx] = slotArrayCache_[slotIdx];
          }
        }
      }
      totalGraphReplays_++;
      seg.executionCount++;
      return Status::OK;
    }

    if (!captureBuffersOk) {
      DSP_DIAG(FALLBACK, "capture buffer shape mismatch for seg[%d-%d], "
               "invalidating for re-capture", seg.startSlot, seg.endSlot);
      clearGraphStreamError(cudaStr);
      for (auto& cb : seg.captureBuffers) {
        if (!cb.directReference) delete cb.buffer;
      }
      seg.captureBuffers.clear();
      seg.cachedGraph.reset();
      seg.capturedExternalAddrs.clear();
    } else {
      DSP_DIAG(FALLBACK, "graph replay failed for seg[%d-%d], "
               "falling back to slot-by-slot", seg.startSlot, seg.endSlot);
      clearGraphStreamError(cudaStr);
      seg.cachedGraph.reset();
      seg.capturedExternalAddrs.clear();
      if (seg.captureWorkspacePtr != nullptr) {
        cudaFree(seg.captureWorkspacePtr);
        seg.captureWorkspacePtr = nullptr;
        seg.captureWorkspaceBytes = 0;
      }
      for (auto* ptr : seg.capturedHostPtrs) {
        cudaFreeHost(ptr);
      }
      seg.capturedHostPtrs.clear();
    }
  }

  // ── WARM-UP ──
  bool shapeChanged = (seg.cachedShapeKey != segShapeKey);

  if (seg.executionCount > 0 && shapeChanged) {
    seg.consecutiveShapeChanges++;
    if (seg.consecutiveShapeChanges >= GraphSegment::INSTABILITY_THRESHOLD) {
      int segSize = seg.endSlot - seg.startSlot + 1;
      if (segSize <= GraphSegment::MIN_SPLIT_SIZE) {
        seg.captureFailed = true;
      } else {
        seg.needsSplit = true;
      }
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  } else if (!shapeChanged) {
    seg.consecutiveShapeChanges = 0;
  }

  if (seg.executionCount == 0 || (shapeChanged && !seg.captureFailed)) {
    if (shapeChanged && seg.cachedGraph) {
      seg.cachedGraph.reset();
      seg.capturedExternalAddrs.clear();
    }
    seg.cachedShapeKey = segShapeKey;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // ── CAPTURE ──
  if (seg.cachedGraph && seg.cachedShapeKey != segShapeKey) {
    seg.cachedGraph.reset();
    seg.capturedExternalAddrs.clear();
  }

  if (seg.captureOomRetries > 0 && seg.executionCount < seg.captureRetryAfterExec) {
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
      seg.captureFailed = true;
      return warmStatus;
    }

    std::memcpy(outputSlots_, preWarmupOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);

    if (seg.executionCount > 0) {
      seg.executionCount--;
    }

    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    seg.cachedShapeKey = segShapeKey;
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
    seg.captureFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }
  if (!scheduler.deviceSupportsGraphs(currentDevice)) {
    seg.captureFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // ── PRE-CAPTURE MEMORY CHECK ──
  bool isOomRetry = (seg.captureOomRetries > 0);
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

  auto handle = std::make_shared<cuda::CudaGraphHandle>(currentDevice);

  cudaGetLastError();
  if (cudaStr != nullptr) {
    auto syncErr = cudaStreamSynchronize(cudaStr);
    if (syncErr != cudaSuccess) {
      cudaGetLastError();
      seg.captureFailed = true;
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  }

  static const size_t CUBLAS_WORKSPACE_SIZE = 256 * 1024 * 1024;
  ensureCublasWorkspace(CUBLAS_WORKSPACE_SIZE);
  setCublasWorkspaceForCapture(stream);

  MmulHelper::resetCastCacheIndices();

  // ── CAPTURE BUFFER CREATION ──
  for (auto& cb : seg.captureBuffers) {
    delete cb.buffer;
  }
  seg.captureBuffers.clear();

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
            GraphSegment::CaptureBuffer cb;
            cb.buffer = src;
            cb.externalInputIndex = extIdx;
            cb.crossSegmentSlotIdx = -1;
            cb.capturedSize = srcBytes;
            cb.directReference = true;
            cb.initialCopyDone = true;
            cb.lastSourcePtr = src->specialBuffer();
            extInputToCaptureIdx[extIdx] = static_cast<int>(seg.captureBuffers.size());
            seg.captureBuffers.push_back(std::move(cb));
          } else {
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
              }
            }

            GraphSegment::CaptureBuffer cb;
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

            extInputToCaptureIdx[extIdx] = static_cast<int>(seg.captureBuffers.size());
            seg.captureBuffers.push_back(std::move(cb));
          }
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
            }
          }

          GraphSegment::CaptureBuffer cb;
          cb.buffer = capBuf;
          cb.externalInputIndex = -1;
          cb.crossSegmentSlotIdx = srcIdx;
          cb.capturedSize = srcBytes;

          crossSlotToCaptureIdx[srcIdx] = static_cast<int>(seg.captureBuffers.size());
          seg.captureBuffers.push_back(std::move(cb));
        }
      }
    }
  }

  if (captureBufferInitFailed) {
    for (auto& cb : seg.captureBuffers) {
      if (!cb.directReference) delete cb.buffer;
    }
    seg.captureBuffers.clear();
    cudaGetLastError();
    if (cudaStr != nullptr) {
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();
    }
    seg.captureFailed = true;
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Wire external/cross-segment inputs to capture buffers
  std::vector<std::pair<int, NDArray*>> savedExternalInputs;
  std::vector<std::pair<int, NDArray*>> savedOutputSlots;
  std::vector<NDArray*> preCapOutputSlots(outputSlots_, outputSlots_ + totalOutputSlots_);
  size_t pendingClosePreCapSize = pendingClose_.size();

  std::vector<bool> savedFrozenContextReady(seg.endSlot - seg.startSlot + 1);
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    savedFrozenContextReady[s - seg.startSlot] = slots_[s].frozenContextReady;
    slots_[s].frozenContextReady = false;
  }

  for (auto& [extIdx, cbIdx] : extInputToCaptureIdx) {
    savedExternalInputs.push_back({extIdx, externalArrays[extIdx]});
    externalArrays[extIdx] = seg.captureBuffers[cbIdx].buffer;
  }
  for (auto& [slotIdx, cbIdx] : crossSlotToCaptureIdx) {
    savedOutputSlots.push_back({slotIdx, outputSlots_[slotIdx]});
    outputSlots_[slotIdx] = seg.captureBuffers[cbIdx].buffer;
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
               seg.startSlot, seg.endSlot, seg.captureWorkspacePtr, seg.captureWorkspaceBytes);
  if (seg.captureWorkspacePtr == nullptr) {
    cudaError_t wsErr = cudaMalloc(&seg.captureWorkspacePtr, CAPTURE_WORKSPACE_SIZE);
    if (wsErr == cudaSuccess) {
      seg.captureWorkspaceBytes = CAPTURE_WORKSPACE_SIZE;
      DSP_DIAG_SEG(MEMORY, segIdx, "allocated %zuMB capture workspace for seg[%d-%d]",
                   CAPTURE_WORKSPACE_SIZE / (1024*1024), seg.startSlot, seg.endSlot);
    } else {
      cudaGetLastError();
      seg.captureWorkspacePtr = nullptr;
      seg.captureWorkspaceBytes = 0;
      DSP_DIAG_SEG(FALLBACK, segIdx, "capture workspace alloc failed (%s), graph will contain cudaMallocAsync nodes",
                   cudaGetErrorString(wsErr));
    }
  }
  tl_captureWorkspace = seg.captureWorkspacePtr;
  tl_captureWorkspaceSize = seg.captureWorkspaceBytes;
  tl_captureWorkspaceOffset = 0;
  DSP_DIAG_SEG(MEMORY, segIdx, "tl_captureWorkspace=%p size=%zu for capture",
               tl_captureWorkspace, tl_captureWorkspaceSize);

  tl_graphExecutionActive = true;
  tl_capturedHostPtrs.clear();
  tl_captureReplicateCache.clear();

  if (!handle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed)) {
    DSP_DIAG(FALLBACK, "graph capture begin failed for seg[%d-%d]",
             seg.startSlot, seg.endSlot);
    tl_graphExecutionActive = false;
    tl_graphCaptureStream = prevCaptureStream;
    tl_captureWorkspace = nullptr;
    tl_captureWorkspaceSize = 0;
    tl_captureWorkspaceOffset = 0;
    restoreCublasWorkspaceAfterCapture(stream);
    clearGraphStreamError(cudaStr);
    seg.captureFailed = true;
    for (auto& [extIdx, origPtr] : savedExternalInputs) {
      externalArrays[extIdx] = origPtr;
    }
    for (auto& [slotIdx, origPtr] : savedOutputSlots) {
      outputSlots_[slotIdx] = origPtr;
    }
    invalidateSegmentShapeState(seg);
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].frozenContextReady = savedFrozenContextReady[s - seg.startSlot];
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

      int releaseCount = releaseAtStepCounts_[stepIdx];
      if (releaseCount > 0) {
        for (int r = 0; r < releaseCount; r++) {
          int slotIdx = releaseAtStep_[stepIdx][r];
          outputSlots_[slotIdx] = nullptr;
        }
      }
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

    if (captureOomFailure && seg.captureOomRetries < GraphSegment::MAX_OOM_RETRIES) {
      seg.captureOomRetries++;
      seg.captureRetryAfterExec = seg.executionCount + GraphSegment::RETRY_INTERVAL;
      DSP_DIAG_SEG(MEMORY, segIdx, "graph capture OOM for seg[%d-%d], retry %d/%d after exec %d",
                   seg.startSlot, seg.endSlot,
                   seg.captureOomRetries, GraphSegment::MAX_OOM_RETRIES,
                   seg.captureRetryAfterExec);
    } else {
      seg.captureFailed = true;
      DSP_DIAG_SEG(FALLBACK, segIdx, "graph capture permanently failed for seg[%d-%d] (oom=%s, retries=%d)",
                   seg.startSlot, seg.endSlot,
                   captureOomFailure ? "true" : "false",
                   seg.captureOomRetries);
    }

    {
      std::unordered_set<NDArray*> preCapSet(preCapOutputSlots.begin(), preCapOutputSlots.end());
      for (size_t pi = pendingClosePreCapSize; pi < pendingClose_.size(); pi++) {
        if (preCapSet.find(pendingClose_[pi]) == preCapSet.end()) {
          delete pendingClose_[pi];
        }
      }
      pendingClose_.resize(pendingClosePreCapSize);
    }

    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      for (int o = 0; o < slots_[s].numOutputs; o++) {
        int si = slots_[s].outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_) {
          slotArrayCache_[si] = nullptr;
        }
      }
    }

    for (auto& cb : seg.captureBuffers) {
      if (!cb.directReference) delete cb.buffer;
    }
    seg.captureBuffers.clear();

    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].frozenContextReady = savedFrozenContextReady[s - seg.startSlot];
    }
    invalidateSegmentShapeState(seg);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Helper lambda to clean up capture buffers on failure
  auto cleanupCaptureBuffersOnFailure = [&seg, &preCapOutputSlots, &pendingClosePreCapSize, &savedFrozenContextReady, this]() {
    std::unordered_set<NDArray*> preCapSet(preCapOutputSlots.begin(), preCapOutputSlots.end());
    for (size_t pi = pendingClosePreCapSize; pi < pendingClose_.size(); pi++) {
      if (preCapSet.find(pendingClose_[pi]) == preCapSet.end()) {
        delete pendingClose_[pi];
      }
    }
    pendingClose_.resize(pendingClosePreCapSize);
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      for (int o = 0; o < slots_[s].numOutputs; o++) {
        int si = slots_[s].outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_) {
          slotArrayCache_[si] = nullptr;
        }
      }
    }
    for (auto& cb : seg.captureBuffers) {
      if (!cb.directReference) delete cb.buffer;
    }
    seg.captureBuffers.clear();
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].frozenContextReady = savedFrozenContextReady[s - seg.startSlot];
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
    seg.captureFailed = true;
    cleanupCaptureBuffersOnFailure();
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  if (!handle->instantiate()) {
    cudaGetLastError();
    DSP_DIAG(FALLBACK, "graph instantiate failed for seg[%d-%d]",
             seg.startSlot, seg.endSlot);
    for (auto* ptr : tl_capturedHostPtrs) {
      if (ptr != nullptr) cudaFreeHost(ptr);
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();
    clearGraphStreamError(cudaStr);
    seg.captureFailed = true;
    cleanupCaptureBuffersOnFailure();
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
    seg.captureFailed = true;
    cleanupCaptureBuffersOnFailure();
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  for (auto* ptr : tl_capturedHostPtrs) {
    handle->addCapturedHostPtr(ptr);
  }
  tl_capturedHostPtrs.clear();
  tl_captureReplicateCache.clear();

  seg.cachedGraph = handle;
  seg.cachedShapeKey = segShapeKey;
  seg.executionCount++;
  totalGraphReplays_++;

  for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
    NativeSlot& slot = slots_[stepIdx];
    for (int i = 0; i < slot.numOutputs; i++) {
      int slotIdx = slot.outputSlotIndices[i];
      if (slotIdx >= 0 && slotIdx < totalOutputSlots_ && slotArrayCache_[slotIdx] != nullptr) {
        outputSlots_[slotIdx] = slotArrayCache_[slotIdx];
      }
    }
  }

  if (seg.captureOomRetries > 0) {
    DSP_DIAG_SEG(MEMORY, segIdx, "graph capture SUCCEEDED on OOM retry %d for seg[%d-%d]",
                 seg.captureOomRetries, seg.startSlot, seg.endSlot);
    seg.captureOomRetries = 0;
    seg.captureRetryAfterExec = 0;
  }

  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    slots_[s].frozenContextReady = savedFrozenContextReady[s - seg.startSlot];
  }

  if (executionTimingEnabled_) {
    auto stats = handle->getStatistics();
    double wsUtilPct = seg.captureWorkspaceBytes > 0
        ? (100.0 * captureWorkspaceUsed / seg.captureWorkspaceBytes) : 0.0;
    DSP_DIAG_SEG(TIMING, segIdx, "captured CUDA graph seg[%d-%d] (%zu nodes, %zu edges) "
                 "[%d kern, %d memcpy, %d memset, %d alloc, %d free] ws=%zuKB/%zuKB (%.1f%%)",
                 seg.startSlot, seg.endSlot,
                 handle->getNumNodes(), handle->getNumEdges(),
                 stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
                 stats.numMemAllocs, stats.numMemFrees,
                 seg.captureWorkspacePtr ? (captureWorkspaceUsed / 1024) : 0,
                 seg.captureWorkspaceBytes / 1024, wsUtilPct);

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
  int savedSegExecCount = seg.executionCount;
  bool savedCaptureFailed = seg.captureFailed;
  seg.captureFailed = true;
  seg.executionCount = 999;

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
  std::vector<bool> savedFrozenCtx(seg.endSlot - seg.startSlot + 1);
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    savedFrozenCtx[s - seg.startSlot] = slots_[s].frozenContextReady;
    slots_[s].frozenContextReady = false;
  }
  // Set executeCount_ to 0 so shape inference runs fresh
  int savedExecCountGlobal = executeCount_;
  executeCount_ = 0;

  // Dump VARIABLE externals before fresh re-execution
  dspDumpVariableExternals(externalArrays, numExt, externalInputIsVariable_,
                           externalInputNames_, "before-fresh");

  auto freshStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);

  // Restore all state
  releaseAtStep_ = savedReleaseAtStep;
  releaseAtStepCounts_ = savedReleaseAtStepCounts;
  delete[] zeroedCounts;
  delete[] dummyRelease;
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    slots_[s].frozenContextReady = savedFrozenCtx[s - seg.startSlot];
  }
  executeCount_ = savedExecCountGlobal;
  seg.executionCount = savedSegExecCount;
  seg.captureFailed = savedCaptureFailed;

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
