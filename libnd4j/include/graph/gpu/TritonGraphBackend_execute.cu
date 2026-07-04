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

#include <config.h>
#if HAVE_TRITON

#include <graph/gpu/TritonGraphBackend.h>
#include <graph/gpu/CapturedModuleRegistry.h>
#include <graph/gpu/TritonGraphBackend_internal.h>
#include <graph/gpu/TritonTargetDispatch.h>
#include <graph/DspDiagnostics.h>
#include <graph/LegacyOpTypeCodes.h>
#include <system/Environment.h>
#include <helpers/logger.h>
#include <helpers/ShapeUtils.h>
#include <array/DataTypeUtils.h>
#include <array/ShapeList.h>
#include <array/DataBuffer.h>
#include <memory/cuda/CudaMemoryPool.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

using namespace triton_internal;

// ─── Diagnostic helpers ─────────────────────────────────────────────────────


// Compute a host-side metadata fingerprint without reading device contents.
// VERIFY diagnostics must not introduce blocking D2H reads in DSP execution.
static uint64_t hashSlotGpuContent(NDArray* arr, cudaStream_t stream) {
  if (!arr || !arr->specialBuffer() || !arr->dataBuffer()) return 0;
  size_t bytes = arr->lengthOf() * arr->sizeOfT();
  uint64_t hash = FNV1A64_OFFSET_BASIS;
  void* sbuf = arr->specialBuffer();
  mixFNV1a(hash, &sbuf, sizeof(sbuf));
  mixFNV1a(hash, &bytes, sizeof(bytes));
  auto dt = static_cast<int>(arr->dataType());
  mixFNV1a(hash, &dt, sizeof(dt));
  return hash;
}

// Log hashes of all output slots in a range [startSlot, endSlot].
// label = "GAP" or "TRITON" or "SKIP" to identify the execution phase.
static void logSlotHashes(const char* label, int startSlot, int endSlot,
                          NativeSlot* slots, NDArray** outputSlots, int totalOutputSlots,
                          cudaStream_t stream, int execCount) {
  if (!DSP_DIAG_ENABLED(VERIFY)) return;
  std::unordered_set<int> loggedSlots;
  for (int si = startSlot; si <= endSlot; si++) {
    for (int o = 0; o < slots[si].wiring.numOutputs; o++) {
      int outIdx = slots[si].wiring.outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
      if (!loggedSlots.insert(outIdx).second) continue;  // already logged
      NDArray* arr = outputSlots[outIdx];
      if (!arr) continue;
      uint64_t hash = hashSlotGpuContent(arr, stream);
      DSP_DIAG(VERIFY, "HASH %s [%d-%d] exec=%d slot=%d (%s) hash=%016llx len=%lld dt=%d",
               label, startSlot, endSlot, execCount,
               outIdx, slots[si].ident.opName.c_str(), (unsigned long long)hash,
               (long long)arr->lengthOf(), static_cast<int>(arr->dataType()));
    }
  }
}

// Log the actuality state (isPrimaryActual / isSpecialActual) for slots in a range.
static void logActualityState(const char* label, int startSlot, int endSlot,
                              NativeSlot* slots, NDArray** outputSlots, int totalOutputSlots,
                              NDArray** externalInputs, int numExternalInputs) {
  if (!DSP_DIAG_ENABLED(VERIFY)) return;
  std::unordered_set<DataBuffer*> seen;
  int hostPrimaryCount = 0, devicePrimaryCount = 0, bothActualCount = 0, neitherCount = 0;
  for (int si = startSlot; si <= endSlot; si++) {
    // Check inputs
    for (int i = 0; i < slots[si].wiring.numInputs; i++) {
      int srcIdx = slots[si].wiring.inputSourceIndices[i];
      NDArray* arr = resolveRangeArray(srcIdx, externalInputs, numExternalInputs,
                                        outputSlots, totalOutputSlots);
      if (!arr || !arr->dataBuffer()) continue;
      auto* db = arr->dataBuffer();
      if (!seen.insert(db).second) continue;
      bool pAct = db->isPrimaryActual();
      bool sAct = db->isSpecialActual();
      if (pAct && sAct) bothActualCount++;
      else if (pAct && !sAct) hostPrimaryCount++;
      else if (!pAct && sAct) devicePrimaryCount++;
      else neitherCount++;
      // Log individual problematic cases: host-primary but not device-actual
      if (pAct && !sAct) {
        DSP_DIAG(VERIFY, "ACTUALITY %s: slot %d input[%d] srcIdx=%d HOST_PRIMARY (device stale!) "
                 "len=%lld dt=%d",
                 label, si, i, srcIdx,
                 (long long)arr->lengthOf(), static_cast<int>(arr->dataType()));
      }
    }
    // Check outputs
    for (int o = 0; o < slots[si].wiring.numOutputs; o++) {
      int outIdx = slots[si].wiring.outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
      NDArray* arr = outputSlots[outIdx];
      if (!arr || !arr->dataBuffer()) continue;
      auto* db = arr->dataBuffer();
      if (!seen.insert(db).second) continue;
      bool pAct = db->isPrimaryActual();
      bool sAct = db->isSpecialActual();
      if (pAct && sAct) bothActualCount++;
      else if (pAct && !sAct) hostPrimaryCount++;
      else if (!pAct && sAct) devicePrimaryCount++;
      else neitherCount++;
    }
  }
  DSP_DIAG(VERIFY, "ACTUALITY %s [%d-%d]: %d buffers — %d host-primary(STALE) %d device-primary "
           "%d both %d neither",
           label, startSlot, endSlot,
           hostPrimaryCount + devicePrimaryCount + bothActualCount + neitherCount,
           hostPrimaryCount, devicePrimaryCount, bothActualCount, neitherCount);
}

// Detect buffer aliasing within and across a sub-kernel's arg mappings
// Takes individual fields because CompiledKernel is private to TritonGraphBackend.
static void detectBufferAliasing(int ki,
                                 const std::vector<TritonKernelArg>& argSlotMapping,
                                 int skStartSlot, int skEndSlot,
                                 NDArray** externalInputs, int numExternalInputs,
                                 NDArray** outputSlots, int totalOutputSlots,
                                 NativeSlot* slots, int segEndSlot) {
  if (!DSP_DIAG_ENABLED(VERIFY)) return;

  struct AddrInfo { int argIdx; int slotIndex; bool isOutput; void* addr; };
  std::vector<AddrInfo> addrInfos;

  for (int ai = 0; ai < static_cast<int>(argSlotMapping.size()); ai++) {
    auto& argMap = argSlotMapping[ai];
    NDArray* arr = nullptr;
    if (argMap.slotIndex < 0) {
      int extIdx = -(argMap.slotIndex + 1);
      if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
    } else {
      if (argMap.slotIndex < totalOutputSlots) arr = outputSlots[argMap.slotIndex];
    }
    if (arr && arr->specialBuffer()) {
      AddrInfo info;
      info.argIdx = ai;
      info.slotIndex = argMap.slotIndex;
      info.isOutput = argMap.isOutput;
      info.addr = arr->specialBuffer();
      addrInfos.push_back(info);
    }
  }

  // Intra-kernel: check if any two args share the same address but different slots
  for (size_t a = 0; a < addrInfos.size(); a++) {
    for (size_t b = a + 1; b < addrInfos.size(); b++) {
      if (addrInfos[a].addr == addrInfos[b].addr &&
          addrInfos[a].slotIndex != addrInfos[b].slotIndex) {
        bool outputInvolved = addrInfos[a].isOutput || addrInfos[b].isOutput;
        if (outputInvolved) {
          DSP_DIAG(VERIFY, "BUFFER ALIAS: subK[%d] [%d-%d] arg[%d](slot=%d,%s) and "
                   "arg[%d](slot=%d,%s) share GPU addr %p",
                   ki, skStartSlot, skEndSlot,
                   addrInfos[a].argIdx, addrInfos[a].slotIndex,
                   addrInfos[a].isOutput ? "OUT" : "in",
                   addrInfos[b].argIdx, addrInfos[b].slotIndex,
                   addrInfos[b].isOutput ? "OUT" : "in",
                   addrInfos[a].addr);
        }
      }
    }
  }

  // Cross-slot: check if kernel output overlaps any slot OUTSIDE the kernel range
  struct OutputRange {
    uintptr_t start, end;
    int slotIndex, argIdx;
  };
  std::vector<OutputRange> outputRanges;
  for (auto& info : addrInfos) {
    if (info.isOutput && info.addr) {
      NDArray* outArr = (info.slotIndex >= 0 && info.slotIndex < totalOutputSlots)
                            ? outputSlots[info.slotIndex] : nullptr;
      size_t bytes = outArr ? (outArr->lengthOf() * outArr->sizeOfT()) : 0;
      if (bytes > 0) {
        OutputRange range;
        range.start = reinterpret_cast<uintptr_t>(info.addr);
        range.end = range.start + bytes;
        range.slotIndex = info.slotIndex;
        range.argIdx = info.argIdx;
        outputRanges.push_back(range);
      }
    }
  }

  if (!outputRanges.empty()) {
    int aliasCount = 0;
    for (int si = 0; si < totalOutputSlots; si++) {
      if (si >= skStartSlot && si <= skEndSlot) continue;
      if (!outputSlots[si] || !outputSlots[si]->specialBuffer()) continue;
      uintptr_t extStart = reinterpret_cast<uintptr_t>(outputSlots[si]->specialBuffer());
      size_t extBytes = outputSlots[si]->lengthOf() * outputSlots[si]->sizeOfT();
      if (extBytes == 0) continue;
      uintptr_t extEnd = extStart + extBytes;
      for (auto& outRange : outputRanges) {
        if (outRange.start < extEnd && extStart < outRange.end) {
          aliasCount++;
          if (aliasCount <= 20) {
            const char* extSlotName = (si <= segEndSlot)
                ? slots[si].ident.opName.c_str() : "?";
            DSP_DIAG(VERIFY, "BUFFER OVERLAP: subK[%d] [%d-%d] output arg[%d](slot=%d) "
                     "[%p-%p] overlaps slot %d (%s) [%p-%p]",
                     ki, skStartSlot, skEndSlot,
                     outRange.argIdx, outRange.slotIndex,
                     reinterpret_cast<void*>(outRange.start),
                     reinterpret_cast<void*>(outRange.end),
                     si, extSlotName,
                     reinterpret_cast<void*>(extStart),
                     reinterpret_cast<void*>(extEnd));
          }
        }
      }
    }
    if (aliasCount > 0) {
      DSP_DIAG(VERIFY, "BUFFER OVERLAP SUMMARY: subK[%d] [%d-%d] %d overlapping external slots",
               ki, skStartSlot, skEndSlot, aliasCount);
    }

    // Also check DataBuffer-level aliasing: different slot arrays sharing same DataBuffer
    std::unordered_map<DataBuffer*, int> outputDbSlots;
    for (auto& info : addrInfos) {
      if (info.isOutput && info.slotIndex >= 0 && info.slotIndex < totalOutputSlots) {
        NDArray* arr = outputSlots[info.slotIndex];
        if (arr && arr->dataBuffer()) {
          outputDbSlots[arr->dataBuffer()] = info.slotIndex;
        }
      }
    }
    for (int si = 0; si < totalOutputSlots; si++) {
      if (si >= skStartSlot && si <= skEndSlot) continue;
      if (!outputSlots[si] || !outputSlots[si]->dataBuffer()) continue;
      auto it = outputDbSlots.find(outputSlots[si]->dataBuffer());
      if (it != outputDbSlots.end()) {
        DSP_DIAG(VERIFY, "DATABUFFER ALIAS: subK[%d] [%d-%d] output slot %d and "
                 "external slot %d (%s) share same DataBuffer %p",
                 ki, skStartSlot, skEndSlot,
                 it->second, si,
                 (si <= segEndSlot) ? slots[si].ident.opName.c_str() : "?",
                 static_cast<void*>(outputSlots[si]->dataBuffer()));
      }
    }
  }
}

// ─── executeSegment ─────────────────────────────────────────────────────────

Status TritonGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                          NDArray** externalInputs, int numExternalInputs,
                                          NDArray** outputSlots, int totalOutputSlots,
                                          void* stream) {
  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;
  int execDevice = -1;
  int targetDevice = -1;
  if (seg.def.startSlot >= 0) {
    targetDevice = slots[seg.def.startSlot].targetDeviceId;
  }
  bool streamCaptureActive = false;

  cudaError_t execDeviceErr = cudaGetDevice(&execDevice);
  if (execDeviceErr != cudaSuccess) {
    DSP_DIAG(BACKEND, "TritonGraphBackend::executeSegment: cudaGetDevice failed for segment [%d-%d]: %s",
              seg.def.startSlot, seg.def.endSlot, cudaGetErrorString(execDeviceErr));
    cudaGetLastError();
    return Status::KERNEL_FAILURE;
  }
  if (actualStream != nullptr) {
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    auto capErr = cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(actualStream), &captureStatus);
    if (capErr == cudaSuccess && captureStatus != cudaStreamCaptureStatusNone) {
      streamCaptureActive = true;
    }
  }

  auto& execEnv = Environment::getInstance();
  SegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, seg.def.shapeKeyState.compiledShapeKey, execDevice,
                      execEnv.tritonCompileAll(),
                      std::hash<std::string>()(execEnv.tritonExcludeOps()),
                      std::hash<std::string>()(execEnv.tritonIncludeTypes()),
                      execEnv.tritonGraphCapture()};

  CompiledSegment* compiledSeg = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      int cachedDeviceId = -999;
      for (const auto& entry : cache_) {
        if (entry.first.startSlot == seg.def.startSlot &&
            entry.first.endSlot == seg.def.endSlot &&
            entry.first.shapeKey == seg.def.shapeKeyState.compiledShapeKey) {
          cachedDeviceId = entry.first.deviceId;
          break;
        }
      }
      if (cachedDeviceId != -999) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: kernel cache miss for segment [%d-%d] "
                 "(shapeKey=%lld, activeDevice=%d, targetDeviceId=%d). "
                 "Found compiled kernel for deviceId=%d but cross-device module reuse is disallowed.",
                 seg.def.startSlot, seg.def.endSlot, seg.def.shapeKeyState.compiledShapeKey, execDevice, targetDevice, cachedDeviceId);
      } else {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: no compiled kernel for segment [%d-%d] "
                 "(shapeKey=%lld, deviceId=%d, compileAll=%d, excludeHash=%zu). "
                 "Cache has %zu entries:",
                 seg.def.startSlot, seg.def.endSlot, seg.def.shapeKeyState.compiledShapeKey, execDevice,
                 key.compileAll ? 1 : 0, key.excludeOpsHash, cache_.size());
        for (const auto& entry : cache_) {
          DSP_DIAG(EXECUTE, "  cached: [%d-%d] shapeKey=%lld dev=%d compileAll=%d excHash=%zu",
                   entry.first.startSlot, entry.first.endSlot, entry.first.shapeKey,
                   entry.first.deviceId, entry.first.compileAll ? 1 : 0, entry.first.excludeOpsHash);
        }
      }
      return Status::KERNEL_FAILURE;
    }
	    compiledSeg = &it->second;
	  }

#ifdef SD_CUDA
  if (compiledSeg->preallocReadyEvent != nullptr && !streamCaptureActive) {
    auto waitErr = cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(actualStream),
                                       reinterpret_cast<cudaEvent_t>(compiledSeg->preallocReadyEvent), 0);
    if (waitErr != cudaSuccess) {
      DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: pre-allocation event wait failed for [%d-%d]: %s",
                seg.def.startSlot, seg.def.endSlot, cudaGetErrorString(waitErr));
      cudaGetLastError();
      return Status::KERNEL_FAILURE;
    }
  } else if (compiledSeg->preallocReadyEvent != nullptr) {
    DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: pre-allocation event already ordered before capture for [%d-%d]",
              seg.def.startSlot, seg.def.endSlot);
  }
#endif

	  if (streamCaptureActive && !compiledSeg->orderedRanges.empty()) {
    DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: capturing ordered native ranges during CUDA graph capture for [%d-%d] (%d ranges)",
             seg.def.startSlot, seg.def.endSlot, static_cast<int>(compiledSeg->orderedRanges.size()));
  }

  bool tritonSkipKernels = Environment::getInstance().tritonSkipKernels();
  bool tritonVerifyKernels = Environment::getInstance().tritonVerifyKernels();
  int tritonMaxSubKernelIndex = Environment::getInstance().tritonMaxSubKernelIndex();
  bool tritonVerifyFullSnapshot = Environment::getInstance().tritonVerifyFullSnapshot();
  if (tritonVerifyKernels || tritonVerifyFullSnapshot) {
    // The env flags were explicitly set, but host-snapshot verify is unsupported on the async-only
    // DSP path. Surface the ignored request once (not per-segment) rather than silently, then disable.
    static bool verifyUnsupportedWarned = false;
    if (!verifyUnsupportedWarned) {
      verifyUnsupportedWarned = true;
      sd_printf("TRITON VERIFY: ND4J_TRITON_VERIFY_KERNELS / _FULL_SNAPSHOT are not supported on the "
                "async-only DSP execution path; verify is disabled for this run.\n");
    }
    tritonVerifyKernels = false;
    tritonVerifyFullSnapshot = false;
  }

  DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "TritonGraphBackend::executeSegment: segment [%d-%d] launching %d sub-kernels "
               "(orderedRanges=%d, targetDeviceId=%d, activeDevice=%d, skipKernels=%d, verifyKernels=%d, "
               "maxSubKernelIdx=%d, fullSnapshot=%d, execCount=%d)",
               seg.def.startSlot, seg.def.endSlot,
               static_cast<int>(compiledSeg->subKernels.size()),
               static_cast<int>(compiledSeg->orderedRanges.size()),
               targetDevice, execDevice,
               tritonSkipKernels ? 1 : 0, tritonVerifyKernels ? 1 : 0,
               tritonMaxSubKernelIndex, tritonVerifyFullSnapshot ? 1 : 0,
               seg.exec.executionCount);

  // Consolidated arg table: do ONE H2D memcpy for all sub-kernels' arg tables.
  // IMPORTANT: Must pre-allocate output arrays and sync inputs BEFORE populating
  // the arg table, because allocateSpecial() in syncToDevice/syncToSpecial may
  // change specialBuffer() pointers.  Without this prepare pass, the consolidated
  // H2D would bake stale/null pointers into the CUDA graph.
  bool useConsolidated = compiledSeg->useConsolidatedArgTable;
  bool consolidatedArgsCopied = false;
  if (useConsolidated && compiledSeg->consolidatedArgTableHostPinned &&
      compiledSeg->consolidatedArgTableDevice &&
      compiledSeg->consolidatedArgTableBytes > 0) {

    // ── Phase 1: Prepare pass — pre-allocate outputs + sync inputs ──
    // This ensures all arrays have valid specialBuffer() pointers before
    // we populate the consolidated arg table.
    //  During CUDA graph capture, skip pre-allocation and input syncing.
    // The pre-capture warmup execution already allocated all outputs and synced
    // all inputs. Allocating during capture creates MemAlloc graph nodes with
    // addresses that become stale on replay. Input syncing is also unnecessary
    // since the pre-capture path already synced everything.
    if (!streamCaptureActive) {
      for (size_t ki = 0; ki < compiledSeg->subKernels.size(); ki++) {
        auto& sk = compiledSeg->subKernels[ki];
        if (!sk.useIndirectArgs) continue;

        // Pre-allocate output arrays for slots that don't have arrays yet
        for (auto& argMapping : sk.argSlotMapping) {
          if (!argMapping.isOutput) continue;
          if (argMapping.slotIndex < 0 || argMapping.slotIndex >= totalOutputSlots) continue;
          if (outputSlots[argMapping.slotIndex] != nullptr) continue;
          if (argMapping.shape.empty()) continue;
          std::vector<LongType> shapeVec(argMapping.shape.begin(), argMapping.shape.end());
          auto* newArr = new NDArray('c', shapeVec, argMapping.dtype, LaunchContext::defaultContext());
          outputSlots[argMapping.slotIndex] = newArr;
          if (seg.slotArrayCache) {
            seg.slotArrayCache[argMapping.slotIndex] = newArr;
          }
          void* rawSpecial = newArr->dataBuffer() != nullptr ? newArr->dataBuffer()->special() : nullptr;
          DSP_DIAG(MEMORY, "CONSOL PRE-ALLOC: subK[%zu] slot %d shape=[%s] dtype=%d specialBuf=%p",
                   ki, argMapping.slotIndex,
                   [&]() -> std::string {
                     std::string s;
                     for (size_t d = 0; d < shapeVec.size(); d++) {
                       if (d > 0) s += ",";
                       s += std::to_string(shapeVec[d]);
                     }
                     return s;
                   }().c_str(),
                   static_cast<int>(argMapping.dtype), rawSpecial);
        }

        std::vector<NDArray*> prepareReads;
        std::vector<NDArray*> prepareWrites;
        for (auto& argMapping : sk.argSlotMapping) {
          NDArray* arr = nullptr;
          bool isExternal = (argMapping.slotIndex < 0);
          if (isExternal) {
            int extIdx = -(argMapping.slotIndex + 1);
            if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
          } else {
            if (argMapping.slotIndex < totalOutputSlots) arr = outputSlots[argMapping.slotIndex];
          }
          if (arr && arr->lengthOf() > 0) {
            if (argMapping.isOutput) {
              prepareWrites.push_back(arr);
            } else {
              prepareReads.push_back(arr);
            }
          }
        }
        if (!prepareReads.empty() || !prepareWrites.empty()) {
          NDArray::prepareSpecialUse(prepareWrites, prepareReads);
        }
      }
    }

    // Same-stream ordering is sufficient here: input syncs and allocation
    // work were enqueued before the arg-table H2D, so no host stream drain is
    // needed before reading the stable device pointer values.

    // ── Phase 2: Populate consolidated arg table with post-sync pointers ──
    for (size_t ki = 0; ki < compiledSeg->subKernels.size(); ki++) {
      auto& sk = compiledSeg->subKernels[ki];
      if (!sk.useIndirectArgs || !sk.cachedArgTableHostPinned) continue;
      auto* hostPinned = static_cast<int64_t*>(sk.cachedArgTableHostPinned);
      int numArgs = static_cast<int>(sk.argSlotMapping.size());

      for (int ai = 0; ai < numArgs; ai++) {
        auto& argMap = sk.argSlotMapping[ai];
        NDArray* arr = nullptr;
        if (argMap.slotIndex < 0) {
          int extIdx = -(argMap.slotIndex + 1);
          if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
        } else {
          if (argMap.slotIndex < totalOutputSlots) arr = outputSlots[argMap.slotIndex];
        }
        if (arr) {
          void* sbuf = arr->specialBuffer();
          if (sbuf) hostPinned[ai] = reinterpret_cast<int64_t>(sbuf);
        }
      }

      // Log all arg addresses for ALL sub-kernels when VERIFY is enabled.
      // SKIP during stream capture: a->specialBuffer() can call syncToDevice()
      // if bufferDeviceId != currentDeviceId, issuing a cudaMemcpyAsync on a
      // non-captured stream that poisons the capture ("previous error during
      // capture") — later arg-table H2D and endCapture both fail.
      if (DSP_DIAG_ENABLED(VERIFY) && !streamCaptureActive) {
        for (int ai = 0; ai < numArgs; ai++) {
          auto& am = sk.argSlotMapping[ai];
          NDArray* a = nullptr;
          if (am.slotIndex < 0) {
            int ei = -(am.slotIndex + 1);
            if (ei < numExternalInputs) a = externalInputs[ei];
          } else {
            if (am.slotIndex < totalOutputSlots) a = outputSlots[am.slotIndex];
          }
          DSP_DIAG(VERIFY, "CONSOL ARG TABLE: subK[%d] [%d-%d] arg[%d] slot=%d %s gpuAddr=%p "
                   "len=%lld bytes=%zu dbAddr=%p dbBytes=%lld pAct=%d sAct=%d",
                   (int)ki, sk.startSlot_, sk.endSlot_, ai, am.slotIndex,
                   am.isOutput ? "OUT" : "in",
                   a ? a->specialBuffer() : nullptr,
                   a ? (long long)a->lengthOf() : 0LL,
                   a ? (size_t)(a->lengthOf() * a->sizeOfT()) : 0,
                   a ? static_cast<void*>(a->dataBuffer()) : nullptr,
                   a && a->dataBuffer() ? (long long)a->dataBuffer()->getLenInBytes() : 0LL,
                   a && a->dataBuffer() ? (a->dataBuffer()->isPrimaryActual() ? 1 : 0) : -1,
                   a && a->dataBuffer() ? (a->dataBuffer()->isSpecialActual() ? 1 : 0) : -1);
        }
      }

      // Buffer aliasing detection for this sub-kernel.
      // SKIP during capture: calls a->specialBuffer() → syncToDevice() poisons capture.
      if (!streamCaptureActive) {
        detectBufferAliasing(static_cast<int>(ki),
                             sk.argSlotMapping, sk.startSlot_, sk.endSlot_,
                             externalInputs, numExternalInputs,
                             outputSlots, totalOutputSlots,
                             slots, seg.def.endSlot);
      }
    }

    // ── Phase 3: Single consolidated H2D ──
    // SKIP during stream capture: each island's CUDA graph must NOT bake in a
    // full-segment consolidated H2D. At replay, gap ops between islands may
    // change output buffer addresses. If the graph re-executes the consolidated
    // H2D, it overwrites the device arg table with stale pre-loop addresses,
    // causing island B+ to read wrong data. Per-kernel H2D (line ~707) is safe
    // during capture — it uses per-kernel pinned/device buffers that are
    // interior pointers into the consolidated table, updated by
    // refreshArgTablesForReplay() + copyConsolidatedArgTableToDevice() at replay.
    if (!streamCaptureActive) {
      auto memcpyErr = cudaMemcpyAsync(
          compiledSeg->consolidatedArgTableDevice,
          compiledSeg->consolidatedArgTableHostPinned,
          compiledSeg->consolidatedArgTableBytes,
          cudaMemcpyHostToDevice,
          reinterpret_cast<cudaStream_t>(actualStream));
      if (memcpyErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: consolidated arg table H2D failed (%zu bytes): %s",
                  compiledSeg->consolidatedArgTableBytes, cudaGetErrorString(memcpyErr));
        cudaGetLastError();
      } else {
        consolidatedArgsCopied = true;
        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "TritonGraphBackend: consolidated arg table H2D: 1 copy of %zu bytes "
                     "(replaces %d per-kernel copies) for seg[%d-%d]",
                     compiledSeg->consolidatedArgTableBytes,
                     static_cast<int>(compiledSeg->subKernels.size()),
                     seg.def.startSlot, seg.def.endSlot);
      }
    } else {
      // During capture: per-kernel H2D via executeSingleKernel (argTablePreCopied=false)
      DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                   "TritonGraphBackend: SKIP consolidated H2D during capture — per-kernel H2D will be used");
    }
  }

  // Island slot range filter: when set, only launch sub-kernels within
  // [tl_islandSlotMin, tl_islandSlotMax]. Used by composite capture to capture
  // one Triton island at a time without capturing other islands in the segment.
  // tl_islandSlotMin > tl_islandSlotMax means no filter (normal execution).
  bool islandFilterActive = (tl_islandSlotMin <= tl_islandSlotMax);

  if (streamCaptureActive) {
    DSP_DIAG(EXECUTE, "CAPTURE_EXEC_ENTRY: seg[%d-%d] islandFilter=%d filterRange=[%d-%d] "
             "subKernels=%d consolidatedArgsCopied=%d",
             seg.def.startSlot, seg.def.endSlot,
             islandFilterActive ? 1 : 0, tl_islandSlotMin, tl_islandSlotMax,
             static_cast<int>(compiledSeg->subKernels.size()),
             consolidatedArgsCopied ? 1 : 0);
  }

  int captureFilteredCount = 0, captureLaunchedCount = 0, captureSkippedCount = 0;
  int nextSlotToRun = seg.def.startSlot;

  for (int i = 0; i < (int)compiledSeg->subKernels.size(); i++) {
    auto& subKernel = compiledSeg->subKernels[i];

    // Island capture filter: skip sub-kernels outside the requested island range.
    // During normal execution (no filter), islandFilterActive is false and all
    // sub-kernels execute. During per-island capture, only the target island's
    // sub-kernels are captured; others are skipped to avoid capturing incorrect
    // kernels into the island-specific graph.
    if (islandFilterActive &&
        (subKernel.endSlot_ < tl_islandSlotMin ||
         subKernel.startSlot_ > tl_islandSlotMax)) {
      DSP_DIAG(EXECUTE, "ISLAND_FILTER_SKIP: sub-kernel [%d-%d] outside island [%d-%d] — skipped",
               subKernel.startSlot_, subKernel.endSlot_, tl_islandSlotMin, tl_islandSlotMax);
      captureFilteredCount++;
      nextSlotToRun = subKernel.endSlot_ + 1;
      continue;
    }

    // When island filter is active, any filtered sub-kernels that precede the island
    // set nextSlotToRun past their end slot (into the pre-island gap). Those gap slots
    // were already executed by the outer composite loop BEFORE capture was started,
    // so re-executing them here — while tl_graphExecutionActive=true — would run
    // non-capture-safe ops on the active capture stream and invalidate it (err=901).
    // Clamp nextSlotToRun to tl_islandSlotMin so the gap-check below only fires for
    // gaps that are genuinely INSIDE the island, not for pre-island gaps.
    if (islandFilterActive && nextSlotToRun < tl_islandSlotMin) {
      DSP_DIAG(EXECUTE, "ISLAND_GAP_CLAMP: nextSlotToRun %d clamped to tl_islandSlotMin %d (pre-island gap already executed by outer loop)",
               nextSlotToRun, tl_islandSlotMin);
      nextSlotToRun = tl_islandSlotMin;
    }

    if (nextSlotToRun < subKernel.startSlot_) {
      // Log actuality state BEFORE gap execution
      logActualityState("PRE_GAP", nextSlotToRun, subKernel.startSlot_ - 1, slots,
                        outputSlots, totalOutputSlots, externalInputs, numExternalInputs);
      if (!orderedRangeExecutor_) {
        DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: missing ordered range executor for gap [%d-%d]",
                  nextSlotToRun, subKernel.startSlot_ - 1);
        return Status::KERNEL_FAILURE;
      }
      auto gapStatus = orderedRangeExecutor_(nextSlotToRun, subKernel.startSlot_ - 1);
      if (gapStatus != Status::OK) {
        DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: ordered native range [%d-%d] failed with status=%d",
                  nextSlotToRun, subKernel.startSlot_ - 1, static_cast<int>(gapStatus));
        return gapStatus;
      }
      // markOrderedRangeDeviceCurrent DISABLED: orderedRangeExecutor_ already handles
      // actuality via prepareSpecialUse/registerSpecialUse inside executeSegmentSlotBySlot.
      // readSpecial()/writeSpecial() here is redundant and poisons frozen constant flags.
      DSP_DIAG(EXECUTE, "markOrderedRangeDeviceCurrent SKIPPED (orderedRangeExecutor handled actuality) [%d-%d]",
               nextSlotToRun, subKernel.startSlot_ - 1);

      // ── Post-gap consolidated arg table re-copy ──
      // Gap ops (reshape, permute, view ops) update outputSlots[] with new NDArray
      // wrappers whose specialBuffer() may differ from the addresses baked into the
      // consolidated arg table during Phase 2 (before any sub-kernels launched).
      // Without this re-copy, subsequent Triton sub-kernels read from the stale GPU
      // arg table and use capture-time buffer addresses instead of the fresh ones
      // installed by the gap op. This causes the first post-capture direct execution
      // to produce identical output to the capture — the "stale RESHAPE_MATMUL" bug.
      if (consolidatedArgsCopied && useConsolidated &&
          compiledSeg->consolidatedArgTableHostPinned &&
          compiledSeg->consolidatedArgTableDevice &&
          compiledSeg->consolidatedArgTableBytes > 0 &&
          !streamCaptureActive) {
        // Re-populate host-pinned arg table entries for ALL subsequent sub-kernels
        // (gap may have changed slot pointers that are inputs to any of them).
        for (size_t rki = i; rki < compiledSeg->subKernels.size(); rki++) {
          auto& rsk = compiledSeg->subKernels[rki];
          if (!rsk.useIndirectArgs || !rsk.cachedArgTableHostPinned) continue;
          auto* hostPinned = static_cast<int64_t*>(rsk.cachedArgTableHostPinned);
          int numArgs = static_cast<int>(rsk.argSlotMapping.size());
          for (int ai = 0; ai < numArgs; ai++) {
            auto& argMap = rsk.argSlotMapping[ai];
            NDArray* arr = nullptr;
            if (argMap.slotIndex < 0) {
              int extIdx = -(argMap.slotIndex + 1);
              if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
            } else {
              if (argMap.slotIndex < totalOutputSlots) arr = outputSlots[argMap.slotIndex];
            }
            if (arr) {
              void* sbuf = arr->specialBuffer();
              if (sbuf) hostPinned[ai] = reinterpret_cast<int64_t>(sbuf);
            }
          }
        }
        // Re-copy entire consolidated arg table to device
        auto reErr = cudaMemcpyAsync(
            compiledSeg->consolidatedArgTableDevice,
            compiledSeg->consolidatedArgTableHostPinned,
            compiledSeg->consolidatedArgTableBytes,
            cudaMemcpyHostToDevice,
            reinterpret_cast<cudaStream_t>(actualStream));
        if (reErr != cudaSuccess) {
          DSP_DIAG(MEMORY, "TritonGraphBackend: post-gap arg table re-copy FAILED: %s",
                    cudaGetErrorString(reErr));
          cudaGetLastError();
        } else {
          DSP_DIAG(EXECUTE, "POST_GAP_ARG_TABLE_RECOPY: re-copied %zu bytes after gap [%d-%d] for seg[%d-%d]",
                   compiledSeg->consolidatedArgTableBytes,
                   nextSlotToRun, subKernel.startSlot_ - 1,
                   seg.def.startSlot, seg.def.endSlot);
        }
      }

      // ── Post-gap shape re-validation ──
      // After a gap executes view-producing ops (reshape_no_copy, permute, etc.),
      // downstream Triton sub-kernel outputs may have been pre-allocated with the
      // wrong shape. The pre-exec code in gpubackend.cpp allocates outputs using
      // the input-source fallback shape, but at pre-exec time the gap hasn't run
      // yet, so the input source's shape is the un-reshaped original shape.
      // Now that the gap has executed and outputSlots_ for gap slots have correct
      // shapes, re-validate each output in the upcoming sub-kernel. If the shape
      // doesn't match what shape inference produces from the actual inputs,
      // re-allocate the output with the correct shape.
      //
      // IMPORTANT: For element-count-preserving reshapes (the common case), the
      // Triton kernel writes correct values to the buffer regardless of shape
      // metadata. We create a new NDArray that wraps the SAME DataBuffer with
      // the correct shape, so the consolidated arg table's device pointer remains
      // valid. Only when element counts differ do we allocate a fresh buffer.
      //
      // SKIP during CUDA graph capture: gaps are not executed during capture (they
      // return OK immediately), so there is nothing to re-validate. Allocating new
      // NDArrays during capture would inject cudaMalloc nodes into the graph and
      // create buffer address mismatches on replay — causing SIGSEGV.
      if (streamCaptureActive) {
        DSP_DIAG(EXECUTE, "POST_GAP_RESHAPE SKIPPED during capture [gap %d-%d]",
                 nextSlotToRun, subKernel.startSlot_ - 1);
      } else {
      for (int si = subKernel.startSlot_; si <= subKernel.endSlot_; si++) {
        auto& slot = slots[si];
        // Check if any input comes from the gap range
        bool hasInputFromGap = false;
        for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
          int srcIdx = slot.wiring.inputSourceIndices[inp];
          if (srcIdx >= 0 && srcIdx >= nextSlotToRun && srcIdx < subKernel.startSlot_) {
            hasInputFromGap = true;
            break;
          }
        }
        if (!hasInputFromGap) continue;

        // Resolve actual input arrays (post-gap, so shapes are correct)
        std::vector<NDArray*> inputArrays(slot.wiring.numInputs, nullptr);
        bool allInputsAvailable = true;
        for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
          int srcIdx = slot.wiring.inputSourceIndices[inp];
          inputArrays[inp] = resolveRangeArray(srcIdx, externalInputs, numExternalInputs,
                                               outputSlots, totalOutputSlots);
          if (inputArrays[inp] == nullptr) {
            allInputsAvailable = false;
            break;
          }
        }
        if (!allInputsAvailable) continue;

        // Build input shape list and run shape inference
        ShapeList inputShapes;
        for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
          inputShapes.push_back(inputArrays[inp]->shapeInfo());
        }

        Context inferCtx(1);
        for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
          inferCtx.setInputArray(inp, inputArrays[inp]);
        }
        if (slot.args.numIArgs > 0) inferCtx.setIArguments(slot.args.iArgs, slot.args.numIArgs);
        if (slot.args.numTArgs > 0) inferCtx.setTArguments(slot.args.tArgs, slot.args.numTArgs);
        if (slot.args.numBArgs > 0) inferCtx.setBArguments(slot.args.bArgs, slot.args.numBArgs);
        if (slot.args.numDArgs > 0) inferCtx.setDArguments(slot.args.dArgs, slot.args.numDArgs);

        // Legacy reduce/broadcast ops read reduction dims from block.getAxis(), not iArgs.
        // Mirror iArgs into axis so calculateOutputShape picks up the right dims.
        if (legacyOpReadsAxisFromIArgs(slot.legacy.legacyOpType)) {
          inferCtx.getAxis()->clear();
          for (int i = 0; i < slot.args.numIArgs; i++) {
            inferCtx.getAxis()->emplace_back(static_cast<sd::LongType>(slot.args.iArgs[i]));
          }
        }

        ShapeList* inferredShapes = nullptr;
        try {
          inferredShapes = slot.ident.op->calculateOutputShape(&inputShapes, inferCtx);
        } catch (...) {
          continue;
        }
        if (inferredShapes == nullptr || inferredShapes->size() == 0) {
          delete inferredShapes;
          continue;
        }

        // Check each output — fix shape if mismatched
        for (int o = 0; o < slot.wiring.numOutputs && o < static_cast<int>(inferredShapes->size()); o++) {
          int outSlotIdx = slot.wiring.outputSlotIndices[o];
          if (outSlotIdx < 0 || outSlotIdx >= totalOutputSlots) continue;

          const LongType* inferredShape = inferredShapes->at(o);
          NDArray* existingOut = outputSlots[outSlotIdx];
          if (existingOut == nullptr) continue;

          // Compare shapes — skip if already correct
          if (shape::equalsSoft(existingOut->shapeInfo(), inferredShape)) continue;

          auto dt = ArrayOptions::dataType(inferredShape);
          auto order = shape::order(inferredShape);
          LongType rank = shape::rank(inferredShape);
          std::vector<LongType> newShapeVec(rank);
          for (int d = 0; d < rank; d++) newShapeVec[d] = inferredShape[d + 1];

          // Compute element counts
          LongType existingLen = existingOut->lengthOf();
          LongType inferredLen = 1;
          for (int d = 0; d < rank; d++) inferredLen *= newShapeVec[d];

          DSP_DIAG(SHAPE, "POST_GAP_RESHAPE: slot %d (%s) output slot %d shape mismatch: "
                   "existing=%s inferred=%s existingLen=%lld inferredLen=%lld",
                   si, slot.ident.opName.c_str(), outSlotIdx,
                   ShapeUtils::shapeAsString(existingOut).c_str(),
                   ShapeUtils::shapeAsString(inferredShape).c_str(),
                   (long long)existingLen, (long long)inferredLen);

          if (inferredLen == existingLen && existingOut->dataBuffer() != nullptr) {
            // Same element count: create a view of the same buffer with correct shape.
            // This preserves the specialBuffer pointer so consolidated arg tables
            // and CUDA graph captured pointers remain valid.
            auto* reshapedArr = new NDArray(existingOut->dataBuffer(), order, newShapeVec);
            reshapedArr->tickWriteDevice();  // Mark device as current
            outputSlots[outSlotIdx] = reshapedArr;
            if (seg.slotArrayCache) {
              seg.slotArrayCache[outSlotIdx] = reshapedArr;
            }
          } else {
            // Different element count: must allocate a new buffer
            auto* newArr = new NDArray(order, newShapeVec, dt, LaunchContext::defaultContext());
            outputSlots[outSlotIdx] = newArr;
            if (seg.slotArrayCache) {
              seg.slotArrayCache[outSlotIdx] = newArr;
            }
          }
        }
        delete inferredShapes;
      }
      }  // end else (!streamCaptureActive) — POST_GAP_RESHAPE
      // Hash gap outputs
      if (!streamCaptureActive) {
        logSlotHashes("GAP", nextSlotToRun, subKernel.startSlot_ - 1, slots,
                      outputSlots, totalOutputSlots,
                      reinterpret_cast<cudaStream_t>(actualStream), seg.exec.executionCount);
      }

    }

    // Decide whether to skip this sub-kernel: global skip OR per-index cutoff
    bool skipThisKernel = tritonSkipKernels ||
        (tritonMaxSubKernelIndex >= 0 && i > tritonMaxSubKernelIndex);

    if (skipThisKernel) {
      if (streamCaptureActive) captureSkippedCount++;
      if (orderedRangeExecutor_) {
        if (!tritonSkipKernels && tritonMaxSubKernelIndex >= 0) {
          DSP_DIAG(EXECUTE, "TritonGraphBackend: subK[%d] [%d-%d] SKIPPED (index > maxSubKernelIndex=%d)",
                   i, subKernel.startSlot_, subKernel.endSlot_, tritonMaxSubKernelIndex);
        }
        auto skipStatus = orderedRangeExecutor_(subKernel.startSlot_, subKernel.endSlot_);
        if (skipStatus != Status::OK) {
          DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: native ordered range for skipped kernel [%d-%d] failed with status=%d",
                 subKernel.startSlot_, subKernel.endSlot_, static_cast<int>(skipStatus));
          return skipStatus;
        }
        // markOrderedRangeDeviceCurrent DISABLED: orderedRangeExecutor_ already handles
        // actuality via prepareSpecialUse/registerSpecialUse inside executeSegmentSlotBySlot.
        // readSpecial()/writeSpecial() here is redundant and poisons frozen constant flags.
        DSP_DIAG(EXECUTE, "markOrderedRangeDeviceCurrent SKIPPED (orderedRangeExecutor handled actuality) [%d-%d]",
                 subKernel.startSlot_, subKernel.endSlot_);
        if (!streamCaptureActive) {
          logSlotHashes("SKIP", subKernel.startSlot_, subKernel.endSlot_, slots,
                        outputSlots, totalOutputSlots,
                        reinterpret_cast<cudaStream_t>(actualStream), seg.exec.executionCount);
        }
      }
    } else {
      // Log sub-kernel entry with op details
      {
        std::string opSummary;
        for (int si = subKernel.startSlot_; si <= subKernel.endSlot_ && opSummary.size() < 400; si++) {
          if (!opSummary.empty()) opSummary += ", ";
          opSummary += std::to_string(si) + ":" + slots[si].ident.opName;
        }
        DSP_DIAG(EXECUTE, "SUBKERNEL ENTRY: subK[%d] [%d-%d] numArgs=%d indirect=%d cooperative=%d "
                 "dynamicGrid=%d multiPhase=%d ops=[%s]",
                 i, subKernel.startSlot_, subKernel.endSlot_,
                 static_cast<int>(subKernel.argSlotMapping.size()),
                 subKernel.useIndirectArgs ? 1 : 0,
                 subKernel.useCooperativeLaunch ? 1 : 0,
                 subKernel.useDynamicGrid ? 1 : 0,
                 subKernel.useMultiPhaseLaunch ? 1 : 0,
                 opSummary.c_str());
      }

      if (i > 0 && !streamCaptureActive) {
        cudaError_t launchErr = cudaPeekAtLastError();
        if (launchErr != cudaSuccess) {
          DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: pre sub-kernel %d/%d [%d-%d] "
                    "observed CUDA launch error without blocking sync: %s",
                    i + 1, (int)compiledSeg->subKernels.size(),
                    subKernel.startSlot_, subKernel.endSlot_,
                    cudaGetErrorString(launchErr));
          cudaGetLastError();
        }
      }

      // Log actuality state BEFORE Triton kernel
      if (!streamCaptureActive) {
        logActualityState("PRE_TRITON", subKernel.startSlot_, subKernel.endSlot_, slots,
                          outputSlots, totalOutputSlots, externalInputs, numExternalInputs);
      }

      // ── Verify mode ──
      std::unordered_map<int, NDArray*> savedOutputs;
      std::unordered_map<int, std::vector<uint8_t>> fullSnapshotBefore;
      if (tritonVerifyKernels && !streamCaptureActive) {
        std::string opNames;
        for (int si = subKernel.startSlot_; si <= subKernel.endSlot_; si++) {
          if (!opNames.empty()) opNames += ", ";
          opNames += std::to_string(si) + ":" + slots[si].ident.opName;
        }
        DSP_DIAG(VERIFY, "TRITON VERIFY ENTRY: subK[%d] [%d-%d] execCount=%d ops: %s",
                 i, subKernel.startSlot_, subKernel.endSlot_, seg.exec.executionCount, opNames.c_str());

        // Full snapshot: save ALL slot GPU contents
        if (tritonVerifyFullSnapshot) {
          for (int si = 0; si < totalOutputSlots; si++) {
            if (outputSlots[si] && outputSlots[si]->dataBuffer() && outputSlots[si]->specialBuffer()) {
              size_t bytes = outputSlots[si]->lengthOf() * outputSlots[si]->sizeOfT();
              if (bytes > 0 && bytes <= 64 * 1024 * 1024) {
                fullSnapshotBefore[si].resize(bytes);
                cudaMemcpyAsync(fullSnapshotBefore[si].data(), outputSlots[si]->specialBuffer(),
                                bytes, cudaMemcpyDeviceToHost,
                                reinterpret_cast<cudaStream_t>(actualStream));
              }
            }
          }
          for (int ei = 0; ei < numExternalInputs; ei++) {
            if (externalInputs[ei] && externalInputs[ei]->dataBuffer() && externalInputs[ei]->specialBuffer()) {
              size_t bytes = externalInputs[ei]->lengthOf() * externalInputs[ei]->sizeOfT();
              if (bytes > 0 && bytes <= 64 * 1024 * 1024) {
                int snapKey = -(ei + 1);
                fullSnapshotBefore[snapKey].resize(bytes);
                cudaMemcpyAsync(fullSnapshotBefore[snapKey].data(), externalInputs[ei]->specialBuffer(),
                                bytes, cudaMemcpyDeviceToHost,
                                reinterpret_cast<cudaStream_t>(actualStream));
              }
            }
          }
          DSP_DIAG(VERIFY, "TRITON VERIFY FULL SNAPSHOT: saved %d slot snapshots before subK[%d]",
                   static_cast<int>(fullSnapshotBefore.size()), i);
        }

        // Save copies of output arrays for restore after Triton.
        // When fullSnapshot is enabled, we already have host-side copies in fullSnapshotBefore
        // so we DON'T need GPU dup() copies — avoids doubling GPU memory usage (OOM fix).
        if (!tritonVerifyFullSnapshot) {
          // Non-fullSnapshot: only save ACTUAL kernel output slots via dup().
          // Fused kernels only write buffers in argSlotMapping with isOutput=true;
          // intermediate slots are computed in registers and never stored.
          for (auto& argMap : subKernel.argSlotMapping) {
            if (!argMap.isOutput || argMap.slotIndex < 0) continue;
            int outIdx = argMap.slotIndex;
            if (outIdx < totalOutputSlots && outputSlots[outIdx] &&
                outputSlots[outIdx]->lengthOf() > 0 && !outputSlots[outIdx]->isEmpty()) {
              try {
                savedOutputs[outIdx] = new NDArray(outputSlots[outIdx]->dup());
              } catch (...) {
                DSP_DIAG(VERIFY, "TRITON VERIFY: dup() failed for slot %d — skipping", outIdx);
              }
            }
          }
          DSP_DIAG(VERIFY, "TRITON VERIFY: saved %d kernel output arrays via GPU dup()", static_cast<int>(savedOutputs.size()));
        } else {
          DSP_DIAG(VERIFY, "TRITON VERIFY: using host-side fullSnapshot for restore (%d snapshots, no GPU dup)",
                   static_cast<int>(fullSnapshotBefore.size()));
        }
      }
      // Buffer aliasing detection for non-consolidated path.
      // SKIP during capture: calls a->specialBuffer() → syncToDevice() poisons capture.
      if (!consolidatedArgsCopied && !streamCaptureActive) {
        detectBufferAliasing(i,
                             subKernel.argSlotMapping, subKernel.startSlot_, subKernel.endSlot_,
                             externalInputs, numExternalInputs,
                             outputSlots, totalOutputSlots,
                             slots, seg.def.endSlot);
      }

      auto status = executeSingleKernel(subKernel, slots,
                                         externalInputs, numExternalInputs,
                                         outputSlots, totalOutputSlots,
                                         stream,
                                         consolidatedArgsCopied,
                                         seg.slotArrayCache
                                         );
      if (status != Status::OK) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: sub-kernel %d/%d [%d-%d] FAILED status=%d",
                  i + 1, (int)compiledSeg->subKernels.size(),
                  subKernel.startSlot_, subKernel.endSlot_, static_cast<int>(status));
        for (auto& kv : savedOutputs) delete kv.second;
        return status;
      }

      if (streamCaptureActive) captureLaunchedCount++;
      DSP_DIAG(EXECUTE, "SUBKERNEL EXIT OK: subK[%d] [%d-%d] execCount=%d capturing=%d",
               i, subKernel.startSlot_, subKernel.endSlot_, seg.exec.executionCount,
               streamCaptureActive ? 1 : 0);

      // Hash Triton outputs
      if (!streamCaptureActive) {
        logSlotHashes("TRITON", subKernel.startSlot_, subKernel.endSlot_, slots,
                      outputSlots, totalOutputSlots,
                      reinterpret_cast<cudaStream_t>(actualStream), seg.exec.executionCount);
      }

      // ── Verify mode: run native and compare ──
      do {
      if (tritonVerifyKernels && !streamCaptureActive && orderedRangeExecutor_) {
        // Save Triton outputs (raw bytes).
        // IMPORTANT: Only save slots that are actual kernel outputs (in argSlotMapping
        // with isOutput=true). Fused kernels compute intermediate slots in registers
        // and only write the final output buffer. Comparing intermediate slot buffers
        // produces false positives because the Triton kernel never writes them.
        struct RawBuffer { std::vector<uint8_t> data; DataType dtype; LongType len; };
        std::unordered_map<int, RawBuffer> tritonRawOutputs;
        std::unordered_set<int> kernelOutputSlots;
        for (auto& argMap : subKernel.argSlotMapping) {
          if (argMap.isOutput && argMap.slotIndex >= 0) {
            kernelOutputSlots.insert(argMap.slotIndex);
          }
        }
        for (int outIdx : kernelOutputSlots) {
          if (outIdx < totalOutputSlots && outputSlots[outIdx]) {
            auto* arr = outputSlots[outIdx];
            void* sbuf = arr->specialBuffer();
            if (sbuf && arr->dataBuffer() && arr->lengthOf() > 0) {
              size_t byteLen = arr->lengthOf() * arr->sizeOfT();
              RawBuffer rb;
              rb.data.resize(byteLen);
              rb.dtype = arr->dataType();
              rb.len = arr->lengthOf();
              cudaMemcpyAsync(rb.data.data(), sbuf, byteLen, cudaMemcpyDeviceToHost,
                              reinterpret_cast<cudaStream_t>(actualStream));
              tritonRawOutputs[outIdx] = std::move(rb);
            }
          }
        }

        // Full snapshot corruption detection: find non-output slots modified by Triton
        if (tritonVerifyFullSnapshot && !fullSnapshotBefore.empty()) {
          int corruptedCount = 0;
          // Build set of expected output slots
          std::unordered_set<int> expectedOutputs;
          for (int si = subKernel.startSlot_; si <= subKernel.endSlot_; si++) {
            for (int o = 0; o < slots[si].wiring.numOutputs; o++) {
              int outIdx = slots[si].wiring.outputSlotIndices[o];
              if (outIdx >= 0 && outIdx < totalOutputSlots)
                expectedOutputs.insert(outIdx);
            }
          }

          // Log kernel output buffer addresses for overlap analysis
          for (int outIdx : expectedOutputs) {
            if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx] &&
                outputSlots[outIdx]->specialBuffer()) {
              auto* arr = outputSlots[outIdx];
              size_t bytes = arr->lengthOf() * arr->sizeOfT();
              DSP_DIAG(VERIFY, "KERNEL OUTPUT ADDR: subK[%d] slot=%d gpuAddr=%p len=%lld bytes=%zu "
                       "dbAddr=%p dbLen=%lld",
                       i, outIdx, arr->specialBuffer(), (long long)arr->lengthOf(), bytes,
                       static_cast<void*>(arr->dataBuffer()),
                       (long long)(arr->dataBuffer() ? arr->dataBuffer()->getLenInBytes() : 0));
            }
          }

          for (auto& snap : fullSnapshotBefore) {
            int slotIdx = snap.first;
            auto& beforeBytes = snap.second;
            void* currentBuf = nullptr;
            size_t currentLen = 0;

            if (slotIdx >= 0) {
              if (slotIdx < totalOutputSlots && outputSlots[slotIdx] &&
                  outputSlots[slotIdx]->specialBuffer()) {
                currentBuf = outputSlots[slotIdx]->specialBuffer();
                currentLen = outputSlots[slotIdx]->lengthOf() * outputSlots[slotIdx]->sizeOfT();
              }
            } else {
              int ei = -(slotIdx + 1);
              if (ei < numExternalInputs && externalInputs[ei] &&
                  externalInputs[ei]->specialBuffer()) {
                currentBuf = externalInputs[ei]->specialBuffer();
                currentLen = externalInputs[ei]->lengthOf() * externalInputs[ei]->sizeOfT();
              }
            }

            if (!currentBuf || currentLen != beforeBytes.size()) continue;

            std::vector<uint8_t> afterBytes(currentLen);
            cudaMemcpyAsync(afterBytes.data(), currentBuf, currentLen, cudaMemcpyDeviceToHost,
                            reinterpret_cast<cudaStream_t>(actualStream));

            bool differs = (memcmp(beforeBytes.data(), afterBytes.data(), currentLen) != 0);
            bool isExpected = (slotIdx >= 0 && expectedOutputs.count(slotIdx) > 0);

            if (differs && !isExpected) {
              corruptedCount++;
              if (corruptedCount <= 30) {
                int firstDiffByte = -1;
                for (size_t b = 0; b < currentLen; b++) {
                  if (beforeBytes[b] != afterBytes[b]) { firstDiffByte = (int)b; break; }
                }
                const char* slotName = "external";
                if (slotIdx >= 0 && slotIdx <= seg.def.endSlot) {
                  slotName = slots[slotIdx].ident.opName.c_str();
                }
                DSP_DIAG(VERIFY, "TRITON VERIFY CORRUPTION: subK[%d] [%d-%d] damaged %s slot %d (%s) "
                        "len=%zu firstDiffByte=%d gpuAddr=%p",
                        i, subKernel.startSlot_, subKernel.endSlot_,
                        slotIdx < 0 ? "EXTERNAL" : "NON-OUTPUT",
                        slotIdx, slotName, currentLen, firstDiffByte, currentBuf);
              }
            }
          }
          if (corruptedCount > 0) {
            DSP_DIAG(VERIFY, "TRITON VERIFY CORRUPTION SUMMARY: subK[%d] [%d-%d] corrupted %d non-output slots",
                    i, subKernel.startSlot_, subKernel.endSlot_, corruptedCount);
          } else {
            DSP_DIAG(VERIFY, "TRITON VERIFY CORRUPTION: subK[%d] [%d-%d] ZERO non-output slots corrupted (clean)",
                    i, subKernel.startSlot_, subKernel.endSlot_);
          }
        }

        // Restore pre-Triton state before native execution
        if (tritonVerifyFullSnapshot && !fullSnapshotBefore.empty()) {
          // Restore ALL slots from host-side snapshot (H2D) — no GPU dup needed
          for (auto& snap : fullSnapshotBefore) {
            void* dstBuf = nullptr;
            if (snap.first >= 0) {
              if (snap.first < totalOutputSlots && outputSlots[snap.first] &&
                  outputSlots[snap.first]->specialBuffer()) {
                dstBuf = outputSlots[snap.first]->specialBuffer();
                if (dstBuf) {
                  cudaMemcpyAsync(dstBuf, snap.second.data(), snap.second.size(),
                                  cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(actualStream));
                  outputSlots[snap.first]->dataBuffer()->writeSpecial();
                }
              }
            } else {
              int ei = -(snap.first + 1);
              if (ei < numExternalInputs && externalInputs[ei] && externalInputs[ei]->specialBuffer()) {
                dstBuf = externalInputs[ei]->specialBuffer();
                if (dstBuf) {
                  cudaMemcpyAsync(dstBuf, snap.second.data(), snap.second.size(),
                                  cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(actualStream));
                  externalInputs[ei]->dataBuffer()->writeSpecial();
                }
              }
            }
          }
        } else {
          // Non-fullSnapshot: restore from GPU dup copies (D2D)
          for (auto& kv : savedOutputs) {
            if (kv.first >= 0 && kv.first < totalOutputSlots && outputSlots[kv.first]) {
              auto* dst = outputSlots[kv.first];
              auto* src = kv.second;
              void* dstBuf = dst->specialBuffer();
              void* srcBuf = src->specialBuffer();
              if (dstBuf && srcBuf) {
                size_t bytes = dst->lengthOf() * dst->sizeOfT();
                cudaMemcpyAsync(dstBuf, srcBuf, bytes, cudaMemcpyDeviceToDevice,
                                reinterpret_cast<cudaStream_t>(actualStream));
                dst->dataBuffer()->writeSpecial();
              }
            }
          }
        }
        // Run native slot-by-slot
        auto nativeStatus = orderedRangeExecutor_(subKernel.startSlot_, subKernel.endSlot_);
        if (nativeStatus != Status::OK) {
          DSP_DIAG(VERIFY, "TRITON VERIFY: native ordered range for [%d-%d] FAILED (status=%d)",
                   subKernel.startSlot_, subKernel.endSlot_, static_cast<int>(nativeStatus));
          // Don't abort — continue with Triton results
          // Restore Triton outputs since native failed
          for (auto& kv2 : tritonRawOutputs) {
            if (kv2.first < totalOutputSlots && outputSlots[kv2.first]) {
              auto* arr = outputSlots[kv2.first];
              void* sbuf = arr->specialBuffer();
              if (sbuf) {
                cudaMemcpyAsync(sbuf, kv2.second.data.data(), kv2.second.data.size(),
                                cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(actualStream));
                arr->dataBuffer()->writeSpecial();
              }
            }
          }
          for (auto& kv : savedOutputs) delete kv.second;
          break;  // Skip comparison, exit verify block
        }

        markOrderedRangeDeviceCurrent(subKernel.startSlot_, subKernel.endSlot_, slots,
                                      externalInputs, numExternalInputs,
                                      outputSlots, totalOutputSlots);
        // Compare native outputs against Triton raw outputs
        int mismatches = 0;
        double overallMaxDiff = 0;

        auto readDouble = [](const uint8_t* buf, DataType dt, int idx) -> double {
          if (dt == FLOAT32) return static_cast<double>(reinterpret_cast<const float*>(buf)[idx]);
          if (dt == INT64) return static_cast<double>(reinterpret_cast<const LongType*>(buf)[idx]);
          if (dt == INT32) return static_cast<double>(reinterpret_cast<const int*>(buf)[idx]);
          if (dt == HALF) return static_cast<double>(reinterpret_cast<const float16*>(buf)[idx]);
          if (dt == DOUBLE) return reinterpret_cast<const double*>(buf)[idx];
          if (dt == INT8) return static_cast<double>(reinterpret_cast<const int8_t*>(buf)[idx]);
          if (dt == BOOL) return static_cast<double>(buf[idx]);
          return 0.0;
        };

        for (auto& kv : tritonRawOutputs) {
          int outIdx = kv.first;
          auto& rb = kv.second;
          NDArray* nativeArr = (outIdx < totalOutputSlots) ? outputSlots[outIdx] : nullptr;
          if (!nativeArr || !nativeArr->specialBuffer()) continue;

          size_t nativeByteLen = nativeArr->lengthOf() * nativeArr->sizeOfT();
          std::vector<uint8_t> nativeRaw(nativeByteLen);
          cudaMemcpyAsync(nativeRaw.data(), nativeArr->specialBuffer(), nativeByteLen,
                          cudaMemcpyDeviceToHost,
                          reinterpret_cast<cudaStream_t>(actualStream));

          LongType len = std::min(rb.len, nativeArr->lengthOf());

          double maxAbsDiff = 0;
          int maxDiffIdx = -1;
          for (int e = 0; e < len; e++) {
            double tVal = readDouble(rb.data.data(), rb.dtype, e);
            double nVal = readDouble(nativeRaw.data(), nativeArr->dataType(), e);
            double diff = std::abs(tVal - nVal);
            if (diff > maxAbsDiff) {
              maxAbsDiff = diff;
              maxDiffIdx = e;
            }
          }
          if (maxAbsDiff > overallMaxDiff) overallMaxDiff = maxAbsDiff;

          if (maxAbsDiff > 1e-3) {
            mismatches++;
            double tVal = readDouble(rb.data.data(), rb.dtype, maxDiffIdx);
            double nVal = readDouble(nativeRaw.data(), nativeArr->dataType(), maxDiffIdx);
            DSP_DIAG(VERIFY, "TRITON VERIFY MISMATCH: subK[%d] [%d-%d] slot %d: "
                     "maxDiff=%.6f at idx %d (triton=%.6f, native=%.6f, len=%lld, dtype=%d)",
                     i, subKernel.startSlot_, subKernel.endSlot_,
                     outIdx, maxAbsDiff, maxDiffIdx, tVal, nVal,
                     (long long)len, static_cast<int>(nativeArr->dataType()));
          }
        }

        if (mismatches == 0) {
          DSP_DIAG(VERIFY, "TRITON VERIFY OK: subK[%d] [%d-%d] %d outputs (maxDiff=%.8e) execCount=%d",
                   i, subKernel.startSlot_, subKernel.endSlot_,
                   static_cast<int>(tritonRawOutputs.size()), overallMaxDiff, seg.exec.executionCount);
        } else {
          DSP_DIAG(VERIFY, "TRITON VERIFY: subK[%d] [%d-%d] %d/%d MISMATCHED (maxDiff=%.8e) execCount=%d",
                   i, subKernel.startSlot_, subKernel.endSlot_,
                   mismatches, static_cast<int>(tritonRawOutputs.size()), overallMaxDiff, seg.exec.executionCount);
        }

        bool keepNative = Environment::getInstance().tritonVerifyKeepNative();
        if (!keepNative) {
          // Restore Triton outputs to device
          for (auto& kv2 : tritonRawOutputs) {
            if (kv2.first < totalOutputSlots && outputSlots[kv2.first]) {
              auto* arr = outputSlots[kv2.first];
              void* sbuf = arr->specialBuffer();
              if (sbuf) {
                cudaMemcpyAsync(sbuf, kv2.second.data.data(), kv2.second.data.size(),
                                cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(actualStream));
                arr->dataBuffer()->writeSpecial();
              }
            }
          }
        } else {
          DSP_DIAG(VERIFY, "TRITON VERIFY: keeping NATIVE outputs subK[%d] [%d-%d] execCount=%d",
                  i, subKernel.startSlot_, subKernel.endSlot_, seg.exec.executionCount);
        }

        for (auto& kv : savedOutputs) delete kv.second;
      } else {
        for (auto& kv : savedOutputs) delete kv.second;
      }
      } while (false);
    }
    totalKernelLaunches_++;

    if (!streamCaptureActive) {
      cudaError_t launchErr = cudaPeekAtLastError();
      if (launchErr != cudaSuccess) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: CUDA launch error after sub-kernel [%d-%d]: %s",
                  subKernel.startSlot_, subKernel.endSlot_, cudaGetErrorString(launchErr));
        cudaGetLastError();
        return Status::KERNEL_FAILURE;
      }
    }

    if (!tritonSkipKernels) {
      std::vector<NDArray*> registerReads;
      std::vector<NDArray*> registerWrites;
      std::unordered_set<DataBuffer*> seenInputs;
      for (auto& argMap : subKernel.argSlotMapping) {
        if (argMap.isOutput) continue;
        NDArray* arr = resolveRangeArray(argMap.slotIndex, externalInputs, numExternalInputs,
                                          outputSlots, totalOutputSlots);
        if (arr && arr->dataBuffer() && seenInputs.insert(arr->dataBuffer()).second) {
          registerReads.push_back(arr);
        }
      }
      std::unordered_set<DataBuffer*> seenOutputs;
      for (auto& argMap : subKernel.argSlotMapping) {
        if (!argMap.isOutput) continue;
        int outIdx = argMap.slotIndex;
        if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
          auto* db = outputSlots[outIdx]->dataBuffer();
          if (db && seenOutputs.insert(db).second) {
            registerWrites.push_back(outputSlots[outIdx]);
          }
        }
      }
      if (!registerReads.empty() || !registerWrites.empty()) {
        NDArray::registerSpecialUse(registerWrites, registerReads);
      }
    }

    if (!streamCaptureActive && !skipThisKernel &&
        sd::Environment::getInstance().isDebugAndVerbose()) {
      for (auto& argMap : subKernel.argSlotMapping) {
        if (!argMap.isOutput) continue;
        int outIdx = argMap.slotIndex;
        if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
          auto* arr = outputSlots[outIdx];
          if (!arr->isEmpty() && arr->lengthOf() > 0) {
            DSP_DIAG(VERIFY, "DSP_FINGERPRINT_TRITON subkernel=%d-%d slot=%d op=%s "
                     "shape=%s dtype=%s len=%lld asyncValues=true",
                     subKernel.startSlot_, subKernel.endSlot_, outIdx,
                     slots[outIdx].ident.opName.c_str(),
                     ShapeUtils::shapeAsString(arr).c_str(),
                     DataTypeUtils::asString(arr->dataType()).c_str(),
                     (long long)arr->lengthOf());
          }
        }
      }
    }

    // Log actuality state AFTER Triton kernel + markDeviceCurrent
    if (!streamCaptureActive && !skipThisKernel) {
      logActualityState("POST_TRITON", subKernel.startSlot_, subKernel.endSlot_, slots,
                        outputSlots, totalOutputSlots, externalInputs, numExternalInputs);
    }

    if (subKernel.endSlot_ + 1 > nextSlotToRun) {
      nextSlotToRun = subKernel.endSlot_ + 1;
    }
  }

  if (nextSlotToRun <= seg.def.endSlot) {
    if (!streamCaptureActive) {
      logActualityState("PRE_TRAILING_GAP", nextSlotToRun, seg.def.endSlot, slots,
                        outputSlots, totalOutputSlots, externalInputs, numExternalInputs);
    }
    if (!orderedRangeExecutor_) {
      DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: missing ordered range executor for trailing gap [%d-%d]",
                nextSlotToRun, seg.def.endSlot);
      return Status::KERNEL_FAILURE;
    }
    auto gapStatus = orderedRangeExecutor_(nextSlotToRun, seg.def.endSlot);
    if (gapStatus != Status::OK) {
      DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: trailing ordered native range [%d-%d] failed with status=%d",
                nextSlotToRun, seg.def.endSlot, static_cast<int>(gapStatus));
      return gapStatus;
    }
    markOrderedRangeDeviceCurrent(nextSlotToRun, seg.def.endSlot, slots,
                                  externalInputs, numExternalInputs,
                                  outputSlots, totalOutputSlots);
    if (!streamCaptureActive) {
      logSlotHashes("TRAILING_GAP", nextSlotToRun, seg.def.endSlot, slots,
                    outputSlots, totalOutputSlots,
                    reinterpret_cast<cudaStream_t>(actualStream), seg.exec.executionCount);
    }
  }

  // Compose attention present_key / present_value outputs
  int attnCount = 0;
  for (int si = seg.def.startSlot; si <= seg.def.endSlot; si++) {
    if (islandFilterActive && (si < tl_islandSlotMin || si > tl_islandSlotMax)) {
      continue;
    }
    if (slots[si].ident.opName.empty()) continue;
    bool isAttn = slots[si].ident.op != nullptr &&
                  slots[si].ident.op->getOpDescriptor() != nullptr &&
                  slots[si].ident.op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_ATTENTION);
    if (!isAttn) continue;
    if (slots[si].wiring.numInputs <= 4 || slots[si].wiring.numOutputs < 2) continue;

    int currentKeySrc = slots[si].wiring.inputSourceIndices[1];
    int currentValueSrc = (slots[si].wiring.numInputs > 2) ? slots[si].wiring.inputSourceIndices[2] : -1;
    int presentKeyOut = slots[si].wiring.outputSlotIndices[1];
    int presentValueOut = (slots[si].wiring.numOutputs >= 3) ? slots[si].wiring.outputSlotIndices[2] : -1;

    DSP_DIAG(EXECUTE, "composePresentKv: attn slot=%d currentKeySrc=%d currentValueSrc=%d "
             "presentKeyOut=%d presentValueOut=%d",
             si, currentKeySrc, currentValueSrc, presentKeyOut, presentValueOut);

    auto scatterCurrentToPresent = [&](int currentSlot, int presentSlot, const char* label) {
      NDArray* currentArr = nullptr;
      if (currentSlot < 0) {
        int extIdx = -(currentSlot + 1);
        if (extIdx >= 0 && extIdx < numExternalInputs && externalInputs[extIdx])
          currentArr = externalInputs[extIdx];
      } else if (currentSlot >= 0 && currentSlot < totalOutputSlots && outputSlots[currentSlot]) {
        currentArr = outputSlots[currentSlot];
      }
      if (!currentArr) return;

      if (presentSlot < 0 || presentSlot >= totalOutputSlots || !outputSlots[presentSlot]) return;
      auto* presentArr = outputSlots[presentSlot];

      auto currentBuf = currentArr->dataBuffer();
      auto presentBuf = presentArr->dataBuffer();
      if (!currentBuf || !presentBuf || !currentBuf->special() || !presentBuf->special()) return;

      if (presentArr->rankOf() != 4) return;
      int numHeads = static_cast<int>(presentArr->sizeAt(1));
      int seqLen = static_cast<int>(presentArr->sizeAt(2));
      int headDim = static_cast<int>(presentArr->sizeAt(3));
      int lastPos = seqLen - 1;

      size_t elemSize = presentArr->sizeOfT();
      char* dstBase = static_cast<char*>(presentBuf->special());
      char* srcBase = static_cast<char*>(currentBuf->special());

      for (int h = 0; h < numHeads; h++) {
        size_t dstOffset = static_cast<size_t>(h * seqLen + lastPos) * headDim * elemSize;
        size_t srcOffset = static_cast<size_t>(h) * headDim * elemSize;
        cudaMemcpyAsync(dstBase + dstOffset, srcBase + srcOffset, headDim * elemSize,
                        cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(actualStream));
      }
      DSP_DIAG(EXECUTE, "composePresentKv %s: scatter %d heads x %d headDim at lastPos=%d",
               label, numHeads, headDim, lastPos);
    };

    scatterCurrentToPresent(currentKeySrc, presentKeyOut, "KEY");
    scatterCurrentToPresent(currentValueSrc, presentValueOut, "VAL");
    attnCount++;
  }

  if (!streamCaptureActive) {
    auto launchErr = cudaPeekAtLastError();
    if (launchErr != cudaSuccess) {
      DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: CUDA launch error for [%d-%d]: %s",
                seg.def.startSlot, seg.def.endSlot, cudaGetErrorString(launchErr));
      cudaGetLastError();
      return Status::KERNEL_FAILURE;
    }
  }

  if (!streamCaptureActive && DSP_DIAG_ENABLED(VERIFY)) {
    DSP_DIAG_SEG(VERIFY, seg.def.startSlot,
        "seg[%d-%d] exec=%d skip=%d attn=%d",
        seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
        tritonSkipKernels ? 1 : 0, attnCount);
  }


  if (streamCaptureActive) {
    DSP_DIAG(EXECUTE, "CAPTURE_EXEC_SUMMARY: seg[%d-%d] launched=%d filtered=%d skipped=%d "
             "total_subK=%d islandFilter=[%d-%d]",
             seg.def.startSlot, seg.def.endSlot,
             captureLaunchedCount, captureFilteredCount, captureSkippedCount,
             static_cast<int>(compiledSeg->subKernels.size()),
             tl_islandSlotMin, tl_islandSlotMax);
  }

  return Status::OK;
}

// ─── Cache invalidation ────────────────────────────────────────────────────

std::unordered_set<int> TritonGraphBackend::getGapSlots(const GraphSegment& seg, NativeSlot* slots) const {
  std::unordered_set<int> gapSlots;

  int activeDevice = 0;
  cudaGetDevice(&activeDevice);
  auto& gapEnv = sd::Environment::getInstance();
  bool compileAll = gapEnv.tritonCompileAll();
  size_t excludeOpsHash = std::hash<std::string>()(gapEnv.tritonExcludeOps());
  size_t includeTypesHash = std::hash<std::string>()(gapEnv.tritonIncludeTypes());
  bool graphCapture = gapEnv.tritonGraphCapture();
  SegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, seg.def.shapeKeyState.compiledShapeKey, activeDevice, compileAll, excludeOpsHash, includeTypesHash, graphCapture};

  std::lock_guard<std::mutex> lock(cacheMtx_);
  auto it = cache_.find(key);
  if (it == cache_.end()) {
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      gapSlots.insert(s);
    }
    return gapSlots;
  }

  std::unordered_set<int> coveredSlots;
  for (const auto& sk : it->second.subKernels) {
    for (int s = sk.startSlot_; s <= sk.endSlot_; s++) {
      coveredSlots.insert(s);
    }
  }

  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    if (coveredSlots.find(s) == coveredSlots.end()) {
      gapSlots.insert(s);
    }
  }

  DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "NativeDSP: getGapSlots: seg[%d-%d] %d subKernels, %d covered, %d gap slots (of %d total)",
               seg.def.startSlot, seg.def.endSlot,
               static_cast<int>(it->second.subKernels.size()),
               static_cast<int>(coveredSlots.size()),
               static_cast<int>(gapSlots.size()),
               seg.def.endSlot - seg.def.startSlot + 1);

  return gapSlots;
}

void TritonGraphBackend::clearFailedSegmentCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  if (!failedCache_.empty()) {
    DSP_DIAG(COMPILE, "TritonGraphBackend::clearFailedSegmentCache: clearing %d failed entries",
              static_cast<int>(failedCache_.size()));
    failedCache_.clear();
  }
}

void TritonGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  // Pull all sub-kernels out of the residency tracking list before we tear
  // down their modules.  unregisterLoadedKernel acquires loadedKernelsMtx_
  // briefly; lock order (cacheMtx_ -> loadedKernelsMtx_) is consistent with
  // every other call site (evictIfOverBudget never takes cacheMtx_).
  for (auto& entry : cache_) {
    for (auto& kernel : entry.second.subKernels) {
      unregisterLoadedKernel(&kernel);
    }
  }
	  for (auto& entry : cache_) {
	    auto& seg = entry.second;
#ifdef SD_CUDA
    if (seg.preallocReadyEvent != nullptr) {
      cudaEventDestroy(reinterpret_cast<cudaEvent_t>(seg.preallocReadyEvent));
      seg.preallocReadyEvent = nullptr;
    }
#endif
	    // Determine device for memory tracking
	    int segDeviceId = 0;
    if (!seg.subKernels.empty() && seg.subKernels[0].cachedArgTableDeviceId >= 0)
      segDeviceId = seg.subKernels[0].cachedArgTableDeviceId;

    // Free consolidated arg table buffers FIRST (before per-kernel cleanup,
    // because per-kernel pointers are offsets into these buffers).
    if (seg.useConsolidatedArgTable) {
      if (seg.consolidatedArgTableDevice != nullptr) {
        recordModuleFree(seg.consolidatedArgTableDeviceId >= 0 ? seg.consolidatedArgTableDeviceId : segDeviceId,
                         seg.consolidatedArgTableBytes);
        // Guard: capture workspace interior pointers cannot be individually freed.
        // The workspace base is freed separately by releaseWorkspace/unregisterCaptureWorkspace.
        if (!sd::memory::CudaMemoryPool::getInstance().isInCaptureWorkspace(seg.consolidatedArgTableDevice)) {
          cudaFree(seg.consolidatedArgTableDevice);
        }
        seg.consolidatedArgTableDevice = nullptr;
        seg.consolidatedArgTableBytes = 0;
      }
      if (seg.consolidatedArgTableHostPinned != nullptr) {
        auto& memPool = sd::memory::CudaMemoryPool::getInstance();
        // Capture-owned: destruction belongs to the replay handle (baked H2D src).
        if (!seg.consolidatedArgTableCaptureOwned) {
          memPool.freePinnedHost(seg.consolidatedArgTableHostPinned);
        }
        seg.consolidatedArgTableHostPinned = nullptr;
        seg.consolidatedArgTableCaptureOwned = false;
      }
      // Null out per-kernel pointers (they were offsets into consolidated buffer,
      // NOT independent allocations — do NOT cudaFree them!)
      for (auto& kernel : seg.subKernels) {
        kernel.cachedArgTableDevice = nullptr;
        kernel.cachedArgTableBytes = 0;
        kernel.cachedArgTableDeviceId = -1;
        kernel.cachedArgTableHostPinned = nullptr;
        kernel.cachedArgTableHostPinnedBytes = 0;
      }
    }
    for (auto& kernel : seg.subKernels) {
      int kDevId = kernel.cachedArgTableDeviceId >= 0 ? kernel.cachedArgTableDeviceId : segDeviceId;
      // Only free per-kernel arg tables if NOT consolidated (consolidated
      // arg tables were freed above; per-kernel pointers are interior offsets).
      if (!seg.useConsolidatedArgTable && kernel.cachedArgTableDevice != nullptr) {
        recordModuleFree(kDevId, kernel.cachedArgTableBytes);
        // Guard: capture workspace interior pointers cannot be individually freed.
        if (!sd::memory::CudaMemoryPool::getInstance().isInCaptureWorkspace(kernel.cachedArgTableDevice)) {
          cudaFree(kernel.cachedArgTableDevice);
        }
        kernel.cachedArgTableDevice = nullptr;
        kernel.cachedArgTableBytes = 0;
        kernel.cachedArgTableDeviceId = -1;
      }
      if (!seg.useConsolidatedArgTable && kernel.cachedArgTableHostPinned != nullptr) {
        auto& memPool = sd::memory::CudaMemoryPool::getInstance();
        if (!kernel.cachedArgTableCaptureOwned) {
          memPool.freePinnedHost(kernel.cachedArgTableHostPinned);
        }
        kernel.cachedArgTableHostPinned = nullptr;
        kernel.cachedArgTableHostPinnedBytes = 0;
        kernel.cachedArgTableCaptureOwned = false;
      }
      if (kernel.cachedSyncCounterDevice != nullptr) {
        recordModuleFree(kernel.cachedSyncCounterDeviceId >= 0 ? kernel.cachedSyncCounterDeviceId : segDeviceId,
                         sizeof(int));
        // Guard: capture workspace interior pointers cannot be individually freed.
        if (!sd::memory::CudaMemoryPool::getInstance().isInCaptureWorkspace(kernel.cachedSyncCounterDevice)) {
          cudaFree(kernel.cachedSyncCounterDevice);
        }
        kernel.cachedSyncCounterDevice = nullptr;
        kernel.cachedSyncCounterDeviceId = -1;
      }
      if (kernel.cachedGlobalScratchDevice != nullptr) {
        recordModuleFree(kernel.cachedGlobalScratchDeviceId >= 0 ? kernel.cachedGlobalScratchDeviceId : segDeviceId,
                         kernel.cachedGlobalScratchBytes);
        // Guard: capture workspace interior pointers cannot be individually freed.
        if (!sd::memory::CudaMemoryPool::getInstance().isInCaptureWorkspace(kernel.cachedGlobalScratchDevice)) {
          cudaFree(kernel.cachedGlobalScratchDevice);
        }
        kernel.cachedGlobalScratchDevice = nullptr;
        kernel.cachedGlobalScratchBytes = 0;
        kernel.cachedGlobalScratchDeviceId = -1;
      }
      if (kernel.gpuModule) {
        const int moduleDevId = kernel.loadedDeviceId >= 0 ? kernel.loadedDeviceId : segDeviceId;
        recordModuleFree(moduleDevId, kernel.estimatedModuleBytes);
        if (sd::graph::modreg::releaseFromOwner(kernel.gpuModule)) {
          TritonTargetDispatch::unloadModule(kernel.gpuModule);
        }
        kernel.moduleCaptureOwned = false;
      }
    }
  }
  cache_.clear();
  failedCache_.clear();
  lastCompilationAudit_.clear();
}

void TritonGraphBackend::invalidateCacheForSegments(const std::vector<std::pair<int,int>>& segmentRanges) {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  int freedEntries = 0;
  int freedModules = 0;

  auto it = cache_.begin();
  while (it != cache_.end()) {
    bool overlaps = false;
    for (auto& [segStart, segEnd] : segmentRanges) {
      if (it->first.startSlot >= segStart && it->first.endSlot <= segEnd) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) {
      ++it;
      continue;
    }

	    auto& seg = it->second;
#ifdef SD_CUDA
    if (seg.preallocReadyEvent != nullptr) {
      cudaEventDestroy(reinterpret_cast<cudaEvent_t>(seg.preallocReadyEvent));
      seg.preallocReadyEvent = nullptr;
    }
#endif
	    // Determine device for memory tracking (use first kernel's cached device, or 0)
	    int segDeviceId = 0;
    if (!seg.subKernels.empty() && seg.subKernels[0].cachedArgTableDeviceId >= 0)
      segDeviceId = seg.subKernels[0].cachedArgTableDeviceId;

    // Drop residency tracking for these kernels before unloading their
    // modules, otherwise a concurrent eviction sweep could chase the stale
    // pointers we are about to delete.
    for (auto& kernel : seg.subKernels) {
      unregisterLoadedKernel(&kernel);
    }

    // Free resources (same logic as invalidateCache)
    if (seg.useConsolidatedArgTable) {
      if (seg.consolidatedArgTableDevice != nullptr) {
        recordModuleFree(seg.consolidatedArgTableDeviceId >= 0 ? seg.consolidatedArgTableDeviceId : segDeviceId,
                         seg.consolidatedArgTableBytes);
        // Guard: capture workspace interior pointers cannot be individually freed with cudaFree.
        // The workspace base is freed by releaseWorkspace/unregisterCaptureWorkspace (called
        // AFTER invalidateCacheForSegments in platformFreePlanResources). Skip cudaFree here;
        // the memory will be reclaimed when the workspace block is freed.
        if (!sd::memory::CudaMemoryPool::getInstance().isInCaptureWorkspace(seg.consolidatedArgTableDevice)) {
          cudaFree(seg.consolidatedArgTableDevice);
        }
      }
      if (seg.consolidatedArgTableHostPinned != nullptr) {
        auto& memPool = sd::memory::CudaMemoryPool::getInstance();
        if (!seg.consolidatedArgTableCaptureOwned) {
          memPool.freePinnedHost(seg.consolidatedArgTableHostPinned);
        }
      }
      for (auto& kernel : seg.subKernels) {
        kernel.cachedArgTableDevice = nullptr;
        kernel.cachedArgTableBytes = 0;
        kernel.cachedArgTableHostPinned = nullptr;
        kernel.cachedArgTableHostPinnedBytes = 0;
      }
    }
    for (auto& kernel : seg.subKernels) {
      int kDevId = kernel.cachedArgTableDeviceId >= 0 ? kernel.cachedArgTableDeviceId : segDeviceId;
      if (!seg.useConsolidatedArgTable && kernel.cachedArgTableDevice != nullptr) {
        recordModuleFree(kDevId, kernel.cachedArgTableBytes);
        // Guard: capture workspace interior pointers cannot be individually freed.
        if (!sd::memory::CudaMemoryPool::getInstance().isInCaptureWorkspace(kernel.cachedArgTableDevice)) {
          cudaFree(kernel.cachedArgTableDevice);
        }
      }
      if (!seg.useConsolidatedArgTable && kernel.cachedArgTableHostPinned != nullptr) {
        auto& memPool = sd::memory::CudaMemoryPool::getInstance();
        if (!kernel.cachedArgTableCaptureOwned) {
          memPool.freePinnedHost(kernel.cachedArgTableHostPinned);
        }
      }
      if (kernel.cachedSyncCounterDevice != nullptr) {
        recordModuleFree(kernel.cachedSyncCounterDeviceId >= 0 ? kernel.cachedSyncCounterDeviceId : segDeviceId,
                         sizeof(int));
        // Guard: capture workspace interior pointers cannot be individually freed.
        if (!sd::memory::CudaMemoryPool::getInstance().isInCaptureWorkspace(kernel.cachedSyncCounterDevice)) {
          cudaFree(kernel.cachedSyncCounterDevice);
        }
      }
      if (kernel.cachedGlobalScratchDevice != nullptr) {
        recordModuleFree(kernel.cachedGlobalScratchDeviceId >= 0 ? kernel.cachedGlobalScratchDeviceId : segDeviceId,
                         kernel.cachedGlobalScratchBytes);
        // Guard: capture workspace interior pointers cannot be individually freed.
        if (!sd::memory::CudaMemoryPool::getInstance().isInCaptureWorkspace(kernel.cachedGlobalScratchDevice)) {
          cudaFree(kernel.cachedGlobalScratchDevice);
        }
      }
      if (kernel.gpuModule) {
        const int moduleDevId = kernel.loadedDeviceId >= 0 ? kernel.loadedDeviceId : segDeviceId;
        recordModuleFree(moduleDevId, kernel.estimatedModuleBytes);
        if (sd::graph::modreg::releaseFromOwner(kernel.gpuModule)) {
          TritonTargetDispatch::unloadModule(kernel.gpuModule);
        }
        kernel.moduleCaptureOwned = false;
        freedModules++;
      }
    }
    freedEntries++;
    it = cache_.erase(it);
  }

  // Also remove matching failed cache entries
  auto fit = failedCache_.begin();
  while (fit != failedCache_.end()) {
    bool overlaps = false;
    for (auto& [segStart, segEnd] : segmentRanges) {
      if (fit->startSlot >= segStart && fit->endSlot <= segEnd) {
        overlaps = true;
        break;
      }
    }
    if (overlaps) {
      fit = failedCache_.erase(fit);
    } else {
      ++fit;
    }
  }

  if (freedEntries > 0) {
    DSP_DIAG(MEMORY, "TritonGraphBackend::invalidateCacheForSegments: freed %d cache entries (%d GPU modules) "
             "for %d segment ranges",
             freedEntries, freedModules, static_cast<int>(segmentRanges.size()));
  }
}

// ─── Per-device Triton module memory tracking ───────────────────────────────

void TritonGraphBackend::recordModuleAlloc(int deviceId, size_t bytes) {
  if (deviceId >= 0 && deviceId < kMaxTritonDevices)
    tritonDeviceMemory_[deviceId].fetch_add(bytes, std::memory_order_relaxed);
}

void TritonGraphBackend::recordModuleFree(int deviceId, size_t bytes) {
  if (deviceId >= 0 && deviceId < kMaxTritonDevices)
    tritonDeviceMemory_[deviceId].fetch_sub(bytes, std::memory_order_relaxed);
}

size_t TritonGraphBackend::getTritonModuleMemory(int deviceId) const {
  if (deviceId >= 0 && deviceId < kMaxTritonDevices)
    return tritonDeviceMemory_[deviceId].load(std::memory_order_relaxed);
  return 0;
}

size_t TritonGraphBackend::getTotalTritonModuleMemory() const {
  size_t total = 0;
  for (int i = 0; i < kMaxTritonDevices; i++)
    total += tritonDeviceMemory_[i].load(std::memory_order_relaxed);
  return total;
}

// ─── Compilation audit ──────────────────────────────────────────────────────

std::vector<CompilationAuditEntry> TritonGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

#endif // HAVE_TRITON
