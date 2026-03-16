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
#include <graph/gpu/TritonGraphBackend_internal.h>
#include <graph/gpu/TritonTargetDispatch.h>
#include <graph/DspDiagnostics.h>
#include <system/Environment.h>
#include <helpers/logger.h>

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


// Compute FNV-1a hash of GPU buffer contents (D2H + hash on host).
// Only runs when VERIFY category is enabled. Returns 0 on failure.
static uint64_t hashSlotGpuContent(NDArray* arr, cudaStream_t stream) {
  if (!arr || !arr->specialBuffer() || !arr->dataBuffer()) return 0;
  size_t bytes = arr->lengthOf() * arr->sizeOfT();
  if (bytes == 0 || bytes > 64 * 1024 * 1024) return 0;
  std::vector<uint8_t> buf(bytes);
  auto err = cudaMemcpy(buf.data(), arr->specialBuffer(), bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  uint64_t hash = FNV1A64_OFFSET_BASIS;
  mixFNV1a(hash, buf.data(), bytes);
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
    for (int o = 0; o < slots[si].numOutputs; o++) {
      int outIdx = slots[si].outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
      if (!loggedSlots.insert(outIdx).second) continue;  // already logged
      NDArray* arr = outputSlots[outIdx];
      if (!arr) continue;
      uint64_t hash = hashSlotGpuContent(arr, stream);
      DSP_DIAG(VERIFY, "HASH %s [%d-%d] exec=%d slot=%d (%s) hash=%016llx len=%lld dt=%d",
               label, startSlot, endSlot, execCount,
               outIdx, slots[si].opName.c_str(), (unsigned long long)hash,
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
    for (int i = 0; i < slots[si].numInputs; i++) {
      int srcIdx = slots[si].inputSourceIndices[i];
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
    for (int o = 0; o < slots[si].numOutputs; o++) {
      int outIdx = slots[si].outputSlotIndices[o];
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
                ? slots[si].opName.c_str() : "?";
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
                 (si <= segEndSlot) ? slots[si].opName.c_str() : "?",
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
  if (seg.startSlot >= 0) {
    targetDevice = slots[seg.startSlot].targetDeviceId;
  }
  bool streamCaptureActive = false;

  cudaError_t execDeviceErr = cudaGetDevice(&execDevice);
  if (execDeviceErr != cudaSuccess) {
    DSP_DIAG(BACKEND, "TritonGraphBackend::executeSegment: cudaGetDevice failed for segment [%d-%d]: %s",
              seg.startSlot, seg.endSlot, cudaGetErrorString(execDeviceErr));
    cudaGetLastError();
    return Status::KERNEL_FAILURE;
  }
  if (actualStream != nullptr) {
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    auto capErr = cudaStreamIsCapturing(static_cast<cudaStream_t>(actualStream), &captureStatus);
    if (capErr == cudaSuccess && captureStatus != cudaStreamCaptureStatusNone) {
      streamCaptureActive = true;
    }
  }

  auto& execEnv = Environment::getInstance();
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey, execDevice,
                      execEnv.tritonCompileAll(),
                      std::hash<std::string>()(execEnv.tritonExcludeOps())};

  CompiledSegment* compiledSeg = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      int cachedDeviceId = -999;
      for (const auto& entry : cache_) {
        if (entry.first.startSlot == seg.startSlot &&
            entry.first.endSlot == seg.endSlot &&
            entry.first.shapeKey == seg.shapeKey) {
          cachedDeviceId = entry.first.deviceId;
          break;
        }
      }
      if (cachedDeviceId != -999) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: kernel cache miss for segment [%d-%d] "
                 "(shapeKey=%lld, activeDevice=%d, targetDeviceId=%d). "
                 "Found compiled kernel for deviceId=%d but cross-device module reuse is disallowed.",
                 seg.startSlot, seg.endSlot, seg.shapeKey, execDevice, targetDevice, cachedDeviceId);
      } else {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: no compiled kernel for segment [%d-%d] "
                 "(shapeKey=%lld, deviceId=%d)",
                 seg.startSlot, seg.endSlot, seg.shapeKey, execDevice);
      }
      return Status::KERNEL_FAILURE;
    }
    compiledSeg = &it->second;
  }

  if (streamCaptureActive && !compiledSeg->fallbackRanges.empty()) {
    if (!Environment::getInstance().tritonAllowFallbackCapture()) {
      DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: refusing slot fallback during CUDA graph capture for [%d-%d] (%d fallback ranges)",
               seg.startSlot, seg.endSlot, static_cast<int>(compiledSeg->fallbackRanges.size()));
      return Status::KERNEL_FAILURE;
    }
    DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: allowing fallback during CUDA graph capture for [%d-%d] (%d fallback ranges)",
             seg.startSlot, seg.endSlot, static_cast<int>(compiledSeg->fallbackRanges.size()));
  }

  bool tritonSkipKernels = Environment::getInstance().tritonSkipKernels();
  bool tritonVerifyKernels = Environment::getInstance().tritonVerifyKernels();
  int tritonMaxSubKernelIndex = Environment::getInstance().tritonMaxSubKernelIndex();
  bool tritonVerifyFullSnapshot = Environment::getInstance().tritonVerifyFullSnapshot();

  DSP_DIAG_SEG(EXECUTE, seg.startSlot, "TritonGraphBackend::executeSegment: segment [%d-%d] launching %d sub-kernels "
               "(fallbackRanges=%d, targetDeviceId=%d, activeDevice=%d, skipKernels=%d, verifyKernels=%d, "
               "maxSubKernelIdx=%d, fullSnapshot=%d, execCount=%d)",
               seg.startSlot, seg.endSlot,
               static_cast<int>(compiledSeg->subKernels.size()),
               static_cast<int>(compiledSeg->fallbackRanges.size()),
               targetDevice, execDevice,
               tritonSkipKernels ? 1 : 0, tritonVerifyKernels ? 1 : 0,
               tritonMaxSubKernelIndex, tritonVerifyFullSnapshot ? 1 : 0,
               seg.executionCount);

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
    // CRITICAL: During CUDA graph capture, skip pre-allocation and input syncing.
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
                   static_cast<int>(argMapping.dtype), newArr->specialBuffer());
        }

        // Sync all INPUT arrays to device
        for (auto& argMapping : sk.argSlotMapping) {
          if (argMapping.isOutput) continue;
          NDArray* arr = nullptr;
          if (argMapping.slotIndex < 0) {
            int extIdx = -(argMapping.slotIndex + 1);
            if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
          } else {
            if (argMapping.slotIndex < totalOutputSlots) arr = outputSlots[argMapping.slotIndex];
          }
          if (arr && arr->lengthOf() > 0) {
            if (argMapping.slotIndex < 0 && arr->dataBuffer() != nullptr) {
              bool pAct = arr->dataBuffer()->isPrimaryActual();
              bool sAct = arr->dataBuffer()->isSpecialActual();
              if (pAct && !sAct) {
                arr->dataBuffer()->syncToSpecial(true);
              } else {
                arr->syncToDevice();
              }
            } else {
              arr->syncToDevice();
            }
          }
        }
      }
    }

    // Synchronize the execution stream to ensure all prior allocations
    // (via CudaMemoryPool / cudaMallocAsync) and data transfers are
    // complete before reading specialBuffer() pointers.
    // CRITICAL: cudaStreamSynchronize is ILLEGAL during stream capture —
    // it returns cudaErrorStreamCaptureUnsupported and invalidates the capture,
    // causing all subsequent CUDA operations to fail with "operation failed
    // due to a previous error during capture". During capture, output arrays
    // are already pre-allocated from the warmup execution and input syncs
    // are complete (the pre-capture warmup + cudaStreamSynchronize before
    // beginCapture handles this). Skip the sync during capture.
    if (!streamCaptureActive) {
      cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));
    }

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

      // Log all arg addresses for ALL sub-kernels when VERIFY is enabled
      if (DSP_DIAG_ENABLED(VERIFY)) {
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

      // Buffer aliasing detection for this sub-kernel
      detectBufferAliasing(static_cast<int>(ki),
                           sk.argSlotMapping, sk.startSlot_, sk.endSlot_,
                           externalInputs, numExternalInputs,
                           outputSlots, totalOutputSlots,
                           slots, seg.endSlot);
    }

    // ── Phase 3: Single consolidated H2D ──
    auto memcpyErr = cudaMemcpyAsync(
        compiledSeg->consolidatedArgTableDevice,
        compiledSeg->consolidatedArgTableHostPinned,
        compiledSeg->consolidatedArgTableBytes,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(actualStream));
    if (memcpyErr != cudaSuccess) {
      DSP_DIAG(MEMORY, "TritonGraphBackend: consolidated arg table H2D failed (%zu bytes): %s",
                compiledSeg->consolidatedArgTableBytes, cudaGetErrorString(memcpyErr));
      cudaGetLastError();
    } else {
      consolidatedArgsCopied = true;
      DSP_DIAG_SEG(EXECUTE, seg.startSlot, "TritonGraphBackend: consolidated arg table H2D: 1 copy of %zu bytes "
                   "(replaces %d per-kernel copies) for seg[%d-%d]",
                   compiledSeg->consolidatedArgTableBytes,
                   static_cast<int>(compiledSeg->subKernels.size()),
                   seg.startSlot, seg.endSlot);
    }
  }

  int nextSlotToRun = seg.startSlot;
  for (int i = 0; i < (int)compiledSeg->subKernels.size(); i++) {
    auto& subKernel = compiledSeg->subKernels[i];

    if (nextSlotToRun < subKernel.startSlot_) {
      if (streamCaptureActive && !Environment::getInstance().tritonAllowFallbackCapture()) {
        DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: refusing leading gap [%d-%d] during CUDA graph capture",
                 nextSlotToRun, subKernel.startSlot_ - 1);
        return Status::KERNEL_FAILURE;
      }

      // Log actuality state BEFORE gap execution
      logActualityState("PRE_GAP", nextSlotToRun, subKernel.startSlot_ - 1, slots,
                        outputSlots, totalOutputSlots, externalInputs, numExternalInputs);
      if (!fallbackRangeExecutor_) {
        DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: missing fallback executor for gap [%d-%d]",
                  nextSlotToRun, subKernel.startSlot_ - 1);
        return Status::KERNEL_FAILURE;
      }
      auto gapStatus = fallbackRangeExecutor_(nextSlotToRun, subKernel.startSlot_ - 1);
      if (gapStatus != Status::OK) {
        DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: slot-by-slot gap [%d-%d] failed with status=%d",
                  nextSlotToRun, subKernel.startSlot_ - 1, static_cast<int>(gapStatus));
        return gapStatus;
      }
      markFallbackRangeDeviceCurrent(nextSlotToRun, subKernel.startSlot_ - 1, slots,
                                     externalInputs, numExternalInputs,
                                     outputSlots, totalOutputSlots);
      // Hash gap outputs
      if (!streamCaptureActive) {
        logSlotHashes("GAP", nextSlotToRun, subKernel.startSlot_ - 1, slots,
                      outputSlots, totalOutputSlots,
                      static_cast<cudaStream_t>(actualStream), seg.executionCount);
      }

      // Debug: dump external input 1331 after gap completes, before Triton kernel launch
      if (sd::Environment::getInstance().isDebug() && numExternalInputs > 1331
          && subKernel.startSlot_ == 347) {
        NDArray* ext = externalInputs[1331];
        if (ext && ext->specialBuffer() && ext->lengthOf() > 0 && ext->dataType() == FLOAT32) {
          cudaDeviceSynchronize();
          int dc = std::min((int)ext->lengthOf(), 8);
          std::vector<float> hb(dc);
          cudaMemcpy(hb.data(), ext->specialBuffer(), dc * 4, cudaMemcpyDeviceToHost);
          std::string vs;
          for (int v = 0; v < dc; v++) {
            if (v > 0) vs += ",";
            char b[32]; snprintf(b, sizeof(b), "%.6f", hb[v]); vs += b;
          }
          float v322 = 0;
          if (ext->lengthOf() > 322)
            cudaMemcpy(&v322, static_cast<char*>(ext->specialBuffer()) + 322*4, 4, cudaMemcpyDeviceToHost);
          DSP_DIAG(VERIFY, "POST_GAP_EXT1331: exec=%d addr=%p len=%lld values: %s [322]=%.6f",
                   seg.executionCount, ext->specialBuffer(), (long long)ext->lengthOf(), vs.c_str(), v322);
        }
      }
    }

    // Decide whether to skip this sub-kernel: global skip OR per-index cutoff
    bool skipThisKernel = tritonSkipKernels ||
        (tritonMaxSubKernelIndex >= 0 && i > tritonMaxSubKernelIndex);

    if (skipThisKernel) {
      if (fallbackRangeExecutor_) {
        if (!tritonSkipKernels && tritonMaxSubKernelIndex >= 0) {
          DSP_DIAG(EXECUTE, "TritonGraphBackend: subK[%d] [%d-%d] SKIPPED (index > maxSubKernelIndex=%d)",
                   i, subKernel.startSlot_, subKernel.endSlot_, tritonMaxSubKernelIndex);
        }
        auto skipStatus = fallbackRangeExecutor_(subKernel.startSlot_, subKernel.endSlot_);
        if (skipStatus != Status::OK) {
          DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: native fallback for skipped kernel [%d-%d] failed with status=%d",
                 subKernel.startSlot_, subKernel.endSlot_, static_cast<int>(skipStatus));
          return skipStatus;
        }
        markFallbackRangeDeviceCurrent(subKernel.startSlot_, subKernel.endSlot_, slots,
                                       externalInputs, numExternalInputs,
                                       outputSlots, totalOutputSlots);
        if (!streamCaptureActive) {
          logSlotHashes("SKIP", subKernel.startSlot_, subKernel.endSlot_, slots,
                        outputSlots, totalOutputSlots,
                        static_cast<cudaStream_t>(actualStream), seg.executionCount);
        }
      }
    } else {
      // Log sub-kernel entry with op details
      {
        std::string opSummary;
        for (int si = subKernel.startSlot_; si <= subKernel.endSlot_ && opSummary.size() < 400; si++) {
          if (!opSummary.empty()) opSummary += ", ";
          opSummary += std::to_string(si) + ":" + slots[si].opName;
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
        cudaError_t syncErr = cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));
        if (syncErr != cudaSuccess) {
          DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: stream sync before sub-kernel %d/%d [%d-%d] "
                    "detected prior async error: %s",
                    i + 1, (int)compiledSeg->subKernels.size(),
                    subKernel.startSlot_, subKernel.endSlot_,
                    cudaGetErrorString(syncErr));
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
          opNames += std::to_string(si) + ":" + slots[si].opName;
        }
        DSP_DIAG(VERIFY, "TRITON VERIFY ENTRY: subK[%d] [%d-%d] execCount=%d ops: %s",
                 i, subKernel.startSlot_, subKernel.endSlot_, seg.executionCount, opNames.c_str());

        cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));

        // Full snapshot: save ALL slot GPU contents
        if (tritonVerifyFullSnapshot) {
          for (int si = 0; si < totalOutputSlots; si++) {
            if (outputSlots[si] && outputSlots[si]->dataBuffer() && outputSlots[si]->specialBuffer()) {
              size_t bytes = outputSlots[si]->lengthOf() * outputSlots[si]->sizeOfT();
              if (bytes > 0 && bytes <= 64 * 1024 * 1024) {
                fullSnapshotBefore[si].resize(bytes);
                cudaMemcpy(fullSnapshotBefore[si].data(), outputSlots[si]->specialBuffer(),
                           bytes, cudaMemcpyDeviceToHost);
              }
            }
          }
          for (int ei = 0; ei < numExternalInputs; ei++) {
            if (externalInputs[ei] && externalInputs[ei]->dataBuffer() && externalInputs[ei]->specialBuffer()) {
              size_t bytes = externalInputs[ei]->lengthOf() * externalInputs[ei]->sizeOfT();
              if (bytes > 0 && bytes <= 64 * 1024 * 1024) {
                int snapKey = -(ei + 1);
                fullSnapshotBefore[snapKey].resize(bytes);
                cudaMemcpy(fullSnapshotBefore[snapKey].data(), externalInputs[ei]->specialBuffer(),
                           bytes, cudaMemcpyDeviceToHost);
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
          // Non-fullSnapshot: only save the sub-kernel's declared output slots via dup()
          for (int si = subKernel.startSlot_; si <= subKernel.endSlot_; si++) {
            for (int o = 0; o < slots[si].numOutputs; o++) {
              int outIdx = slots[si].outputSlotIndices[o];
              if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx] &&
                  outputSlots[outIdx]->lengthOf() > 0 && !outputSlots[outIdx]->isEmpty()) {
                try {
                  savedOutputs[outIdx] = new NDArray(outputSlots[outIdx]->dup());
                } catch (...) {
                  DSP_DIAG(VERIFY, "TRITON VERIFY: dup() failed for slot %d — skipping", outIdx);
                }
              }
            }
          }
          DSP_DIAG(VERIFY, "TRITON VERIFY: saved %d output arrays via GPU dup()", static_cast<int>(savedOutputs.size()));
        } else {
          DSP_DIAG(VERIFY, "TRITON VERIFY: using host-side fullSnapshot for restore (%d snapshots, no GPU dup)",
                   static_cast<int>(fullSnapshotBefore.size()));
        }
      }
      // Buffer aliasing detection for non-consolidated path
      if (!consolidatedArgsCopied) {
        detectBufferAliasing(i,
                             subKernel.argSlotMapping, subKernel.startSlot_, subKernel.endSlot_,
                             externalInputs, numExternalInputs,
                             outputSlots, totalOutputSlots,
                             slots, seg.endSlot);
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

      DSP_DIAG(EXECUTE, "SUBKERNEL EXIT OK: subK[%d] [%d-%d] execCount=%d",
               i, subKernel.startSlot_, subKernel.endSlot_, seg.executionCount);

      // Hash Triton outputs
      if (!streamCaptureActive) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));
        logSlotHashes("TRITON", subKernel.startSlot_, subKernel.endSlot_, slots,
                      outputSlots, totalOutputSlots,
                      static_cast<cudaStream_t>(actualStream), seg.executionCount);
      }

      // ── Verify mode: run native and compare ──
      if (tritonVerifyKernels && !streamCaptureActive && fallbackRangeExecutor_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));

        // Save Triton outputs (raw bytes)
        struct RawBuffer { std::vector<uint8_t> data; DataType dtype; LongType len; };
        std::unordered_map<int, RawBuffer> tritonRawOutputs;
        for (int si = subKernel.startSlot_; si <= subKernel.endSlot_; si++) {
          for (int o = 0; o < slots[si].numOutputs; o++) {
            int outIdx = slots[si].outputSlotIndices[o];
            if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
              auto* arr = outputSlots[outIdx];
              void* sbuf = arr->specialBuffer();
              if (sbuf && arr->dataBuffer() && arr->lengthOf() > 0) {
                size_t byteLen = arr->lengthOf() * arr->sizeOfT();
                RawBuffer rb;
                rb.data.resize(byteLen);
                rb.dtype = arr->dataType();
                rb.len = arr->lengthOf();
                cudaMemcpy(rb.data.data(), sbuf, byteLen, cudaMemcpyDeviceToHost);
                tritonRawOutputs[outIdx] = std::move(rb);
              }
            }
          }
        }

        // Full snapshot corruption detection: find non-output slots modified by Triton
        if (tritonVerifyFullSnapshot && !fullSnapshotBefore.empty()) {
          int corruptedCount = 0;
          // Build set of expected output slots
          std::unordered_set<int> expectedOutputs;
          for (int si = subKernel.startSlot_; si <= subKernel.endSlot_; si++) {
            for (int o = 0; o < slots[si].numOutputs; o++) {
              int outIdx = slots[si].outputSlotIndices[o];
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
            cudaMemcpy(afterBytes.data(), currentBuf, currentLen, cudaMemcpyDeviceToHost);

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
                if (slotIdx >= 0 && slotIdx <= seg.endSlot) {
                  slotName = slots[slotIdx].opName.c_str();
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
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(actualStream));
                  outputSlots[snap.first]->dataBuffer()->writeSpecial();
                }
              }
            } else {
              int ei = -(snap.first + 1);
              if (ei < numExternalInputs && externalInputs[ei] && externalInputs[ei]->specialBuffer()) {
                dstBuf = externalInputs[ei]->specialBuffer();
                if (dstBuf) {
                  cudaMemcpyAsync(dstBuf, snap.second.data(), snap.second.size(),
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(actualStream));
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
                                static_cast<cudaStream_t>(actualStream));
                dst->dataBuffer()->writeSpecial();
              }
            }
          }
        }
        cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));

        // Run native slot-by-slot
        auto nativeStatus = fallbackRangeExecutor_(subKernel.startSlot_, subKernel.endSlot_);
        if (nativeStatus != Status::OK) {
          DSP_DIAG(VERIFY, "TRITON VERIFY: native fallback for [%d-%d] FAILED (status=%d)",
                   subKernel.startSlot_, subKernel.endSlot_, static_cast<int>(nativeStatus));
          // Don't abort — continue with Triton results
          // Restore Triton outputs since native failed
          for (auto& kv2 : tritonRawOutputs) {
            if (kv2.first < totalOutputSlots && outputSlots[kv2.first]) {
              auto* arr = outputSlots[kv2.first];
              void* sbuf = arr->specialBuffer();
              if (sbuf) {
                cudaMemcpyAsync(sbuf, kv2.second.data.data(), kv2.second.data.size(),
                                cudaMemcpyHostToDevice, static_cast<cudaStream_t>(actualStream));
                arr->dataBuffer()->writeSpecial();
              }
            }
          }
          cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));
          for (auto& kv : savedOutputs) delete kv.second;
          goto end_verify;
        }
        cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));

        markFallbackRangeDeviceCurrent(subKernel.startSlot_, subKernel.endSlot_, slots,
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
          cudaMemcpy(nativeRaw.data(), nativeArr->specialBuffer(), nativeByteLen,
                     cudaMemcpyDeviceToHost);

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
                   static_cast<int>(tritonRawOutputs.size()), overallMaxDiff, seg.executionCount);
        } else {
          DSP_DIAG(VERIFY, "TRITON VERIFY: subK[%d] [%d-%d] %d/%d MISMATCHED (maxDiff=%.8e) execCount=%d",
                   i, subKernel.startSlot_, subKernel.endSlot_,
                   mismatches, static_cast<int>(tritonRawOutputs.size()), overallMaxDiff, seg.executionCount);
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
                                cudaMemcpyHostToDevice, static_cast<cudaStream_t>(actualStream));
                arr->dataBuffer()->writeSpecial();
              }
            }
          }
          cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));
        } else {
          DSP_DIAG(VERIFY, "TRITON VERIFY: keeping NATIVE outputs subK[%d] [%d-%d] execCount=%d",
                  i, subKernel.startSlot_, subKernel.endSlot_, seg.executionCount);
        }

        for (auto& kv : savedOutputs) delete kv.second;
      } else {
        for (auto& kv : savedOutputs) delete kv.second;
      }
      end_verify:
    }
    totalKernelLaunches_++;

    if (!streamCaptureActive) {
      cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));
    }

    // Mark Triton-written outputs as device-current so downstream gap ops
    // don't overwrite fresh GPU data with stale host values via syncToDevice().
    if (!tritonSkipKernels) {
      markFallbackRangeDeviceCurrent(subKernel.startSlot_, subKernel.endSlot_, slots,
                                     externalInputs, numExternalInputs,
                                     outputSlots, totalOutputSlots);
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

  if (nextSlotToRun <= seg.endSlot) {
    if (streamCaptureActive && !Environment::getInstance().tritonAllowFallbackCapture()) {
      DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: refusing trailing gap [%d-%d] during CUDA graph capture",
               nextSlotToRun, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }

    if (!streamCaptureActive) {
      logActualityState("PRE_TRAILING_GAP", nextSlotToRun, seg.endSlot, slots,
                        outputSlots, totalOutputSlots, externalInputs, numExternalInputs);
    }
    if (!fallbackRangeExecutor_) {
      DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: missing fallback executor for trailing gap [%d-%d]",
                nextSlotToRun, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }
    auto gapStatus = fallbackRangeExecutor_(nextSlotToRun, seg.endSlot);
    if (gapStatus != Status::OK) {
      DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSegment: trailing slot-by-slot gap [%d-%d] failed with status=%d",
                nextSlotToRun, seg.endSlot, static_cast<int>(gapStatus));
      return gapStatus;
    }
    markFallbackRangeDeviceCurrent(nextSlotToRun, seg.endSlot, slots,
                                   externalInputs, numExternalInputs,
                                   outputSlots, totalOutputSlots);
    if (!streamCaptureActive) {
      logSlotHashes("TRAILING_GAP", nextSlotToRun, seg.endSlot, slots,
                    outputSlots, totalOutputSlots,
                    static_cast<cudaStream_t>(actualStream), seg.executionCount);
    }
  }

  // Compose attention present_key / present_value outputs
  int attnCount = 0;
  for (int si = seg.startSlot; si <= seg.endSlot; si++) {
    if (slots[si].opName.empty()) continue;
    bool isAttn = (slots[si].opName == "onnx_multi_head_attention" ||
                   slots[si].opName == "multi_head_attention");
    if (!isAttn) continue;
    if (slots[si].numInputs <= 4 || slots[si].numOutputs < 2) continue;

    int currentKeySrc = slots[si].inputSourceIndices[1];
    int currentValueSrc = (slots[si].numInputs > 2) ? slots[si].inputSourceIndices[2] : -1;
    int presentKeyOut = slots[si].outputSlotIndices[1];
    int presentValueOut = (slots[si].numOutputs >= 3) ? slots[si].outputSlotIndices[2] : -1;

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
                        cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(actualStream));
      }
      DSP_DIAG(EXECUTE, "composePresentKv %s: scatter %d heads x %d headDim at lastPos=%d",
               label, numHeads, headDim, lastPos);
    };

    scatterCurrentToPresent(currentKeySrc, presentKeyOut, "KEY");
    scatterCurrentToPresent(currentValueSrc, presentValueOut, "VAL");
    attnCount++;
  }

  if (!streamCaptureActive) {
    auto syncErr = cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));
    if (syncErr != cudaSuccess) {
      DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSegment: stream sync failed for [%d-%d]: %s",
                seg.startSlot, seg.endSlot, cudaGetErrorString(syncErr));
      cudaGetLastError();
      return Status::KERNEL_FAILURE;
    }
  }

  if (!streamCaptureActive && DSP_DIAG_ENABLED(VERIFY)) {
    DSP_DIAG_SEG(VERIFY, seg.startSlot,
        "seg[%d-%d] exec=%d skip=%d attn=%d",
        seg.startSlot, seg.endSlot, seg.executionCount,
        tritonSkipKernels ? 1 : 0, attnCount);
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
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey, activeDevice, compileAll, excludeOpsHash};

  std::lock_guard<std::mutex> lock(cacheMtx_);
  auto it = cache_.find(key);
  if (it == cache_.end()) {
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
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

  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    if (coveredSlots.find(s) == coveredSlots.end()) {
      gapSlots.insert(s);
    }
  }

  DSP_DIAG_SEG(EXECUTE, seg.startSlot, "NativeDSP: getGapSlots: seg[%d-%d] %d subKernels, %d covered, %d gap slots (of %d total)",
               seg.startSlot, seg.endSlot,
               static_cast<int>(it->second.subKernels.size()),
               static_cast<int>(coveredSlots.size()),
               static_cast<int>(gapSlots.size()),
               seg.endSlot - seg.startSlot + 1);

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
  for (auto& entry : cache_) {
    auto& seg = entry.second;
    // Free consolidated arg table buffers FIRST (before per-kernel cleanup,
    // because per-kernel pointers are offsets into these buffers).
    if (seg.useConsolidatedArgTable) {
      if (seg.consolidatedArgTableDevice != nullptr) {
        cudaFree(seg.consolidatedArgTableDevice);
        seg.consolidatedArgTableDevice = nullptr;
        seg.consolidatedArgTableBytes = 0;
      }
      if (seg.consolidatedArgTableHostPinned != nullptr) {
        cudaFreeHost(seg.consolidatedArgTableHostPinned);
        seg.consolidatedArgTableHostPinned = nullptr;
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
      // Only free per-kernel arg tables if NOT consolidated (consolidated
      // arg tables were freed above; per-kernel pointers are interior offsets).
      if (!seg.useConsolidatedArgTable && kernel.cachedArgTableDevice != nullptr) {
        cudaFree(kernel.cachedArgTableDevice);
        kernel.cachedArgTableDevice = nullptr;
        kernel.cachedArgTableBytes = 0;
        kernel.cachedArgTableDeviceId = -1;
      }
      if (!seg.useConsolidatedArgTable && kernel.cachedArgTableHostPinned != nullptr) {
        cudaFreeHost(kernel.cachedArgTableHostPinned);
        kernel.cachedArgTableHostPinned = nullptr;
        kernel.cachedArgTableHostPinnedBytes = 0;
      }
      if (kernel.cachedSyncCounterDevice != nullptr) {
        cudaFree(kernel.cachedSyncCounterDevice);
        kernel.cachedSyncCounterDevice = nullptr;
        kernel.cachedSyncCounterDeviceId = -1;
      }
      if (kernel.gpuModule) {
        TritonTargetDispatch::unloadModule(kernel.gpuModule);
      }
    }
  }
  cache_.clear();
  failedCache_.clear();
  lastCompilationAudit_.clear();
}

// ─── Compilation audit ──────────────────────────────────────────────────────

std::vector<CompilationAuditEntry> TritonGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

#endif // HAVE_TRITON
