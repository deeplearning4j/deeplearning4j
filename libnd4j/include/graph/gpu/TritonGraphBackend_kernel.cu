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
#include <execution/LaunchContext.h>
#include <helpers/DebugHelper.h>
#include <array/DataBuffer.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <graph/gpu/CapturedModuleRegistry.h>
#include <system/Environment.h>
#include <helpers/logger.h>

#include <vector>
#include <unordered_map>

namespace sd {
namespace graph {

using namespace triton_internal;

// Cached CUDA device ID for arg table operations.
// Device never changes during a replay step — caching avoids redundant
// cudaGetDevice() calls (~5-10us each) in refreshArgTablesForReplay and
// copyConsolidatedArgTableToDevice.
static thread_local int tl_cachedCudaDevice = -1;

static inline int getCachedCudaDevice() {
  if (tl_cachedCudaDevice < 0) {
    cudaGetDevice(&tl_cachedCudaDevice);
  }
  return tl_cachedCudaDevice;
}

Status TritonGraphBackend::executeSingleKernel(CompiledKernel& compiled, NativeSlot* slots,
                                                NDArray** externalInputs, int numExternalInputs,
                                                NDArray** outputSlots, int totalOutputSlots,
                                                void* stream, bool argTablePreCopied,
                                                NDArray** slotArrayCache) {
  int numBufferArgs = static_cast<int>(compiled.argSlotMapping.size());
  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;
  auto failKernel = [&](const std::string& reason) {
    std::string message = reason + " [Triton kernel " +
                          std::to_string(compiled.startSlot_) + "-" +
                          std::to_string(compiled.endSlot_) +
                          ", status=KERNEL_FAILURE (50)]";
    auto* errorRef = LaunchContext::defaultContext()->errorReference();
    errorRef->setErrorCode(static_cast<int>(Status::KERNEL_FAILURE));
    errorRef->setErrorMessage(message);
    return Status::KERNEL_FAILURE;
  };

  // ── ModuleResidencyCache reload + LRU touch ──
  // If the kernel module was evicted by a prior over-budget sweep, reload it
  // from the disk cache before we touch any of its launch state.  Touch the
  // LRU tick on every launch so eviction picks the actually-coldest module.
  //
  // We deliberately reload BEFORE the stream-capture check below: a reload
  // during initial capture is fine (the new function pointer goes into the
  // captured graph), but evicting some OTHER module mid-capture would
  // invalidate the in-flight graph.  reloadModuleIfEvicted re-registers the
  // kernel which can trigger an eviction sweep — guard that by checking
  // capture status first and skipping the reload if we're already capturing.
  if (compiled.gpuModule == nullptr) {
    bool earlyStreamIsCapturing = false;
    if (actualStream != nullptr) {
      cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
      if (cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(actualStream), &capStat) == cudaSuccess &&
          capStat != cudaStreamCaptureStatusNone) {
        earlyStreamIsCapturing = true;
      }
    }
    if (earlyStreamIsCapturing) {
      DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: cannot reload evicted kernel "
                "[%d-%d] hash=%s while stream is capturing",
                compiled.startSlot_, compiled.endSlot_, compiled.diskCacheHash.c_str());
      return failKernel("module was evicted before capture");
    }
    Status reloadStatus = reloadModuleIfEvicted(&compiled);
    if (reloadStatus != Status::OK) {
      DSP_DIAG(EXECUTE,
               "TritonGraphBackend::executeSingleKernel: reload failed for evicted "
               "kernel [%d-%d] (hash=%s)",
               compiled.startSlot_, compiled.endSlot_, compiled.diskCacheHash.c_str());
      return reloadStatus;
    }
  }
  touchModule(&compiled);

  // Clear any sticky CUDA errors left by prior sub-kernel failures.
  // Without this, a device-side error (e.g., misaligned access) from an earlier
  // kernel execution contaminates the CUDA context and causes ALL subsequent
  // operations (memcpy, launch, etc.) to report the same stale error.
  {
    cudaError_t staleErr = cudaGetLastError();
    if (staleErr != cudaSuccess) {
      DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: cleared stale CUDA error before [%d-%d]: %s",
               compiled.startSlot_, compiled.endSlot_, cudaGetErrorString(staleErr));
    }
  }

  cudaStream_t cudaExecStream = reinterpret_cast<cudaStream_t>(actualStream);
  int currentDevice = -1;
  auto devErr = cudaGetDevice(&currentDevice);
  if (devErr != cudaSuccess) {
    DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: cudaGetDevice failed: %s",
              cudaGetErrorString(devErr));
    return failKernel("cudaGetDevice failed");
  }

  bool streamIsCapturing = false;
  if (actualStream != nullptr) {
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    auto capErr = cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(actualStream), &captureStatus);
    if (capErr == cudaSuccess && captureStatus != cudaStreamCaptureStatusNone) {
      streamIsCapturing = true;
    }
  }

  // Pre-allocate output arrays for slots that don't have arrays yet.
  // In compileAll mode, non-elementwise sections (reductions, gathers, etc.) are
  // compiled into the Triton kernel instead of running via fallback. Their output
  // arrays must exist before the kernel launches since args are resolved by pointer.
  for (auto& argMapping : compiled.argSlotMapping) {
    if (!argMapping.isOutput) continue;
    if (argMapping.slotIndex < 0 || argMapping.slotIndex >= totalOutputSlots) continue;
    if (outputSlots[argMapping.slotIndex] != nullptr) continue;
    // Need to allocate — use shape/dtype from the arg mapping
    if (argMapping.shape.empty()) {
      DSP_DIAG(MEMORY, "TritonGraphBackend::executeSingleKernel: cannot pre-allocate slot %d — no shape info "
                "(sub-segment [%d-%d])",
                argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_);
      return failKernel("output shape is unavailable");
    }
    std::vector<LongType> shapeVec(argMapping.shape.begin(), argMapping.shape.end());
    auto* newArr = new NDArray('c', shapeVec, argMapping.dtype, LaunchContext::defaultContext());
    outputSlots[argMapping.slotIndex] = newArr;
    if (slotArrayCache) {
      NDArray* oldCached = slotArrayCache[argMapping.slotIndex];
      if (oldCached != nullptr && oldCached != newArr) {
        // Old cached array is being replaced by the pre-allocated one.
        // Update the cache so the release schedule can find and free newArr
        // instead of orphaning it when outputSlots is later nullified.
        DSP_DIAG(MEMORY, "PRE-ALLOC CACHE UPDATE: [%d-%d] slot %d replacing cached %p with new %p",
                 compiled.startSlot_, compiled.endSlot_, argMapping.slotIndex,
                 oldCached, newArr);
      }
      slotArrayCache[argMapping.slotIndex] = newArr;
    }
    DSP_DIAG(MEMORY, "PRE-ALLOC: [%d-%d] slot %d allocated shape=[%s] dtype=%d specialBuf=%p",
             compiled.startSlot_, compiled.endSlot_, argMapping.slotIndex,
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

  // Prepare all Triton inputs/outputs through the standard NDArray ownership
  // contract before resolving device pointers.
  std::vector<NDArray*> tritonReadList;
  std::vector<NDArray*> tritonWriteList;
  int prepareReadCount = 0, prepareWriteCount = 0, prepareSkippedCount = 0;
  for (auto& argMapping : compiled.argSlotMapping) {
    NDArray* arr = nullptr;
    if (argMapping.slotIndex < 0) {
      int extIdx = -(argMapping.slotIndex + 1);
      if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
    } else {
      if (argMapping.slotIndex < totalOutputSlots) arr = outputSlots[argMapping.slotIndex];
    }
    if (arr && arr->lengthOf() > 0) {
      if (argMapping.isOutput) {
        tritonWriteList.push_back(arr);
        prepareWriteCount++;
      } else {
        tritonReadList.push_back(arr);
        prepareReadCount++;
      }
    } else {
      prepareSkippedCount++;
    }
  }
  if (!tritonReadList.empty() || !tritonWriteList.empty()) {
    NDArray::prepareSpecialUse(tritonWriteList, tritonReadList);
  }
  DSP_DIAG(EXECUTE, "TRITON_PREPARE: [%d-%d] reads=%d writes=%d skipped=%d (of %d args)",
           compiled.startSlot_, compiled.endSlot_,
           prepareReadCount, prepareWriteCount, prepareSkippedCount, numBufferArgs);

  // Resolve all buffer pointers from the arg slot mapping
  std::vector<void*> bufferPtrs;
  bufferPtrs.reserve(numBufferArgs);

  for (auto& argMapping : compiled.argSlotMapping) {
    NDArray* arr = nullptr;
    if (argMapping.slotIndex < 0) {
      int extIdx = -(argMapping.slotIndex + 1);
      if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
    } else {
      if (argMapping.slotIndex < totalOutputSlots) arr = outputSlots[argMapping.slotIndex];
      // Fallback: gap slots (CONST_GEN, SHAPE_MANIP, etc.) may have been skipped
      // by the frozen constant optimization during gap execution, leaving
      // outputSlots_[si] null. When the pre-exec restoration is also skipped
      // (frozen replay with valid replayHandle), the slot stays null. Restore
      // from outputSlots_ which retains the array from the warmup step.
      if (!arr && slotArrayCache && argMapping.slotIndex < totalOutputSlots) {
        arr = slotArrayCache[argMapping.slotIndex];
        if (arr) {
          // Validate the cached array's DataBuffer is still alive
          auto* db = arr->dataBuffer();
          if (db != nullptr && db->isValid()) {
            outputSlots[argMapping.slotIndex] = arr;
            DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: restored arg slot %d "
                      "from slotArrayCache (sub-segment [%d-%d])",
                      argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_);
          } else {
            arr = nullptr;  // Invalid cache entry — still null
          }
        }
      }
    }

    if (!arr) {
      DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: null array for arg slot %d "
                "(sub-segment [%d-%d])",
                argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_);
      return failKernel("argument array is null");
    }
    // Validate DataBuffer before accessing specialBuffer() — Java close() may have
    // deleted the NDArray or its DataBuffer, leaving outputSlots_ with a dangling pointer.
    // Empty arrays (isEmpty=true, length=0) legitimately have no DataBuffer — they
    // represent optional/unused inputs (e.g., attention mask placeholders). Handle them
    // with a dummy pointer below (same as the zero-length specialBuffer() path).
    auto* db = arr->dataBuffer();
    if ((db == nullptr || !db->isValid()) && !arr->isEmpty() && arr->lengthOf() > 0) {
      DSP_DIAG(MEMORY, "TritonGraphBackend::executeSingleKernel: INVALID DataBuffer for arg slot %d "
                "(sub-segment [%d-%d], isOutput=%d, arr=%p, db=%p, dbValid=%d, "
                "rank=%d, length=%lld, dtype=%d, isEmpty=%d)",
                argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_,
                argMapping.isOutput ? 1 : 0, (void*)arr, (void*)db,
                db ? (db->isValid() ? 1 : 0) : -1,
                arr->rankOf(), (long long)arr->lengthOf(),
                static_cast<int>(arr->dataType()), arr->isEmpty() ? 1 : 0);
      // Log which slots in this sub-kernel consume this external input
      if (argMapping.slotIndex < 0) {
        for (int si = compiled.startSlot_; si <= compiled.endSlot_; si++) {
          for (int inp = 0; inp < slots[si].wiring.numInputs; inp++) {
            if (slots[si].wiring.inputSourceIndices[inp] == argMapping.slotIndex) {
              DSP_DIAG(EXECUTE, "  -> consumed by slot %d op='%s' (input #%d)",
                        si, slots[si].ident.opName.c_str(), inp);
            }
          }
        }
      }
      return failKernel("argument DataBuffer is invalid");
    }
    void* sbuf = arr->specialBuffer();
    if (!sbuf) {
      // Empty arrays (e.g., unused optional attention mask inputs) legitimately have
      // no device buffer. ND4J's empty scalar descriptor reports rank=0/length=1, so
      // length alone is not an emptiness test. Match the DataBuffer validation above:
      // any isEmpty() array receives the preallocated dummy pointer. The attention
      // emitter ignores these optional arguments; the pointer only satisfies the
      // kernel ABI and must be stable across CUDA graph capture/replay.
      if (arr->isEmpty() || arr->lengthOf() == 0) {
        sbuf = getDummyDevicePtrForDevice(currentDevice, streamIsCapturing);
        if (!sbuf) {
          DSP_DIAG(MEMORY, "TritonGraphBackend::executeSingleKernel: null specialBuffer for empty arg slot %d "
                    "(sub-segment [%d-%d], dtype=%d, device=%d, capturing=%d) and dummy pointer unavailable",
                    argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_,
                    static_cast<int>(arr->dataType())
                    , currentDevice, streamIsCapturing ? 1 : 0
                    );
          return failKernel("empty argument dummy pointer is unavailable");
        }
      } else {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: null specialBuffer for arg slot %d "
                  "(sub-segment [%d-%d], length=%lld, dtype=%d)",
                  argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_,
                  (long long)arr->lengthOf(), static_cast<int>(arr->dataType()));
        return failKernel("argument special buffer is null");
      }
    }
    bufferPtrs.push_back(sbuf);
  }

  // ── Comprehensive kernel execution diagnostics ──
  // Logs EVERY kernel's full state: args, shapes, pointers, dtypes, op names,
  // grid/block config, iArgs/tArgs. Zero cost when DSP_DIAG is disabled.
  if (DSP_DIAG_ENABLED(EXECUTE)) {
    // Kernel identity and launch config
    {
      // Collect op names covered by this kernel
      std::string opList;
      for (int si = compiled.startSlot_; si <= compiled.endSlot_ && si >= 0; si++) {
        if (!opList.empty()) opList += ",";
        opList += slots[si].ident.opName;
        if (opList.size() > 200) { opList += "..."; break; }
      }
      DSP_DIAG(EXECUTE, "KERNEL_EXEC: slots[%d-%d] grid=(%u,%u,%u) block=(%u,%u,%u) "
                "shmem=%u warps=%d numArgs=%d indirect=%d coop=%d ops=[%s]",
                compiled.startSlot_, compiled.endSlot_,
                compiled.gridX, compiled.gridY, compiled.gridZ,
                compiled.blockX, compiled.blockY, compiled.blockZ,
                compiled.sharedMemBytes, compiled.numWarps, numBufferArgs,
                compiled.useIndirectArgs ? 1 : 0,
                compiled.useCooperativeLaunch ? 1 : 0,
                opList.c_str());
    }

    // Every arg: slot, isOutput, pointer, length, rank, shape, dtype, op name
    for (int i = 0; i < numBufferArgs; i++) {
      auto& am = compiled.argSlotMapping[i];
      NDArray* dbgArr = nullptr;
      if (am.slotIndex < 0) {
        int ei = -(am.slotIndex + 1);
        if (ei < numExternalInputs) dbgArr = externalInputs[ei];
      } else if (am.slotIndex < totalOutputSlots) {
        dbgArr = outputSlots[am.slotIndex];
      }

      // Format shape string
      char shapeBuf[128] = {0};
      int soff = 0;
      if (dbgArr) {
        for (int d = 0; d < dbgArr->rankOf() && soff < 120; d++) {
          soff += snprintf(shapeBuf + soff, sizeof(shapeBuf) - soff,
                           "%s%lld", d > 0 ? "," : "", (long long)dbgArr->sizeAt(d));
        }
      }

      DSP_DIAG(EXECUTE, "  ARG[%d] slot=%d %s ptr=%p len=%lld rank=%d shape=[%s] "
                "dtype=%d empty=%d compiledShape=[%s]",
                i, am.slotIndex, am.isOutput ? "OUT" : "IN ",
                bufferPtrs[i],
                dbgArr ? (long long)dbgArr->lengthOf() : -1LL,
                dbgArr ? dbgArr->rankOf() : -1,
                shapeBuf,
                dbgArr ? static_cast<int>(dbgArr->dataType()) : -1,
                dbgArr ? (dbgArr->isEmpty() ? 1 : 0) : -1,
                [&]() -> std::string {
                  std::string s;
                  for (size_t d = 0; d < am.shape.size(); d++) {
                    if (d) s += ",";
                    s += std::to_string(am.shape[d]);
                  }
                  return s.empty() ? "none" : s;
                }().c_str());
    }

    // Per-slot detail: iArgs, tArgs, bArgs for each slot in the kernel range
    for (int si = compiled.startSlot_; si <= compiled.endSlot_ && si >= 0; si++) {
      auto& slot = slots[si];
      // Only log slots that have interesting args
      if (slot.args.numIArgs > 0 || slot.args.numTArgs > 0 || slot.args.numBArgs > 0) {
        char iArgBuf[256] = {0};
        int ioff = 0;
        for (int a = 0; a < slot.args.numIArgs && a < 16 && ioff < 240; a++) {
          ioff += snprintf(iArgBuf + ioff, sizeof(iArgBuf) - ioff,
                           "%s%lld", a > 0 ? "," : "", (long long)slot.args.iArgs[a]);
        }
        char tArgBuf[256] = {0};
        int toff = 0;
        for (int a = 0; a < slot.args.numTArgs && a < 8 && toff < 240; a++) {
          toff += snprintf(tArgBuf + toff, sizeof(tArgBuf) - toff,
                           "%s%.4g", a > 0 ? "," : "", slot.args.tArgs[a]);
        }
        DSP_DIAG(EXECUTE, "  SLOT[%d] op='%s' inputs=%d outputs=%d "
                  "iArgs=[%s](%d) tArgs=[%s](%d) bArgs=%d identity=%d view=%d fused=%d",
                  si, slot.ident.opName.c_str(), slot.wiring.numInputs, slot.wiring.numOutputs,
                  iArgBuf, slot.args.numIArgs, tArgBuf, slot.args.numTArgs,
                  slot.args.numBArgs, slot.isIdentityOp() ? 1 : 0,
                  slot.isViewCapableOp() ? 1 : 0, slot.isInPlaceFused() ? 1 : 0);
      }

      // Log input wiring for every slot
      if (slot.wiring.numInputs > 0) {
        char wireBuf[512] = {0};
        int woff = 0;
        for (int inp = 0; inp < slot.wiring.numInputs && woff < 480; inp++) {
          int srcIdx = slot.wiring.inputSourceIndices[inp];
          NDArray* srcArr = nullptr;
          if (srcIdx < 0) {
            int ei = -(srcIdx + 1);
            if (ei < numExternalInputs) srcArr = externalInputs[ei];
          } else if (srcIdx < totalOutputSlots) {
            srcArr = outputSlots[srcIdx];
          }
          woff += snprintf(wireBuf + woff, sizeof(wireBuf) - woff,
                           "%sin[%d]=%d(len=%lld)", inp > 0 ? " " : "",
                           inp, srcIdx,
                           srcArr ? (long long)srcArr->lengthOf() : -1LL);
        }
        DSP_DIAG(EXECUTE, "  SLOT[%d] wiring: %s", si, wireBuf);
      }
    }
  }

  // ── INT64 input buffer metadata diagnostic ──
  if (DSP_DIAG_ENABLED(VERIFY) && !streamIsCapturing) {
    for (int i = 0; i < numBufferArgs; i++) {
      auto& am = compiled.argSlotMapping[i];
      if (am.isOutput) continue;
      NDArray* dbgArr = nullptr;
      if (am.slotIndex < 0) {
        int ei = -(am.slotIndex + 1);
        if (ei < numExternalInputs) dbgArr = externalInputs[ei];
      } else if (am.slotIndex < totalOutputSlots) {
        dbgArr = outputSlots[am.slotIndex];
      }
      if (!dbgArr || dbgArr->lengthOf() == 0) continue;
      // Only dump INT64 inputs with len <= 64 (attention mask sized)
      if (dbgArr->dataType() != INT64 || dbgArr->lengthOf() > 64) continue;

      LongType len = dbgArr->lengthOf();
      DSP_DIAG(VERIFY, "INT64_INPUT_META: subK[%d-%d] ARG[%d] slot=%d len=%lld ptr=%p bytes=%zu",
               compiled.startSlot_, compiled.endSlot_, i, am.slotIndex,
               (long long)len, bufferPtrs[i], static_cast<size_t>(len) * sizeof(int64_t));
    }
  }

  // ── Buffer aliasing detection and resolution ──
  // When an output buffer pointer falls within any input buffer's address range,
  // the Triton kernel's stores can race with loads from the same memory.
  // This happens when DSP allocates an output slot's NDArray at the same address
  // as an input slot's NDArray (e.g., identity casts, in-place views).
  // Fix: allocate fresh temporary buffers for aliased outputs, run the kernel,
  // then copy the results back after the kernel completes.
  // NOTE: During CUDA graph capture, we cannot allocate temp buffers (creates
  // MemAlloc nodes that become stale on replay). The graph capture path should
  // use pre-allocated alias resolution buffers from compileSegment.
  struct AliasedOutput {
    int argIdx;         // Index in bufferPtrs/argSlotMapping
    void* origPtr;      // Original (aliased) output pointer
    void* tempPtr;      // Freshly allocated temporary buffer
    size_t bytes;       // Buffer size in bytes
    int slotIdx;        // Output slot index for post-copy
  };
  std::vector<AliasedOutput> aliasedOutputs;

  if (!streamIsCapturing) {
    // Build input address ranges
    struct InputRange {
      uintptr_t start;
      uintptr_t end;  // exclusive
      int argIdx;
    };
    std::vector<InputRange> inputRanges;
    for (int i = 0; i < numBufferArgs; i++) {
      if (compiled.argSlotMapping[i].isOutput) continue;
      auto& am = compiled.argSlotMapping[i];
      NDArray* a = nullptr;
      if (am.slotIndex < 0) {
        int ei = -(am.slotIndex + 1);
        if (ei < numExternalInputs) a = externalInputs[ei];
      } else {
        if (am.slotIndex < totalOutputSlots) a = outputSlots[am.slotIndex];
      }
      if (!a || !a->specialBuffer()) continue;
      size_t bytes = a->lengthOf() * a->sizeOfT();
      if (bytes == 0) continue;
      InputRange ir;
      ir.start = reinterpret_cast<uintptr_t>(bufferPtrs[i]);
      ir.end = ir.start + bytes;
      ir.argIdx = i;
      inputRanges.push_back(ir);
    }

    // Check each output against all input ranges
    for (int i = 0; i < numBufferArgs; i++) {
      if (!compiled.argSlotMapping[i].isOutput) continue;
      auto& am = compiled.argSlotMapping[i];
      uintptr_t outAddr = reinterpret_cast<uintptr_t>(bufferPtrs[i]);
      NDArray* outArr = nullptr;
      if (am.slotIndex >= 0 && am.slotIndex < totalOutputSlots) outArr = outputSlots[am.slotIndex];
      size_t outBytes = outArr ? (outArr->lengthOf() * outArr->sizeOfT()) : 0;
      if (outBytes == 0) continue;

      for (auto& ir : inputRanges) {
        // Check if output buffer overlaps with input buffer range
        uintptr_t outEnd = outAddr + outBytes;
        if (outAddr < ir.end && outEnd > ir.start) {
          // Overlap detected — allocate temporary buffer
          DSP_DIAG(VERIFY, "BUFFER ALIAS DETECTED: [%d-%d] output arg[%d] slot=%d addr=%p bytes=%zu "
                   "overlaps input arg[%d] slot=%d range=[%p,%p)",
                   compiled.startSlot_, compiled.endSlot_, i, am.slotIndex,
                   (void*)outAddr, outBytes,
                   ir.argIdx, compiled.argSlotMapping[ir.argIdx].slotIndex,
                   (void*)ir.start, (void*)ir.end);

          void* tempBuf = nullptr;
          auto allocErr = allocateDeviceBufferAsync(&tempBuf, outBytes, cudaExecStream);
          if (allocErr != cudaSuccess) {
            DSP_DIAG(MEMORY, "BUFFER ALIAS: failed to alloc temp buffer (%zu bytes) for slot %d: %s",
                     outBytes, am.slotIndex, cudaGetErrorString(allocErr));
            // Fall through without fix — kernel may produce wrong results but won't crash
            break;
          }

          AliasedOutput ao;
          ao.argIdx = i;
          ao.origPtr = bufferPtrs[i];
          ao.tempPtr = tempBuf;
          ao.bytes = outBytes;
          ao.slotIdx = am.slotIndex;
          aliasedOutputs.push_back(ao);

          // Redirect the kernel to use the temp buffer
          bufferPtrs[i] = tempBuf;

          DSP_DIAG(VERIFY, "BUFFER ALIAS FIX: [%d-%d] output arg[%d] slot=%d redirected to temp=%p",
                   compiled.startSlot_, compiled.endSlot_, i, am.slotIndex, tempBuf);
          break;  // One aliasing match is enough to redirect
        }
      }
    }
  }

  // Log ALL resolved buffer pointers for every sub-kernel
  // SKIP during stream capture: a->specialBuffer() can call syncToDevice() →
  // cross-stream cudaMemcpyAsync poisons the capture (same bug as CONSOL ARG TABLE).
  if (DSP_DIAG_ENABLED(VERIFY) && !streamIsCapturing) {
    for (int ai = 0; ai < numBufferArgs; ai++) {
      auto& am = compiled.argSlotMapping[ai];
      NDArray* a = nullptr;
      if (am.slotIndex < 0) {
        int ei = -(am.slotIndex + 1);
        if (ei < numExternalInputs) a = externalInputs[ei];
      } else {
        if (am.slotIndex < totalOutputSlots) a = outputSlots[am.slotIndex];
      }
      DSP_DIAG(VERIFY, "RESOLVED ARG: [%d-%d] arg[%d] slot=%d %s resolvedPtr=%p arrSpecial=%p "
               "len=%lld bytes=%zu dbPtr=%p dbBytes=%lld pAct=%d sAct=%d",
               compiled.startSlot_, compiled.endSlot_, ai, am.slotIndex,
               am.isOutput ? "OUT" : "in",
               bufferPtrs[ai],
               a ? a->specialBuffer() : nullptr,
               a ? (long long)a->lengthOf() : 0LL,
               a ? (size_t)(a->lengthOf() * a->sizeOfT()) : 0,
               a ? static_cast<void*>(a->dataBuffer()) : nullptr,
               a && a->dataBuffer() ? (long long)a->dataBuffer()->getLenInBytes() : 0LL,
               a && a->dataBuffer() ? (a->dataBuffer()->isPrimaryActual() ? 1 : 0) : -1,
               a && a->dataBuffer() ? (a->dataBuffer()->isSpecialActual() ? 1 : 0) : -1);
    }
  }

  // Debug: log input metadata and slot info for the first Triton section.
  if (sd::Environment::getInstance().isDebug() && compiled.startSlot_ == 347) {
    for (int ai = 0; ai < numBufferArgs; ai++) {
      auto& am = compiled.argSlotMapping[ai];
      if (am.isOutput) continue;
      NDArray* arr = nullptr;
      if (am.slotIndex < 0) {
        int ei = -(am.slotIndex + 1);
        if (ei < numExternalInputs) arr = externalInputs[ei];
      } else {
        if (am.slotIndex < totalOutputSlots) arr = outputSlots[am.slotIndex];
      }
      if (!arr || !arr->specialBuffer()) continue;
      size_t len = arr->lengthOf();
      if (len == 0) continue;
      DSP_DIAG(VERIFY, "INPUT META: [%d-%d] arg[%d] slot=%d dtype=%d len=%lld addr=%p bytes=%zu",
               compiled.startSlot_, compiled.endSlot_,
               ai, am.slotIndex, (int)arr->dataType(), (long long)len,
               arr->specialBuffer(), len * arr->sizeOfT());
    }
    for (int si = compiled.startSlot_; si <= compiled.endSlot_; si++) {
      auto& slot = slots[si];
      std::string srcStr;
      for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
        if (inp > 0) srcStr += ",";
        char buf[32]; snprintf(buf, sizeof(buf), "%d", slot.wiring.inputSourceIndices[inp]); srcStr += buf;
      }
      DSP_DIAG(VERIFY, "SLOT INFO: slot=%d op='%s' numInputs=%d inputSources=[%s] numOutputs=%d",
               si, slot.ident.opName.c_str(), slot.wiring.numInputs, srcStr.c_str(), slot.wiring.numOutputs);
    }
  }

  // Compute n_elements from the LARGEST output to ensure all elements are computed.
  // When a fused kernel covers multiple independent chains with different output sizes
  // (e.g., main hidden state [1,960] + RoPE frequencies [1,480]), using the first output's
  // count could under-launch threads, leaving larger outputs partially computed.
  // Per-output masks in the IR handle the reverse case (preventing overflow on smaller outputs).
  LongType nElements = 0;
  for (auto& argMapping : compiled.argSlotMapping) {
    if (argMapping.isOutput) {
      int slotIdx = argMapping.slotIndex;
      if (slotIdx >= 0 && slotIdx < totalOutputSlots && outputSlots[slotIdx]) {
        LongType len = outputSlots[slotIdx]->lengthOf();
        if (len > nElements) nElements = len;
      }
    }
  }
  int nElem32 = static_cast<int>(nElements);

  DSP_DIAG(EXECUTE, "N_ELEMENTS: [%d-%d] nElements=%lld (from largest output) numBufferArgs=%d "
           "indirect=%d cooperative=%d multiPhase=%d dynamicGrid=%d",
           compiled.startSlot_, compiled.endSlot_, (long long)nElements, numBufferArgs,
           compiled.useIndirectArgs ? 1 : 0, compiled.useCooperativeLaunch ? 1 : 0,
           compiled.useMultiPhaseLaunch ? 1 : 0, compiled.useDynamicGrid ? 1 : 0);

  // Compute grid size.
  // Only simple 1D element-wise kernels derive gridX from n_elements at launch.
  // Sectioned/matmul/attention kernels must use the IR-precomputed launch grid.
  unsigned int actualGridX = compiled.gridX;
  unsigned int actualGridY = compiled.gridY;
  unsigned int actualGridZ = compiled.gridZ;
  if (compiled.useDynamicGrid) {
    actualGridX = (nElements + compiled.blockX - 1) / compiled.blockX;
  }
  if (actualGridX == 0) actualGridX = 1;

  DSP_DIAG(EXECUTE, "LAUNCH CONFIG: [%d-%d] grid=%ux%ux%u block=%ux%ux%u sharedMem=%u "
           "warps=%d kernelFunc=%p globalScratchBytes=%d",
           compiled.startSlot_, compiled.endSlot_,
           actualGridX, actualGridY, actualGridZ,
           compiled.numWarps * 32, compiled.blockY, compiled.blockZ,
           compiled.sharedMemBytes, compiled.numWarps, compiled.kernelFunction,
           compiled.globalScratchBytes);

  // Build kernel args — either direct (each ptr is a separate arg) or indirect
  // (all ptrs packed into a device-side i64 array, kernel receives 1 pointer)
  std::vector<void*> kernelArgs;
  void* argTableDevice = nullptr;
  void* syncCounterDevice = nullptr;

  if (compiled.useIndirectArgs) {
    // Pack all buffer pointers as int64 values into a device-side array.
    // The kernel signature is:
    //   non-cooperative: @kernel(%argTable: !tt.ptr<i64>, %n_elements: i32)
    //   cooperative:     @kernel(%argTable: !tt.ptr<i64>, %n_elements: i32, %sync_counter: !tt.ptr<i32>)
    // It loads each buffer pointer from argTable[i] and casts via tt.int_to_ptr.

    // Reuse a persistent device arg table per compiled kernel.
    size_t tableBytes = numBufferArgs * sizeof(int64_t);
    bool deviceChanged = (compiled.cachedArgTableDeviceId != currentDevice);
    bool needsAlloc = deviceChanged || compiled.cachedArgTableDevice == nullptr ||
                      compiled.cachedArgTableBytes < tableBytes;
    if (needsAlloc) {
      if (streamIsCapturing) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: indirect arg table was not pre-allocated "
                  "for captured launch [%d-%d] (deviceChanged=%d, cachedPtr=%p, cachedBytes=%zu, "
                  "tableBytes=%zu, cachedDeviceId=%d, currentDevice=%d)",
                  compiled.startSlot_, compiled.endSlot_,
                  deviceChanged ? 1 : 0, compiled.cachedArgTableDevice,
                  compiled.cachedArgTableBytes, tableBytes,
                  compiled.cachedArgTableDeviceId, currentDevice);
        return failKernel("indirect argument table was not preallocated before capture");
      }
      if (compiled.cachedArgTableDevice != nullptr) {
        recordModuleFree(compiled.cachedArgTableDeviceId >= 0 ? compiled.cachedArgTableDeviceId : currentDevice,
                         compiled.cachedArgTableBytes);
        auto freeErr = freeDeviceBufferAsync(compiled.cachedArgTableDevice, cudaExecStream);
        if (freeErr != cudaSuccess) {
          DSP_DIAG(MEMORY, "TritonGraphBackend: failed to free stale arg table for [%d-%d]: %s",
                    compiled.startSlot_, compiled.endSlot_, cudaGetErrorString(freeErr));
          return failKernel("stale indirect argument table could not be freed");
        }
        compiled.cachedArgTableDevice = nullptr;
        compiled.cachedArgTableBytes = 0;
        compiled.cachedArgTableDeviceId = -1;
      }
      auto allocErr = allocateDeviceBufferAsync(&compiled.cachedArgTableDevice, tableBytes, cudaExecStream);
      if (allocErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed to allocate arg table (%d bytes): %s",
                  (int)tableBytes, cudaGetErrorString(allocErr));
        return failKernel("indirect argument table allocation failed");
      }
      compiled.cachedArgTableBytes = tableBytes;
      compiled.cachedArgTableDeviceId = currentDevice;
      recordModuleAlloc(currentDevice, tableBytes);
    }

    // Use persistent PINNED host buffer for the arg table source.
    // CUDA graph capture records the cudaMemcpyAsync source address — if we use
    // a stack-local vector, the graph replay reads from dead stack memory → SIGSEGV.
    // Pinned memory survives across graph replays.
    if (compiled.cachedArgTableHostPinned == nullptr ||
        compiled.cachedArgTableHostPinnedBytes < tableBytes) {
      auto& memPool = sd::memory::CudaMemoryPool::getInstance();
      if (compiled.cachedArgTableHostPinned != nullptr) {
        // Capture-owned blocks belong to a live captured graph — the replay
        // handle frees them at its death. Never pool-free here; just detach.
        if (!compiled.cachedArgTableCaptureOwned) {
          memPool.freePinnedHost(compiled.cachedArgTableHostPinned);
        }
        compiled.cachedArgTableHostPinned = nullptr;
        compiled.cachedArgTableHostPinnedBytes = 0;
        compiled.cachedArgTableCaptureOwned = false;
      }
      compiled.cachedArgTableHostPinned = static_cast<char*>(memPool.allocatePinnedHost(tableBytes));
      if (compiled.cachedArgTableHostPinned == nullptr) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed to allocate pinned arg table host (%d bytes) via pool",
                  (int)tableBytes);
        return failKernel("pinned argument table allocation failed");
      }
      compiled.cachedArgTableHostPinnedBytes = tableBytes;
    }

    // Write buffer pointers into the persistent pinned host buffer
    auto* argTableHostPinned = static_cast<int64_t*>(compiled.cachedArgTableHostPinned);
    for (int i = 0; i < numBufferArgs; i++) {
      argTableHostPinned[i] = reinterpret_cast<int64_t>(bufferPtrs[i]);
    }

    argTableDevice = compiled.cachedArgTableDevice;

    if (!argTablePreCopied) {
      // Per-kernel H2D copy (standard path, or when consolidated copy is not in use).
      // Each per-kernel copy creates a separate CUDA graph node during capture.
      // Validate arg table pointer before copy
      if (argTableDevice == nullptr) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend: arg table device pointer is NULL for [%d-%d] "
                  "(tableBytes=%d, cachedDeviceId=%d, currentDevice=%d)",
                  compiled.startSlot_, compiled.endSlot_,
                  (int)tableBytes, compiled.cachedArgTableDeviceId, currentDevice);
        return failKernel("indirect argument table device pointer is null");
      }

      // Check pointer alignment (CUDA requires at least 4-byte alignment for memcpy)
      if (reinterpret_cast<uintptr_t>(argTableDevice) % 4 != 0) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend: arg table device pointer %p is misaligned for [%d-%d] "
                  "(alignment=%zu, cachedDeviceId=%d)",
                  argTableDevice, compiled.startSlot_, compiled.endSlot_,
                  reinterpret_cast<uintptr_t>(argTableDevice) % 256,
                  compiled.cachedArgTableDeviceId);
        return failKernel("indirect argument table device pointer is misaligned");
      }

      // Copy host → device (async on the execution stream)
      // Uses the persistent pinned host buffer — safe for CUDA graph capture/replay.
      auto memcpyErr = cudaMemcpyAsync(argTableDevice, argTableHostPinned, tableBytes,
                                       cudaMemcpyHostToDevice, cudaExecStream);
      // Under capture this memcpy becomes a GRAPH NODE that BAKES
      // argTableHostPinned as its host source. From that moment the block's
      // lifetime must match the graph's: move destruction ownership to the
      // capture machinery (tl_capturedHostPtrs -> replay handle) and drop pool
      // bookkeeping. The block stays cached here as the per-step arg mailbox.
      // Without this, compiled-kernel teardown pool-frees the block under the
      // live graph and the next cudaGraphLaunch dies host-side (pointer-matched
      // on DspBufferAliasAccuracyTest sharedBufferViewFanout/AUTO).
      if (memcpyErr == cudaSuccess) {
        cudaStream_t execStreamCopy = cudaExecStream;
        if (DebugHelper::streamIsCapturing(&execStreamCopy)) {
          if (!compiled.cachedArgTableCaptureOwned) {
            // Non-consolidated tables are independently pool-tracked bases.
            // Consolidated tables are transferred once by executeSegment and
            // mark every interior view capture-owned before reaching here.
            auto& memPool = sd::memory::CudaMemoryPool::getInstance();
            if (!memPool.relinquishPinnedHost(compiled.cachedArgTableHostPinned)) {
              return failKernel("argument table ownership transfer failed");
            }
            tl_capturedHostPtrs.push_back(compiled.cachedArgTableHostPinned);
            compiled.cachedArgTableCaptureOwned = true;
            DSP_DIAG(EXECUTE, "TritonGraphBackend: arg table pinned %p baked into capture "
                     "for [%d-%d] — ownership -> captured graph",
                     compiled.cachedArgTableHostPinned, compiled.startSlot_, compiled.endSlot_);
          }
          // Module lifetime is independent from arg-table ownership. A
          // consolidated table marks its interior views capture-owned before
          // this call, but the captured kernel node still needs its module.
          if (!compiled.moduleCaptureOwned && compiled.gpuModule != nullptr) {
            sd::graph::modreg::retainForHandle(compiled.gpuModule);
            tl_capturedModules.push_back(compiled.gpuModule);
            compiled.moduleCaptureOwned = true;
            DSP_DIAG(EXECUTE, "TritonGraphBackend: module %p baked into capture for [%d-%d] "
                     "— ownership -> captured graph",
                     compiled.gpuModule, compiled.startSlot_, compiled.endSlot_);
          }
        }
      }
      if (memcpyErr != cudaSuccess) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend: failed to copy arg table (%d bytes) for [%d-%d]: %s "
                  "(devicePtr=%p, hostPtr=%p, cachedDeviceId=%d, currentDevice=%d, stream=%p)",
                  (int)tableBytes, compiled.startSlot_, compiled.endSlot_,
                  cudaGetErrorString(memcpyErr),
                  argTableDevice, argTableHostPinned,
                  compiled.cachedArgTableDeviceId, currentDevice, (void*)cudaExecStream);
        cudaGetLastError();  // Clear the error so subsequent operations aren't poisoned
        return failKernel("indirect argument table H2D copy failed");
      }
    }
    // When argTablePreCopied=true, the consolidated copy in executeSegment already
    // transferred the entire consolidated buffer to device. Per-kernel copy is skipped,
    // reducing ~N graph nodes to 1.

    // Kernel args: [argTablePtr, n_elements]
    kernelArgs.push_back(&argTableDevice);
    kernelArgs.push_back(&nElem32);
  } else {
    // Direct mode: each buffer pointer is a separate kernel arg + n_elements
    // cuLaunchKernel expects void** where each entry points to the actual param value.
    // bufferPtrs[i] IS the void* value; &bufferPtrs[i] is the pointer-to-pointer.
    for (int i = 0; i < numBufferArgs; i++) {
      kernelArgs.push_back(&bufferPtrs[i]);
    }
    kernelArgs.push_back(&nElem32);
  }

  if (compiled.useCooperativeLaunch) {
    bool deviceChanged = (compiled.cachedSyncCounterDeviceId != currentDevice);
    if (deviceChanged && compiled.cachedSyncCounterDevice != nullptr) {
      if (streamIsCapturing) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: cooperative sync counter device mismatch during capture [%d-%d]",
                  compiled.startSlot_, compiled.endSlot_);
        return failKernel("cooperative sync counter device changed during capture");
      }
      recordModuleFree(compiled.cachedSyncCounterDeviceId >= 0 ? compiled.cachedSyncCounterDeviceId : currentDevice,
                       sizeof(int));
      auto freeErr = freeDeviceBufferAsync(compiled.cachedSyncCounterDevice, cudaExecStream);
      if (freeErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed to free stale cooperative sync counter for [%d-%d]: %s",
                  compiled.startSlot_, compiled.endSlot_, cudaGetErrorString(freeErr));
        return failKernel("stale cooperative sync counter could not be freed");
      }
      compiled.cachedSyncCounterDevice = nullptr;
      compiled.cachedSyncCounterDeviceId = -1;
    }
    if (compiled.cachedSyncCounterDevice == nullptr) {
      if (streamIsCapturing) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: cooperative sync counter was not pre-allocated for captured launch [%d-%d]",
                  compiled.startSlot_, compiled.endSlot_);
        return failKernel("cooperative sync counter was not preallocated before capture");
      }
      auto allocErr = allocateDeviceBufferAsync(&compiled.cachedSyncCounterDevice, sizeof(int), cudaExecStream);
      if (allocErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed to allocate cooperative sync counter: %s",
                  cudaGetErrorString(allocErr));
        return failKernel("cooperative sync counter allocation failed");
      }
      compiled.cachedSyncCounterDeviceId = currentDevice;
      recordModuleAlloc(currentDevice, sizeof(int));
    }
    syncCounterDevice = compiled.cachedSyncCounterDevice;

    auto memsetErr = cudaMemsetAsync(syncCounterDevice, 0, sizeof(int),
                                     cudaExecStream);
    if (memsetErr != cudaSuccess) {
      DSP_DIAG(EXECUTE, "TritonGraphBackend: failed to initialize cooperative sync counter: %s",
                cudaGetErrorString(memsetErr));
      return failKernel("cooperative sync counter initialization failed");
    }
    // Cooperative sectioned kernels expect sync counter arg after n_elements.
    kernelArgs.push_back(&syncCounterDevice);
  }

  // Multi-phase launch: phase_id argument (set per-launch, before implicit Triton args)
  int phaseId = 0;  // Default phase 0 (overridden in multi-launch loop)
  if (compiled.useMultiPhaseLaunch) {
    kernelArgs.push_back(&phaseId);  // Will be updated per-phase in the launch loop
  }

  // ── Triton 3.6.0 implicit kernel arguments ──
  // The TritonGPUToLLVM FuncOp conversion adds 2 extra pointer arguments to every
  // kernel function (see FuncOpToLLVM.cpp, NOTE: [Additional Function Arguments]):
  //   1. global_scratch_ptr — pointer to per-program scratch memory
  //   2. profile_ptr — pointer to profiling data (unused, pass nullptr)
  // These MUST be appended to kernelArgs in the same order as the lowering adds them.
  void* globalScratchPtr = nullptr;
  void* profilePtr = nullptr;

  if (compiled.globalScratchBytes > 0) {
    // Allocate or reuse persistent global scratch buffer
    // Triton partitions scratch by program ID, so total size = scratchBytes * numPrograms
    unsigned int numPrograms = actualGridX * actualGridY * actualGridZ;
    size_t totalScratchBytes = static_cast<size_t>(compiled.globalScratchBytes) * numPrograms;
    bool deviceChanged = (compiled.cachedGlobalScratchDeviceId != currentDevice);
    bool needsAlloc = deviceChanged || compiled.cachedGlobalScratchDevice == nullptr ||
                      compiled.cachedGlobalScratchBytes < totalScratchBytes;
    if (needsAlloc) {
      if (streamIsCapturing) {
        // During CUDA graph capture, raw cudaMallocAsync/cudaFreeAsync create MemAlloc/MemFree
        // graph nodes with addresses that become stale on replay → SIGSEGV on cudaGraphLaunch.
        // Global scratch must be pre-allocated before capture (in compileSegment prealloc loop).
        DSP_DIAG(FALLBACK, "TritonGraphBackend::executeSingleKernel: global scratch needs realloc during capture "
                  "[%d-%d] (deviceChanged=%d, ptr=%p, cached=%zu, needed=%zu) — falling back",
                  compiled.startSlot_, compiled.endSlot_,
                  deviceChanged ? 1 : 0, compiled.cachedGlobalScratchDevice,
                  compiled.cachedGlobalScratchBytes, totalScratchBytes);
        return failKernel("global scratch was not preallocated before capture");
      }
      if (compiled.cachedGlobalScratchDevice != nullptr) {
        recordModuleFree(compiled.cachedGlobalScratchDeviceId >= 0 ? compiled.cachedGlobalScratchDeviceId : currentDevice,
                         compiled.cachedGlobalScratchBytes);
        freeDeviceBufferAsync(compiled.cachedGlobalScratchDevice, cudaExecStream);
        compiled.cachedGlobalScratchDevice = nullptr;
        compiled.cachedGlobalScratchBytes = 0;
        compiled.cachedGlobalScratchDeviceId = -1;
      }
      auto allocErr = allocateDeviceBufferAsync(&compiled.cachedGlobalScratchDevice,
                                                 totalScratchBytes, cudaExecStream);
      if (allocErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed to allocate global scratch (%zu bytes) for [%d-%d]: %s",
                  totalScratchBytes, compiled.startSlot_, compiled.endSlot_,
                  cudaGetErrorString(allocErr));
        return failKernel("global scratch allocation failed");
      }
      compiled.cachedGlobalScratchBytes = totalScratchBytes;
      compiled.cachedGlobalScratchDeviceId = currentDevice;
      recordModuleAlloc(currentDevice, totalScratchBytes);
    }
    globalScratchPtr = compiled.cachedGlobalScratchDevice;
  }

  kernelArgs.push_back(&globalScratchPtr);
  kernelArgs.push_back(&profilePtr);

  // Re-apply shared memory opt-in at launch time as a safety net.
  // The attribute was set during compilation, but if the CUfunction was compiled
  // on a different thread (parallel compilation pool), the attribute may not have
  // persisted to the execution context. Re-applying is cheap and prevents
  // cuLaunchKernel from failing with CUDA_ERROR_INVALID_VALUE for >48KB shared mem.
  if (compiled.sharedMemBytes > 49152u) {
    if (!configureCudaKernelSharedMemory(compiled.kernelFunction, compiled.sharedMemBytes)) {
      DSP_DIAG(COMPILE, "TritonGraphBackend::executeSingleKernel: shared memory re-configuration failed "
                "for [%d-%d] (requested=%u bytes, device=%d)",
                compiled.startSlot_, compiled.endSlot_, compiled.sharedMemBytes, currentDevice);
      return failKernel("shared-memory configuration failed");
    }
  }

  // Clear any error that might have been set by the shared memory configuration
  cudaGetLastError();

  // Launch
  bool ok;
  if (compiled.useMultiPhaseLaunch && !compiled.launchPhases.empty()) {
    // Multi-phase launch: launch the SAME kernel once per phase, with different
    // phase_id and grid size. Each kernel launch provides implicit global sync.
    ok = true;
    for (size_t p = 0; p < compiled.launchPhases.size() && ok; p++) {
      phaseId = static_cast<int>(p);  // Update phase_id in kernelArgs (points to &phaseId)
      unsigned int phaseGridX = static_cast<unsigned int>(
          std::max(1, compiled.launchPhases[p].gridX));
      ok = TritonTargetDispatch::launchKernel(
          compiled.kernelFunction,
          phaseGridX, actualGridY, actualGridZ,
          compiled.numWarps * 32,
          compiled.blockY, compiled.blockZ,
          compiled.sharedMemBytes,
          actualStream,
          kernelArgs.data(),
          static_cast<int>(kernelArgs.size()));
    }
  } else if (compiled.useCooperativeLaunch) {
    ok = TritonTargetDispatch::launchCooperativeKernel(
        compiled.kernelFunction,
        actualGridX, actualGridY, actualGridZ,
        compiled.numWarps * 32,
        compiled.blockY, compiled.blockZ,
        compiled.sharedMemBytes,
        actualStream,
        kernelArgs.data(),
        static_cast<int>(kernelArgs.size()));
  } else {
    ok = TritonTargetDispatch::launchKernel(
        compiled.kernelFunction,
        actualGridX, actualGridY, actualGridZ,
        compiled.numWarps * 32,
        compiled.blockY, compiled.blockZ,
        compiled.sharedMemBytes,
        actualStream,
        kernelArgs.data(),
        static_cast<int>(kernelArgs.size()));
  }

  if (!ok) {
    // Log detailed diagnostic info for launch failures
    int maxSharedOptIn = 0, maxSharedDefault = 0;
    cudaDeviceGetAttribute(&maxSharedOptIn, cudaDevAttrMaxSharedMemoryPerBlockOptin, currentDevice);
    cudaDeviceGetAttribute(&maxSharedDefault, cudaDevAttrMaxSharedMemoryPerBlock, currentDevice);
    DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: kernel launch failed for [%d-%d] "
              "(cooperative=%d, dynamicGrid=%d, grid=%ux%ux%u, block=%ux%ux%u, sharedMem=%u, "
              "deviceSharedDefault=%d, deviceSharedOptIn=%d, kernelFunc=%p)",
              compiled.startSlot_, compiled.endSlot_,
              compiled.useCooperativeLaunch ? 1 : 0, compiled.useDynamicGrid ? 1 : 0,
              actualGridX, actualGridY, actualGridZ,
              compiled.numWarps * 32, compiled.blockY, compiled.blockZ,
              compiled.sharedMemBytes,
              maxSharedDefault, maxSharedOptIn, compiled.kernelFunction);
    cudaGetLastError();  // Clear the error from the failed launch
    // Clean up temp buffers from aliasing fix
    for (auto& ao : aliasedOutputs) {
      freeDeviceBufferAsync(ao.tempPtr, cudaExecStream);
    }
    return failKernel("kernel launch failed");
  }

  DSP_DIAG(EXECUTE, "LAUNCH OK: [%d-%d] kernel launched successfully, marking actuality",
           compiled.startSlot_, compiled.endSlot_);

  // ── Copy back aliased output results from temp buffers ──
  // The kernel wrote to temporary buffers to avoid data races with aliased inputs.
  // Now copy the results to the original (aliased) output buffer locations.
  if (!aliasedOutputs.empty()) {
    for (auto& ao : aliasedOutputs) {
      auto copyErr = cudaMemcpyAsync(ao.origPtr, ao.tempPtr, ao.bytes,
                                      cudaMemcpyDeviceToDevice, cudaExecStream);
      if (copyErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "BUFFER ALIAS COPYBACK FAILED: slot=%d temp=%p → orig=%p bytes=%zu: %s",
                 ao.slotIdx, ao.tempPtr, ao.origPtr, ao.bytes, cudaGetErrorString(copyErr));
      } else {
        DSP_DIAG(VERIFY, "BUFFER ALIAS COPYBACK: [%d-%d] slot=%d temp=%p → orig=%p bytes=%zu",
                 compiled.startSlot_, compiled.endSlot_, ao.slotIdx, ao.tempPtr, ao.origPtr, ao.bytes);
      }
      freeDeviceBufferAsync(ao.tempPtr, cudaExecStream);
    }
  }

  if (!tritonReadList.empty() || !tritonWriteList.empty()) {
    NDArray::registerSpecialUse(tritonWriteList, tritonReadList);
  }
  DSP_DIAG(EXECUTE, "TRITON_REGISTER: [%d-%d] argMap: %d writes, %d reads",
           compiled.startSlot_, compiled.endSlot_,
           prepareWriteCount, prepareReadCount);

  return Status::OK;
}

// ─── Arg table refresh for CUDA graph replay ───────────────────────────────

Status TritonGraphBackend::refreshArgTablesForReplay(
    GraphSegment& seg,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* execStream) {

  // Phase diagnostic: detect when refresh is called despite arg table being stable.
  // This indicates the caller skipped the fast-replay gate, which is a performance bug.
  // Not a hard assert because verify mode intentionally bypasses the gate.
  if (!seg.exec.needsArgRefresh()) {
    DSP_DIAG(EXECUTE, "PHASE_WARN: refreshArgTablesForReplay called while needsArgRefresh()=false "
             "for seg[%d-%d] execCount=%d — caller should have used fast-replay path",
             seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
  }

  int currentDevice = getCachedCudaDevice();

  auto& refreshEnv = Environment::getInstance();
  SegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, seg.def.shapeKeyState.compiledShapeKey, currentDevice,
                      refreshEnv.tritonCompileAll(),
                      std::hash<std::string>()(refreshEnv.tritonExcludeOps()),
                      std::hash<std::string>()(refreshEnv.tritonIncludeTypes()),
                      refreshEnv.tritonGraphCapture(), &seg};
  CompiledSegment* compiledSeg = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    // Recover the segInternalDtypeHash via the secondary dtype index (populated at
    // compile time) since this function does not receive the slots array needed
    // for computeSegInternalDtypeHash directly.
    key.segInternalDtypeHash = lookupDtypeHash(seg.def.startSlot, seg.def.endSlot,
                                                seg.def.shapeKeyState.compiledShapeKey,
                                                currentDevice, &seg);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      compiledSeg = &it->second;
    } else {
      // Recovered hash may be stale/0 (dtypeIndex_ entry evicted between compile and replay).
      // A silent miss here would falsely mark args current and replay with STALE arg tables —
      // fall back to a loose match ignoring the dtype hash before concluding "no sub-kernels".
      compiledSeg = findCompiledSegmentAnyDtype(key);
      if (compiledSeg == nullptr) {
        compiledSeg = findCompiledSegmentForLiveSegment(key);
      }
      if (compiledSeg == nullptr) {
        // Last resort: does an entry exist for THIS SEGMENT + compiled shape key under a
        // different device? That is a device mismatch (this lookup was keyed on the
        // calling thread's cached device while compile published under the segment's
        // compile-time device), not a missing kernel. Same-arch GPUs share the binary;
        // arg-table refresh only patches host-side pointer values, so the entry's device
        // does not change which pointers get refreshed.
        for (auto& entry : cache_) {
          const auto& k = entry.first;
          if (k.startSlot == seg.def.startSlot &&
              k.endSlot == seg.def.endSlot &&
              k.shapeKey == seg.def.shapeKeyState.compiledShapeKey &&
              k.segmentInstance == &seg) {
            DSP_DIAG(EXECUTE, "TritonGraphBackend::refreshArgTablesForReplay: device mismatch for [%d-%d]: "
                     "lookup device=%d, published entry deviceId=%d — using published entry",
                     seg.def.startSlot, seg.def.endSlot, currentDevice, k.deviceId);
            compiledSeg = &entry.second;
            break;
          }
        }
      }
      if (compiledSeg == nullptr) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::refreshArgTablesForReplay: no compiled segment for [%d-%d] "
                  "(shapeKey=%lld, device=%d) → marking args current (no arg tables to refresh)",
                  seg.def.startSlot, seg.def.endSlot, seg.def.shapeKeyState.compiledShapeKey, currentDevice);
        // No Triton sub-kernels = no arg tables to refresh. Mark stable so fast replay
        // path can be used (skip iterating over all external inputs for sync checks).
        seg.exec.markArgsCurrent();
        const std::string message =
            "Triton argument-table refresh found no compiled segment for range [" +
            std::to_string(seg.def.startSlot) + "-" +
            std::to_string(seg.def.endSlot) + "], shapeKey=" +
            std::to_string(seg.def.shapeKeyState.compiledShapeKey) +
            ", lookupDevice=" + std::to_string(currentDevice) +
            " [DSP status=KERNEL_FAILURE (50)]";
        auto* errorReference = LaunchContext::defaultContext()->errorReference();
        errorReference->setErrorCode(static_cast<int>(Status::KERNEL_FAILURE));
        errorReference->setErrorMessage(message);
        return Status::KERNEL_FAILURE;
      }
    }
  }

  bool useDirtyTracking = Environment::getInstance().tritonArgDirtyTracking()
                          && compiledSeg->hasDirtyTrackingInfo();
  // specialBuffer() addresses are CPU-side pointer values set during allocation
  // (cudaMallocAsync returns pointers synchronously). No stream sync needed
  // to read them — actual data ordering is handled by graph launch on cudaStr.

  int refreshedCount = 0;
  int skippedCount = 0;
  int dirtySkippedCount = 0;
  int totalChangedPtrs = 0;
  int totalChangedInternalPtrs = 0;
  for (size_t ki = 0; ki < compiledSeg->subKernels.size(); ki++) {
    auto& subKernel = compiledSeg->subKernels[ki];
    if (!subKernel.useIndirectArgs || subKernel.cachedArgTableHostPinned == nullptr) {
      skippedCount++;
      continue;
    }

    bool isStaticByDirtyTracking = useDirtyTracking && compiledSeg->isSubKernelStatic(ki);

    auto* argTableHostPinned = static_cast<int64_t*>(subKernel.cachedArgTableHostPinned);
    int numBufferArgs = static_cast<int>(subKernel.argSlotMapping.size());

    int changedPtrs = 0;
    int changedInternalPtrs = 0;
    for (int i = 0; i < numBufferArgs; i++) {
      auto& argMapping = subKernel.argSlotMapping[i];
      NDArray* arr = nullptr;
      bool isExternal = (argMapping.slotIndex < 0);
      if (isExternal) {
        int extIdx = -(argMapping.slotIndex + 1);
        if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
      } else {
        if (argMapping.slotIndex < totalOutputSlots) {
          arr = outputSlots[argMapping.slotIndex];
        }
      }

      if (arr != nullptr) {
        void* sbuf = arr->specialBuffer();
        if (sbuf != nullptr) {
          int64_t newVal = reinterpret_cast<int64_t>(sbuf);
          if (argTableHostPinned[i] != newVal) {
            changedPtrs++;
            if (!isExternal) changedInternalPtrs++;
          }
          argTableHostPinned[i] = newVal;
        } else {
          // arr exists but specialBuffer is null — zero the slot rather than
          // leaving a stale pointer baked in from a prior plan instance.
          // Count it as changed so arg table cannot be marked stable while
          // the slot still holds a freed/foreign pointer.
          if (argTableHostPinned[i] != 0) {
            changedPtrs++;
            if (!isExternal) changedInternalPtrs++;
          }
          argTableHostPinned[i] = 0;
          DSP_DIAG(EXECUTE, "REFRESH_NULL_SBUF: seg[%d-%d] subK[%zu] arg[%d] arr=%p has null specialBuffer, zeroing slot",
                   seg.def.startSlot, seg.def.endSlot, ki, i, (void*)arr);
        }
      } else {
        // arr is null — zero the slot rather than leaving a stale pointer
        // from a prior plan instance baked in.
        if (argTableHostPinned[i] != 0) {
          changedPtrs++;
          if (!isExternal) changedInternalPtrs++;
        }
        argTableHostPinned[i] = 0;
      }
    }
    totalChangedPtrs += changedPtrs;
    totalChangedInternalPtrs += changedInternalPtrs;
    if (changedPtrs > 0) {
      if (isStaticByDirtyTracking) {
        DSP_DIAG(EXECUTE, "DIRTY_TRACK_BUG: subK[%zu] [%d-%d] classified STATIC but %d/%d ptrs changed!",
                 ki, subKernel.startSlot_, subKernel.endSlot_, changedPtrs, numBufferArgs);
      } else {
        DSP_DIAG(EXECUTE, "REFRESH: subK[%zu] [%d-%d] %d/%d ptrs changed",
                 ki, subKernel.startSlot_, subKernel.endSlot_, changedPtrs, numBufferArgs);
      }
    }
    if (isStaticByDirtyTracking) dirtySkippedCount++;
    refreshedCount++;
  }

  if (refreshedCount > 0 || dirtySkippedCount > 0) {
    DSP_DIAG(EXECUTE, "TritonGraphBackend::refreshArgTablesForReplay: refreshed %d sub-kernels "
             "(skipped %d non-indirect, %d static-only, changedPtrs=%d extChanged=%d intChanged=%d) for seg[%d-%d]",
             refreshedCount, skippedCount, dirtySkippedCount, totalChangedPtrs,
             totalChangedPtrs - totalChangedInternalPtrs, totalChangedInternalPtrs,
             seg.def.startSlot, seg.def.endSlot);
  }

  // ── TRIPWIRE: scan arg tables for NULL (0) pointer entries ──────────
  // A NULL device pointer in the arg table will cause SIGSEGV at address 0x0
  // during cudaGraphLaunch when the kernel tries to dereference it.
  // Only runs when DSP_DIAG EXECUTE is enabled — zero cost in production.
  if (DspDiagnostics::getInstance().isEnabled(DSP_DIAG_EXECUTE)) {
    int nullArgEntries = 0;
    for (size_t ki = 0; ki < compiledSeg->subKernels.size(); ki++) {
      auto& subKernel = compiledSeg->subKernels[ki];
      if (!subKernel.useIndirectArgs || subKernel.cachedArgTableHostPinned == nullptr) continue;
      auto* argTableHostPinned = static_cast<int64_t*>(subKernel.cachedArgTableHostPinned);
      int numBufferArgs = static_cast<int>(subKernel.argSlotMapping.size());
      for (int i = 0; i < numBufferArgs; i++) {
        if (argTableHostPinned[i] == 0) {
          auto& argMapping = subKernel.argSlotMapping[i];
          int slotIdx = argMapping.slotIndex;
          const char* kind = (slotIdx < 0) ? "ext" : "slot";
          int resolvedIdx = (slotIdx < 0) ? -(slotIdx + 1) : slotIdx;
          NDArray* arr = nullptr;
          if (slotIdx < 0 && resolvedIdx < numExternalInputs) {
            arr = externalInputs[resolvedIdx];
          } else if (slotIdx >= 0 && slotIdx < totalOutputSlots) {
            arr = outputSlots[slotIdx];
          }
          DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_ARGTABLE: seg[%d-%d] subK[%zu] arg[%d]=%s#%d "
                   "value=0 (NULL device ptr) arr=%p specialBuf=%p — graph will SIGSEGV!",
                   seg.def.startSlot, seg.def.endSlot, ki, i, kind, resolvedIdx,
                   (void*)arr, arr ? arr->specialBuffer() : nullptr);
          nullArgEntries++;
        }
      }
    }
    if (nullArgEntries > 0) {
      DSP_DIAG(EXECUTE, "TRIPWIRE_ARGTABLE_SUMMARY: seg[%d-%d] %d NULL arg table entries detected!",
               seg.def.startSlot, seg.def.endSlot, nullArgEntries);
    }
  }
  // ── END TRIPWIRE ───────────────────────────────────────────────────

  // Mark arg table stable ONLY when ALL pointers (internal AND external) are unchanged.
  // The fast replay path in compositeReplay skips refreshArgTablesForReplay entirely
  // when needsArgRefresh()=false. If external pointers changed but we mark stable, the
  // CUDA graph reads stale external pointers → reads freed GPU memory → heap corruption.
  // Previously this only checked internal pointers, assuming external changes were
  // "handled by the arg table refresh + D2D copy." But the fast replay path SKIPS
  // that refresh — so ALL pointer changes must prevent fast replay.
  if (totalChangedPtrs == 0 && refreshedCount > 0) {
    seg.exec.markArgsCurrent();
    DSP_DIAG(EXECUTE, "ARG_TABLE_STABLE: seg[%d-%d] %d sub-kernels, fast-replay enabled "
             "(extChanges=%d internalChanges=%d)",
             seg.def.startSlot, seg.def.endSlot, refreshedCount,
             totalChangedPtrs - totalChangedInternalPtrs, totalChangedInternalPtrs);
  } else {
    seg.exec.markArgsStale();
    if (totalChangedPtrs > 0) {
      DSP_DIAG(EXECUTE, "ARG_TABLE_UNSTABLE: seg[%d-%d] %d ptrs changed (ext=%d internal=%d)",
               seg.def.startSlot, seg.def.endSlot, totalChangedPtrs,
               totalChangedPtrs - totalChangedInternalPtrs, totalChangedInternalPtrs);
    }
  }
  return Status::OK;
}

void TritonGraphBackend::copyConsolidatedArgTableToDevice(GraphSegment& seg, void* stream) {
  int currentDevice = getCachedCudaDevice();

  // Dereference void* → cudaStream_t (stream is a pointer-to-cudaStream_t)
  cudaStream_t cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;

  auto& refreshEnv = Environment::getInstance();
  SegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, seg.def.shapeKeyState.compiledShapeKey, currentDevice,
                      refreshEnv.tritonCompileAll(),
                      std::hash<std::string>()(refreshEnv.tritonExcludeOps()),
                      std::hash<std::string>()(refreshEnv.tritonIncludeTypes()),
                      refreshEnv.tritonGraphCapture(), &seg};

  CompiledSegment* compiledSeg = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    // Recover the segInternalDtypeHash via the secondary dtype index since this
    // function does not receive outputSlots (needed for computeSegInternalDtypeHash).
    key.segInternalDtypeHash = lookupDtypeHash(seg.def.startSlot, seg.def.endSlot,
                                                seg.def.shapeKeyState.compiledShapeKey,
                                                currentDevice, &seg);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      compiledSeg = &it->second;
    } else {
      // Recovered hash may be stale/0 — a silent skip here leaves the device arg table stale.
      // Fall back to a loose match ignoring the dtype hash before concluding nothing to copy.
      compiledSeg = findCompiledSegmentAnyDtype(key);
      if (compiledSeg == nullptr) {
        compiledSeg = findCompiledSegmentForLiveSegment(key);
      }
      if (compiledSeg == nullptr) {
        // Same cross-device recovery as refreshArgTablesForReplay: an entry for THIS
        // SEGMENT + compiled shape key published under a different device is a device
        // mismatch (lookup keyed on the calling thread's cached device), not a missing
        // kernel. Skipping silently here would leave the device arg table stale.
        for (auto& entry : cache_) {
          const auto& k = entry.first;
          if (k.startSlot == seg.def.startSlot &&
              k.endSlot == seg.def.endSlot &&
              k.shapeKey == seg.def.shapeKeyState.compiledShapeKey &&
              k.segmentInstance == &seg) {
            DSP_DIAG(EXECUTE, "TritonGraphBackend::copyConsolidatedArgTableToDevice: device mismatch for [%d-%d]: "
                     "lookup device=%d, published entry deviceId=%d — using published entry",
                     seg.def.startSlot, seg.def.endSlot, currentDevice, k.deviceId);
            compiledSeg = &entry.second;
            break;
          }
        }
      }
      if (compiledSeg == nullptr) {
        // No compiled segment - nothing to copy
        return;
      }
    }
  }

  // Do consolidated H2D copy if available
  if (compiledSeg->useConsolidatedArgTable &&
      compiledSeg->consolidatedArgTableHostPinned &&
      compiledSeg->consolidatedArgTableDevice &&
      compiledSeg->consolidatedArgTableBytes > 0) {

    auto memcpyErr = cudaMemcpyAsync(
        compiledSeg->consolidatedArgTableDevice,
        compiledSeg->consolidatedArgTableHostPinned,
        compiledSeg->consolidatedArgTableBytes,
        cudaMemcpyHostToDevice,
        cudaStr);
    
    if (memcpyErr != cudaSuccess) {
      DSP_DIAG(MEMORY, "TritonGraphBackend: consolidated arg table H2D failed (%zu bytes): %s",
                compiledSeg->consolidatedArgTableBytes, cudaGetErrorString(memcpyErr));
      cudaGetLastError();
    } else {
      DSP_DIAG(EXECUTE, "TritonGraphBackend: consolidated arg table H2D: 1 copy of %zu bytes "
                   "(replaces %d per-kernel copies) for seg[%d-%d]",
                   compiledSeg->consolidatedArgTableBytes,
                   static_cast<int>(compiledSeg->subKernels.size()),
                   seg.def.startSlot, seg.def.endSlot);
      // Same capture-ownership contract as the per-kernel arg table: once a
      // capture bakes this pinned block as an H2D source, the replay handle
      // owns destruction; pool bookkeeping is dropped and teardown must skip.
      if (!compiledSeg->consolidatedArgTableCaptureOwned) {
        cudaStream_t consolStreamCopy = cudaStr;
        if (DebugHelper::streamIsCapturing(&consolStreamCopy)) {
          tl_capturedHostPtrs.push_back(compiledSeg->consolidatedArgTableHostPinned);
          sd::memory::CudaMemoryPool::getInstance().relinquishPinnedHost(
              compiledSeg->consolidatedArgTableHostPinned);
          compiledSeg->consolidatedArgTableCaptureOwned = true;
          DSP_DIAG(EXECUTE, "TritonGraphBackend: consolidated arg table pinned %p baked into "
                   "capture for seg[%d-%d] — ownership -> captured graph",
                   compiledSeg->consolidatedArgTableHostPinned,
                   seg.def.startSlot, seg.def.endSlot);
        }
      }
    }
  }
}

}  // namespace graph
}  // namespace sd

#endif // HAVE_TRITON
