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
#include <system/Environment.h>
#include <helpers/logger.h>

#include <vector>
#include <unordered_map>

namespace sd {
namespace graph {

using namespace triton_internal;

Status TritonGraphBackend::executeSingleKernel(CompiledKernel& compiled, NativeSlot* slots,
                                                NDArray** externalInputs, int numExternalInputs,
                                                NDArray** outputSlots, int totalOutputSlots,
                                                void* stream, bool argTablePreCopied,
                                                NDArray** slotArrayCache) {
  int numBufferArgs = static_cast<int>(compiled.argSlotMapping.size());
  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;

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

  cudaStream_t cudaExecStream = static_cast<cudaStream_t>(actualStream);
  int currentDevice = -1;
  auto devErr = cudaGetDevice(&currentDevice);
  if (devErr != cudaSuccess) {
    DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: cudaGetDevice failed: %s",
              cudaGetErrorString(devErr));
    return Status::KERNEL_FAILURE;
  }

  bool streamIsCapturing = false;
  if (actualStream != nullptr) {
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    auto capErr = cudaStreamIsCapturing(static_cast<cudaStream_t>(actualStream), &captureStatus);
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
      return Status::KERNEL_FAILURE;
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

  // Sync all INPUT arrays to device before resolving buffer pointers.
  // Without this, internal slot arrays whose host buffer was updated (e.g., by
  // shape ops, constant generation, or Java-side modifications during warmup/gap
  // ops) would still have stale GPU buffers.  The native SBS path calls
  // prepareSpecialUse() per-op which handles this implicitly; the Triton path
  // skips that, so we must sync explicitly here.
  //
  // We use syncToDevice() (NOT forceSync) because force-syncing would also
  // overwrite correct GPU data written by prior Triton sub-kernels with stale
  // host data.  The real fix for stale cross-segment arrays is handled by
  // invalidating device actuality after native gap ops (see executeSegment).
  int syncForcedCount = 0, syncNormalCount = 0, syncSkippedCount = 0;
  for (auto& argMapping : compiled.argSlotMapping) {
    if (argMapping.isOutput) continue;  // outputs will be written, not read
    NDArray* arr = nullptr;
    if (argMapping.slotIndex < 0) {
      int extIdx = -(argMapping.slotIndex + 1);
      if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
    } else {
      if (argMapping.slotIndex < totalOutputSlots) arr = outputSlots[argMapping.slotIndex];
    }
    if (arr && arr->lengthOf() > 0) {
      // For external inputs (negative slot indices), only force H2D sync when the
      // host buffer was actually updated (pAct=1) and device is stale (sAct=0).
      // Previously this always forced H2D, which overwrote valid GPU data (written
      // by native gap ops) with stale/zero host buffers — causing all-zero inputs
      // for ops like pow(inputs_embeds, 2) in the first Triton section.
      if (argMapping.slotIndex < 0 && arr->dataBuffer() != nullptr) {
        bool wasPrimaryActual = arr->dataBuffer()->isPrimaryActual();
        bool wasSpecialActual = arr->dataBuffer()->isSpecialActual();
        if (wasPrimaryActual && !wasSpecialActual) {
          // Host was updated by Java, device is stale — force H2D
          arr->dataBuffer()->syncToSpecial(true);
          syncForcedCount++;
          DSP_DIAG(EXECUTE, "INPUT SYNC FORCED: [%d-%d] ext slot %d (HOST_WAS_PRIMARY, DEVICE_WAS_STALE) "
                   "len=%lld buf=%p",
                   compiled.startSlot_, compiled.endSlot_, argMapping.slotIndex,
                   (long long)arr->lengthOf(), arr->specialBuffer());
        } else {
          // Device already actual — do not overwrite valid GPU data with stale host
          arr->syncToDevice();  // respects actuality flags
          syncNormalCount++;
        }
      } else {
        arr->syncToDevice();
        syncNormalCount++;
      }
    } else {
      syncSkippedCount++;
    }
  }
  DSP_DIAG(EXECUTE, "INPUT SYNC: [%d-%d] forced=%d normal=%d skipped=%d (of %d input args)",
           compiled.startSlot_, compiled.endSlot_, syncForcedCount, syncNormalCount, syncSkippedCount,
           numBufferArgs);

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
      // (seg.exec.executionCount > 2 optimization), the slot stays null. Restore
      // from slotArrayCache_ which retains the array from the warmup step.
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
      return Status::KERNEL_FAILURE;
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
          for (int inp = 0; inp < slots[si].numInputs; inp++) {
            if (slots[si].inputSourceIndices[inp] == argMapping.slotIndex) {
              DSP_DIAG(EXECUTE, "  -> consumed by slot %d op='%s' (input #%d)",
                        si, slots[si].opName.c_str(), inp);
            }
          }
        }
      }
      return Status::KERNEL_FAILURE;
    }
    void* sbuf = arr->specialBuffer();
    if (!sbuf) {
      // Zero-length arrays (e.g., unused optional attention mask inputs) have no
      // device buffer. Provide a dummy pointer so the arg table has a valid address.
      // The kernel won't actually read/write this slot since the element count is 0.
      if (arr->lengthOf() == 0) {
        sbuf = getDummyDevicePtrForDevice(currentDevice, streamIsCapturing);
        if (!sbuf) {
          DSP_DIAG(MEMORY, "TritonGraphBackend::executeSingleKernel: null specialBuffer for zero-length arg slot %d "
                    "(sub-segment [%d-%d], dtype=%d, device=%d, capturing=%d) and dummy pointer unavailable",
                    argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_,
                    static_cast<int>(arr->dataType())
                    , currentDevice, streamIsCapturing ? 1 : 0
                    );
          return Status::KERNEL_FAILURE;
        }
      } else {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: null specialBuffer for arg slot %d "
                  "(sub-segment [%d-%d], length=%lld, dtype=%d)",
                  argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_,
                  (long long)arr->lengthOf(), static_cast<int>(arr->dataType()));
        return Status::KERNEL_FAILURE;
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
        opList += slots[si].opName;
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
      if (slot.numIArgs > 0 || slot.numTArgs > 0 || slot.numBArgs > 0) {
        char iArgBuf[256] = {0};
        int ioff = 0;
        for (int a = 0; a < slot.numIArgs && a < 16 && ioff < 240; a++) {
          ioff += snprintf(iArgBuf + ioff, sizeof(iArgBuf) - ioff,
                           "%s%lld", a > 0 ? "," : "", (long long)slot.iArgs[a]);
        }
        char tArgBuf[256] = {0};
        int toff = 0;
        for (int a = 0; a < slot.numTArgs && a < 8 && toff < 240; a++) {
          toff += snprintf(tArgBuf + toff, sizeof(tArgBuf) - toff,
                           "%s%.4g", a > 0 ? "," : "", slot.tArgs[a]);
        }
        DSP_DIAG(EXECUTE, "  SLOT[%d] op='%s' inputs=%d outputs=%d "
                  "iArgs=[%s](%d) tArgs=[%s](%d) bArgs=%d identity=%d view=%d fused=%d",
                  si, slot.opName.c_str(), slot.numInputs, slot.numOutputs,
                  iArgBuf, slot.numIArgs, tArgBuf, slot.numTArgs,
                  slot.numBArgs, slot.isIdentityOp ? 1 : 0,
                  slot.isViewCapableOp ? 1 : 0, slot.inPlaceFused ? 1 : 0);
      }

      // Log input wiring for every slot
      if (slot.numInputs > 0) {
        char wireBuf[512] = {0};
        int woff = 0;
        for (int inp = 0; inp < slot.numInputs && woff < 480; inp++) {
          int srcIdx = slot.inputSourceIndices[inp];
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
  if (DSP_DIAG_ENABLED(VERIFY)) {
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

  // Debug: dump input values and slot info for the first Triton section
  if (sd::Environment::getInstance().isDebug() && compiled.startSlot_ == 347) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));
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
      int dumpCount = std::min((int)len, 8);
      int dumpBytes = dumpCount * arr->sizeOfT();
      std::vector<uint8_t> hostBuf(dumpBytes);
      auto cpyErr = cudaMemcpy(hostBuf.data(), arr->specialBuffer(), dumpBytes, cudaMemcpyDeviceToHost);
      std::string valStr;
      if (cpyErr != cudaSuccess) {
        char errBuf[128]; snprintf(errBuf, sizeof(errBuf), "(cudaMemcpy FAILED: %s)", cudaGetErrorString(cpyErr));
        valStr = errBuf;
      } else if (arr->dataType() == FLOAT32) {
        float* fp = reinterpret_cast<float*>(hostBuf.data());
        for (int e = 0; e < dumpCount; e++) {
          if (e > 0) valStr += ",";
          char buf[32]; snprintf(buf, sizeof(buf), "%.6f", fp[e]); valStr += buf;
        }
      } else if (arr->dataType() == INT64) {
        int64_t* ip = reinterpret_cast<int64_t*>(hostBuf.data());
        for (int e = 0; e < dumpCount; e++) {
          if (e > 0) valStr += ",";
          char buf[32]; snprintf(buf, sizeof(buf), "%lld", (long long)ip[e]); valStr += buf;
        }
      } else {
        valStr = "(non-f32/i64 dtype)";
      }
      DSP_DIAG(VERIFY, "INPUT DUMP: [%d-%d] arg[%d] slot=%d dtype=%d len=%lld addr=%p: %s",
               compiled.startSlot_, compiled.endSlot_,
               ai, am.slotIndex, (int)arr->dataType(), (long long)len, arr->specialBuffer(), valStr.c_str());
    }
    for (int si = compiled.startSlot_; si <= compiled.endSlot_; si++) {
      auto& slot = slots[si];
      std::string srcStr;
      for (int inp = 0; inp < slot.numInputs; inp++) {
        if (inp > 0) srcStr += ",";
        char buf[32]; snprintf(buf, sizeof(buf), "%d", slot.inputSourceIndices[inp]); srcStr += buf;
      }
      DSP_DIAG(VERIFY, "SLOT INFO: slot=%d op='%s' numInputs=%d inputSources=[%s] numOutputs=%d",
               si, slot.opName.c_str(), slot.numInputs, srcStr.c_str(), slot.numOutputs);
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
        return Status::KERNEL_FAILURE;
      }
      if (compiled.cachedArgTableDevice != nullptr) {
        auto freeErr = freeDeviceBufferAsync(compiled.cachedArgTableDevice, cudaExecStream);
        if (freeErr != cudaSuccess) {
          DSP_DIAG(MEMORY, "TritonGraphBackend: failed to free stale arg table for [%d-%d]: %s",
                    compiled.startSlot_, compiled.endSlot_, cudaGetErrorString(freeErr));
          return Status::KERNEL_FAILURE;
        }
        compiled.cachedArgTableDevice = nullptr;
        compiled.cachedArgTableBytes = 0;
        compiled.cachedArgTableDeviceId = -1;
      }
      auto allocErr = allocateDeviceBufferAsync(&compiled.cachedArgTableDevice, tableBytes, cudaExecStream);
      if (allocErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed to allocate arg table (%d bytes): %s",
                  (int)tableBytes, cudaGetErrorString(allocErr));
        return Status::KERNEL_FAILURE;
      }
      compiled.cachedArgTableBytes = tableBytes;
      compiled.cachedArgTableDeviceId = currentDevice;
    }

    // Use persistent PINNED host buffer for the arg table source.
    // CUDA graph capture records the cudaMemcpyAsync source address — if we use
    // a stack-local vector, the graph replay reads from dead stack memory → SIGSEGV.
    // Pinned memory survives across graph replays.
    if (compiled.cachedArgTableHostPinned == nullptr ||
        compiled.cachedArgTableHostPinnedBytes < tableBytes) {
      if (compiled.cachedArgTableHostPinned != nullptr) {
        cudaFreeHost(compiled.cachedArgTableHostPinned);
        compiled.cachedArgTableHostPinned = nullptr;
        compiled.cachedArgTableHostPinnedBytes = 0;
      }
      auto pinnedErr = cudaMallocHost(&compiled.cachedArgTableHostPinned, tableBytes);
      if (pinnedErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed to allocate pinned arg table host (%d bytes): %s",
                  (int)tableBytes, cudaGetErrorString(pinnedErr));
        return Status::KERNEL_FAILURE;
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
        return Status::KERNEL_FAILURE;
      }

      // Check pointer alignment (CUDA requires at least 4-byte alignment for memcpy)
      if (reinterpret_cast<uintptr_t>(argTableDevice) % 4 != 0) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend: arg table device pointer %p is misaligned for [%d-%d] "
                  "(alignment=%zu, cachedDeviceId=%d)",
                  argTableDevice, compiled.startSlot_, compiled.endSlot_,
                  reinterpret_cast<uintptr_t>(argTableDevice) % 256,
                  compiled.cachedArgTableDeviceId);
        return Status::KERNEL_FAILURE;
      }

      // Copy host → device (async on the execution stream)
      // Uses the persistent pinned host buffer — safe for CUDA graph capture/replay.
      auto memcpyErr = cudaMemcpyAsync(argTableDevice, argTableHostPinned, tableBytes,
                                       cudaMemcpyHostToDevice, cudaExecStream);
      if (memcpyErr != cudaSuccess) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend: failed to copy arg table (%d bytes) for [%d-%d]: %s "
                  "(devicePtr=%p, hostPtr=%p, cachedDeviceId=%d, currentDevice=%d, stream=%p)",
                  (int)tableBytes, compiled.startSlot_, compiled.endSlot_,
                  cudaGetErrorString(memcpyErr),
                  argTableDevice, argTableHostPinned,
                  compiled.cachedArgTableDeviceId, currentDevice, (void*)cudaExecStream);
        cudaGetLastError();  // Clear the error so subsequent operations aren't poisoned
        return Status::KERNEL_FAILURE;
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
        return Status::KERNEL_FAILURE;
      }
      auto freeErr = freeDeviceBufferAsync(compiled.cachedSyncCounterDevice, cudaExecStream);
      if (freeErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed to free stale cooperative sync counter for [%d-%d]: %s",
                  compiled.startSlot_, compiled.endSlot_, cudaGetErrorString(freeErr));
        return Status::KERNEL_FAILURE;
      }
      compiled.cachedSyncCounterDevice = nullptr;
      compiled.cachedSyncCounterDeviceId = -1;
    }
    if (compiled.cachedSyncCounterDevice == nullptr) {
      if (streamIsCapturing) {
        DSP_DIAG(EXECUTE, "TritonGraphBackend::executeSingleKernel: cooperative sync counter was not pre-allocated for captured launch [%d-%d]",
                  compiled.startSlot_, compiled.endSlot_);
        return Status::KERNEL_FAILURE;
      }
      auto allocErr = allocateDeviceBufferAsync(&compiled.cachedSyncCounterDevice, sizeof(int), cudaExecStream);
      if (allocErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed to allocate cooperative sync counter: %s",
                  cudaGetErrorString(allocErr));
        return Status::KERNEL_FAILURE;
      }
      compiled.cachedSyncCounterDeviceId = currentDevice;
    }
    syncCounterDevice = compiled.cachedSyncCounterDevice;

    auto memsetErr = cudaMemsetAsync(syncCounterDevice, 0, sizeof(int),
                                     cudaExecStream);
    if (memsetErr != cudaSuccess) {
      DSP_DIAG(EXECUTE, "TritonGraphBackend: failed to initialize cooperative sync counter: %s",
                cudaGetErrorString(memsetErr));
      return Status::KERNEL_FAILURE;
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
        return Status::KERNEL_FAILURE;
      }
      if (compiled.cachedGlobalScratchDevice != nullptr) {
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
        return Status::KERNEL_FAILURE;
      }
      compiled.cachedGlobalScratchBytes = totalScratchBytes;
      compiled.cachedGlobalScratchDeviceId = currentDevice;
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
      return Status::KERNEL_FAILURE;
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
    return Status::KERNEL_FAILURE;
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
      // Free the temporary buffer
      freeDeviceBufferAsync(ao.tempPtr, cudaExecStream);
    }
  }

  // ─── Update ND4J actuality tracking after successful Triton kernel launch ───
  //
  // The Triton kernel writes directly to GPU buffers via cuLaunchKernel, bypassing
  // ND4J's DataBuffer actuality tracking system.  Without this, ND4J doesn't know
  // that the GPU (special) buffer was written and may:
  //  1. Return stale host-side data when subsequent ops call syncToPrimary() or dup()
  //  2. Skip D2H sync because isPrimaryActual() returns true (stale host = "current")
  //  3. Overwrite correct GPU results with stale host data on next syncToSpecial()
  //
  // Native ops handle this via registerSpecialUse() / prepareSpecialUse() which call
  // writeSpecial()/readSpecial() internally.  The Triton path must do the equivalent.
  int writeSpecialCount = 0, readSpecialCount = 0;
  for (auto& argMapping : compiled.argSlotMapping) {
    NDArray* arr = nullptr;
    if (argMapping.slotIndex < 0) {
      int extIdx = -(argMapping.slotIndex + 1);
      if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
    } else {
      if (argMapping.slotIndex < totalOutputSlots) arr = outputSlots[argMapping.slotIndex];
    }
    if (arr && arr->dataBuffer() != nullptr) {
      if (argMapping.isOutput) {
        // Mark GPU buffer as authoritative — the Triton kernel just wrote to it
        arr->dataBuffer()->writeSpecial();
        writeSpecialCount++;
      } else {
        // Mark GPU buffer as read — prevents stale host→device sync from overwriting
        // input data that was consumed by this kernel
        arr->dataBuffer()->readSpecial();
        readSpecialCount++;
      }
    }
  }

  // Also mark ALL output slots in this sub-kernel's range as device-written.
  // The argSlotMapping only covers kernel arguments (externally-visible outputs),
  // but some output slots might be written by the kernel yet not appear in
  // argSlotMapping (e.g., cross-section intermediates that were merged into
  // external outputs during IR building). Missing writeSpecial() on these
  // causes subsequent native gap ops to overwrite fresh GPU data with stale
  // host data when they call prepareSpecialUse → syncToDevice.
  int rangeWriteSpecialCount = 0;
  for (int si = compiled.startSlot_; si <= compiled.endSlot_; si++) {
    for (int o = 0; o < slots[si].numOutputs; o++) {
      int outIdx = slots[si].outputSlotIndices[o];
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
        auto* db = outputSlots[outIdx]->dataBuffer();
        if (db) {
          db->writeSpecial();
          rangeWriteSpecialCount++;
        }
      }
    }
  }

  DSP_DIAG(EXECUTE, "ACTUALITY MARK: [%d-%d] argMap: %d writeSpecial, %d readSpecial; "
           "range scan: %d writeSpecial",
           compiled.startSlot_, compiled.endSlot_,
           writeSpecialCount, readSpecialCount, rangeWriteSpecialCount);

  return Status::OK;
}

// ─── Arg table refresh for CUDA graph replay ───────────────────────────────

Status TritonGraphBackend::refreshArgTablesForReplay(
    GraphSegment& seg,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* execStream) {
  int currentDevice = -1;
  cudaGetDevice(&currentDevice);

  auto& refreshEnv = Environment::getInstance();
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey, currentDevice,
                      refreshEnv.tritonCompileAll(),
                      std::hash<std::string>()(refreshEnv.tritonExcludeOps())};

  CompiledSegment* compiledSeg = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      DSP_DIAG(EXECUTE, "TritonGraphBackend::refreshArgTablesForReplay: no compiled segment for [%d-%d] "
                "(shapeKey=%lld, device=%d)",
                seg.startSlot, seg.endSlot, seg.shapeKey, currentDevice);
      return Status::KERNEL_FAILURE;
    }
    compiledSeg = &it->second;
  }

  bool useDirtyTracking = Environment::getInstance().tritonArgDirtyTracking()
                          && !compiledSeg->hasDynamicArgs.empty();

  // specialBuffer() addresses are CPU-side pointer values set during allocation
  // (cudaMallocAsync returns pointers synchronously). No stream sync needed
  // to read them — actual data ordering is handled by graph launch on cudaStr.

  int refreshedCount = 0;
  int skippedCount = 0;
  int dirtySkippedCount = 0;
  int totalChangedPtrs = 0;
  for (size_t ki = 0; ki < compiledSeg->subKernels.size(); ki++) {
    auto& subKernel = compiledSeg->subKernels[ki];
    if (!subKernel.useIndirectArgs || subKernel.cachedArgTableHostPinned == nullptr) {
      skippedCount++;
      continue;
    }

    bool isStaticByDirtyTracking = useDirtyTracking
        && ki < compiledSeg->hasDynamicArgs.size()
        && !compiledSeg->hasDynamicArgs[ki];

    auto* argTableHostPinned = static_cast<int64_t*>(subKernel.cachedArgTableHostPinned);
    int numBufferArgs = static_cast<int>(subKernel.argSlotMapping.size());

    int changedPtrs = 0;
    for (int i = 0; i < numBufferArgs; i++) {
      auto& argMapping = subKernel.argSlotMapping[i];
      NDArray* arr = nullptr;
      if (argMapping.slotIndex < 0) {
        int extIdx = -(argMapping.slotIndex + 1);
        if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
      } else {
        if (argMapping.slotIndex < totalOutputSlots) arr = outputSlots[argMapping.slotIndex];
      }

      if (arr != nullptr) {
        void* sbuf = arr->specialBuffer();
        if (sbuf != nullptr) {
          int64_t newVal = reinterpret_cast<int64_t>(sbuf);
          if (argTableHostPinned[i] != newVal) changedPtrs++;
          argTableHostPinned[i] = newVal;
        }
      }
    }
    totalChangedPtrs += changedPtrs;
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
             "(skipped %d non-indirect, %d static-only, changedPtrs=%d) for seg[%d-%d]",
             refreshedCount, skippedCount, dirtySkippedCount, totalChangedPtrs,
             seg.startSlot, seg.endSlot);
  }

  // Mark arg table stable when no pointers changed — enables fast-replay path.
  if (totalChangedPtrs == 0 && refreshedCount > 0) {
    seg.exec.argTableStable = true;
    DSP_DIAG(EXECUTE, "ARG_TABLE_STABLE: seg[%d-%d] %d sub-kernels, fast-replay enabled",
             seg.startSlot, seg.endSlot, refreshedCount);
  } else {
    seg.exec.argTableStable = false;
  }
  return Status::OK;
}

void TritonGraphBackend::copyConsolidatedArgTableToDevice(GraphSegment& seg, void* stream) {
  int currentDevice = -1;
  cudaGetDevice(&currentDevice);

  auto& refreshEnv = Environment::getInstance();
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey, currentDevice,
                      refreshEnv.tritonCompileAll(),
                      std::hash<std::string>()(refreshEnv.tritonExcludeOps())};

  CompiledSegment* compiledSeg = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      // No compiled segment - nothing to copy
      return;
    }
    compiledSeg = &it->second;
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
        static_cast<cudaStream_t>(stream));
    
    if (memcpyErr != cudaSuccess) {
      DSP_DIAG(MEMORY, "TritonGraphBackend: consolidated arg table H2D failed (%zu bytes): %s",
                compiledSeg->consolidatedArgTableBytes, cudaGetErrorString(memcpyErr));
      cudaGetLastError();
    } else {
      DSP_DIAG(EXECUTE, "TritonGraphBackend: consolidated arg table H2D: 1 copy of %zu bytes "
                   "(replaces %d per-kernel copies) for seg[%d-%d]",
                   compiledSeg->consolidatedArgTableBytes,
                   static_cast<int>(compiledSeg->subKernels.size()),
                   seg.startSlot, seg.endSlot);
    }
  }
}

}  // namespace graph
}  // namespace sd

#endif // HAVE_TRITON
