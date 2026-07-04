/* ******************************************************************************
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

//
// Unified pre-replay synchronization + staleness detection for DSP.
//
// This file consolidates the 4 divergent "sync inputs then launch graph"
// preambles that were previously spread across:
//   - platformTryFrozenFastPath      (NativeDynamicShapePlan_cuda.cu)
//   - executeSegmentWithCudaGraph    (NativeDynamicShapePlan_cudagraph.cu)
//   - compositeReplay                (NativeDynamicShapePlan_gpubackend.cu)
//
// All three paths now call performPreReplaySync() which handles:
//   1. Cross-stream ordering (default stream → DSP stream)
//   2. H2D sync for variable external inputs
//   3. D2D copy into plan-owned staging buffers
//   4. Staleness verification (when DSP diagnostics enabled)
//
// The function is idempotent: PlanExecutionContext dedup flags ensure each
// step runs at most once per execute() call, regardless of how many segments
// or paths invoke it.
//

#ifdef SD_CUDA

#include <graph/NativeDynamicShapePlan.h>
#include <graph/PlanExecutionContext.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspStreamGuard.h>
#include <helpers/DebugHelper.h>
#include <system/Environment.h>
#include <system/common.h>

#include <cuda_runtime.h>
#include <cstring>

namespace sd {
namespace graph {

// ═══════════════════════════════════════════════════════════════════════════
// Staleness detection — verifies that variable inputs flowing into graph
// replay are fresh and that staging buffers received the D2D copy.
//
// Three checks, all gated behind DSP_DIAG(VERIFY):
//
//   CHECK 1: Staging buffer content verification
//     After D2D copy, staging[i] must byte-match source ext[i] for every
//     variable input. Mismatch = D2D failed or targeted wrong buffer.
//
//   CHECK 2: Variable input mutation across steps
//     At least one variable input must differ from the previous step.
//     If ALL variable inputs are identical to last step, the decode loop
//     is feeding stale data (e.g. input_ids not updated, position_ids stuck).
//
//   CHECK 3: Staging address stability
//     Staging buffer device addresses must match what the graph captured.
//     If an address changed, the graph replays against the old address
//     (baked into the graph) and reads garbage.
//
// When a check fails with diagnostics enabled, it throws std::runtime_error
// so the failure is immediately visible rather than producing silent wrong
// output 14 steps later.
// ═══════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::verifyStagingNotStale(
    NDArray** externalArrays, NDArray** effectiveArrays,
    int numExt, void* stream, const char* diagTag) {

  if (!DSP_DIAG_ENABLED(VERIFY)) return;

  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Identify variable inputs to check
  const std::vector<int>& varIndices =
      !cachedVariableExtIndices_.empty() ? cachedVariableExtIndices_
      : variableExternalInputIndices_;

  if (varIndices.empty()) {
    DSP_DIAG(VERIFY, "%s STALENESS: no variable indices cached — "
             "externalInputIsVariable_.size()=%d, cannot verify",
             diagTag, static_cast<int>(externalInputIsVariable_.size()));
    return;
  }

  // ── CHECK 1: Staging buffer content matches source ────────────────────
  // If staging buffers exist, verify D2D actually copied the right data.
  if (effectiveExternals_ != nullptr && placeholderStagingBuffers_ != nullptr) {
    for (int idx : varIndices) {
      if (idx < 0 || idx >= numExt) continue;
      NDArray* ext = externalArrays[idx];
      NDArray* staged = effectiveArrays[idx];
      if (ext == nullptr || staged == nullptr) continue;
      if (ext == staged) continue;  // not staged, passthrough

      void* srcBuf = ext->specialBuffer();
      void* dstBuf = staged->specialBuffer();
      if (srcBuf == nullptr || dstBuf == nullptr) continue;

      size_t bytes = static_cast<size_t>(ext->lengthOf()) * ext->sizeOfT();
      if (bytes == 0) continue;

      DSP_DIAG(VERIFY, "%s STAGING_QUEUED: ext[%d] name='%s' srcBuf=%p dstBuf=%p "
               "bytes=%zu (async path: content compare skipped)",
               diagTag, idx,
               (idx < static_cast<int>(externalInputNames_.size()))
                   ? externalInputNames_[idx].c_str() : "?",
               srcBuf, dstBuf, bytes);
    }
  }

  // ── CHECK 2: Variable input mutation across steps ─────────────────────
  // Content fingerprinting requires a host-visible D2H completion point. Keep
  // DSP replay async here and rely on address-stability plus replay tests for
  // the hot path.
  if (executeCount_ >= 3) {
    DSP_DIAG(VERIFY,
             "%s MUTATION_CHECK_SKIPPED: execCount=%d, %d variable inputs "
             "(async path: device fingerprint D2H compare skipped)",
             diagTag, executeCount_, static_cast<int>(varIndices.size()));
    prevStepFingerprints_.clear();
    prevStepFingerprints_[-2] = static_cast<uint64_t>(executeCount_);
  }

  // ── CHECK 3: Staging address stability ────────────────────────────────
  // If staging buffers exist, their device addresses must not have changed
  // since the graph was captured. The CUDA graph bakes in device addresses —
  // if a staging buffer was reallocated, the graph reads from the old address.
  if (placeholderStagingBuffers_ != nullptr && !prevStagingAddresses_.empty()) {
    for (int idx : varIndices) {
      if (idx < 0 || idx >= numExt) continue;
      NDArray* staging = placeholderStagingBuffers_[idx];
      if (staging == nullptr) continue;

      void* currentAddr = staging->specialBuffer();
      auto it = prevStagingAddresses_.find(idx);
      if (it != prevStagingAddresses_.end() && it->second != currentAddr) {
        const char* name = (idx < static_cast<int>(externalInputNames_.size()))
                           ? externalInputNames_[idx].c_str() : "?";
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "STALENESS CHECK 3 FAILED: %s ext[%d] name='%s' — staging buffer "
                 "device address changed! prev=%p current=%p execCount=%d. "
                 "The CUDA graph has the old address baked in and will read garbage.",
                 diagTag, idx, name, it->second, currentAddr, executeCount_);
        DSP_DIAG(VERIFY, "%s", msg);
        THROW_EXCEPTION(msg);
      }
    }

    DSP_DIAG(VERIFY, "%s ADDR_STABLE: all staging addresses unchanged", diagTag);
  }

  // Record current staging addresses for next step's check 3
  if (placeholderStagingBuffers_ != nullptr) {
    prevStagingAddresses_.clear();
    for (int idx : varIndices) {
      if (idx < 0 || idx >= numExt) continue;
      NDArray* staging = placeholderStagingBuffers_[idx];
      if (staging == nullptr) continue;
      prevStagingAddresses_[idx] = staging->specialBuffer();
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// performPreReplaySync — unified pre-replay synchronization
//
// Handles all three sync concerns in a single call:
//   Step 1: Cross-stream sync (default stream → DSP stream via event)
//   Step 2: H2D sync variable external inputs (isPrimaryActual guard)
//   Step 3: D2D copy into plan-owned staging buffers
//   Step 4: Staleness verification (when DSP VERIFY enabled)
//
// Deduplication: all steps check PlanExecutionContext flags and are no-ops
// when already done in this execute() call. Safe to call from every replay
// path — only the first call per step does real work.
//
// PRECONDITION: activeExecutionContext() returns a valid PlanExecutionContext*.
//               DspStreamGuard is active (caller owns it).
// POSTCONDITION: execCtx->syncPhase == STAGING_DONE
//                (implies cross-stream, H2D, and D2D all complete)
//                effectiveExternals_ is up to date for this step
// ═══════════════════════════════════════════════════════════════════════════
NDArray** NativeDynamicShapePlan::performPreReplaySync(
    NDArray** externalArrays, int numExt, void* stream, const char* diagTag) {

  auto* execCtx = static_cast<PlanExecutionContext*>(activeExecutionContext());
  if (execCtx == nullptr) {
    DSP_DIAG(EXECUTE,
             "%s performPreReplaySync: NO PlanExecutionContext — falling back to "
             "prepareSpecialUse for all %d ext inputs. "
             "This path should NOT occur in production.",
             diagTag, numExt);
    std::vector<NDArray*> readList;
    readList.reserve(static_cast<size_t>(numExt));
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] != nullptr && !externalArrays[ei]->isEmpty() &&
          externalArrays[ei]->lengthOf() > 0) {
        readList.push_back(externalArrays[ei]);
      }
    }
    if (!readList.empty()) {
      NDArray::prepareSpecialUse({}, readList);
      NDArray::registerSpecialUse({}, readList);
    }
    return externalArrays;
  }

  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  ExecTarget target = execCtx->execTarget;
  bool needsCrossStream = (target == ExecTarget::GRAPH_CAPTURE ||
                           target == ExecTarget::GRAPH_REPLAY);
  bool needsStaging     = (target == ExecTarget::GRAPH_CAPTURE ||
                           target == ExecTarget::GRAPH_REPLAY);

  DSP_DIAG(EXECUTE,
           "%s performPreReplaySync: execTarget=%s syncPhase=%s",
           diagTag, execCtx->execTargetName(), execCtx->syncPhaseName());

  // ── Step 1: Cross-stream ordering ──────────────────────────────────────
  // Java's .assign() and putScalar() run on the default stream. Graph replay
  // and capture run on cudaStr (the DSP stream). Make cudaStr wait on the
  // default stream so those paths see the updated data.
  //
  // SBS_ON_LC_STREAM: ops execute on the LC stream — same as assign(). Same-
  // stream ordering is inherent, no cross-stream sync needed.
  if (needsCrossStream && !execCtx->isCrossStreamSynced()) {
    cudaStream_t defaultStream = nullptr;
    auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
    if (defaultStreamPtr != nullptr) {
      defaultStream = *defaultStreamPtr;
    }
    cudaEvent_t crossEvt = reinterpret_cast<cudaEvent_t>(execCtx->crossStreamEvent);
    if (defaultStream != nullptr && defaultStream != cudaStr &&
        crossEvt != nullptr) {
      cudaEventRecord(crossEvt, defaultStream);
      cudaStreamWaitEvent(cudaStr, crossEvt, 0);
      DSP_DIAG(STREAM_SYNC,
               "%s cross-stream sync: recordedOn=defaultStream=%p waitedOn=dspStream=%p",
               diagTag, (void*)defaultStream, (void*)cudaStr);
    } else {
      DSP_DIAG(STREAM_SYNC,
               "%s cross-stream sync SKIPPED: defaultStream=%p dspStream=%p "
               "crossStreamEvent=%p — reason: %s",
               diagTag, (void*)defaultStream, (void*)cudaStr,
               execCtx->crossStreamEvent,
               (defaultStream == nullptr) ? "defaultStream is null" :
               (defaultStream == cudaStr)  ? "same stream — no ordering needed" :
                                             "crossStreamEvent is null");
    }
    execCtx->markCrossStreamSynced();
  } else if (needsCrossStream) {
    DSP_DIAG(STREAM_SYNC,
             "%s cross-stream sync: already done (syncPhase=%s) — dedup skip",
             diagTag, execCtx->syncPhaseName());
  }
  // SBS path: skip cross-stream entirely. If the phase is still UNSYNCED,
  // advance it so the H2D step can proceed (it asserts UNSYNCED→CROSS_STREAM_DONE).
  if (!needsCrossStream && !execCtx->isCrossStreamSynced()) {
    execCtx->markCrossStreamSynced();
    DSP_DIAG(STREAM_SYNC,
             "%s cross-stream sync: SKIPPED (SBS_ON_LC_STREAM — same stream, inherent ordering)",
             diagTag);
  }

  // ── Step 2: Prepare external inputs through NDArray ownership ───────────
  // execute() owns normal H2D readiness. This path remains as a defensive
  // fallback for callers that enter replay preamble without that preparation.
  if (!execCtx->isExtInputsSynced()) {
    int prepared = 0, skipped = 0;
    std::vector<NDArray*> readList;
    readList.reserve(static_cast<size_t>(numExt));
    bool useVariableFilter = !planLifecycle_.isSlotBySlot() &&
                             !externalInputIsVariable_.empty();

    auto queueRead = [&](int idx, const char* reason) {
      if (idx < 0 || idx >= numExt || externalArrays[idx] == nullptr) return;
      NDArray* arr = externalArrays[idx];
      if (arr->isEmpty() || arr->lengthOf() == 0) {
        skipped++;
        return;
      }
      if (idx < static_cast<int>(deviceWritePending_.size()) && deviceWritePending_[idx]) {
        skipped++;
        void* rawSpecial = arr->dataBuffer() != nullptr ? arr->dataBuffer()->special() : nullptr;
        DSP_DIAG(STREAM_SYNC,
                 "%s H2D[%d] SKIPPED — deviceWritePending (JNI direct write): "
                 "name='%s' buf=%p",
                 diagTag, idx,
                 (idx < static_cast<int>(externalInputNames_.size()))
                     ? externalInputNames_[idx].c_str() : "?",
                 rawSpecial);
        return;
      }
      readList.push_back(arr);
      prepared++;
      DSP_LIFECYCLE_EVENT(executeCount_, idx, "H2D_PREPARE_QUEUED", arr);
      DSP_DIAG(STREAM_SYNC,
               "%s H2D[%d] PREPARE_QUEUED (%s): name='%s' len=%lld",
               diagTag, idx, reason,
               (idx < static_cast<int>(externalInputNames_.size()))
                   ? externalInputNames_[idx].c_str() : "?",
               (long long)arr->lengthOf());
    };

    if (useVariableFilter) {
      if (!variableIndicesCached_ && !externalInputIsVariable_.empty()) {
        variableExternalInputIndices_.clear();
        for (int i = 0; i < static_cast<int>(externalInputIsVariable_.size()); ++i) {
          if (externalInputIsVariable_[i]) {
            variableExternalInputIndices_.push_back(i);
          }
        }
        variableIndicesCached_ = true;
        DSP_DIAG(EXECUTE, "%s: cached %d variable ext input indices out of %d total",
                 diagTag, static_cast<int>(variableExternalInputIndices_.size()),
                 static_cast<int>(externalInputIsVariable_.size()));
      }

      // On the FIRST frozen execution of an externally-frozen plan (executeCount_==0,
      // meaning setPlanShapesFrozen was called before this plan ever ran), CONSTANT-type
      // external inputs have externalInputIsVariable_[i]=false and would be SKIPPED by
      // the variable-filter path below. But this is the plan's FIRST execution — its
      // CUDA graph hasn't been captured yet (phaseWarmup runs here). All inputs,
      // including CONSTANT/SOURCE_CONSTANT weight buffers (e.g. HALF proj_weight), must
      // be H2D-synced so that CUDA graph capture bakes valid device pointers and the
      // warmup slot-by-slot execution reads correct data.
      //
      // Broadcast to all inputs when executeCount_==0 (first frozen exec). The
      // broadPrepare gate in buildExternalReadList (execute.cpp ~line 2421) already
      // handles the execute()-level ExternalInputSpecialUseGuard with broadPrepare=true
      // when executeCount_<=1. This fallback ensures the defensive prereplay path is
      // consistent with that gate when reached independently (e.g. from phaseWarmup
      // dispatch before the execute-level guard has marked extInputsSynced).
      // consumeBroadPreReplaySync: a plan-cache hit handed this REPLAYING plan to a
      // NEW executor whose weight DataBuffers were rebound (marked by
      // refreshProtectedWeightBuffers). executeCount_>0 makes the first-frozen gate
      // false, and the variable-filter branch below would skip the new WEIGHT buffers
      // entirely → ops read stale device memory → batch-only wrong results
      // (testFreshInputCloseBetween[bgeEncoder][5], ~7% divergence). Broad-sync once;
      // the accessor clears the flag and logs the consumption.
      const bool weightRebindBroad = consumeBroadPreReplaySync("performPreReplaySync");
      const bool firstFrozenExec =
          (!planLifecycle_.isSlotBySlot() && (executeCount_ == 0)) || weightRebindBroad;
      if (firstFrozenExec) {
        DSP_DIAG(EXECUTE,
                 "%s H2D prepare: broad sync (executeCount_=%d weightRebind=%d) — "
                 "preparing ALL %d ext inputs (not just variable-filter) to "
                 "ensure CONSTANT weight buffers are device-synced before capture",
                 diagTag, executeCount_, weightRebindBroad ? 1 : 0, numExt);
        for (int ei = 0; ei < numExt; ei++) {
          queueRead(ei, "first-frozen-all");
        }
      } else if (!variableExternalInputIndices_.empty()) {
        for (int idx : variableExternalInputIndices_) {
          queueRead(idx, "variable-filter");
        }
        skipped = numExt - static_cast<int>(variableExternalInputIndices_.size());
        if (skipped < 0) skipped = 0;
      } else {
        for (int ei = 0; ei < numExt; ei++) {
          queueRead(ei, "no-variable-index-fallback");
        }
      }
    } else {
      DSP_DIAG(EXECUTE,
               "%s H2D prepare: no variable filter (warmup/non-frozen path) — preparing all %d ext inputs",
               diagTag, numExt);
      for (int ei = 0; ei < numExt; ei++) {
        queueRead(ei, "warmup-all");
      }
    }

    if (!readList.empty()) {
      NDArray::prepareSpecialUse({}, readList);
      NDArray::registerSpecialUse({}, readList);
    }
    DSP_DIAG(EXECUTE, "%s: H2D prepare done — prepared=%d skipped=%d total=%d useVarFilter=%s execTarget=%s",
             diagTag, prepared, skipped, numExt,
             useVariableFilter ? "YES" : "NO",
             execCtx->execTargetName());
    execCtx->markExtInputsSynced();
  } else {
    DSP_DIAG(EXECUTE,
             "%s H2D sync: already done (syncPhase=%s) — dedup skip",
             diagTag, execCtx->syncPhaseName());
  }

  // ── Step 3: D2D copy variable inputs into staging buffers ──────────────
  // Only for GRAPH_CAPTURE and GRAPH_REPLAY targets. SBS reads directly from
  // the raw external arrays on the LC stream — no staging needed.
  NDArray** result = externalArrays;
  if (needsStaging && !execCtx->isStagingBuffersSynced()) {
    if (!planLifecycle_.isSlotBySlot() && !externalInputIsVariable_.empty()) {
      NDArray** staged = ensureAndSyncStagingBuffers(externalArrays, numExt, stream);
      if (staged != nullptr) {
        // Cross-stream ordering: DSP stream → LC stream.
        // ensureAndSyncStagingBuffers enqueues D2D copies on cudaStr (DSP stream).
        // For GRAPH_REPLAY: slot execution (composite replay gaps) may run kernels
        // on the LC stream. For GRAPH_CAPTURE: the capture stream IS the DSP stream
        // so the D2D copy naturally precedes captured kernels by same-stream order.
        if (target == ExecTarget::GRAPH_REPLAY) {
          auto* lcStreamPtr = LaunchContext::defaultContext()->getCudaStream();
          cudaStream_t lcStream = (lcStreamPtr != nullptr) ? *lcStreamPtr : nullptr;
          cudaEvent_t stageEvt = reinterpret_cast<cudaEvent_t>(execCtx->crossStreamEvent);
          if (lcStream != nullptr && lcStream != cudaStr && cudaStr != nullptr) {
            if (stageEvt != nullptr) {
              cudaEventRecord(stageEvt, cudaStr);
              cudaStreamWaitEvent(lcStream, stageEvt, 0);
	      DSP_DIAG(STREAM_SYNC,
	               "%s: D2D→slot ordering: event on dspStream=%p, wait on lcStream=%p",
	               diagTag, (void*)cudaStr, (void*)lcStream);
	    } else {
	      cudaEvent_t localEvent = nullptr;
	      cudaEventCreateWithFlags(&localEvent, cudaEventDisableTiming);
	      cudaEventRecord(localEvent, cudaStr);
	      cudaStreamWaitEvent(lcStream, localEvent, 0);
	      cudaEventDestroy(localEvent);
	      DSP_DIAG(STREAM_SYNC,
	               "%s: D2D→slot ordering: local event on dspStream=%p, wait on lcStream=%p",
	               diagTag, (void*)cudaStr, (void*)lcStream);
	    }
          }
        } else if (target == ExecTarget::GRAPH_CAPTURE) {
          DSP_DIAG(STREAM_SYNC,
                   "%s: GRAPH_CAPTURE: D2D staging ordered on capture stream=%p "
                   "(no blocking stream sync)",
                   diagTag, (void*)cudaStr);
        }
        DSP_DIAG(EXECUTE, "%s: staging buffers synced for %d ext inputs — "
                 "using staged pointers (effectiveExternals_=%p) execTarget=%s",
                 diagTag, numExt, (void*)staged, execCtx->execTargetName());
        result = staged;
      } else {
        DSP_DIAG(EXECUTE,
                 "%s: ensureAndSyncStagingBuffers returned NULL — "
                 "using raw externalArrays (no staging). isSlotBySlot=%s varEmpty=%s",
                 diagTag,
                 planLifecycle_.isSlotBySlot() ? "true" : "false",
                 externalInputIsVariable_.empty() ? "true" : "false");
      }
    } else {
      DSP_DIAG(EXECUTE,
               "%s: staging D2D skipped — isSlotBySlot=%s externalInputIsVariable_.empty=%s",
               diagTag,
               planLifecycle_.isSlotBySlot() ? "true" : "false",
               externalInputIsVariable_.empty() ? "true" : "false");
    }
    execCtx->markStagingBuffersSynced();
  } else if (needsStaging && effectiveExternals_ != nullptr && !planLifecycle_.isSlotBySlot() &&
             !externalInputIsVariable_.empty()) {
    DSP_DIAG(EXECUTE,
             "%s: staging buffers already synced (syncPhase=%s) — "
             "reusing effectiveExternals_=%p (dedup skip)",
             diagTag, execCtx->syncPhaseName(), (void*)effectiveExternals_);
    result = effectiveExternals_;
  } else if (!needsStaging) {
    // SBS_ON_LC_STREAM: no staging. Advance sync phase past STAGING_DONE
    // so downstream code doesn't re-enter staging logic.
    if (!execCtx->isStagingBuffersSynced()) {
      execCtx->markStagingBuffersSynced();
    }
    DSP_DIAG(EXECUTE,
             "%s: SBS_ON_LC_STREAM — no staging, using raw ext arrays",
             diagTag);
  } else {
    DSP_DIAG(EXECUTE,
             "%s: staging D2D dedup skip (syncPhase=%s) but effectiveExternals_ "
             "not reusable — isSlotBySlot=%s varEmpty=%s effectiveExternals_=%p",
             diagTag, execCtx->syncPhaseName(),
             planLifecycle_.isSlotBySlot() ? "true" : "false",
             externalInputIsVariable_.empty() ? "true" : "false",
             (void*)effectiveExternals_);
  }

  // ── Step 4: Staleness verification ─────────────────────────────────────
  // Only run for graph targets where staging matters.
  if (needsStaging) {
    verifyStagingNotStale(externalArrays, result, numExt, stream, diagTag);
  }

  return result;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
