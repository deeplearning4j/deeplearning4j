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

// FNV-1a hash of first N bytes from a device buffer (syncs stream first).
static uint64_t deviceBufFingerprint(void* devPtr, size_t totalBytes,
                                     cudaStream_t str, size_t maxBytes = 256) {
  if (devPtr == nullptr || totalBytes == 0) return 0;
  size_t n = std::min(totalBytes, maxBytes);
  uint8_t buf[256];
  if (n > 256) n = 256;
  cudaMemcpyAsync(buf, devPtr, n, cudaMemcpyDeviceToHost, str);
  cudaStreamSynchronize(str);
  uint64_t h = 0xcbf29ce484222325ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= static_cast<uint64_t>(buf[i]);
    h *= 0x100000001b3ULL;
  }
  return h;
}

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

      // Compare first 256 bytes
      size_t cmpBytes = std::min(bytes, static_cast<size_t>(256));
      uint8_t srcSnap[256], dstSnap[256];
      cudaMemcpyAsync(srcSnap, srcBuf, cmpBytes, cudaMemcpyDeviceToHost, cudaStr);
      cudaMemcpyAsync(dstSnap, dstBuf, cmpBytes, cudaMemcpyDeviceToHost, cudaStr);
      cudaStreamSynchronize(cudaStr);

      if (std::memcmp(srcSnap, dstSnap, cmpBytes) != 0) {
        const char* name = (idx < static_cast<int>(externalInputNames_.size()))
                           ? externalInputNames_[idx].c_str() : "?";
        // Find first differing byte
        int diffAt = -1;
        for (size_t b = 0; b < cmpBytes; b++) {
          if (srcSnap[b] != dstSnap[b]) { diffAt = static_cast<int>(b); break; }
        }
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "STALENESS CHECK 1 FAILED: %s ext[%d] name='%s' — staging buffer "
                 "content does NOT match source after D2D. srcBuf=%p dstBuf=%p "
                 "bytes=%zu firstDiffByte=%d execCount=%d. "
                 "The graph will replay against stale staging data.",
                 diagTag, idx, name, srcBuf, dstBuf, bytes, diffAt, executeCount_);
        DSP_DIAG(VERIFY, "%s", msg);
        throw std::runtime_error(msg);
      }

      DSP_DIAG(VERIFY, "%s STAGING_OK: ext[%d] name='%s' srcBuf=%p dstBuf=%p "
               "bytes=%zu — content matches after D2D",
               diagTag, idx,
               (idx < static_cast<int>(externalInputNames_.size()))
                   ? externalInputNames_[idx].c_str() : "?",
               srcBuf, dstBuf, bytes);
    }
  }

  // ── CHECK 2: Variable input mutation across steps ─────────────────────
  // After warmup (execCount >= 3), at least one variable input's device data
  // should differ from the previous step in a decode loop.  If ALL inputs are
  // identical for N or more consecutive steps it indicates a stagnant loop
  // (e.g. input_ids / position_ids not advancing) and replay will produce
  // wrong output.
  //
  // Threshold kStaleThreshold=3: a single identical-step pair is tolerated
  // because vision encoder tiles can legitimately share pixel data (blank or
  // uniform regions) or identical attention masks.  A real decode-loop
  // stagnation produces dozens of unchanged steps, so threshold=3 catches it
  // while suppressing false positives from occasional duplicate vision frames.
  //
  // The per-plan consecutive-unchanged counter is stored in prevStepFingerprints_
  // under the sentinel key -1 (all real external-input indices are >= 0).
  static constexpr int kStaleThreshold = 3;

  if (executeCount_ >= 3) {
    // Sentinel key -2 stores the executeCount_ value of the execute() invocation
    // at which fingerprints were last saved.  Multiple compositeReplay calls within
    // the same execute() invocation share the same executeCount_; only the FIRST
    // call per invocation should run the staleness check and update fingerprints.
    // Subsequent calls for other segments skip to prevent false-positive
    // consecutive-unchanged accumulation across segments of a single frame
    // (e.g. vision encoder issuing N segment replays per encode() call).
    auto it2 = prevStepFingerprints_.find(-2);
    bool skipCheck2 = (it2 != prevStepFingerprints_.end() &&
                       static_cast<int>(it2->second) == executeCount_);
    if (!skipCheck2) {
      bool anyChanged = false;
      bool anyChecked = false;
      std::vector<std::pair<int, uint64_t>> currentFingerprints;

      // Read the consecutive-unchanged counter from the sentinel slot.
      uint64_t consecutiveUnchanged = 0;
      {
        auto it = prevStepFingerprints_.find(-1);
        if (it != prevStepFingerprints_.end()) consecutiveUnchanged = it->second;
      }

      for (int idx : varIndices) {
        if (idx < 0 || idx >= numExt) continue;
        NDArray* arr = effectiveArrays[idx];
        if (arr == nullptr) continue;

        void* devBuf = arr->specialBuffer();
        size_t bytes = static_cast<size_t>(arr->lengthOf()) * arr->sizeOfT();
        uint64_t fp = deviceBufFingerprint(devBuf, bytes, cudaStr);
        currentFingerprints.push_back({idx, fp});
        anyChecked = true;

        // Compare against previous step's fingerprint (ignore sentinel keys < 0).
        auto it = prevStepFingerprints_.find(idx);
        if (it != prevStepFingerprints_.end() && it->second != fp) {
          anyChanged = true;
        }
      }

      if (anyChecked) {
        // prevStepFingerprints_ is non-empty only when we have a prior step's data.
        // The sentinel key -1 also keeps it non-empty, so gate on having at least
        // one real fingerprint (any key >= 0).
        bool hasPriorStep = false;
        for (auto& kv : prevStepFingerprints_) {
          if (kv.first >= 0) { hasPriorStep = true; break; }
        }

        if (!anyChanged && hasPriorStep) {
          ++consecutiveUnchanged;

          if (static_cast<int>(consecutiveUnchanged) >= kStaleThreshold) {
            // Enough consecutive unchanged steps to be confident this is a real bug.
            std::string detail;
            for (auto& [idx, fp] : currentFingerprints) {
              const char* name = (idx < static_cast<int>(externalInputNames_.size()))
                                 ? externalInputNames_[idx].c_str() : "?";
              char buf[128];
              snprintf(buf, sizeof(buf), "  ext[%d]='%s' fp=0x%016llx (UNCHANGED)\n",
                       idx, name, static_cast<unsigned long long>(fp));
              detail += buf;
            }

            char msg[1024];
            snprintf(msg, sizeof(msg),
                     "STALENESS CHECK 2 FAILED: %s execCount=%d — ALL %d variable "
                     "inputs have been identical for %llu consecutive steps "
                     "(threshold=%d). Inputs are not being updated between steps; "
                     "graph replay will produce stale/wrong output.\n%s",
                     diagTag, executeCount_,
                     static_cast<int>(currentFingerprints.size()),
                     static_cast<unsigned long long>(consecutiveUnchanged),
                     kStaleThreshold,
                     detail.c_str());
            DSP_DIAG(VERIFY, "%s", msg);
            throw std::runtime_error(msg);
          } else {
            // Warn but tolerate: may be a vision-encoder with identical tiles.
            DSP_DIAG(VERIFY,
                     "%s MUTATION_WARN: execCount=%d, %d variable inputs unchanged "
                     "(consecutive=%llu, threshold=%d — tolerated)",
                     diagTag, executeCount_,
                     static_cast<int>(currentFingerprints.size()),
                     static_cast<unsigned long long>(consecutiveUnchanged),
                     kStaleThreshold);
          }
        } else {
          // At least one input changed — reset the counter.
          consecutiveUnchanged = 0;
          DSP_DIAG(VERIFY, "%s MUTATION_OK: execCount=%d, %d variable inputs checked, "
                   "anyChanged=%d",
                   diagTag, executeCount_, static_cast<int>(currentFingerprints.size()),
                   anyChanged ? 1 : 0);
        }
      }

      // Save fingerprints for next step; persist consecutive counter in sentinel -1.
      // Record this execute() invocation in sentinel -2 so that subsequent
      // compositeReplay calls for other segments skip this check.
      prevStepFingerprints_.clear();
      for (auto& [idx, fp] : currentFingerprints) {
        prevStepFingerprints_[idx] = fp;
      }
      if (consecutiveUnchanged > 0) {
        prevStepFingerprints_[-1] = consecutiveUnchanged;
      }
      prevStepFingerprints_[-2] = static_cast<uint64_t>(executeCount_);
    } // end if (!skipCheck2)
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
        throw std::runtime_error(msg);
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
// POSTCONDITION: execCtx->crossStreamSynced = true
//                execCtx->extInputsSynced   = true
//                execCtx->stagingBuffersSynced = true
//                effectiveExternals_ is up to date for this step
// ═══════════════════════════════════════════════════════════════════════════
NDArray** NativeDynamicShapePlan::performPreReplaySync(
    NDArray** externalArrays, int numExt, void* stream, const char* diagTag) {

  auto* execCtx = static_cast<PlanExecutionContext*>(activeExecutionContext());
  if (execCtx == nullptr) {
    // Fallback: no execution context (should not happen in production).
    // Do a basic sync and return raw externals.
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] != nullptr) {
        externalArrays[ei]->syncToDevice();
      }
    }
    return externalArrays;
  }

  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // ── Step 1: Cross-stream ordering ──────────────────────────────────────
  // Java's .assign() and putScalar() run on the default stream. Graph replay
  // launches on cudaStr (the DSP stream). Make cudaStr wait on the default
  // stream so replay sees the updated data.
  if (!execCtx->crossStreamSynced) {
    cudaStream_t defaultStream = nullptr;
    auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
    if (defaultStreamPtr != nullptr) {
      defaultStream = *defaultStreamPtr;
    }
    if (defaultStream != nullptr && defaultStream != cudaStr &&
        execCtx->crossStreamEvent != nullptr) {
      cudaEventRecord(execCtx->crossStreamEvent, defaultStream);
      cudaStreamWaitEvent(cudaStr, execCtx->crossStreamEvent, 0);
      DSP_DIAG(STREAM_SYNC,
               "%s cross-stream sync: recordedOn=defaultStream=%p waitedOn=dspStream=%p",
               diagTag, (void*)defaultStream, (void*)cudaStr);
    }
    execCtx->crossStreamSynced = true;
  }

  // ── Step 2: H2D sync variable external inputs ──────────────────────────
  // Only sync inputs classified as variable (placeholders that Java writes
  // to host). Weights are device-authoritative and never need H2D.
  // Uses isPrimaryActual() guard: only syncs when host has newer data.
  // This is critical for the native decode loop where CUDA kernels write
  // directly to device buffers — forcing H2D would clobber fresh device data.
  if (!execCtx->extInputsSynced) {
    int synced = 0, skipped = 0;
    bool useVariableFilter = !planLifecycle_.isSlotBySlot() &&
                             !externalInputIsVariable_.empty();

    if (useVariableFilter) {
      // Lazily build variable index cache if needed
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

      // Fast path: iterate only the 2-3 variable inputs
      if (!variableExternalInputIndices_.empty()) {
        for (int idx : variableExternalInputIndices_) {
          if (idx < 0 || idx >= numExt || externalArrays[idx] == nullptr) continue;
          auto* db = externalArrays[idx]->dataBuffer();
          if (db != nullptr && db->isPrimaryActual()) {
            db->syncToSpecial(true);  // Force H2D: host has newer data
          }
          synced++;
        }
        skipped = numExt - static_cast<int>(variableExternalInputIndices_.size());
        if (skipped < 0) skipped = 0;
      } else {
        // Variable filter enabled but no variable indices cached — sync all
        for (int ei = 0; ei < numExt; ei++) {
          if (externalArrays[ei] == nullptr) continue;
          auto* db = externalArrays[ei]->dataBuffer();
          if (db != nullptr && db->isPrimaryActual()) {
            db->syncToSpecial(true);
          }
          synced++;
        }
      }
    } else {
      // No variable filter (warmup / non-frozen): sync all external inputs
      for (int ei = 0; ei < numExt; ei++) {
        if (externalArrays[ei] == nullptr) continue;
        externalArrays[ei]->syncToDevice();
        synced++;
      }
    }

    DSP_DIAG(EXECUTE, "%s: H2D sync done — synced=%d skipped=%d total=%d",
             diagTag, synced, skipped, numExt);
    execCtx->extInputsSynced = true;
  }

  // ── Step 3: D2D copy variable inputs into staging buffers ──────────────
  // The CUDA graph was captured using staging buffer device addresses (stable,
  // plan-lifetime). Each step we D2D copy fresh data from the Java-side device
  // buffer (just H2D-synced above) into the staging buffer the graph reads from.
  // Without this, graph replay reads stale capture-time data.
  NDArray** result = externalArrays;
  if (!execCtx->stagingBuffersSynced) {
    if (!planLifecycle_.isSlotBySlot() && !externalInputIsVariable_.empty()) {
      NDArray** staged = ensureAndSyncStagingBuffers(externalArrays, numExt, stream);
      if (staged != nullptr) {
        DSP_DIAG(EXECUTE, "%s: staging buffers synced for %d ext inputs", diagTag, numExt);
        result = staged;
      }
    }
    execCtx->stagingBuffersSynced = true;
  } else if (effectiveExternals_ != nullptr && !planLifecycle_.isSlotBySlot() &&
             !externalInputIsVariable_.empty()) {
    // Already synced this step — reuse effectiveExternals
    result = effectiveExternals_;
  }

  // ── Step 4: Staleness verification ─────────────────────────────────────
  // When DSP VERIFY diagnostics are enabled, run all three staleness checks.
  // This catches bugs immediately at the sync point rather than 14 steps later
  // when wrong tokens finally appear.
  verifyStagingNotStale(externalArrays, result, numExt, stream, diagTag);

  return result;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
