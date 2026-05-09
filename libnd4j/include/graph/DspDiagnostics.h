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

#ifndef LIBND4J_DSP_DIAGNOSTICS_H
#define LIBND4J_DSP_DIAGNOSTICS_H

#include <system/common.h>
#include <system/env_functions.h>
#include <array/NDArray.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

// ─── Diagnostic categories (bitfield) ────────────────────────────────────────

enum DspDiagCategory : uint32_t {
  DSP_DIAG_COMPILE  = (1u << 0),   // Backend compilation (Triton, MLIR, NNAPI, ARM)
  DSP_DIAG_JIT      = (1u << 1),   // Kernel generation, PTX/cubin, cache hits/misses
  DSP_DIAG_EXECUTE  = (1u << 2),   // Per-step execution flow, segment dispatch
  DSP_DIAG_TIMING   = (1u << 3),   // Detailed timing breakdowns
  DSP_DIAG_MEMORY   = (1u << 4),   // Allocations, OOM, failover, pool state
  DSP_DIAG_BACKEND  = (1u << 5),   // Backend selection, device placement
  DSP_DIAG_SHAPE    = (1u << 6),   // Shape analysis, static/dynamic, frozen detection
  DSP_DIAG_SEGMENT  = (1u << 7),   // Segment building, boundaries, capturable analysis
  DSP_DIAG_FUSION   = (1u << 8),   // Op fusion, identity elimination, section merging
  DSP_DIAG_VERIFY   = (1u << 9),   // Golden comparison, output validation
  DSP_DIAG_KV_CACHE = (1u << 10),  // KV cache config, retention, scattering
  DSP_DIAG_FALLBACK = (1u << 11),  // Fallback events, error recovery
  DSP_DIAG_TRANSFER = (1u << 12),  // Device transfers (H2D, D2H, D2D)
  DSP_DIAG_EMULATED_REPLAY = (1u << 13),  // Emulated graph replay lifecycle diagnostics
  DSP_DIAG_STREAM_SYNC     = (1u << 14),  // Stream synchronization events (cudaStreamSync, event waits, ordering)
  DSP_DIAG_MULTI_DEVICE    = (1u << 15),  // Multi-device orchestration (device selection, P2P, migrations)
  DSP_DIAG_GRAPH_REPLAY    = (1u << 16),  // Graph replay phases (capture, instantiate, launch, address validation)
  DSP_DIAG_SEGMENT_BUCKETS = (1u << 17),  // Invalid segment bucket classification and gap analysis
  DSP_DIAG_LIFECYCLE       = (1u << 18),  // FrozenPlan/SegmentExecutor state transitions (build/seal/replace)

  DSP_DIAG_NONE     = 0u,
  DSP_DIAG_ALL      = 0x7FFFFu     // All 19 categories
};

// ─── Detail level ────────────────────────────────────────────────────────────

enum DspDiagLevel : int {
  DSP_LEVEL_SUMMARY  = 0,  // Category stats only (default)
  DSP_LEVEL_DETAILED = 1,  // Per-step info
  DSP_LEVEL_FULL     = 2   // Every event echoed to stderr
};

// ─── Event struct (fixed-size, ring buffer friendly) ─────────────────────────

static constexpr int DSP_DIAG_OPNAME_LEN = 32;
static constexpr int DSP_DIAG_MSG_LEN    = 512;

struct DspDiagEvent {
  uint32_t    category;                     // DspDiagCategory value
  int64_t     timestampUs;                  // Microseconds since plan start
  uint64_t    threadId;                     // std::thread::id hash
  int         slotId;                       // -1 if not slot-specific
  int         segmentId;                    // -1 if not segment-specific
  int         deviceId;                     // -1 if not device-specific
  int64_t     timingUs;                     // 0 if no timing
  char        opName[DSP_DIAG_OPNAME_LEN]; // Truncated op name
  char        message[DSP_DIAG_MSG_LEN];   // Truncated message
};

// ─── Per-category aggregate stats ────────────────────────────────────────────

static constexpr int DSP_DIAG_NUM_CATEGORIES = 19;

struct DspDiagCategoryStats {
  int64_t eventCount;
  int64_t totalTimingUs;
  int64_t minTimingUs;
  int64_t maxTimingUs;

  DspDiagCategoryStats()
      : eventCount(0), totalTimingUs(0),
        minTimingUs(std::numeric_limits<int64_t>::max()),
        maxTimingUs(0) {}
};

// ─── Ring buffer size ────────────────────────────────────────────────────────

static constexpr int DSP_DIAG_RING_SIZE = 65536;
static constexpr int DSP_DIAG_RING_MASK = DSP_DIAG_RING_SIZE - 1;

// ─── Singleton ───────────────────────────────────────────────────────────────

class SD_LIB_EXPORT DspDiagnostics {
 public:
  static DspDiagnostics& getInstance();

  // ── Configuration ──
  void setCategories(uint32_t mask);
  void enableCategories(uint32_t mask);
  void disableCategories(uint32_t mask);
  uint32_t getEnabledMask() const;
  void setLevel(DspDiagLevel level);
  DspDiagLevel getLevel() const;
  void setJsonPath(const std::string& path);

  // ── Fast-path check (inlined) ──
  // When Environment debug+verbose is ON, ALL DSP diagnostics are enabled
  // so that DSP_DIAG macros produce output alongside standard op debug logging.
  bool isEnabled(uint32_t category) const {
    if (sd::env_isDebugAndVerbose()) return true;
    return (enabledMask_.load(std::memory_order_relaxed) & category) != 0;
  }

  // ── Event recording ──
  void recordEvent(uint32_t category, int slotId, int segmentId,
                   int deviceId, const char* opName,
                   int64_t timingUs, const char* fmt, ...)
#ifdef __GNUC__
      __attribute__((format(printf, 8, 9)))
#endif
      ;

  void recordEventV(uint32_t category, int slotId, int segmentId,
                    int deviceId, const char* opName,
                    int64_t timingUs, const char* fmt, va_list args);

  // ── Plan lifecycle ──
  void beginPlanExecution(int numSlots, int numSegments);
  void endPlanExecution();
  void beginStep(int stepNumber);
  void endStep(int stepNumber);

  // ── Reports ──
  std::string generatePlanReport() const;
  std::string generateJsonReport() const;
  void printPlanReport() const;
  void flushJsonReport() const;

  // ── Misc ──
  void clear();
  void applyDspConfig();

  // ── Slot buffer diagnostics ──
  // Dumps first N float values from a device buffer via D2H copy.
  // Only fires when both the EXECUTE category is enabled AND debug mode is on.
  // Safe to call outside capture (caller must ensure stream is not capturing).
  // tag: caller-provided label (e.g. "capture-post-endCapture", "replay", "direct")
  // slotIdx: which output slot
  // devicePtr: GPU buffer to read
  // numElements: total element count of the buffer
  // sampleCount: how many leading elements to dump (clamped to numElements)
  void dumpSlotBuffer(const char* tag, int slotIdx, const void* devicePtr,
                      int64_t numElements, int sampleCount = 10);

  // ── Segment output dump (replaces copy-pasted topVal blocks) ──
  // Synchronizes the stream, D2H copies sampleCount floats from devicePtr,
  // and logs them via recordEvent with the given tag.
  void dumpSegmentOutput(const char* tag, int endSlot, const void* devicePtr,
                         int64_t numElements, int execCount, void* stream = nullptr,
                         int sampleCount = 4);

  // ── Invalid segment bucket summary ──
  // Classifies a segment's gap slots by op type and materialization behavior.
  // Emits a structured diagnostic that maps each gap range to its bucket type
  // (view-only, shape-only, materializing) and the op types involved.
  struct GapClassification {
    int startSlot;
    int endSlot;
    const char* primaryOpType;    // e.g., "reshape", "gather", "concat"
    bool isViewOnly;              // true if op produces a view (no allocation)
    bool isShapeOnly;             // true if op only computes shape/meta (no payload)
    bool wouldMaterialize;        // true if op allocates new buffer
    const char* bucketLabel;      // e.g., "simple_const_gather", "concat_ladder"
  };
  void reportSegmentBucketSummary(int segStartSlot, int segEndSlot,
                                  const GapClassification* classifications,
                                  int numClassifications,
                                  const char* combinedBucketLabel,
                                  bool isInvalidForReplay);

  // ── Array content fingerprint ──
  // Records an FNV-1a hash over the full host-side payload of arr so step-to-step
  // content drift of placeholders / external inputs / output slots is visible.
  // If two consecutive calls for the same array produce the same hash, the
  // content is identical — this is how we detect "stuck input" bugs where a
  // placeholder retains last step's bytes.
  //
  // Caller contract: must be invoked OUTSIDE CUDA graph capture. The helper
  // calls arr->syncToHost() so device data is visible on the host; inside
  // capture that sync would issue a cross-stream memcpy and poison the
  // capture (see capture-safe accessors rule).
  void fingerprintArray(const char* tag, int idx, const char* name,
                        NDArray* arr, int execCount);

  // ── External input actuality state dump ──
  // Logs pAct/sAct/bytes/addr for each external input and returns sync counts.
  struct ExtInputSyncResult {
    int synced;    // pAct=1, sAct=0 — H2D transfer needed
    int skipped;   // sAct=1 — device already current
    int total;
  };
  ExtInputSyncResult dumpExternalInputState(NDArray** externalArrays, int numExt,
                                            int execCount, int maxToDump = 5);

  // ── Address snapshot for graph replay validation ──
  // Records all device buffer addresses (outputSlots + externals) at a given
  // execution point.  Compare capture-time snapshot with replay-time snapshot
  // to detect stale addresses that the graph would read/write incorrectly.
  //
  // tag: identifies the snapshot (e.g. "capture-entry", "replay-entry")
  // Stored internally; call compareAddressSnapshots() to diff two tags.
  // Only active when EXECUTE category is enabled AND debug mode is on.
  struct AddrEntry {
    int index;       // slot index (>=0) or external index (negative encoding)
    void* addr;      // specialBuffer address
    int64_t lenBytes;
  };
  void snapshotAddresses(const char* tag, void** outputSlots, int numOutputSlots,
                         void** externalArrays, int numExternals);
  // Returns number of mismatches (0 = identical). Logs each mismatch via DSP_DIAG.
  int compareAddressSnapshots(const char* tagA, const char* tagB);
  void clearAddressSnapshots();

  // ── Configurable limits ──
  // Max execution count for verbose output dumps (0 = no limit when diag enabled)
  int diagExecLimit() const { return diagExecLimit_; }
  // Max mismatch/detail entries to log before summarizing
  int diagDetailLimit() const { return diagDetailLimit_; }
  // Specific external input index to trace (-1 = none)
  int traceExtInput() const { return traceExtInput_; }
  // Specific slot index to trace (-1 = none, read from ND4J_DSP_TRACE_SLOT)
  int traceSlot() const { return traceSlot_; }

  // Check if a given execution count is within the diagnostic dump limit.
  // Returns true when diag is enabled and either no limit is set (0) or
  // execCount <= limit.
  bool withinExecLimit(int execCount) const {
    return diagExecLimit_ == 0 || execCount <= diagExecLimit_;
  }

  // ── Category name helpers ──
  static const char* categoryName(uint32_t category);
  static int categoryIndex(uint32_t category);
  static uint32_t parseCategories(const char* str);

 private:
  DspDiagnostics();
  ~DspDiagnostics() = default;
  DspDiagnostics(const DspDiagnostics&) = delete;
  DspDiagnostics& operator=(const DspDiagnostics&) = delete;

  std::atomic<uint32_t> enabledMask_;
  std::atomic<int>      level_;

  // Ring buffer
  DspDiagEvent events_[DSP_DIAG_RING_SIZE];
  std::atomic<int64_t>  writePos_;
  mutable std::mutex    eventMutex_;

  // Per-category stats
  DspDiagCategoryStats  categoryStats_[DSP_DIAG_NUM_CATEGORIES];

  // Plan-level info
  int     planNumSlots_;
  int     planNumSegments_;
  int     stepsExecuted_;
  int64_t planStartUs_;
  int64_t planTotalUs_;
  int64_t lastStepStartUs_;

  // JSON output path
  std::string jsonPath_;

  // Configurable limits (read from DspConfig in applyDspConfig)
  int diagExecLimit_;     // 0 = no limit
  int diagDetailLimit_;   // default 20
  int traceExtInput_;     // -1 = none
  int traceSlot_;         // -1 = none

  // Address snapshots for graph replay validation
  std::unordered_map<std::string, std::vector<AddrEntry>> addrSnapshots_;
  mutable std::mutex addrMutex_;
};

}  // namespace graph
}  // namespace sd

// ─── Macro family ────────────────────────────────────────────────────────────
//
// All macros compile to nothing under __CUDA_ARCH__ (device code).
// When disabled: single atomic load + bitmask AND + predicted-not-taken branch.

#ifndef __CUDA_ARCH__

#define DSP_DIAG_ENABLED(CAT) \
  (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_##CAT))

#define DSP_DIAG(CAT, FMT, ...) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_##CAT)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_##CAT, -1, -1, -1, nullptr, 0, FMT, ##__VA_ARGS__); \
    } \
  } while (0)

#define DSP_DIAG_SLOT(CAT, SLOT, FMT, ...) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_##CAT)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_##CAT, (SLOT), -1, -1, nullptr, 0, FMT, ##__VA_ARGS__); \
    } \
  } while (0)

#define DSP_DIAG_SEG(CAT, SEG, FMT, ...) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_##CAT)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_##CAT, -1, (SEG), -1, nullptr, 0, FMT, ##__VA_ARGS__); \
    } \
  } while (0)

#define DSP_DIAG_TIMED(CAT, SEG, SLOT, OP, US, FMT, ...) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_##CAT)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_##CAT, (SLOT), (SEG), -1, (OP), (US), FMT, ##__VA_ARGS__); \
    } \
  } while (0)

#define DSP_DIAG_DEV(CAT, DEV, FMT, ...) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_##CAT)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_##CAT, -1, -1, (DEV), nullptr, 0, FMT, ##__VA_ARGS__); \
    } \
  } while (0)

// Snapshot all buffer addresses for graph replay validation.
// Gated on EXECUTE category + debug mode.
#define DSP_DIAG_SNAPSHOT_ADDRS(TAG, OUT_SLOTS, NUM_OUT, EXT_ARRAYS, NUM_EXT) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_EXECUTE) \
        && sd::env_isDebug()) { \
      sd::graph::DspDiagnostics::getInstance().snapshotAddresses( \
          (TAG), (void**)(OUT_SLOTS), (NUM_OUT), (void**)(EXT_ARRAYS), (NUM_EXT)); \
    } \
  } while (0)

#define DSP_DIAG_COMPARE_ADDRS(TAG_A, TAG_B) \
  (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_EXECUTE) \
   && sd::env_isDebug() \
   ? sd::graph::DspDiagnostics::getInstance().compareAddressSnapshots((TAG_A), (TAG_B)) : 0)

// Dump leading float values of an output slot's device buffer.
// Gated on EXECUTE category + debug mode. Caller must be outside stream capture.
#define DSP_DIAG_DUMP_SLOT(TAG, SLOT_IDX, DEV_PTR, NUM_ELEMS) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_EXECUTE) \
        && sd::env_isDebug()) { \
      sd::graph::DspDiagnostics::getInstance().dumpSlotBuffer( \
          (TAG), (SLOT_IDX), (DEV_PTR), (NUM_ELEMS)); \
    } \
  } while (0)

// Dump segment output (topVal + first N floats) — replaces copy-pasted fprintf blocks.
// Gated on EXECUTE category. Stream-synchronized D2H copy.
#define DSP_DIAG_DUMP_SEG_OUTPUT(TAG, END_SLOT, DEV_PTR, NUM_ELEMS, EXEC_COUNT, STREAM) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_EXECUTE)) { \
      sd::graph::DspDiagnostics::getInstance().dumpSegmentOutput( \
          (TAG), (END_SLOT), (DEV_PTR), (NUM_ELEMS), (EXEC_COUNT), (STREAM)); \
    } \
  } while (0)

// Dump external input actuality states and return sync/skip counts.
// Gated on EXECUTE category.
#define DSP_DIAG_DUMP_EXT_INPUTS(EXT_ARRAYS, NUM_EXT, EXEC_COUNT, RESULT_VAR) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_EXECUTE)) { \
      (RESULT_VAR) = sd::graph::DspDiagnostics::getInstance().dumpExternalInputState( \
          (EXT_ARRAYS), (NUM_EXT), (EXEC_COUNT)); \
    } \
  } while (0)

// Record a content fingerprint (FNV-1a hash) for an NDArray so repeated-content
// bugs (stuck placeholders, non-updating KV caches) are visible across steps.
// Gated on EXECUTE category. Must be called outside stream capture.
#define DSP_DIAG_FINGERPRINT(TAG, IDX, NAME, ARR, EXEC_COUNT) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_EXECUTE)) { \
      sd::graph::DspDiagnostics::getInstance().fingerprintArray( \
          (TAG), (IDX), (NAME), (ARR), (EXEC_COUNT)); \
    } \
  } while (0)

// Log a slot WRITE event: who wrote to SLOT_IDX, when, on what STREAM, during PHASE.
// Gated on MEMORY category. Emits a structured DSP_DIAG_SLOT row that records the
// code-path tag (e.g. "fast-frozen", "fused-chain-head", "alloc-output") alongside
// the op name, byte count, and stream pointer.  Used at every slot write site in
// the slot executor so we can reconstruct which code path touched a slot.
#define DSP_DIAG_SLOT_WRITE(SLOT_IDX, OP, BYTES, STREAM, PHASE) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_MEMORY)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_MEMORY, (SLOT_IDX), -1, -1, (OP), 0, \
          "SLOT_WRITE slot=%d op=%s bytes=%lld stream=%p phase=%s", \
          (int)(SLOT_IDX), (OP) ? (OP) : "?", (long long)(BYTES), \
          (void*)(STREAM), (PHASE) ? (PHASE) : "?"); \
    } \
  } while (0)

// Log a slot ZERO event: who zeroed SLOT_IDX and WHY (batch-zero / prezero / nullify).
// Gated on MEMORY category.  Emits the reason string so "output zeroed then written"
// races can be traced back to the exact zeroing site.
#define DSP_DIAG_SLOT_ZERO(SLOT_IDX, REASON, STREAM, PHASE) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_MEMORY)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_MEMORY, (SLOT_IDX), -1, -1, nullptr, 0, \
          "SLOT_ZERO slot=%d reason=%s stream=%p phase=%s", \
          (int)(SLOT_IDX), (REASON) ? (REASON) : "?", \
          (void*)(STREAM), (PHASE) ? (PHASE) : "?"); \
    } \
  } while (0)

// ── CUDA graph capture status probe ───────────────────────────────────────
// Check whether a CUDA stream's capture status is still valid. If the capture
// has been INVALIDATED, log the probe TAG, SLOT, and OP via GRAPH_REPLAY diag.
// Returns true if capture was invalidated (caller should break/abort).
// Gated on GRAPH_REPLAY category — zero overhead when diagnostics are disabled.
// Safe to call from any thread during active capture.
//
// Usage:
//   bool bad = false;
//   DSP_CAPTURE_PROBE(stream, slotIdx, "AFTER_STEP1", opName, bad);
//   if (bad) { /* handle invalidation */ }
//
#if defined(SD_CUDA)
#define DSP_CAPTURE_PROBE(STREAM, SLOT, TAG, OP, OUT_INVALID) \
  do { \
    (OUT_INVALID) = false; \
    if (tl_graphExecutionActive && \
        sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_GRAPH_REPLAY)) { \
      cudaStreamCaptureStatus _dsp_cap_stat = cudaStreamCaptureStatusNone; \
      cudaError_t _dsp_cap_err = cudaStreamGetCaptureInfo_v2( \
          (STREAM), &_dsp_cap_stat, nullptr, nullptr, nullptr, nullptr); \
      if (_dsp_cap_err != cudaSuccess || \
          _dsp_cap_stat == cudaStreamCaptureStatusInvalidated) { \
        sd::graph::DspDiagnostics::getInstance().recordEvent( \
            sd::graph::DSP_DIAG_GRAPH_REPLAY, (SLOT), -1, -1, (OP), 0, \
            "CAPTURE_INVALIDATED: slot=%d tag=%s op=%s capErr=%d capStat=%d", \
            (int)(SLOT), (TAG), (OP) ? (OP) : "?", \
            (int)_dsp_cap_err, (int)_dsp_cap_stat); \
        cudaGetLastError(); \
        (OUT_INVALID) = true; \
      } \
    } \
  } while (0)
#else
#define DSP_CAPTURE_PROBE(STREAM, SLOT, TAG, OP, OUT_INVALID) \
  do { (OUT_INVALID) = false; } while (0)
#endif

// Emit a titled summary banner as a single ring-buffer event.
// Fuses a section title and formatted body into one recordEvent call,
// replacing multi-call blocks like "=== TITLE === \n details \n ===".
#define DSP_DIAG_BANNER(CAT, TITLE, FMT, ...) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_##CAT)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_##CAT, -1, -1, -1, nullptr, 0, \
          "=== %s === " FMT, (TITLE), ##__VA_ARGS__); \
    } \
  } while (0)

// Emit a standardized LIFECYCLE state transition event.
// STATE_NAME_FN: a callable that maps lifecycleState to const char* (e.g. stateName).
// OLD_STATE: the current lifecycle state value.
// NEW_STATE_STR: string literal for the destination state.
// FMT/...: optional additional context appended after the arrow.
// NOTE: renamed from DSP_DIAG_LIFECYCLE to avoid collision with DSP_DIAG_LIFECYCLE category constant.
#define DSP_DIAG_STATE_TRANSITION(STATE_NAME_FN, OLD_STATE, NEW_STATE_STR, FMT, ...) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_EXECUTE)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_EXECUTE, -1, -1, -1, nullptr, 0, \
          "LIFECYCLE: %s -> " NEW_STATE_STR " " FMT, \
          (STATE_NAME_FN)(OLD_STATE), ##__VA_ARGS__); \
    } \
  } while (0)
// Backward compat alias (callers being migrated)
#define DSP_DIAG_LIFECYCLE_TRANSITION DSP_DIAG_STATE_TRANSITION

// ─── Throw helpers ──────────────────────────────────────────────────────────
//
// Consolidate the pervasive  char buf[N]; snprintf(...); THROW_EXCEPTION(buf)
// triple into single-call macros.  Every throw site automatically records a
// diagnostic event (when the category is enabled) so errors are visible in the
// ring buffer even when the exception is caught upstream.

// General throw — logs to category CAT then throws.
#define DSP_THROW(CAT, FMT, ...) \
  do { \
    char _dsp_throw_buf[512]; \
    snprintf(_dsp_throw_buf, sizeof(_dsp_throw_buf), FMT, ##__VA_ARGS__); \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_##CAT)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_##CAT, -1, -1, -1, nullptr, 0, "%s", _dsp_throw_buf); \
    } \
    THROW_EXCEPTION(_dsp_throw_buf); \
  } while (0)

// Segment-tagged throw — embeds seg=[startSlot-endSlot] in the diagnostic event.
#define DSP_THROW_SEG(CAT, SEG_START, FMT, ...) \
  do { \
    char _dsp_throw_buf[512]; \
    snprintf(_dsp_throw_buf, sizeof(_dsp_throw_buf), FMT, ##__VA_ARGS__); \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_##CAT)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_##CAT, -1, (SEG_START), -1, nullptr, 0, "%s", _dsp_throw_buf); \
    } \
    THROW_EXCEPTION(_dsp_throw_buf); \
  } while (0)

// CUDA error throw — appends cudaGetErrorString(ERR) and clears the sticky error.
#ifdef SD_CUDA
#define DSP_THROW_CUDA(CAT, ERR, FMT, ...) \
  do { \
    char _dsp_throw_buf[512]; \
    snprintf(_dsp_throw_buf, sizeof(_dsp_throw_buf), FMT ": %s", \
             ##__VA_ARGS__, cudaGetErrorString(ERR)); \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_##CAT)) { \
      sd::graph::DspDiagnostics::getInstance().recordEvent( \
          sd::graph::DSP_DIAG_##CAT, -1, -1, -1, nullptr, 0, "%s", _dsp_throw_buf); \
    } \
    cudaGetLastError(); \
    THROW_EXCEPTION(_dsp_throw_buf); \
  } while (0)
#else
#define DSP_THROW_CUDA(CAT, ERR, FMT, ...) DSP_THROW(CAT, FMT, ##__VA_ARGS__)
#endif

#else  // __CUDA_ARCH__

#define DSP_DIAG_ENABLED(CAT)                     false
#define DSP_DIAG(CAT, FMT, ...)                   ((void)0)
#define DSP_DIAG_SLOT(CAT, SLOT, FMT, ...)        ((void)0)
#define DSP_DIAG_SEG(CAT, SEG, FMT, ...)          ((void)0)
#define DSP_DIAG_TIMED(CAT, SEG, SLOT, OP, US, FMT, ...) ((void)0)
#define DSP_DIAG_DEV(CAT, DEV, FMT, ...)          ((void)0)
#define DSP_DIAG_DUMP_SLOT(TAG, SLOT_IDX, DEV_PTR, NUM_ELEMS) ((void)0)
#define DSP_DIAG_DUMP_SEG_OUTPUT(TAG, END_SLOT, DEV_PTR, NUM_ELEMS, EXEC_COUNT, STREAM) ((void)0)
#define DSP_DIAG_DUMP_EXT_INPUTS(EXT_ARRAYS, NUM_EXT, EXEC_COUNT, RESULT_VAR) ((void)0)
#define DSP_DIAG_FINGERPRINT(TAG, IDX, NAME, ARR, EXEC_COUNT) ((void)0)
#define DSP_DIAG_SNAPSHOT_ADDRS(TAG, OUT_SLOTS, NUM_OUT, EXT_ARRAYS, NUM_EXT) ((void)0)
#define DSP_DIAG_COMPARE_ADDRS(TAG_A, TAG_B) (0)
#define DSP_DIAG_SLOT_WRITE(SLOT_IDX, OP, BYTES, STREAM, PHASE) ((void)0)
#define DSP_DIAG_SLOT_ZERO(SLOT_IDX, REASON, STREAM, PHASE) ((void)0)
#define DSP_CAPTURE_PROBE(STREAM, SLOT, TAG, OP, OUT_INVALID) do { (OUT_INVALID) = false; } while (0)
#define DSP_DIAG_BANNER(CAT, TITLE, FMT, ...) ((void)0)
#define DSP_DIAG_STATE_TRANSITION(STATE_NAME_FN, OLD_STATE, NEW_STATE_STR, FMT, ...) ((void)0)
#define DSP_DIAG_LIFECYCLE_TRANSITION DSP_DIAG_STATE_TRANSITION
#define DSP_THROW(CAT, FMT, ...)             do { THROW_EXCEPTION("DSP error"); } while (0)
#define DSP_THROW_SEG(CAT, SEG_START, FMT, ...) do { THROW_EXCEPTION("DSP error"); } while (0)
#define DSP_THROW_CUDA(CAT, ERR, FMT, ...)   do { THROW_EXCEPTION("DSP error"); } while (0)

#endif  // __CUDA_ARCH__

#endif  // LIBND4J_DSP_DIAGNOSTICS_H
