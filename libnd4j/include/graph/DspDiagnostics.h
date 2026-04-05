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

  DSP_DIAG_NONE     = 0u,
  DSP_DIAG_ALL      = 0x3FFFu      // All 14 categories
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

static constexpr int DSP_DIAG_NUM_CATEGORIES = 14;

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
  bool isEnabled(uint32_t category) const {
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
  void applyLegacyFlags();

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

  // ── Category name helpers ──
  static const char* categoryName(uint32_t category);
  static int categoryIndex(uint32_t category);
  static uint32_t parseCategories(const char* str);

 private:
  DspDiagnostics();
  ~DspDiagnostics() = default;
  DspDiagnostics(const DspDiagnostics&) = delete;
  DspDiagnostics& operator=(const DspDiagnostics&) = delete;

  void parseEnvVars();

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
        && sd::Environment::getInstance().isDebug()) { \
      sd::graph::DspDiagnostics::getInstance().snapshotAddresses( \
          (TAG), (void**)(OUT_SLOTS), (NUM_OUT), (void**)(EXT_ARRAYS), (NUM_EXT)); \
    } \
  } while (0)

#define DSP_DIAG_COMPARE_ADDRS(TAG_A, TAG_B) \
  (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_EXECUTE) \
   && sd::Environment::getInstance().isDebug() \
   ? sd::graph::DspDiagnostics::getInstance().compareAddressSnapshots((TAG_A), (TAG_B)) : 0)

// Dump leading float values of an output slot's device buffer.
// Gated on EXECUTE category + debug mode. Caller must be outside stream capture.
#define DSP_DIAG_DUMP_SLOT(TAG, SLOT_IDX, DEV_PTR, NUM_ELEMS) \
  do { \
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(sd::graph::DSP_DIAG_EXECUTE) \
        && sd::Environment::getInstance().isDebug()) { \
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
#define DSP_DIAG_SNAPSHOT_ADDRS(TAG, OUT_SLOTS, NUM_OUT, EXT_ARRAYS, NUM_EXT) ((void)0)
#define DSP_DIAG_COMPARE_ADDRS(TAG_A, TAG_B) (0)

#endif  // __CUDA_ARCH__

#endif  // LIBND4J_DSP_DIAGNOSTICS_H
