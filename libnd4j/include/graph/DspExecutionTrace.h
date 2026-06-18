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

#ifndef LIBND4J_DSP_EXECUTION_TRACE_H
#define LIBND4J_DSP_EXECUTION_TRACE_H

#include <system/common.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace sd {
namespace graph {

/**
 * Structured trace event kinds. Each captures a specific execution decision.
 */
enum class TraceEventKind : uint8_t {
  SEGMENT_DISPATCH = 0,    // Segment dispatched to a backend
  SLOT_WRITTEN = 1,        // Output slot array was written/replaced
  BUFFER_REPLACED = 2,     // Buffer pointer changed for a slot
  EXT_INPUT_SYNCED = 3,    // External input synced to device
  GRAPH_CAPTURED = 4,      // CUDA graph capture completed
  GRAPH_REPLAYED = 5,      // CUDA graph replay launched
  PHASE_TRANSITION = 6,    // Plan or segment phase changed
  ARRAY_FREED = 7,         // Plan-owned array freed
  SHAPE_KEY_CHANGED = 8,   // Segment shape key changed (triggers rebuild)
  CAPTURE_ABORTED = 9,     // CUDA graph capture aborted
  ERROR_OCCURRED = 10,     // Error during execution
  LIFECYCLE_TRANSITION = 11, // Segment lifecycle state transition
};

/**
 * Fixed-size structured trace event. 48 bytes per event.
 * No heap allocation, no strings — pure value types for ring buffer storage.
 */
struct DspTraceEvent {
  TraceEventKind kind;         // 1 byte
  int8_t segIdx;               // Segment index (-1 = plan-level)
  uint8_t backendId;           // SelectedBackend enum value
  uint8_t prevState;           // Previous state (for transitions)
  uint8_t newState;            // New state (for transitions)
  uint8_t pad[3];              // Alignment padding
  int32_t slotIdx;             // Slot index (for slot-level events), -1 if N/A
  int32_t slotRangeEnd;        // End slot (for segment events)
  uint32_t seqNo;              // Monotonic sequence number
  uint32_t execCount;          // Plan executeCount_ at time of event
  uint64_t bufAddrHash;        // FNV-1a of relevant buffer addresses
  uint64_t timestampNs;        // Nanoseconds since plan creation (relative)
  uint64_t detail;             // Event-specific detail (shape key, error code, etc.)
};

static_assert(sizeof(DspTraceEvent) == 48, "DspTraceEvent must be 48 bytes");

/**
 * Lock-free ring buffer of structured execution events.
 * Pre-allocated, fixed capacity, zero-alloc at record time.
 *
 * Usage:
 *   trace.record(TraceEventKind::SEGMENT_DISPATCH, segIdx, backendId, ...);
 *   // ... later, on error:
 *   trace.dump(stderr);
 *   // ... or get structured access:
 *   trace.forEach([](const DspTraceEvent& e) { ... });
 */
class SD_LIB_EXPORT DspExecutionTrace {
 public:
  static constexpr int CAPACITY = 512;  // Power of 2 for fast modulo

  DspExecutionTrace() {
    std::memset(events_, 0, sizeof(events_));
    startTimeNs_ = nowNs();
  }

  /**
   * Record a trace event. Lock-free, O(1), no allocation.
   * Returns the sequence number assigned to this event.
   */
  uint32_t record(TraceEventKind kind, int8_t segIdx, uint8_t backendId,
                  int32_t slotIdx, int32_t slotRangeEnd,
                  uint32_t execCount, uint64_t bufAddrHash = 0,
                  uint64_t detail = 0,
                  uint8_t prevState = 0, uint8_t newState = 0) {
    uint32_t seq = nextSeq_.fetch_add(1, std::memory_order_relaxed);
    uint32_t idx = seq & (CAPACITY - 1);  // Fast modulo for power-of-2

    auto& e = events_[idx];
    e.kind = kind;
    e.segIdx = segIdx;
    e.backendId = backendId;
    e.prevState = prevState;
    e.newState = newState;
    e.pad[0] = 0; e.pad[1] = 0; e.pad[2] = 0;
    e.slotIdx = slotIdx;
    e.slotRangeEnd = slotRangeEnd;
    e.seqNo = seq;
    e.execCount = execCount;
    e.bufAddrHash = bufAddrHash;
    e.timestampNs = nowNs() - startTimeNs_;
    e.detail = detail;
    return seq;
  }

  // ── Convenience recorders ─────────────────────────────────────────

  uint32_t recordSegmentDispatch(int8_t segIdx, uint8_t backendId,
                                  int32_t startSlot, int32_t endSlot,
                                  uint32_t execCount, uint64_t shapeKey = 0) {
    return record(TraceEventKind::SEGMENT_DISPATCH, segIdx, backendId,
                  startSlot, endSlot, execCount, 0, shapeKey);
  }

  uint32_t recordSlotWritten(int8_t segIdx, int32_t slotIdx,
                              uint32_t execCount, uint64_t bufAddr = 0) {
    return record(TraceEventKind::SLOT_WRITTEN, segIdx, 0,
                  slotIdx, -1, execCount, bufAddr);
  }

  uint32_t recordBufferReplaced(int8_t segIdx, int32_t slotIdx,
                                 uint32_t execCount,
                                 uint64_t oldAddr, uint64_t newAddr) {
    return record(TraceEventKind::BUFFER_REPLACED, segIdx, 0,
                  slotIdx, -1, execCount, newAddr, oldAddr);
  }

  uint32_t recordGraphCaptured(int8_t segIdx, uint32_t execCount,
                                int32_t startSlot, int32_t endSlot,
                                uint64_t nodeCount = 0) {
    return record(TraceEventKind::GRAPH_CAPTURED, segIdx, 0,
                  startSlot, endSlot, execCount, 0, nodeCount);
  }

  uint32_t recordGraphReplayed(int8_t segIdx, uint32_t execCount,
                                int32_t startSlot, int32_t endSlot) {
    return record(TraceEventKind::GRAPH_REPLAYED, segIdx, 0,
                  startSlot, endSlot, execCount);
  }

  uint32_t recordPhaseTransition(int8_t segIdx, uint8_t prevPhase, uint8_t newPhase,
                                  uint32_t execCount) {
    return record(TraceEventKind::PHASE_TRANSITION, segIdx, 0,
                  -1, -1, execCount, 0, 0, prevPhase, newPhase);
  }

  uint32_t recordLifecycleTransition(int8_t segIdx, uint8_t prevState, uint8_t newState,
                                      uint32_t execCount) {
    return record(TraceEventKind::LIFECYCLE_TRANSITION, segIdx, 0,
                  -1, -1, execCount, 0, 0, prevState, newState);
  }

  uint32_t recordError(int8_t segIdx, int32_t slotIdx,
                        uint32_t execCount, uint64_t errorCode = 0) {
    return record(TraceEventKind::ERROR_OCCURRED, segIdx, 0,
                  slotIdx, -1, execCount, 0, errorCode);
  }

  // ── Dump/query ────────────────────────────────────────────────────

  /**
   * Dump the most recent `count` events to the given file stream.
   * Events are printed oldest-first. Thread-safe for single-writer scenarios.
   */
  void dump(FILE* out, int count = 64) const {
    uint32_t head = nextSeq_.load(std::memory_order_relaxed);
    if (head == 0) {
      std::fprintf(out, "DSP_TRACE: (empty)\n");
      return;
    }

    int available = static_cast<int>(head < static_cast<uint32_t>(CAPACITY) ? head : CAPACITY);
    if (count > available) count = available;

    uint32_t startSeq = head - static_cast<uint32_t>(count);
    std::fprintf(out, "DSP_TRACE: dumping %d events (seq %u to %u)\n",
                 count, startSeq, head - 1);

    for (uint32_t seq = startSeq; seq < head; seq++) {
      uint32_t idx = seq & (CAPACITY - 1);
      const auto& e = events_[idx];
      // Skip if this slot was overwritten by a newer event (ring wrapped)
      if (e.seqNo != seq) continue;

      std::fprintf(out, "  [%6u] +%8llu us  %-22s seg=%2d backend=%d slots=[%d,%d] "
                        "exec=%u addrHash=%016llx detail=%llu",
                   e.seqNo,
                   (unsigned long long)(e.timestampNs / 1000),
                   traceEventKindName(e.kind),
                   (int)e.segIdx, (int)e.backendId,
                   e.slotIdx, e.slotRangeEnd,
                   e.execCount,
                   (unsigned long long)e.bufAddrHash,
                   (unsigned long long)e.detail);

      // Extra detail for transitions
      if (e.kind == TraceEventKind::PHASE_TRANSITION ||
          e.kind == TraceEventKind::LIFECYCLE_TRANSITION) {
        std::fprintf(out, " [%d->%d]", (int)e.prevState, (int)e.newState);
      }
      if (e.kind == TraceEventKind::BUFFER_REPLACED) {
        std::fprintf(out, " [old=%016llx]", (unsigned long long)e.detail);
      }
      std::fprintf(out, "\n");
    }
  }

  /**
   * Iterate over recent events (newest first). Callback receives const ref.
   * Returns count of events visited.
   */
  template <typename Fn>
  int forEach(Fn&& fn, int maxCount = CAPACITY) const {
    uint32_t head = nextSeq_.load(std::memory_order_relaxed);
    if (head == 0) return 0;

    int available = static_cast<int>(head < static_cast<uint32_t>(CAPACITY) ? head : CAPACITY);
    if (maxCount > available) maxCount = available;
    int visited = 0;

    for (int i = 0; i < maxCount; i++) {
      uint32_t seq = head - 1 - static_cast<uint32_t>(i);
      uint32_t idx = seq & (CAPACITY - 1);
      const auto& e = events_[idx];
      if (e.seqNo != seq) continue;
      fn(e);
      visited++;
    }
    return visited;
  }

  /** Total events recorded (may exceed CAPACITY due to ring wrap). */
  uint32_t totalRecorded() const {
    return nextSeq_.load(std::memory_order_relaxed);
  }

  /** Reset the trace (for tests). NOT thread-safe with concurrent recording. */
  void reset() {
    nextSeq_.store(0, std::memory_order_relaxed);
    std::memset(events_, 0, sizeof(events_));
    startTimeNs_ = nowNs();
  }

  // ── Event kind name helper ────────────────────────────────────────

  static const char* traceEventKindName(TraceEventKind kind) {
    switch (kind) {
      case TraceEventKind::SEGMENT_DISPATCH:      return "SEGMENT_DISPATCH";
      case TraceEventKind::SLOT_WRITTEN:           return "SLOT_WRITTEN";
      case TraceEventKind::BUFFER_REPLACED:        return "BUFFER_REPLACED";
      case TraceEventKind::EXT_INPUT_SYNCED:       return "EXT_INPUT_SYNCED";
      case TraceEventKind::GRAPH_CAPTURED:         return "GRAPH_CAPTURED";
      case TraceEventKind::GRAPH_REPLAYED:         return "GRAPH_REPLAYED";
      case TraceEventKind::PHASE_TRANSITION:       return "PHASE_TRANSITION";
      case TraceEventKind::ARRAY_FREED:            return "ARRAY_FREED";
      case TraceEventKind::SHAPE_KEY_CHANGED:      return "SHAPE_KEY_CHANGED";
      case TraceEventKind::CAPTURE_ABORTED:        return "CAPTURE_ABORTED";
      case TraceEventKind::ERROR_OCCURRED:         return "ERROR_OCCURRED";
      case TraceEventKind::LIFECYCLE_TRANSITION:   return "LIFECYCLE_TRANSITION";
      default:                                     return "UNKNOWN";
    }
  }

 private:
  DspTraceEvent events_[CAPACITY];
  std::atomic<uint32_t> nextSeq_{0};
  uint64_t startTimeNs_ = 0;

  static uint64_t nowNs() {
    auto now = std::chrono::high_resolution_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count());
  }
};

// ── Trace recording macros ──────────────────────────────────────────
// These are no-ops when the trace pointer is null, so call sites
// don't need to check.

#define DSP_TRACE_RECORD(trace, ...) \
  do { if ((trace) != nullptr) (trace)->record(__VA_ARGS__); } while (0)

#define DSP_TRACE_SEG_DISPATCH(trace, segIdx, backendId, startSlot, endSlot, execCount, shapeKey) \
  do { if ((trace) != nullptr) (trace)->recordSegmentDispatch(segIdx, backendId, startSlot, endSlot, execCount, shapeKey); } while (0)

#define DSP_TRACE_SLOT_WRITTEN(trace, segIdx, slotIdx, execCount, bufAddr) \
  do { if ((trace) != nullptr) (trace)->recordSlotWritten(segIdx, slotIdx, execCount, bufAddr); } while (0)

#define DSP_TRACE_GRAPH_CAPTURED(trace, segIdx, execCount, startSlot, endSlot, nodeCount) \
  do { if ((trace) != nullptr) (trace)->recordGraphCaptured(segIdx, execCount, startSlot, endSlot, nodeCount); } while (0)

#define DSP_TRACE_GRAPH_REPLAYED(trace, segIdx, execCount, startSlot, endSlot) \
  do { if ((trace) != nullptr) (trace)->recordGraphReplayed(segIdx, execCount, startSlot, endSlot); } while (0)

#define DSP_TRACE_PHASE(trace, segIdx, prevPhase, newPhase, execCount) \
  do { if ((trace) != nullptr) (trace)->recordPhaseTransition(segIdx, prevPhase, newPhase, execCount); } while (0)

#define DSP_TRACE_LIFECYCLE(trace, segIdx, prevState, newState, execCount) \
  do { if ((trace) != nullptr) (trace)->recordLifecycleTransition(segIdx, prevState, newState, execCount); } while (0)

#define DSP_TRACE_ERROR(trace, segIdx, slotIdx, execCount, errorCode) \
  do { if ((trace) != nullptr) (trace)->recordError(segIdx, slotIdx, execCount, errorCode); } while (0)

#define DSP_TRACE_DUMP(trace, out, count) \
  do { if ((trace) != nullptr) (trace)->dump(out, count); } while (0)

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_EXECUTION_TRACE_H
