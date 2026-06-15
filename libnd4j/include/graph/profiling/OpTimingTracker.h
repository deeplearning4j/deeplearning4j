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

#ifndef LIBND4J_OP_TIMING_TRACKER_H
#define LIBND4J_OP_TIMING_TRACKER_H

#include <system/common.h>
#include <types/types.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

/**
 * Execution phases within an operation
 */
enum class OpPhase {
  VALIDATION = 0,    // Input validation, argument checking, datatype validation
  SHAPE_CALC = 1,    // Output shape calculation
  MEMORY_ALLOC = 2,  // Output array allocation
  HELPER_CHECK = 3,  // Platform helper isUsable() check
  HELPER_EXEC = 4,   // Platform helper execution (oneDNN, cuDNN)
  NATIVE_EXEC = 5,   // Native C++ implementation execution
  TOTAL = 6          // End-to-end operation time
};

static constexpr int OP_PHASE_COUNT = 7;

/**
 * Logarithmic histogram for timing distribution
 * Buckets: <1us, 1-2us, 2-4us, 4-8us, ... up to 1s+
 */
struct TimingHistogram {
  static constexpr int NUM_BUCKETS = 24;  // Covers <1us to >8s
  uint64_t buckets[NUM_BUCKETS];

  // Explicit default constructor to ensure all buckets are zeroed
  TimingHistogram() {
    for (int i = 0; i < NUM_BUCKETS; i++) {
      buckets[i] = 0;
    }
  }

  void record(LongType nanos) {
    int bucket = 0;
    if (nanos > 0) {
      // Convert to microseconds for bucket calculation
      LongType micros = nanos / 1000;
      if (micros > 0) {
        // log2(micros) gives us the bucket
        bucket = 1 + static_cast<int>(std::log2(static_cast<double>(micros)));
        if (bucket >= NUM_BUCKETS) bucket = NUM_BUCKETS - 1;
      }
    }
    buckets[bucket]++;
  }

  void merge(const TimingHistogram& other) {
    for (int i = 0; i < NUM_BUCKETS; i++) {
      buckets[i] += other.buckets[i];
    }
  }

  void reset() {
    for (int i = 0; i < NUM_BUCKETS; i++) {
      buckets[i] = 0;
    }
  }

  // Get percentile value in microseconds
  double getPercentile(double p, uint64_t totalCount) const {
    if (totalCount == 0) return 0.0;
    uint64_t target = static_cast<uint64_t>(p * totalCount);
    uint64_t cumulative = 0;
    for (int i = 0; i < NUM_BUCKETS; i++) {
      cumulative += buckets[i];
      if (cumulative >= target) {
        // Return bucket midpoint in microseconds
        if (i == 0) return 0.5;
        return std::pow(2.0, i - 1) * 1.5;  // midpoint of bucket
      }
    }
    return std::pow(2.0, NUM_BUCKETS - 2);  // max bucket
  }
};

/**
 * Per-thread timing statistics
 */
struct ThreadTimingStats {
  std::thread::id threadId;
  uint64_t callCount;
  LongType totalTimeNanos;
  LongType minTimeNanos;
  LongType maxTimeNanos;

  // Explicit default constructor
  ThreadTimingStats() :
      threadId(),
      callCount(0),
      totalTimeNanos(0),
      minTimeNanos(LLONG_MAX),
      maxTimeNanos(0) {}

  void record(LongType nanos) {
    callCount++;
    totalTimeNanos += nanos;
    if (nanos < minTimeNanos) minTimeNanos = nanos;
    if (nanos > maxTimeNanos) maxTimeNanos = nanos;
  }

  double avgMicros() const {
    return callCount > 0 ? (totalTimeNanos / 1000.0) / callCount : 0.0;
  }
};

/**
 * Maximum length for op name in timing record
 */
static constexpr size_t OP_NAME_MAX_LEN = 64;

// Version marker for ABI compatibility checking
static constexpr uint32_t OP_TIMING_ABI_VERSION = 2;

/**
 * Single execution record for ring buffer (fixed-size)
 */
struct OpTimingRecord {
  LongType hash;                            // Op identifier (hash)
  char name[OP_NAME_MAX_LEN];               // Op name (copied, not pointer - avoids dangling refs)
  LongType phaseNanos[OP_PHASE_COUNT];      // Per-phase timing
  LongType inputBytes;                      // Total input size
  LongType outputBytes;                     // Total output size
  LongType memoryAllocated;                 // Memory allocated during op
  LongType timestampNanos;                  // For trace export (start time)
  std::thread::id threadId;                 // Thread that executed
  bool usedHelper;                          // Helper vs native path
  int deviceId;                             // CUDA device (0 for CPU)

  // Explicit default constructor to ensure all fields are zeroed
  OpTimingRecord() :
      hash(0),
      inputBytes(0),
      outputBytes(0),
      memoryAllocated(0),
      timestampNanos(0),
      threadId(),
      usedHelper(false),
      deviceId(0) {
    std::memset(name, 0, OP_NAME_MAX_LEN);
    for (int i = 0; i < OP_PHASE_COUNT; i++) {
      phaseNanos[i] = 0;
    }
  }
};

/**
 * Chrome trace event for JSON export
 */
struct ChromeTraceEvent {
  std::string name;
  std::string category;
  LongType timestampMicros;
  LongType durationMicros;
  std::thread::id threadId;
  std::string phase;  // "X" for complete event

  std::string toJson() const {
    std::ostringstream ss;
    ss << "{\"name\":\"" << name << "\","
       << "\"cat\":\"" << category << "\","
       << "\"ph\":\"" << phase << "\","
       << "\"ts\":" << timestampMicros << ","
       << "\"dur\":" << durationMicros << ","
       << "\"pid\":1,"
       << "\"tid\":" << std::hash<std::thread::id>{}(threadId) << "}";
    return ss.str();
  }
};

/**
 * Aggregated statistics per op type
 */
struct AggregatedOpStats {
  std::string opName;
  LongType opHash;

  // Counts
  uint64_t callCount;
  uint64_t helperUsedCount;

  // Timing by phase (nanoseconds)
  LongType phaseTime[OP_PHASE_COUNT];

  // Distribution
  TimingHistogram histogram;

  // Welford's algorithm for running variance
  double mean;
  double m2;  // sum of squares of differences from mean

  // Memory tracking
  LongType totalInputBytes;
  LongType totalOutputBytes;
  LongType totalMemoryAllocated;

  // Min/Max
  LongType minTimeNanos;
  LongType maxTimeNanos;

  // Explicit default constructor to ensure all fields are initialized
  AggregatedOpStats() :
      opName(),
      opHash(0),
      callCount(0),
      helperUsedCount(0),
      phaseTime{0},
      histogram(),
      mean(0.0),
      m2(0.0),
      totalInputBytes(0),
      totalOutputBytes(0),
      totalMemoryAllocated(0),
      minTimeNanos(LLONG_MAX),
      maxTimeNanos(0) {
    histogram.reset();  // Ensure histogram buckets are zeroed
  }

  void record(const OpTimingRecord& rec) {
    // Validate record before processing - skip corrupted entries
    LongType totalNanos = rec.phaseNanos[static_cast<int>(OpPhase::TOTAL)];
    constexpr LongType MAX_REASONABLE_NANOS = 3600000000000LL;  // 1 hour max
    if (totalNanos < 0 || totalNanos > MAX_REASONABLE_NANOS) {
      return;  // Skip corrupted record
    }

    callCount++;
    if (rec.usedHelper) helperUsedCount++;

    // Accumulate phase times (with validation)
    for (int i = 0; i < OP_PHASE_COUNT; i++) {
      LongType phaseNanos = rec.phaseNanos[i];
      if (phaseNanos >= 0 && phaseNanos <= MAX_REASONABLE_NANOS) {
        phaseTime[i] += phaseNanos;
      }
    }

    // Update histogram
    histogram.record(totalNanos);

    // Welford's algorithm for running mean and variance
    double delta = totalNanos - mean;
    mean += delta / callCount;
    double delta2 = totalNanos - mean;
    m2 += delta * delta2;

    // Min/Max
    if (totalNanos < minTimeNanos) minTimeNanos = totalNanos;
    if (totalNanos > maxTimeNanos) maxTimeNanos = totalNanos;

    // Memory (with validation - should be non-negative)
    if (rec.inputBytes >= 0) totalInputBytes += rec.inputBytes;
    if (rec.outputBytes >= 0) totalOutputBytes += rec.outputBytes;
    if (rec.memoryAllocated >= 0) totalMemoryAllocated += rec.memoryAllocated;
  }

  double stdDevMicros() const {
    if (callCount < 2) return 0.0;
    return std::sqrt(m2 / (callCount - 1)) / 1000.0;
  }

  double avgMicros() const {
    return mean / 1000.0;
  }

  double helperPercent() const {
    return callCount > 0 ? (100.0 * helperUsedCount / callCount) : 0.0;
  }

  double totalMillis() const {
    return phaseTime[static_cast<int>(OpPhase::TOTAL)] / 1000000.0;
  }
};

/**
 * RAII timer for measuring a single phase
 */
class OpPhaseTimer {
 private:
  OpTimingRecord* _record;
  OpPhase _phase;
  std::chrono::high_resolution_clock::time_point _start;

 public:
  OpPhaseTimer(OpTimingRecord* record, OpPhase phase) : _record(record), _phase(phase) {
    if (_record) {
      _start = std::chrono::high_resolution_clock::now();
    }
  }

  ~OpPhaseTimer() {
    if (_record) {
      auto end = std::chrono::high_resolution_clock::now();
      auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - _start).count();
      _record->phaseNanos[static_cast<int>(_phase)] = nanos;
    }
  }

  // Non-copyable
  OpPhaseTimer(const OpPhaseTimer&) = delete;
  OpPhaseTimer& operator=(const OpPhaseTimer&) = delete;
};

#ifdef SD_CUDA
/**
 * CUDA event-based timer for accurate GPU kernel timing
 */
class CudaEventTimer {
 private:
  void* _startEvent;  // cudaEvent_t
  void* _stopEvent;   // cudaEvent_t
  bool _started = false;

 public:
  CudaEventTimer();
  ~CudaEventTimer();

  void start();
  void stop();
  float elapsedMillis();  // Returns -1 if not properly started/stopped

  // Non-copyable
  CudaEventTimer(const CudaEventTimer&) = delete;
  CudaEventTimer& operator=(const CudaEventTimer&) = delete;
};
#endif

/**
 * Thread-local state for tracking indirect helper usage.
 * When an op internally calls other ops that use helpers,
 * this tracks that usage so it propagates to the parent op.
 */
struct IndirectHelperTracker {
  int depth;              // Nesting depth of op execution
  bool anyHelperUsed;     // Was any helper used at current depth or below?

  IndirectHelperTracker() : depth(0), anyHelperUsed(false) {}

  // Call when entering an op execution
  void enter() { depth++; }

  // Call when exiting an op execution, returns true if any helper was used
  bool exit() {
    depth--;
    if (depth == 0) {
      bool result = anyHelperUsed;
      anyHelperUsed = false;  // Reset for next top-level op
      return result;
    }
    return anyHelperUsed;
  }

  // Call when a helper is used (direct or indirect)
  void markHelperUsed() { anyHelperUsed = true; }

  // Check if we're in a nested op (not top-level)
  bool isNested() const { return depth > 1; }
};

// Thread-local tracker for indirect helper usage
SD_LIB_EXPORT IndirectHelperTracker& getIndirectHelperTracker();

/**
 * Main timing tracker singleton
 *
 * Provides low-overhead op timing with:
 * - Lock-free hot path via ring buffer
 * - Per-phase timing breakdown
 * - Histogram and percentile analysis
 * - Chrome trace export
 * - Per-thread statistics
 * - Indirect helper usage tracking (when ops call other ops that use helpers)
 */
class SD_LIB_EXPORT OpTimingTracker {
 private:
  // Ring buffer for lock-free recording
  static constexpr int RING_SIZE = 8192;
  OpTimingRecord _ringBuffer[RING_SIZE]{};  // Value-initialize all elements to zero
  std::atomic<uint64_t> _ringIndex{0};
  uint64_t _lastFlushedIdx = 0;  // Track last processed index for incremental flush

  // Aggregated stats (protected by mutex)
  std::mutex _statsMutex;
  std::unordered_map<LongType, AggregatedOpStats> _aggregatedStats;
  std::unordered_map<std::thread::id, ThreadTimingStats> _threadStats;

  // Chrome trace events (limited to prevent unbounded growth)
  std::vector<ChromeTraceEvent> _traceEvents;
  static constexpr size_t MAX_TRACE_EVENTS = 100000;

  // Configuration
  std::atomic<bool> _enabled{false};
  std::atomic<bool> _detailedMode{false};  // Phase-level timing
  std::atomic<bool> _traceMode{false};     // Chrome trace recording
  LongType _traceStartTime = 0;

  // Private constructor for singleton - explicitly initialize ring buffer
  OpTimingTracker() {
    // Ensure ring buffer is zero-initialized (defense in depth)
    for (int i = 0; i < RING_SIZE; i++) {
      _ringBuffer[i] = OpTimingRecord{};
    }
  }

 public:
  static OpTimingTracker& getInstance();

  // Configuration
  void enable(bool detailed = false);
  void enableWithTrace(bool detailed = false);
  void disable();
  bool isEnabled() const { return _enabled.load(std::memory_order_relaxed); }
  bool isDetailedMode() const { return _detailedMode.load(std::memory_order_relaxed); }
  bool isTraceMode() const { return _traceMode.load(std::memory_order_relaxed); }

  // Recording (hot path - lock-free)
  void record(const OpTimingRecord& record);

  // Get timestamp relative to trace start (for Chrome trace)
  LongType getTraceTimestamp();

  // Aggregation (call before reading stats)
  void flush();

  // Analysis
  void printHotspots(int topN = 20);
  void printOpBreakdown(const std::string& opName);
  void printHistogram(const std::string& opName);
  void printThreadStats();

  // Export
  bool exportChromeTrace(const std::string& filename);
  bool exportCSV(const std::string& filename);

  // Query
  int getNumOps() const;
  uint64_t getTotalExecutions() const;
  const AggregatedOpStats* getOpStats(const std::string& opName);
  const AggregatedOpStats* getOpStatsByHash(LongType hash);

  // Reset
  void reset();

  // Non-copyable singleton
  OpTimingTracker(const OpTimingTracker&) = delete;
  OpTimingTracker& operator=(const OpTimingTracker&) = delete;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_OP_TIMING_TRACKER_H
