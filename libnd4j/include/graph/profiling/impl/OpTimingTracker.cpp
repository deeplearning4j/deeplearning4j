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

#include <graph/profiling/OpTimingTracker.h>

#include <algorithm>
#include <iomanip>

namespace sd {
namespace graph {

// Thread-local storage for indirect helper tracking
static thread_local IndirectHelperTracker tls_indirectHelperTracker;

IndirectHelperTracker& getIndirectHelperTracker() {
  return tls_indirectHelperTracker;
}

OpTimingTracker& OpTimingTracker::getInstance() {
  static OpTimingTracker instance;
  return instance;
}

void OpTimingTracker::enable(bool detailed) {
  // Print struct sizes for ABI debugging
  std::cerr << "[OpTimingTracker] Struct sizes: OpTimingRecord=" << sizeof(OpTimingRecord)
            << ", AggregatedOpStats=" << sizeof(AggregatedOpStats)
            << ", TimingHistogram=" << sizeof(TimingHistogram) << std::endl;

  // Verify constructors work
  OpTimingRecord testRecord;
  if (testRecord.hash != 0 || testRecord.phaseNanos[0] != 0) {
    std::cerr << "[OpTimingTracker] WARNING: OpTimingRecord constructor not zero-initializing!" << std::endl;
  }
  AggregatedOpStats testStats;
  if (testStats.callCount != 0 || testStats.helperUsedCount != 0) {
    std::cerr << "[OpTimingTracker] WARNING: AggregatedOpStats constructor not zero-initializing! "
              << "callCount=" << testStats.callCount << std::endl;
  }

  // Reset state to ensure clean start
  reset();
  _detailedMode.store(detailed, std::memory_order_relaxed);
  _traceMode.store(false, std::memory_order_relaxed);
  _enabled.store(true, std::memory_order_release);
}

void OpTimingTracker::enableWithTrace(bool detailed) {
  // Reset state to ensure clean start
  reset();
  _detailedMode.store(detailed, std::memory_order_relaxed);
  _traceMode.store(true, std::memory_order_relaxed);
  _traceStartTime =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  _enabled.store(true, std::memory_order_release);
}

void OpTimingTracker::disable() {
  _enabled.store(false, std::memory_order_release);
}

LongType OpTimingTracker::getTraceTimestamp() {
  auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::high_resolution_clock::now().time_since_epoch())
                 .count();
  return now - _traceStartTime;
}

void OpTimingTracker::record(const OpTimingRecord& record) {
  // Lock-free ring buffer insertion
  uint64_t idx = _ringIndex.fetch_add(1, std::memory_order_relaxed) % RING_SIZE;
  _ringBuffer[idx] = record;

  // If trace mode, also record trace event (under lock)
  if (_traceMode.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(_statsMutex);
    if (_traceEvents.size() < MAX_TRACE_EVENTS) {
      ChromeTraceEvent event;
      event.name = (record.name[0] != '\0') ? record.name : "unknown";
      event.category = record.usedHelper ? "helper" : "native";
      event.timestampMicros = record.timestampNanos / 1000;
      event.durationMicros = record.phaseNanos[static_cast<int>(OpPhase::TOTAL)] / 1000;
      event.threadId = record.threadId;
      event.phase = "X";  // Complete event
      _traceEvents.push_back(std::move(event));
    }
  }
}

void OpTimingTracker::flush() {
  std::lock_guard<std::mutex> lock(_statsMutex);

  // Process only NEW records since last flush
  uint64_t currentIdx = _ringIndex.load(std::memory_order_relaxed);

  // Calculate how many new records to process
  uint64_t newRecords = currentIdx - _lastFlushedIdx;
  if (newRecords == 0) return;  // Nothing new to process

  // Cap at ring buffer size (if we wrapped around)
  if (newRecords > RING_SIZE) {
    newRecords = RING_SIZE;
    _lastFlushedIdx = currentIdx - RING_SIZE;
  }

  for (uint64_t i = 0; i < newRecords; i++) {
    uint64_t idx = (_lastFlushedIdx + i) % RING_SIZE;
    const auto& rec = _ringBuffer[idx];

    // Skip empty slots
    if (rec.hash == 0 && rec.name[0] == '\0') continue;

    // Validate record - skip corrupted entries
    // Hash should be a reasonable value (op hashes are typically small integers or computed hashes)
    // Timing values should be reasonable (< 1 hour = 3.6e12 nanoseconds)
    LongType totalNanos = rec.phaseNanos[static_cast<int>(OpPhase::TOTAL)];
    constexpr LongType MAX_REASONABLE_TIME_NANOS = 3600000000000LL;  // 1 hour
    if (totalNanos < 0 || totalNanos > MAX_REASONABLE_TIME_NANOS) continue;  // Skip corrupted

    // Update aggregated stats
    auto it = _aggregatedStats.find(rec.hash);
    if (it == _aggregatedStats.end()) {
      // First time seeing this op - constructor handles zero-initialization
      AggregatedOpStats newStats;  // Uses explicit default constructor

      // Verify constructor worked - callCount should be 0
      if (newStats.callCount != 0) {
        std::cerr << " AggregatedOpStats constructor failed! callCount="
                  << newStats.callCount << " (should be 0)" << std::endl;
      }

      newStats.opName = (rec.name[0] != '\0') ? rec.name : "unknown";
      newStats.opHash = rec.hash;

      auto result = _aggregatedStats.emplace(rec.hash, std::move(newStats));
      it = result.first;

      // Verify after emplace - the value in the map should also be 0
      if (it->second.callCount != 0) {
        std::cerr << " AggregatedOpStats corrupted after emplace! callCount="
                  << it->second.callCount << " for op " << it->second.opName
                  << " wasInserted=" << result.second << std::endl;
      }
    }

    // Snapshot before record
    uint64_t beforeCount = it->second.callCount;
    it->second.record(rec);

    // Verify record() only incremented by 1
    if (it->second.callCount != beforeCount + 1 && it->second.callCount > 0) {
      std::cerr << " record() corrupted callCount! before=" << beforeCount
                << " after=" << it->second.callCount << " for op " << it->second.opName << std::endl;
    }

    // Update thread stats - constructor handles initialization via operator[]
    auto& threadStats = _threadStats[rec.threadId];
    if (threadStats.callCount == 0) {
      threadStats.threadId = rec.threadId;
    }
    threadStats.record(totalNanos);
  }

  _lastFlushedIdx = currentIdx;
}

void OpTimingTracker::printHotspots(int topN) {
  flush();
  std::lock_guard<std::mutex> lock(_statsMutex);

  // Sort by total time
  std::vector<const AggregatedOpStats*> sorted;
  for (const auto& pair : _aggregatedStats) {
    const auto& stats = pair.second;
    if (stats.callCount > 0) {
      sorted.push_back(&stats);
    }
  }
  std::sort(sorted.begin(), sorted.end(), [](const AggregatedOpStats* a, const AggregatedOpStats* b) {
    return a->phaseTime[static_cast<int>(OpPhase::TOTAL)] > b->phaseTime[static_cast<int>(OpPhase::TOTAL)];
  });

  std::cout << "\n=== Op Timing Hotspots (Top " << topN << ") ===" << std::endl;
  std::cout << std::setw(5) << "Rank" << "  " << std::setw(24) << std::left << "Op Name" << std::right << "  "
            << std::setw(10) << "Calls" << "  " << std::setw(12) << "Total(ms)" << "  " << std::setw(10) << "Avg(us)"
            << "  " << std::setw(10) << "StdDev(us)" << "  " << std::setw(8) << "Helper%" << std::endl;
  std::cout << std::string(90, '-') << std::endl;

  int rank = 1;
  for (const auto* stats : sorted) {
    if (rank > topN) break;

    std::cout << std::setw(5) << rank << "  " << std::setw(24) << std::left << stats->opName << std::right << "  "
              << std::setw(10) << stats->callCount << "  " << std::setw(12) << std::fixed << std::setprecision(2)
              << stats->totalMillis() << "  " << std::setw(10) << std::fixed << std::setprecision(2)
              << stats->avgMicros() << "  " << std::setw(10) << std::fixed << std::setprecision(2)
              << stats->stdDevMicros() << "  " << std::setw(7) << std::fixed << std::setprecision(1)
              << stats->helperPercent() << "%" << std::endl;
    rank++;
  }
  std::cout << std::endl;
}

void OpTimingTracker::printOpBreakdown(const std::string& opName) {
  flush();
  std::lock_guard<std::mutex> lock(_statsMutex);

  // Find op by name
  const AggregatedOpStats* found = nullptr;
  for (const auto& pair : _aggregatedStats) {
    if (pair.second.opName == opName) {
      found = &pair.second;
      break;
    }
  }

  if (!found) {
    std::cout << "Op '" << opName << "' not found in timing data" << std::endl;
    return;
  }

  const char* phaseNames[] = {"Validation", "Shape Calc", "Memory Alloc", "Helper Check", "Helper Exec", "Native Exec",
                              "TOTAL"};

  LongType totalTime = found->phaseTime[static_cast<int>(OpPhase::TOTAL)];

  std::cout << "\n=== " << opName << " Breakdown (" << found->callCount << " calls) ===" << std::endl;
  std::cout << std::setw(16) << std::left << "Phase" << std::right << "  " << std::setw(12) << "Total(ms)" << "  "
            << std::setw(10) << "Avg(us)" << "  " << std::setw(8) << "% of Op" << std::endl;
  std::cout << std::string(52, '-') << std::endl;

  for (int i = 0; i < OP_PHASE_COUNT; i++) {
    double phaseMs = found->phaseTime[i] / 1000000.0;
    double phaseAvgUs = found->callCount > 0 ? (found->phaseTime[i] / 1000.0) / found->callCount : 0.0;
    double phasePct = totalTime > 0 ? (100.0 * found->phaseTime[i] / totalTime) : 0.0;

    std::cout << std::setw(16) << std::left << phaseNames[i] << std::right << "  " << std::setw(12) << std::fixed
              << std::setprecision(2) << phaseMs << "  " << std::setw(10) << std::fixed << std::setprecision(2)
              << phaseAvgUs << "  " << std::setw(7) << std::fixed << std::setprecision(1) << phasePct << "%"
              << std::endl;
  }

  std::cout << "\nStatistics:" << std::endl;
  std::cout << "  Helper used: " << std::fixed << std::setprecision(1) << found->helperPercent() << "% of calls"
            << std::endl;
  std::cout << "  Min: " << std::fixed << std::setprecision(2) << (found->minTimeNanos / 1000.0) << " us"
            << ", Max: " << std::fixed << std::setprecision(2) << (found->maxTimeNanos / 1000.0) << " us"
            << ", StdDev: " << std::fixed << std::setprecision(2) << found->stdDevMicros() << " us" << std::endl;
  std::cout << "  p50: " << std::fixed << std::setprecision(1) << found->histogram.getPercentile(0.50, found->callCount)
            << " us"
            << ", p90: " << std::fixed << std::setprecision(1)
            << found->histogram.getPercentile(0.90, found->callCount) << " us"
            << ", p99: " << std::fixed << std::setprecision(1)
            << found->histogram.getPercentile(0.99, found->callCount) << " us" << std::endl;
  std::cout << std::endl;
}

void OpTimingTracker::printHistogram(const std::string& opName) {
  flush();
  std::lock_guard<std::mutex> lock(_statsMutex);

  // Find op by name
  const AggregatedOpStats* found = nullptr;
  for (const auto& pair : _aggregatedStats) {
    if (pair.second.opName == opName) {
      found = &pair.second;
      break;
    }
  }

  if (!found) {
    std::cout << "Op '" << opName << "' not found in timing data" << std::endl;
    return;
  }

  std::cout << "\n=== " << opName << " Timing Histogram (" << found->callCount << " samples) ===" << std::endl;
  std::cout << "\n  Timing Distribution (logarithmic buckets):" << std::endl;
  std::cout << std::setw(22) << std::left << "  Range (us)" << std::right << "  " << std::setw(10) << "Count" << "  "
            << std::setw(8) << "Percent" << "  "
            << "Histogram" << std::endl;
  std::cout << "  " << std::string(70, '-') << std::endl;

  const char* bucketLabels[] = {"< 1",         "1 - 2",        "2 - 4",         "4 - 8",        "8 - 16",
                                "16 - 32",     "32 - 64",      "64 - 128",      "128 - 256",    "256 - 512",
                                "512 - 1024",  "1024 - 2048",  "2048 - 4096",   "4096 - 8192",  "8192 - 16384",
                                "16384 - 32K", "32K - 65K",    "65K - 131K",    "131K - 262K",  "262K - 524K",
                                "524K - 1M",   "1M - 2M",      "2M - 4M",       "> 4M"};

  // Find max count for scaling
  uint64_t maxCount = 0;
  for (int i = 0; i < TimingHistogram::NUM_BUCKETS; i++) {
    if (found->histogram.buckets[i] > maxCount) maxCount = found->histogram.buckets[i];
  }

  for (int i = 0; i < TimingHistogram::NUM_BUCKETS; i++) {
    if (found->histogram.buckets[i] == 0) continue;

    double pct = found->callCount > 0 ? (100.0 * found->histogram.buckets[i] / found->callCount) : 0.0;
    int barLen = maxCount > 0 ? static_cast<int>(40.0 * found->histogram.buckets[i] / maxCount) : 0;

    std::cout << "  " << std::setw(20) << std::left << bucketLabels[i] << std::right << "  " << std::setw(10)
              << found->histogram.buckets[i] << "  " << std::setw(7) << std::fixed << std::setprecision(1) << pct
              << "%  " << std::string(barLen, '#') << std::endl;
  }

  std::cout << "\n  Percentiles: p50=" << std::fixed << std::setprecision(1)
            << found->histogram.getPercentile(0.50, found->callCount) << "us"
            << ", p90=" << found->histogram.getPercentile(0.90, found->callCount) << "us"
            << ", p99=" << found->histogram.getPercentile(0.99, found->callCount) << "us" << std::endl;
  std::cout << std::endl;
}

void OpTimingTracker::printThreadStats() {
  flush();
  std::lock_guard<std::mutex> lock(_statsMutex);

  std::cout << "\n=== Per-Thread Op Timing ===" << std::endl;
  std::cout << std::setw(20) << "Thread ID" << "  " << std::setw(10) << "Calls" << "  " << std::setw(12) << "Total(ms)"
            << "  " << std::setw(10) << "Avg(us)" << "  " << std::setw(10) << "Min(us)" << "  " << std::setw(10)
            << "Max(us)" << std::endl;
  std::cout << std::string(78, '-') << std::endl;

  for (const auto& pair : _threadStats) {
    const auto& stats = pair.second;
    std::cout << std::setw(20) << std::hash<std::thread::id>{}(stats.threadId) << "  " << std::setw(10)
              << stats.callCount << "  " << std::setw(12) << std::fixed << std::setprecision(2)
              << (stats.totalTimeNanos / 1000000.0) << "  " << std::setw(10) << std::fixed << std::setprecision(2)
              << stats.avgMicros() << "  " << std::setw(10) << std::fixed << std::setprecision(2)
              << (stats.minTimeNanos / 1000.0) << "  " << std::setw(10) << std::fixed << std::setprecision(2)
              << (stats.maxTimeNanos / 1000.0) << std::endl;
  }
  std::cout << std::endl;
}

bool OpTimingTracker::exportChromeTrace(const std::string& filename) {
  flush();
  std::lock_guard<std::mutex> lock(_statsMutex);

  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for Chrome trace export: " << filename << std::endl;
    return false;
  }

  file << "{\"traceEvents\":[" << std::endl;

  bool first = true;
  for (const auto& event : _traceEvents) {
    if (!first) file << "," << std::endl;
    file << event.toJson();
    first = false;
  }

  file << std::endl << "]}" << std::endl;
  file.close();

  std::cout << "Chrome trace exported to: " << filename << " (" << _traceEvents.size() << " events)" << std::endl;
  return true;
}

bool OpTimingTracker::exportCSV(const std::string& filename) {
  flush();
  std::lock_guard<std::mutex> lock(_statsMutex);

  // Debug: print all entries in the map
  std::cerr << "[OpTimingTracker] exportCSV: " << _aggregatedStats.size() << " entries in map" << std::endl;
  for (const auto& pair : _aggregatedStats) {
    std::cerr << "  [DEBUG] Op=" << pair.second.opName
              << " hash=" << pair.first
              << " calls=" << pair.second.callCount
              << " helper=" << pair.second.helperUsedCount << std::endl;
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for CSV export: " << filename << std::endl;
    return false;
  }

  // Header
  file << "OpName,Hash,Calls,TotalMs,AvgUs,StdDevUs,MinUs,MaxUs,HelperPct,"
       << "ValidationMs,ShapeCalcMs,MemoryAllocMs,HelperCheckMs,HelperExecMs,NativeExecMs,"
       << "TotalInputBytes,TotalOutputBytes,TotalMemoryAllocated,"
       << "P50Us,P90Us,P99Us" << std::endl;

  int exported = 0;
  int skippedCorrupt = 0;
  for (const auto& pair : _aggregatedStats) {
    const auto& stats = pair.second;
    // Skip entries with no calls (shouldn't happen with proper initialization)
    if (stats.callCount == 0) continue;

    // Validate for corruption - callCount should be reasonable (< 1 billion for any real workload)
    constexpr uint64_t MAX_REASONABLE_CALLS = 1000000000ULL;  // 1 billion max
    if (stats.callCount > MAX_REASONABLE_CALLS || stats.helperUsedCount > stats.callCount) {
      std::cerr << "WARNING: Skipping corrupted stats entry - Op: " << stats.opName
                << ", Hash: " << stats.opHash
                << ", CallCount: " << stats.callCount
                << ", HelperUsed: " << stats.helperUsedCount << std::endl;
      skippedCorrupt++;
      continue;
    }
    exported++;

    file << stats.opName << "," << stats.opHash << "," << stats.callCount << "," << std::fixed << std::setprecision(3)
         << stats.totalMillis() << "," << stats.avgMicros() << "," << stats.stdDevMicros() << ","
         << (stats.minTimeNanos / 1000.0) << "," << (stats.maxTimeNanos / 1000.0) << "," << stats.helperPercent()
         << ",";

    // Phase times
    for (int i = 0; i < OP_PHASE_COUNT - 1; i++) {
      file << (stats.phaseTime[i] / 1000000.0) << ",";
    }

    // Memory
    file << stats.totalInputBytes << "," << stats.totalOutputBytes << "," << stats.totalMemoryAllocated << ",";

    // Percentiles
    file << stats.histogram.getPercentile(0.50, stats.callCount) << ","
         << stats.histogram.getPercentile(0.90, stats.callCount) << ","
         << stats.histogram.getPercentile(0.99, stats.callCount) << std::endl;
  }

  file.close();
  std::cout << "CSV exported to: " << filename << " (" << exported << " ops";
  if (skippedCorrupt > 0) {
    std::cout << ", " << skippedCorrupt << " corrupted entries skipped";
  }
  std::cout << ")" << std::endl;
  return true;
}

int OpTimingTracker::getNumOps() const {
  std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(_statsMutex));
  return static_cast<int>(_aggregatedStats.size());
}

uint64_t OpTimingTracker::getTotalExecutions() const {
  std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(_statsMutex));
  uint64_t total = 0;
  for (const auto& pair : _aggregatedStats) {
    total += pair.second.callCount;
  }
  return total;
}

const AggregatedOpStats* OpTimingTracker::getOpStats(const std::string& opName) {
  std::lock_guard<std::mutex> lock(_statsMutex);
  for (const auto& pair : _aggregatedStats) {
    if (pair.second.opName == opName) {
      return &pair.second;
    }
  }
  return nullptr;
}

const AggregatedOpStats* OpTimingTracker::getOpStatsByHash(LongType hash) {
  std::lock_guard<std::mutex> lock(_statsMutex);
  auto it = _aggregatedStats.find(hash);
  return it != _aggregatedStats.end() ? &it->second : nullptr;
}

void OpTimingTracker::reset() {
  std::lock_guard<std::mutex> lock(_statsMutex);

  // Reset ring buffer
  _ringIndex.store(0, std::memory_order_relaxed);
  _lastFlushedIdx = 0;
  for (int i = 0; i < RING_SIZE; i++) {
    _ringBuffer[i] = OpTimingRecord{};
  }

  // Clear aggregated stats
  _aggregatedStats.clear();
  _threadStats.clear();
  _traceEvents.clear();
}

// CudaEventTimer implementation moved to helpers/cuda/OpTimingTracker_cuda.cu

}  // namespace graph
}  // namespace sd
