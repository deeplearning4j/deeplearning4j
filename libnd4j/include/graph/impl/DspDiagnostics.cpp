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

#include <graph/DspDiagnostics.h>
#include <execution/LaunchContext.h>
#include <system/Environment.h>

#include <algorithm>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <functional>
#include <sstream>
#include <vector>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

namespace sd {
namespace graph {

// ─── Category name table ─────────────────────────────────────────────────────

static const char* const sCategoryNames[DSP_DIAG_NUM_CATEGORIES] = {
    "COMPILE", "JIT",      "EXECUTE",  "TIMING",
    "MEMORY",  "BACKEND",  "SHAPE",    "SEGMENT",
    "FUSION",  "VERIFY",   "KV_CACHE", "FALLBACK",
    "TRANSFER"
};

// ─── Singleton ───────────────────────────────────────────────────────────────

DspDiagnostics& DspDiagnostics::getInstance() {
  static DspDiagnostics instance;
  return instance;
}

DspDiagnostics::DspDiagnostics()
    : enabledMask_(DSP_DIAG_NONE),
      level_(DSP_LEVEL_SUMMARY),
      writePos_(0),
      planNumSlots_(0),
      planNumSegments_(0),
      stepsExecuted_(0),
      planStartUs_(0),
      planTotalUs_(0),
      lastStepStartUs_(0) {
  std::memset(events_, 0, sizeof(events_));
  parseEnvVars();
  applyLegacyFlags();
}

// ─── Timestamp helper ────────────────────────────────────────────────────────

static int64_t nowUs() {
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

// ─── Configuration ───────────────────────────────────────────────────────────

void DspDiagnostics::setCategories(uint32_t mask) {
  enabledMask_.store(mask, std::memory_order_relaxed);
}

void DspDiagnostics::enableCategories(uint32_t mask) {
  enabledMask_.fetch_or(mask, std::memory_order_relaxed);
}

void DspDiagnostics::disableCategories(uint32_t mask) {
  enabledMask_.fetch_and(~mask, std::memory_order_relaxed);
}

uint32_t DspDiagnostics::getEnabledMask() const {
  return enabledMask_.load(std::memory_order_relaxed);
}

void DspDiagnostics::setLevel(DspDiagLevel level) {
  level_.store(static_cast<int>(level), std::memory_order_relaxed);
}

DspDiagLevel DspDiagnostics::getLevel() const {
  return static_cast<DspDiagLevel>(level_.load(std::memory_order_relaxed));
}

void DspDiagnostics::setJsonPath(const std::string& path) {
  std::lock_guard<std::mutex> lock(eventMutex_);
  jsonPath_ = path;
}

// ─── Event recording ─────────────────────────────────────────────────────────

void DspDiagnostics::recordEvent(uint32_t category, int slotId, int segmentId,
                                  int deviceId, const char* opName,
                                  int64_t timingUs, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  recordEventV(category, slotId, segmentId, deviceId, opName, timingUs, fmt, args);
  va_end(args);
}

void DspDiagnostics::recordEventV(uint32_t category, int slotId, int segmentId,
                                   int deviceId, const char* opName,
                                   int64_t timingUs, const char* fmt, va_list args) {
  // Format message
  char msgBuf[DSP_DIAG_MSG_LEN];
  vsnprintf(msgBuf, DSP_DIAG_MSG_LEN, fmt, args);
  msgBuf[DSP_DIAG_MSG_LEN - 1] = '\0';

  int64_t ts = planStartUs_ > 0 ? (nowUs() - planStartUs_) : 0;

  // Store in ring buffer
  {
    std::lock_guard<std::mutex> lock(eventMutex_);
    int64_t pos = writePos_.fetch_add(1, std::memory_order_relaxed);
    int idx = static_cast<int>(pos & DSP_DIAG_RING_MASK);

    DspDiagEvent& ev = events_[idx];
    ev.category    = category;
    ev.timestampUs = ts;
    ev.threadId    = std::hash<std::thread::id>{}(std::this_thread::get_id());
    ev.slotId      = slotId;
    ev.segmentId   = segmentId;
    ev.deviceId    = deviceId;
    ev.timingUs    = timingUs;

    if (opName != nullptr) {
      std::strncpy(ev.opName, opName, DSP_DIAG_OPNAME_LEN - 1);
      ev.opName[DSP_DIAG_OPNAME_LEN - 1] = '\0';
    } else {
      ev.opName[0] = '\0';
    }

    std::strncpy(ev.message, msgBuf, DSP_DIAG_MSG_LEN - 1);
    ev.message[DSP_DIAG_MSG_LEN - 1] = '\0';

    // Update category stats
    int catIdx = categoryIndex(category);
    if (catIdx >= 0 && catIdx < DSP_DIAG_NUM_CATEGORIES) {
      auto& stats = categoryStats_[catIdx];
      stats.eventCount++;
      if (timingUs > 0) {
        stats.totalTimingUs += timingUs;
        if (timingUs < stats.minTimingUs) stats.minTimingUs = timingUs;
        if (timingUs > stats.maxTimingUs) stats.maxTimingUs = timingUs;
      }
    }
  }

  // Echo to stdout if FULL level (stdout is captured by surefire; stderr is not)
  if (getLevel() == DSP_LEVEL_FULL) {
    int catIdx = categoryIndex(category);
    const char* catName = (catIdx >= 0) ? sCategoryNames[catIdx] : "UNKNOWN";
    fprintf(stdout, "[DSP_DIAG] [%s] ", catName);
    if (segmentId >= 0) fprintf(stdout, "seg[%d] ", segmentId);
    if (slotId >= 0)    fprintf(stdout, "slot %d ", slotId);
    if (opName && opName[0]) fprintf(stdout, "(%s) ", opName);
    if (timingUs > 0)   fprintf(stdout, "%lldus ", static_cast<long long>(timingUs));
    fprintf(stdout, "%s\n", msgBuf);
    fflush(stdout);
  }
}

// ─── Plan lifecycle ──────────────────────────────────────────────────────────

void DspDiagnostics::beginPlanExecution(int numSlots, int numSegments) {
  std::lock_guard<std::mutex> lock(eventMutex_);
  planNumSlots_    = numSlots;
  planNumSegments_ = numSegments;
  stepsExecuted_   = 0;
  planStartUs_     = nowUs();
  planTotalUs_     = 0;
}

void DspDiagnostics::endPlanExecution() {
  std::lock_guard<std::mutex> lock(eventMutex_);
  if (planStartUs_ > 0) {
    planTotalUs_ = nowUs() - planStartUs_;
  }
}

void DspDiagnostics::beginStep(int stepNumber) {
  lastStepStartUs_ = nowUs();
}

void DspDiagnostics::endStep(int stepNumber) {
  std::lock_guard<std::mutex> lock(eventMutex_);
  stepsExecuted_++;
}

// ─── Category helpers ────────────────────────────────────────────────────────

const char* DspDiagnostics::categoryName(uint32_t category) {
  int idx = categoryIndex(category);
  if (idx >= 0 && idx < DSP_DIAG_NUM_CATEGORIES) return sCategoryNames[idx];
  return "UNKNOWN";
}

int DspDiagnostics::categoryIndex(uint32_t category) {
  if (category == 0) return -1;
  // Find lowest set bit
  int idx = 0;
  uint32_t v = category;
  while ((v & 1u) == 0 && idx < DSP_DIAG_NUM_CATEGORIES) {
    v >>= 1;
    idx++;
  }
  return (idx < DSP_DIAG_NUM_CATEGORIES) ? idx : -1;
}

// ─── String parsing ──────────────────────────────────────────────────────────

uint32_t DspDiagnostics::parseCategories(const char* str) {
  if (str == nullptr || str[0] == '\0') return DSP_DIAG_NONE;

  std::string s(str);
  // Convert to uppercase
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);

  if (s == "ALL" || s == "*") return DSP_DIAG_ALL;
  if (s == "NONE" || s == "OFF" || s == "0") return DSP_DIAG_NONE;

  uint32_t mask = DSP_DIAG_NONE;
  std::istringstream iss(s);
  std::string token;
  while (std::getline(iss, token, ',')) {
    // Trim whitespace
    size_t start = token.find_first_not_of(" \t");
    size_t end   = token.find_last_not_of(" \t");
    if (start == std::string::npos) continue;
    token = token.substr(start, end - start + 1);

    for (int i = 0; i < DSP_DIAG_NUM_CATEGORIES; i++) {
      if (token == sCategoryNames[i]) {
        mask |= (1u << i);
        break;
      }
    }
  }
  return mask;
}

// ─── Environment variable parsing ────────────────────────────────────────────

void DspDiagnostics::parseEnvVars() {
  const char* cats = std::getenv("ND4J_DSP_DIAGNOSTICS");
  if (cats != nullptr) {
    enabledMask_.store(parseCategories(cats), std::memory_order_relaxed);
  }

  const char* lvl = std::getenv("ND4J_DSP_DIAGNOSTICS_LEVEL");
  if (lvl != nullptr) {
    std::string s(lvl);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "full" || s == "2")     setLevel(DSP_LEVEL_FULL);
    else if (s == "detailed" || s == "1") setLevel(DSP_LEVEL_DETAILED);
    else                              setLevel(DSP_LEVEL_SUMMARY);
  }

  const char* path = std::getenv("ND4J_DSP_DIAGNOSTICS_FILE");
  if (path != nullptr) {
    jsonPath_ = path;
  }
}

// ─── Legacy flag mapping ─────────────────────────────────────────────────────

void DspDiagnostics::applyLegacyFlags() {
  // nd4j.dsp.trace / ND4J_DSP_TRACE -> EXECUTE
  const char* trace = std::getenv("ND4J_DSP_TRACE");
  if (trace != nullptr) {
    enableCategories(DSP_DIAG_EXECUTE);
  }

  // ND4J_TRITON_VERBOSE -> COMPILE + JIT + BACKEND
  const char* tritonVerbose = std::getenv("ND4J_TRITON_VERBOSE");
  if (tritonVerbose != nullptr) {
    enableCategories(DSP_DIAG_COMPILE | DSP_DIAG_JIT | DSP_DIAG_BACKEND);
  }

  // nd4j.dsp.executionTiming -> TIMING
  const char* timing = std::getenv("ND4J_DSP_EXECUTION_TIMING");
  if (timing != nullptr) {
    enableCategories(DSP_DIAG_TIMING);
  }

  // nd4j.dsp.native.dumpOutputs -> VERIFY
  const char* dumpNative = std::getenv("ND4J_DSP_NATIVE_DUMP_OUTPUTS");
  if (dumpNative != nullptr) {
    enableCategories(DSP_DIAG_VERIFY);
  }

  // tritonVerifyKernels -> VERIFY + FULL level (auto-enable diagnostics output when verification is on)
  if (sd::Environment::getInstance().tritonVerifyKernels()) {
    enableCategories(DSP_DIAG_VERIFY);
    setLevel(DSP_LEVEL_FULL);
  }
}

// ─── Report generation ───────────────────────────────────────────────────────

std::string DspDiagnostics::generatePlanReport() const {
  std::lock_guard<std::mutex> lock(eventMutex_);
  std::ostringstream os;

  double totalMs = planTotalUs_ / 1000.0;
  double avgMs   = stepsExecuted_ > 0 ? totalMs / stepsExecuted_ : 0;

  os << "\n";
  os << "+==================================================================+\n";
  os << "|              DSP EXECUTION DIAGNOSTICS REPORT                    |\n";
  os << "+==================================================================+\n";
  os << "| Plan: " << planNumSlots_ << " slots, " << planNumSegments_
     << " segments, " << stepsExecuted_ << " steps\n";
  os << "| Total wall time: " << std::fixed << std::setprecision(1)
     << totalMs << "ms (" << avgMs << "ms/step avg)\n";
  os << "+-----------+--------+------------+----------+--------+-----------+\n";
  os << "| Category  | Events | Total Time | Avg Time | Min    | Max       |\n";
  os << "+-----------+--------+------------+----------+--------+-----------+\n";

  for (int i = 0; i < DSP_DIAG_NUM_CATEGORIES; i++) {
    const auto& s = categoryStats_[i];
    if (s.eventCount == 0) continue;

    os << "| " << std::setw(9) << std::left << sCategoryNames[i] << " | "
       << std::setw(6) << std::right << s.eventCount << " | ";

    if (s.totalTimingUs > 0) {
      double totalCatMs = s.totalTimingUs / 1000.0;
      double avgCatMs   = totalCatMs / s.eventCount;
      double minCatMs   = s.minTimingUs / 1000.0;
      double maxCatMs   = s.maxTimingUs / 1000.0;
      os << std::setw(8) << std::fixed << std::setprecision(1) << totalCatMs << "ms | "
         << std::setw(6) << std::setprecision(1) << avgCatMs << "ms | "
         << std::setw(4) << std::setprecision(1) << minCatMs << "ms | "
         << std::setw(7) << std::setprecision(1) << maxCatMs << "ms |";
    } else {
      os << "      --   |     -- |   --   |      --   |";
    }
    os << "\n";
  }

  os << "+-----------+--------+------------+----------+--------+-----------+\n";

  // Show FALLBACK events (always interesting)
  int64_t totalEvents = writePos_.load(std::memory_order_relaxed);
  int64_t startIdx = totalEvents > DSP_DIAG_RING_SIZE
                         ? totalEvents - DSP_DIAG_RING_SIZE
                         : 0;

  bool hasFallback = false;
  for (int64_t i = startIdx; i < totalEvents; i++) {
    const auto& ev = events_[i & DSP_DIAG_RING_MASK];
    if (ev.category == DSP_DIAG_FALLBACK) {
      if (!hasFallback) {
        os << "| FALLBACK events:                                                 |\n";
        hasFallback = true;
      }
      os << "|  [" << std::setw(8) << std::setprecision(1)
         << ev.timestampUs / 1000.0 << "ms] ";
      if (ev.segmentId >= 0) os << "seg[" << ev.segmentId << "] ";
      if (ev.slotId >= 0)    os << "slot " << ev.slotId << " ";
      if (ev.opName[0])      os << "(" << ev.opName << ") ";
      os << ev.message << "\n";
    }
  }

  if (hasFallback) {
    os << "+==================================================================+\n";
  }

  return os.str();
}

std::string DspDiagnostics::generateJsonReport() const {
  std::lock_guard<std::mutex> lock(eventMutex_);
  std::ostringstream os;

  double totalMs = planTotalUs_ / 1000.0;

  os << "{\n";
  os << "  \"planInfo\": {\n";
  os << "    \"numSlots\": " << planNumSlots_ << ",\n";
  os << "    \"numSegments\": " << planNumSegments_ << ",\n";
  os << "    \"stepsExecuted\": " << stepsExecuted_ << ",\n";
  os << "    \"totalTimeMs\": " << std::fixed << std::setprecision(1) << totalMs << "\n";
  os << "  },\n";

  // Category stats
  os << "  \"categoryStats\": {\n";
  bool first = true;
  for (int i = 0; i < DSP_DIAG_NUM_CATEGORIES; i++) {
    const auto& s = categoryStats_[i];
    if (s.eventCount == 0) continue;
    if (!first) os << ",\n";
    first = false;

    os << "    \"" << sCategoryNames[i] << "\": { "
       << "\"events\": " << s.eventCount;
    if (s.totalTimingUs > 0) {
      os << ", \"totalTimeUs\": " << s.totalTimingUs
         << ", \"avgTimeUs\": " << (s.totalTimingUs / s.eventCount)
         << ", \"minTimeUs\": " << s.minTimingUs
         << ", \"maxTimeUs\": " << s.maxTimingUs;
    }
    os << " }";
  }
  os << "\n  },\n";

  // Events array
  os << "  \"events\": [\n";
  int64_t totalEvents = writePos_.load(std::memory_order_relaxed);
  int64_t startIdx = totalEvents > DSP_DIAG_RING_SIZE
                         ? totalEvents - DSP_DIAG_RING_SIZE
                         : 0;

  bool firstEv = true;
  for (int64_t i = startIdx; i < totalEvents; i++) {
    const auto& ev = events_[i & DSP_DIAG_RING_MASK];
    if (!firstEv) os << ",\n";
    firstEv = false;

    int catIdx = categoryIndex(ev.category);
    const char* catName = (catIdx >= 0) ? sCategoryNames[catIdx] : "UNKNOWN";

    os << "    { \"category\": \"" << catName << "\""
       << ", \"timestampUs\": " << ev.timestampUs;
    if (ev.segmentId >= 0) os << ", \"segmentId\": " << ev.segmentId;
    if (ev.slotId >= 0)    os << ", \"slotId\": " << ev.slotId;
    if (ev.deviceId >= 0)  os << ", \"deviceId\": " << ev.deviceId;
    if (ev.opName[0])      os << ", \"opName\": \"" << ev.opName << "\"";
    if (ev.timingUs > 0)   os << ", \"timingUs\": " << ev.timingUs;

    // Escape message for JSON
    os << ", \"message\": \"";
    for (int c = 0; ev.message[c] != '\0'; c++) {
      char ch = ev.message[c];
      if (ch == '"')       os << "\\\"";
      else if (ch == '\\') os << "\\\\";
      else if (ch == '\n') os << "\\n";
      else                 os << ch;
    }
    os << "\" }";
  }
  os << "\n  ]\n";
  os << "}\n";

  return os.str();
}

void DspDiagnostics::printPlanReport() const {
  if (getEnabledMask() == DSP_DIAG_NONE) return;
  std::string report = generatePlanReport();
  fprintf(stdout, "%s", report.c_str());
  fflush(stdout);
}

void DspDiagnostics::flushJsonReport() const {
  std::string path;
  {
    std::lock_guard<std::mutex> lock(eventMutex_);
    path = jsonPath_;
  }
  if (path.empty()) return;

  std::string json = generateJsonReport();
  std::ofstream ofs(path);
  if (ofs.is_open()) {
    ofs << json;
    ofs.close();
  }
}

// ─── Clear ───────────────────────────────────────────────────────────────────

void DspDiagnostics::clear() {
  std::lock_guard<std::mutex> lock(eventMutex_);
  writePos_.store(0, std::memory_order_relaxed);
  for (int i = 0; i < DSP_DIAG_NUM_CATEGORIES; i++) {
    categoryStats_[i] = DspDiagCategoryStats();
  }
  planNumSlots_    = 0;
  planNumSegments_ = 0;
  stepsExecuted_   = 0;
  planStartUs_     = 0;
  planTotalUs_     = 0;
  lastStepStartUs_ = 0;
}

// ─── Address snapshot for graph replay validation ────────────────────────────

void DspDiagnostics::snapshotAddresses(const char* tag, void** outputSlots,
                                        int numOutputSlots,
                                        void** externalArrays, int numExternals) {
  std::vector<AddrEntry> entries;
  entries.reserve(numOutputSlots + numExternals);

  for (int i = 0; i < numOutputSlots; i++) {
    void* addr = (outputSlots != nullptr && outputSlots[i] != nullptr)
                 ? outputSlots[i] : nullptr;
    entries.push_back({i, addr, 0});
  }
  for (int i = 0; i < numExternals; i++) {
    void* addr = (externalArrays != nullptr && externalArrays[i] != nullptr)
                 ? externalArrays[i] : nullptr;
    entries.push_back({-(i + 1), addr, 0});
  }

  std::string key(tag);
  {
    std::lock_guard<std::mutex> lock(addrMutex_);
    addrSnapshots_[key] = std::move(entries);
  }

  recordEvent(DSP_DIAG_EXECUTE, -1, -1, -1, tag, 0,
              "address snapshot: %d output slots + %d externals",
              numOutputSlots, numExternals);
}

int DspDiagnostics::compareAddressSnapshots(const char* tagA, const char* tagB) {
  std::lock_guard<std::mutex> lock(addrMutex_);

  auto itA = addrSnapshots_.find(std::string(tagA));
  auto itB = addrSnapshots_.find(std::string(tagB));
  if (itA == addrSnapshots_.end() || itB == addrSnapshots_.end()) {
    recordEvent(DSP_DIAG_EXECUTE, -1, -1, -1, nullptr, 0,
                "compareAddressSnapshots: missing snapshot '%s' or '%s'", tagA, tagB);
    return -1;
  }

  auto& a = itA->second;
  auto& b = itB->second;
  int maxLen = std::max(static_cast<int>(a.size()), static_cast<int>(b.size()));
  int mismatches = 0;
  int nullToNonNull = 0;
  int nonNullToNull = 0;
  int ptrChanged = 0;

  for (int i = 0; i < maxLen; i++) {
    void* addrA = (i < static_cast<int>(a.size())) ? a[i].addr : nullptr;
    void* addrB = (i < static_cast<int>(b.size())) ? b[i].addr : nullptr;
    int idx = (i < static_cast<int>(a.size())) ? a[i].index
            : (i < static_cast<int>(b.size())) ? b[i].index : i;

    if (addrA != addrB) {
      mismatches++;
      if (addrA == nullptr) nullToNonNull++;
      else if (addrB == nullptr) nonNullToNull++;
      else ptrChanged++;

      if (mismatches <= 20) {  // cap detailed logging
        const char* kind = (idx >= 0) ? "slot" : "ext";
        int dispIdx = (idx >= 0) ? idx : -(idx + 1);
        recordEvent(DSP_DIAG_EXECUTE, idx, -1, -1, nullptr, 0,
                    "ADDR MISMATCH %s[%d]: %s=%p vs %s=%p",
                    kind, dispIdx, tagA, addrA, tagB, addrB);
      }
    }
  }

  if (mismatches > 0) {
    recordEvent(DSP_DIAG_EXECUTE, -1, -1, -1, nullptr, 0,
                "address diff '%s' vs '%s': %d mismatches (%d ptr-changed, %d null→non-null, %d non-null→null)",
                tagA, tagB, mismatches, ptrChanged, nullToNonNull, nonNullToNull);
  } else {
    recordEvent(DSP_DIAG_EXECUTE, -1, -1, -1, nullptr, 0,
                "address diff '%s' vs '%s': IDENTICAL (%d entries)",
                tagA, tagB, maxLen);
  }

  return mismatches;
}

void DspDiagnostics::clearAddressSnapshots() {
  std::lock_guard<std::mutex> lock(addrMutex_);
  addrSnapshots_.clear();
}

// ─── Helper: safe D2H copy on a specific stream (capture-safe) ───────────────
#ifdef SD_CUDA
static bool safeDtoH(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
  // Never copy during graph capture — would bake a memcpy node into the graph
  if (stream != nullptr) {
    cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &capStat);
    if (capStat != cudaStreamCaptureStatusNone) return false;
  }
  cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return true;
}
#endif

// ─── Slot buffer dump (debug-gated, D2H via stream-async copy) ───────────────
void DspDiagnostics::dumpSlotBuffer(const char* tag, int slotIdx,
                                     const void* devicePtr,
                                     int64_t numElements, int sampleCount) {
  if (devicePtr == nullptr || numElements <= 0) return;

  int n = std::min(sampleCount, static_cast<int>(numElements));
  std::vector<float> host(n);

#ifdef SD_CUDA
  auto* lc = LaunchContext::defaultContext();
  cudaStream_t stream = lc && lc->getCudaStream() ? *lc->getCudaStream() : nullptr;
  if (!safeDtoH(host.data(), devicePtr, n * sizeof(float), stream)) return;
#else
  std::memcpy(host.data(), devicePtr, n * sizeof(float));
#endif

  // Build value string
  char buf[512];
  int off = 0;
  for (int i = 0; i < n && off < static_cast<int>(sizeof(buf)) - 16; i++) {
    off += snprintf(buf + off, sizeof(buf) - off, "%s%.4f", i > 0 ? "," : "", host[i]);
  }

  recordEvent(DSP_DIAG_EXECUTE, slotIdx, -1, -1, tag, 0,
              "slot[%d] len=%lld first%d=[%s]",
              slotIdx, static_cast<long long>(numElements), n, buf);
}

// ─── Segment output dump (replaces copy-pasted topVal+first-N blocks) ────────
void DspDiagnostics::dumpSegmentOutput(const char* tag, int endSlot,
                                        const void* devicePtr,
                                        int64_t numElements, int execCount,
                                        void* stream, int sampleCount) {
  if (devicePtr == nullptr || numElements <= 0) return;

  int n = std::min(sampleCount, static_cast<int>(numElements));
  std::vector<float> host(n, 0.0f);

#ifdef SD_CUDA
  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;
  if (!safeDtoH(host.data(), devicePtr, n * sizeof(float), cudaStr)) return;
#else
  std::memcpy(host.data(), devicePtr, n * sizeof(float));
#endif

  // Build value string
  char valBuf[256];
  int off = 0;
  for (int i = 0; i < n && off < static_cast<int>(sizeof(valBuf)) - 16; i++) {
    off += snprintf(valBuf + off, sizeof(valBuf) - off, "%s%.5f", i > 0 ? ", " : "", host[i]);
  }

  recordEvent(DSP_DIAG_EXECUTE, endSlot, -1, -1, tag, 0,
              "%s topVal[0]=%.5f addr=%p [%s] (endSlot=%d len=%lld execCount=%d)",
              tag, host[0], devicePtr, valBuf, endSlot,
              static_cast<long long>(numElements), execCount);
}

// ─── External input actuality state dump ─────────────────────────────────────
DspDiagnostics::ExtInputSyncResult DspDiagnostics::dumpExternalInputState(
    NDArray** externalArrays, int numExt, int execCount, int maxToDump) {
  ExtInputSyncResult result = {0, 0, 0};
  if (externalArrays == nullptr || numExt <= 0) return result;

#ifdef SD_CUDA
  auto* lc = LaunchContext::defaultContext();
  cudaStream_t diagStream = lc && lc->getCudaStream() ? *lc->getCudaStream() : nullptr;
#endif

  result.total = numExt;
  for (int ei = 0; ei < numExt; ei++) {
    if (externalArrays[ei] == nullptr) continue;
    auto* db = externalArrays[ei]->dataBuffer();
    bool pAct = db ? db->isPrimaryActual() : false;
    bool sAct = db ? db->isSpecialActual() : false;

    if (sAct && !pAct) {
      result.skipped++;
    } else {
      result.synced++;
    }

    if (ei < maxToDump) {
      char valBuf[128] = {0};
#ifdef SD_CUDA
      if (db && db->special() && db->getLenInBytes() >= static_cast<int64_t>(sizeof(float))) {
        float vals[4] = {0};
        int nCopy = std::min(4, static_cast<int>(db->getLenInBytes() / sizeof(float)));
        if (safeDtoH(vals, db->special(), nCopy * sizeof(float), diagStream)) {
          int off = 0;
          for (int i = 0; i < nCopy && off < static_cast<int>(sizeof(valBuf)) - 16; i++) {
            off += snprintf(valBuf + off, sizeof(valBuf) - off, "%s%.5f", i > 0 ? ", " : "", vals[i]);
          }
        }
      }
#endif
      recordEvent(DSP_DIAG_EXECUTE, -(ei + 1), -1, -1, nullptr, 0,
                  "EXT_INPUT[%d] pAct=%d sAct=%d bytes=%lld addr=%p device=[%s] execCount=%d",
                  ei, pAct ? 1 : 0, sAct ? 1 : 0,
                  static_cast<long long>(db ? db->getLenInBytes() : 0),
                  db ? db->special() : nullptr, valBuf, execCount);
    }
  }

  recordEvent(DSP_DIAG_EXECUTE, -1, -1, -1, nullptr, 0,
              "EXT_INPUT_SYNC: %d synced, %d skipped (sAct=1, no H2D) execCount=%d",
              result.synced, result.skipped, execCount);

  return result;
}

}  // namespace graph
}  // namespace sd
