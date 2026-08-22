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
#include <cctype>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <vector>


namespace sd {
namespace graph {

// ─── Category name table ─────────────────────────────────────────────────────

static const char* const sCategoryNames[DSP_DIAG_NUM_CATEGORIES] = {
    "COMPILE", "JIT",      "EXECUTE",  "TIMING",
    "MEMORY",  "BACKEND",  "SHAPE",    "SEGMENT",
    "FUSION",  "VERIFY",   "KV_CACHE", "FALLBACK",
    "TRANSFER", "EMULATED_REPLAY",
    "STREAM_SYNC", "MULTI_DEVICE", "GRAPH_REPLAY",
    "SEGMENT_BUCKETS", "LIFECYCLE", "COLORING"
};

// ─── Singleton ───────────────────────────────────────────────────────────────

DspDiagnostics& DspDiagnostics::getInstance() {
  static DspDiagnostics* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new DspDiagnostics();
  });
  return *instance;
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
      lastStepStartUs_(0),
      diagExecLimit_(0),
      diagDetailLimit_(20),
      traceExtInput_(-1),
      traceSlot_(-1) {
  std::memset(events_, 0, sizeof(events_));
  applyDspConfig();
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

  // Echo to stdout if FULL level OR if debug+verbose is enabled
  // (stdout is captured by surefire; stderr is not)
  if (getLevel() == DSP_LEVEL_FULL || sd::Environment::getInstance().isDebugAndVerbose()) {
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

// ─── Environment-driven configuration ──────────────────────────────────────
//
// All environment variable parsing is centralized in DspConfig::initFromEnvironment().
// DspDiagnostics reads configuration from the Environment singleton — no direct
// std::getenv or EnvHelper calls. This is the single source of truth.

void DspDiagnostics::applyDspConfig() {
  auto& cfg = sd::Environment::getInstance().dsp();

  // Categories
  if (!cfg.diagnosticsCategories().empty()) {
    enabledMask_.store(parseCategories(cfg.diagnosticsCategories().c_str()),
                       std::memory_order_relaxed);
  }

  // Level
  if (!cfg.diagnosticsLevel().empty()) {
    std::string s = cfg.diagnosticsLevel();
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "full" || s == "2")       setLevel(DSP_LEVEL_FULL);
    else if (s == "detailed" || s == "1") setLevel(DSP_LEVEL_DETAILED);
    else                               setLevel(DSP_LEVEL_SUMMARY);
  }

  // File
  if (!cfg.diagnosticsFile().empty()) {
    jsonPath_ = cfg.diagnosticsFile();
  }

  // Legacy boolean flags → categories
  if (cfg.diagnosticsTrace()) {
    enableCategories(DSP_DIAG_EXECUTE);
  }
  if (cfg.diagnosticsTiming()) {
    enableCategories(DSP_DIAG_TIMING);
  }
  if (cfg.diagnosticsNativeDump()) {
    enableCategories(DSP_DIAG_VERIFY);
  }

  // ND4J_TRITON_VERBOSE → COMPILE + JIT + BACKEND
  if (sd::Environment::getInstance().tritonVerbose()) {
    enableCategories(DSP_DIAG_COMPILE | DSP_DIAG_JIT | DSP_DIAG_BACKEND);
  }

  // tritonVerifyKernels → VERIFY + FULL level
  if (sd::Environment::getInstance().tritonVerifyKernels()) {
    enableCategories(DSP_DIAG_VERIFY);
    setLevel(DSP_LEVEL_FULL);
  }

  // Configurable limits
  diagExecLimit_   = cfg.diagExecLimit();
  diagDetailLimit_ = cfg.diagDetailLimit();
  traceExtInput_   = cfg.traceExtInput();
  traceSlot_       = cfg.traceSlot();
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

// ─── Vulkan diagnostic helpers ───────────────────────────────────────────────
//
// Scans the ring buffer for GRAPH_REPLAY events emitted by VulkanReplayHandle
// whose message starts with "vulkan_backend ".  Aggregates the most recent
// CAPTURE_DONE and REPLAY_DONE stats into a compact structure for JSON output.
//
// This is a pure read — no allocations beyond the returned struct, no side
// effects.  Called only from generateJsonReport() while the event lock is held.

namespace {

struct VulkanDiagSummary {
  bool found = false;
  std::string deviceName;
  uint32_t apiVersion = 0;
  int dispatches = 0;
  double captureMs = 0.0;
  size_t workspaceBytes = 0;
  bool umaDetected = false;
  bool fp16Supported = false;
  int replayCount = 0;
  double lastReplayMs = 0.0;
};

// Extract a quoted-string value for key= from a message like:
//   vulkan_backend CAPTURE_DONE device="Adreno 8 Gen 3" api_version=0x...
// Returns empty string if key not found or value not quoted.
static std::string extractStringValue(const std::string& msg, const char* key) {
  std::string needle = std::string(key) + "=\"";
  auto pos = msg.find(needle);
  if (pos == std::string::npos) return "";
  pos += needle.size();
  auto end = msg.find('"', pos);
  if (end == std::string::npos) return "";
  return msg.substr(pos, end - pos);
}

// Extract a numeric value for key= (stops at space or end of string).
static std::string extractNumericValue(const std::string& msg, const char* key) {
  std::string needle = std::string(key) + "=";
  auto pos = msg.find(needle);
  if (pos == std::string::npos) return "";
  pos += needle.size();
  auto end = msg.find_first_of(" \t\n", pos);
  return msg.substr(pos, (end == std::string::npos) ? std::string::npos : end - pos);
}

// Check whether msg is a Vulkan diagnostic event and if so whether it is a
// CAPTURE_DONE or REPLAY_DONE event.
static bool isVulkanEvent(const char* msg) {
  return msg != nullptr && strncmp(msg, "vulkan_backend ", 15) == 0;
}

}  // anonymous namespace

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

  // Segment terminal summaries — persistent, never overwritten by ring buffer
  os << "  \"segmentTerminals\": [\n";
  for (int i = 0; i < segTerminalCount_; i++) {
    const auto& r = segTerminals_[i];
    if (i > 0) os << ",\n";
    os << "    { \"seg\": [" << r.startSlot << ", " << r.endSlot << "]"
       << ", \"execCount\": " << r.execCountAtTransition
       << ", \"outcome\": " << r.outcome
       << ", \"phase\": \"" << r.phase << "\""
       << ", \"reason\": \"" << r.reason << "\""
       << ", \"backend\": \"" << r.backend << "\""
       << ", \"timestampUs\": " << r.timestampUs
       << " }";
  }
  os << "\n  ],\n";

  // Events array — also scans for Vulkan GRAPH_REPLAY events while iterating
  os << "  \"events\": [\n";
  int64_t totalEvents = writePos_.load(std::memory_order_relaxed);
  int64_t startIdx = totalEvents > DSP_DIAG_RING_SIZE
                         ? totalEvents - DSP_DIAG_RING_SIZE
                         : 0;

  // Accumulate the most recent Vulkan diagnostic state while iterating events.
  // We update vulkanSummary in-pass so we only traverse the ring buffer once.
  VulkanDiagSummary vulkanSummary;

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

    // Extract Vulkan stats from GRAPH_REPLAY events emitted by VulkanReplayHandle.
    // Messages start with "vulkan_backend " and contain key=value pairs.
    if (ev.category == DSP_DIAG_GRAPH_REPLAY && isVulkanEvent(ev.message)) {
      std::string msg(ev.message);
      vulkanSummary.found = true;

      // Device name (quoted string) — always present in both event types
      std::string devName = extractStringValue(msg, "device");
      if (!devName.empty()) vulkanSummary.deviceName = devName;

      if (msg.find("CAPTURE_DONE") != std::string::npos) {
        // vulkan_backend CAPTURE_DONE device="..." api_version=0x... dispatches=N
        //   capture_ms=F workspace_bytes=N uma=N fp16=N
        std::string apiStr = extractNumericValue(msg, "api_version");
        if (!apiStr.empty()) {
          vulkanSummary.apiVersion = static_cast<uint32_t>(std::stoul(apiStr, nullptr, 0));
        }
        std::string dStr = extractNumericValue(msg, "dispatches");
        if (!dStr.empty()) vulkanSummary.dispatches = std::stoi(dStr);
        std::string cStr = extractNumericValue(msg, "capture_ms");
        if (!cStr.empty()) vulkanSummary.captureMs = std::stod(cStr);
        std::string wStr = extractNumericValue(msg, "workspace_bytes");
        if (!wStr.empty()) vulkanSummary.workspaceBytes = std::stoull(wStr);
        std::string umaStr = extractNumericValue(msg, "uma");
        if (!umaStr.empty()) vulkanSummary.umaDetected = (umaStr == "1");
        std::string fp16Str = extractNumericValue(msg, "fp16");
        if (!fp16Str.empty()) vulkanSummary.fp16Supported = (fp16Str == "1");

      } else if (msg.find("REPLAY_DONE") != std::string::npos) {
        // vulkan_backend REPLAY_DONE device="..." replay_count=N replay_ms=F dispatches=N
        std::string rcStr = extractNumericValue(msg, "replay_count");
        if (!rcStr.empty()) vulkanSummary.replayCount = std::stoi(rcStr);
        std::string rmStr = extractNumericValue(msg, "replay_ms");
        if (!rmStr.empty()) vulkanSummary.lastReplayMs = std::stod(rmStr);
        std::string dStr2 = extractNumericValue(msg, "dispatches");
        if (!dStr2.empty()) vulkanSummary.dispatches = std::stoi(dStr2);
      }
    }
  }
  os << "\n  ]";

  // Emit a "vulkan" top-level object when any Vulkan GRAPH_REPLAY events were found.
  // This gives a single, easily queryable location for Vulkan backend metrics
  // (jq '.vulkan' returns the full block; jq '.vulkan.replay_ms' returns the metric).
  if (vulkanSummary.found) {
    // Format the Vulkan API version as "major.minor.patch" for readability.
    // Avoid including vulkan.h here — use the bit layout directly:
    //   bits [31:22] = major, [21:12] = minor, [11:0] = patch
    uint32_t vkMajor = (vulkanSummary.apiVersion >> 22u) & 0x7fu;
    uint32_t vkMinor = (vulkanSummary.apiVersion >> 12u) & 0x3ffu;
    uint32_t vkPatch = (vulkanSummary.apiVersion) & 0xfffu;
    char apiVersionStr[32];
    snprintf(apiVersionStr, sizeof(apiVersionStr), "%u.%u.%u", vkMajor, vkMinor, vkPatch);

    os << ",\n  \"vulkan\": {\n";
    os << "    \"backend\": \"vulkan\",\n";
    os << "    \"device_name\": \"" << vulkanSummary.deviceName << "\",\n";
    os << "    \"api_version\": \"" << apiVersionStr << "\",\n";
    os << "    \"memory_budget_mb\": "
       << std::fixed << std::setprecision(2)
       << (vulkanSummary.workspaceBytes / (1024.0 * 1024.0)) << ",\n";
    os << "    \"replay_count\": " << vulkanSummary.replayCount << ",\n";
    os << "    \"num_dispatches\": " << vulkanSummary.dispatches << ",\n";
    os << "    \"capture_ms\": "
       << std::fixed << std::setprecision(3) << vulkanSummary.captureMs << ",\n";
    os << "    \"replay_ms\": "
       << std::fixed << std::setprecision(3) << vulkanSummary.lastReplayMs << ",\n";
    os << "    \"uma_available\": " << (vulkanSummary.umaDetected ? "true" : "false") << ",\n";
    os << "    \"fp16_supported\": " << (vulkanSummary.fp16Supported ? "true" : "false") << "\n";
    os << "  }\n";
  } else {
    os << "\n";
  }
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
  epoch_.fetch_add(1, std::memory_order_relaxed);
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
  segTerminalCount_ = 0;
}

// ─── Segment terminal records ────────────────────────────────────────────────

void DspDiagnostics::recordSegmentTerminal(int startSlot, int endSlot, int execCount,
                                            int outcome, const char* phase,
                                            const char* reason, const char* backend) {
  std::lock_guard<std::mutex> lock(eventMutex_);
  if (segTerminalCount_ >= MAX_SEGMENT_TERMINALS) return;

  auto& r = segTerminals_[segTerminalCount_++];
  r.startSlot = startSlot;
  r.endSlot = endSlot;
  r.execCountAtTransition = execCount;
  r.outcome = outcome;
  r.timestampUs = nowUs();

  if (phase) {
    strncpy(r.phase, phase, sizeof(r.phase) - 1);
    r.phase[sizeof(r.phase) - 1] = '\0';
  }
  if (reason) {
    strncpy(r.reason, reason, sizeof(r.reason) - 1);
    r.reason[sizeof(r.reason) - 1] = '\0';
  }
  if (backend) {
    strncpy(r.backend, backend, sizeof(r.backend) - 1);
    r.backend[sizeof(r.backend) - 1] = '\0';
  }
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

      if (mismatches <= diagDetailLimit_) {
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

// ─── Slot buffer dump (debug-gated metadata only) ────────────────────────────
void DspDiagnostics::dumpSlotBuffer(const char* tag, int slotIdx,
                                     const void* devicePtr,
                                     int64_t numElements, int sampleCount) {
  if (devicePtr == nullptr || numElements <= 0) return;

  recordEvent(DSP_DIAG_EXECUTE, slotIdx, -1, -1, tag, 0,
              "slot[%d] len=%lld addr=%p sampleCount=%d",
              slotIdx, static_cast<long long>(numElements), devicePtr, sampleCount);
}

// ─── Segment output dump (replaces copy-pasted topVal+first-N blocks) ────────
void DspDiagnostics::dumpSegmentOutput(const char* tag, int endSlot,
                                        const void* devicePtr,
                                        int64_t numElements, int execCount,
                                        void* stream, int sampleCount) {
  if (devicePtr == nullptr || numElements <= 0) return;

  recordEvent(DSP_DIAG_EXECUTE, endSlot, -1, -1, tag, 0,
              "%s addr=%p (endSlot=%d len=%lld execCount=%d sampleCount=%d)",
              tag, devicePtr, endSlot,
              static_cast<long long>(numElements), execCount, sampleCount);
}

// ─── External input actuality state dump ─────────────────────────────────────
DspDiagnostics::ExtInputSyncResult DspDiagnostics::dumpExternalInputState(
    NDArray** externalArrays, int numExt, int execCount, int maxToDump) {
  ExtInputSyncResult result = {0, 0, 0};
  if (externalArrays == nullptr || numExt <= 0) return result;

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

// ─── Array metadata fingerprint (FNV-1a) ─────────────────────────────────────
void DspDiagnostics::fingerprintArray(const char* tag, int idx, const char* name,
                                       NDArray* arr, int execCount) {
  if (arr == nullptr) return;
  auto* db = arr->dataBuffer();
  if (db == nullptr) return;

  size_t elemBytes = arr->sizeOfT();
  if (elemBytes == 0) elemBytes = 1;
  size_t totalBytes = static_cast<size_t>(arr->lengthOf()) * elemBytes;

  uint64_t h = 0xcbf29ce484222325ULL;
  uintptr_t primaryPtr = reinterpret_cast<uintptr_t>(db->primary());
  uintptr_t specialPtr = reinterpret_cast<uintptr_t>(db->special());
  h ^= primaryPtr; h *= 0x100000001b3ULL;
  h ^= specialPtr; h *= 0x100000001b3ULL;
  h ^= totalBytes; h *= 0x100000001b3ULL;
  auto offset = static_cast<uint64_t>(arr->offset());
  h ^= offset; h *= 0x100000001b3ULL;

  recordEvent(DSP_DIAG_EXECUTE, -1, -1, -1, tag, 0,
              "ARRAY_FINGERPRINT tag=%s idx=%d name='%s' dtype=%d len=%lld bytes=%zu "
              "hash=0x%016llx host=%p dev=%p execCount=%d",
              tag ? tag : "?", idx, name ? name : "?",
              static_cast<int>(arr->dataType()),
              static_cast<long long>(arr->lengthOf()), totalBytes,
              static_cast<unsigned long long>(h),
              db->primary(), db->special(), execCount);
}

// ── Invalid segment bucket summary ──────────────────────────────────────────

void DspDiagnostics::reportSegmentBucketSummary(
    int segStartSlot, int segEndSlot,
    const GapClassification* classifications,
    int numClassifications,
    const char* combinedBucketLabel,
    bool isInvalidForReplay) {

  // Primary event: summary line
  recordEvent(DSP_DIAG_SEGMENT_BUCKETS, -1, segStartSlot, -1, nullptr, 0,
              "BUCKET_SUMMARY: seg[%d-%d] bucket='%s' invalid=%d gaps=%d",
              segStartSlot, segEndSlot,
              combinedBucketLabel ? combinedBucketLabel : "(none)",
              isInvalidForReplay ? 1 : 0, numClassifications);

  // Per-gap detail events
  for (int i = 0; i < numClassifications; i++) {
    const auto& gc = classifications[i];
    const char* matLabel = gc.isShapeOnly ? "shape-only"
                            : gc.isViewOnly ? "view-only"
                            : gc.wouldMaterialize ? "materializing"
                            : "unknown";

    recordEvent(DSP_DIAG_SEGMENT_BUCKETS, -1, segStartSlot, -1, gc.primaryOpType, 0,
                "BUCKET_GAP[%d]: seg[%d-%d] slots[%d-%d] op='%s' class='%s' bucket='%s'",
                i, segStartSlot, segEndSlot, gc.startSlot, gc.endSlot,
                gc.primaryOpType ? gc.primaryOpType : "(unknown)",
                matLabel,
                gc.bucketLabel ? gc.bucketLabel : combinedBucketLabel);
  }

  // Echo to stdout at FULL level for immediate visibility
  if (getLevel() == DSP_LEVEL_FULL) {
    fprintf(stdout, "[DSP_DIAG] [SEGMENT_BUCKETS] seg[%d-%d] bucket='%s' invalid=%d:\n",
            segStartSlot, segEndSlot,
            combinedBucketLabel ? combinedBucketLabel : "(none)",
            isInvalidForReplay ? 1 : 0);
    for (int i = 0; i < numClassifications; i++) {
      const auto& gc = classifications[i];
      const char* matLabel = gc.isShapeOnly ? "shape-only"
                              : gc.isViewOnly ? "view-only"
                              : gc.wouldMaterialize ? "materializing"
                              : "unknown";
      fprintf(stdout, "  gap[%d] slots[%d-%d] op='%s' -> %s\n",
              i, gc.startSlot, gc.endSlot,
              gc.primaryOpType ? gc.primaryOpType : "(unknown)",
              matLabel);
    }
    fflush(stdout);
  }
}

void DspDiagnostics::recordGraphStateDump(const char* tag, const char* jsonSummary) {
  if (!isEnabled(DSP_DIAG_GRAPH_REPLAY)) return;
  recordEvent(DSP_DIAG_GRAPH_REPLAY, -1, -1, -1, nullptr, 0,
              "GRAPH_STATE_DUMP tag=%s: %s", tag ? tag : "?",
              jsonSummary ? jsonSummary : "{}");
}

}  // namespace graph
}  // namespace sd
