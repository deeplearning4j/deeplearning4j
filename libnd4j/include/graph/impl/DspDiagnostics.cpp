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
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>
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

// Raw tensor artifacts are separate from the diagnostic ring. The wire format is
// DSPTS001, little-endian integers and length-prefixed UTF-8 strings, followed by
// C-logical raw storage bytes (never numeric casts). See TestDspTensorSnapshot.
// saveNpy/cnpy's typed OpaqueDataBuffer path is unsuitable here: it may sync the
// primary buffer and does not carry this capture's strides/source/call metadata.
struct DspDiagnostics::TensorSnapshotState {
  struct Tensor {
    std::string name;
    int dtype, width;
    char order;
    int64_t originalOffset;
    std::vector<LongType> shape, strides;
    size_t offset = 0, bytes = 0;
  };
  struct Call {
    int index;
    int64_t position;
    int phases = 1;
    std::vector<Tensor> tensors;
    void* pinned = nullptr;
    size_t bytes = 0;
    bool ready = false;
  };
  std::string source, prefix;
  std::thread::id owner;
  void* stream = nullptr;
  void* arena = nullptr;
  bool finished = false;
  size_t totalBytes = 0;
  std::vector<Call> calls;
};

namespace {
// Reserve bounded metadata/footer space for all three files, so the complete
// artifact set (not just the DMA payload) fits within 32 MiB.
constexpr size_t tensorSnapshotMetadataBudget = 128u * 1024u;
constexpr size_t tensorSnapshotBudget = 32u * 1024u * 1024u - 3u * tensorSnapshotMetadataBudget;
void snapshotInteger(std::ostream& out, uint64_t value, int bytes) {
  for (int i = 0; i < bytes; ++i) out.put(static_cast<char>((value >> (8 * i)) & 255));
}
void snapshotString(std::ostream& out, const std::string& value) {
  snapshotInteger(out, value.size(), 4);
  out.write(value.data(), value.size());
}
uint64_t snapshotChecksum(const unsigned char* data, size_t bytes) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (size_t i = 0; i < bytes; ++i) { hash ^= data[i]; hash *= UINT64_C(1099511628211); }
  return hash;
}
#if defined(SD_CUDA)
void snapshotCudaCheck(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(std::string("DSP tensor snapshot: ") + cudaGetErrorString(error));
}
#endif
}

bool DspDiagnostics::beginTensorSnapshot(const char* source, void* stream) {
  if (!tensorSnapshotRequested()) return false;
  std::lock_guard<std::mutex> lock(tensorSnapshotMutex_);
  if (tensorSnapshots_ != nullptr) return false;  // never reset by clear()/plan epochs
#if defined(SD_CUDA)
  std::string prefix;
  {
    std::lock_guard<std::mutex> configLock(eventMutex_);
    prefix = jsonPath_; // existing Environment + Java setJsonPath bridge
  }
  if (prefix.empty() || source == nullptr || std::strlen(source) > 1024)
    throw std::runtime_error("TENSOR_SNAPSHOT requires diagnosticsFile and a bounded source label");
  tensorSnapshots_ = new TensorSnapshotState();
  tensorSnapshots_->prefix = prefix;
  tensorSnapshots_->source = source;
  tensorSnapshots_->owner = std::this_thread::get_id();
  tensorSnapshots_->stream = stream;
  tensorSnapshots_->calls.reserve(3);
  // Allocate once at session setup, BEFORE entering the decode loop. Never
  // allocate pinned memory between position/mask writes and predictor execution.
  snapshotCudaCheck(cudaMallocHost(&tensorSnapshots_->arena, tensorSnapshotBudget));
  return true;
#else
  throw std::runtime_error("Asynchronous DSP tensor snapshot capture requires CUDA");
#endif
}

void DspDiagnostics::enqueueTensorSnapshot(int callIndex, int64_t sourcePosition, void* stream,
                                           const std::vector<std::string>& names,
                                           const std::vector<NDArray*>& arrays) {
#if defined(SD_CUDA)
  std::lock_guard<std::mutex> lock(tensorSnapshotMutex_);
  auto* state = tensorSnapshots_;
  if (state == nullptr || callIndex > 3) return;
  const bool append = !state->calls.empty() && callIndex == state->calls.back().index;
  if (state->finished || state->owner != std::this_thread::get_id() || state->stream != stream ||
      callIndex < 1 || (!append && callIndex != static_cast<int>(state->calls.size()) + 1) ||
      (append && (!state->calls.back().ready || state->calls.back().position != sourcePosition)))
    throw std::runtime_error("DSP tensor snapshot owner/stream/call mismatch");
  const size_t priorCount = append ? state->calls.back().tensors.size() : 0;
  if (names.size() != arrays.size() || names.empty() || names.size() + priorCount > 64)
    throw std::runtime_error("DSP tensor snapshot requires 1..64 explicitly named tensors per call");
  auto cudaStream = reinterpret_cast<cudaStream_t>(stream);
  cudaStreamCaptureStatus capture;
  snapshotCudaCheck(cudaStreamIsCapturing(cudaStream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::runtime_error("DSP tensor snapshots must be enqueued outside graph capture");

  TensorSnapshotState::Call batch;
  batch.index = callIndex;
  batch.position = sourcePosition;
  // Validate the entire batch before allocating or issuing any transfer. View
  // bounds use the unshifted DataBuffer; copies use specialBuffer (already shifted).
  for (size_t n = 0; n < arrays.size(); ++n) {
    auto* a = arrays[n];
    if (!a || !a->dataBuffer() || names[n].empty() || names[n].size() > 1024 ||
        a->rankOf() < 0 || a->rankOf() > 32 || a->sizeOfT() < 1 || a->sizeOfT() > 8)
      throw std::runtime_error("Invalid DSP tensor snapshot input metadata");
    for (size_t prior = 0; prior < n; ++prior)
      if (names[prior] == names[n]) throw std::runtime_error("Duplicate DSP snapshot input name");
    if (append) for (const auto& prior : state->calls.back().tensors)
      if (prior.name == names[n]) throw std::runtime_error("Duplicate DSP snapshot phase tensor name");
    TensorSnapshotState::Tensor t;
    t.name = names[n]; t.dtype = static_cast<int>(a->dataType());
    t.width = a->sizeOfT(); t.order = a->ordering(); t.originalOffset = a->offset();
    if (!(a->isR() || a->isZ() || a->isB()) ||
        (a->isEmpty() && a->rankOf() == 0))
      throw std::runtime_error("DSP tensor snapshots require fixed-width numeric/bool tensors with explicit empty shapes");
    const int64_t capacity = a->dataBuffer()->getLenInBytes() / t.width;
    int64_t low = t.originalOffset, high = low;
    size_t count = a->isEmpty() ? 0 : 1;
    for (int d = 0; d < a->rankOf(); ++d) {
      const LongType dim = a->sizeAt(d), stride = a->stridesOf()[d];
      if (dim < 0 || (count && static_cast<uint64_t>(dim) > tensorSnapshotBudget / t.width / count))
        throw std::runtime_error("DSP tensor snapshot size exceeds 32 MiB");
      count *= dim;
      t.shape.push_back(dim); t.strides.push_back(stride);
      if (!a->isEmpty() && dim > 1) {
        // Each contribution must fit the backing allocation before multiplication.
        if (stride == std::numeric_limits<LongType>::min() ||
            (stride < 0 ? -stride : stride) > capacity / (dim - 1))
          throw std::runtime_error("DSP tensor snapshot stride exceeds backing allocation");
        const auto delta = (dim - 1) * stride;
        if (delta < 0) {
          if (low < -delta) throw std::runtime_error("DSP tensor snapshot view precedes backing allocation");
          low += delta;
        } else {
          if (high > capacity - delta) throw std::runtime_error("DSP tensor snapshot view exceeds backing allocation");
          high += delta;
        }
      }
    }
    if (count && (low < 0 || high >= capacity || !a->specialBuffer() || !a->isActualOnDeviceSide()))
      throw std::runtime_error("DSP tensor snapshot requires valid device-current storage");
    t.bytes = count * t.width; t.offset = batch.bytes;
    if (t.bytes > tensorSnapshotBudget - state->totalBytes - batch.bytes)
      throw std::runtime_error("DSP tensor snapshot total payload exceeds 32 MiB; no truncated capture emitted");
    batch.bytes += t.bytes;
    batch.tensors.push_back(std::move(t));
  }
  // State owns storage BEFORE the first async transfer. On any CUDA failure it
  // stays owned, unpublished and bounded until process exit; no destructor frees
  // storage still targeted by queued DMA, and no error path introduces a sync.
  batch.pinned = static_cast<char*>(state->arena) + state->totalBytes;
  state->totalBytes += batch.bytes;
  if (append) {
    auto& previous = state->calls.back();
    previous.ready = false;  // A failed append must never publish a partial frame.
    for (auto& tensor : batch.tensors) {
      tensor.offset += previous.bytes;
      previous.tensors.push_back(std::move(tensor));
    }
    previous.bytes += batch.bytes;
    ++previous.phases;
  } else {
    state->calls.push_back(std::move(batch));
  }
  auto& pending = state->calls.back();
  // The producer owns prepare/register (validated device-current above). Borrow
  // its exact special pointers without re-preparing: syncToDevice can migrate a
  // spill allocation, which would change the state this diagnostic must observe.
  for (size_t n = 0; n < arrays.size(); ++n) {
    const auto& t = pending.tensors[priorCount + n];
    if (t.bytes == 0) continue;
    auto* dst = static_cast<char*>(pending.pinned) + t.offset;
    const auto* src = static_cast<const char*>(arrays[n]->specialBuffer());
    const size_t count = t.bytes / t.width;
    // Coalesce contiguous C-logical runs. Noncontiguous/F/offset/negative-stride
    // views copy only selected elements, not an enclosing span or view-base tail.
    auto offset = [&](size_t logical) {
      LongType result = 0;
      for (int d = static_cast<int>(t.shape.size()) - 1; d >= 0; --d) {
        result += static_cast<LongType>(logical % t.shape[d]) * t.strides[d]; logical /= t.shape[d];
      }
      return result;
    };
    bool contiguous = true;
    LongType expected = 1;
    for (int d = static_cast<int>(t.shape.size()) - 1; d >= 0; --d) {
      if (t.shape[d] > 1 && t.strides[d] != expected) contiguous = false;
      expected *= t.shape[d];
    }
    for (size_t begin = 0; begin < count;) {
      const LongType start = contiguous ? static_cast<LongType>(begin) : offset(begin);
      size_t end = contiguous ? count : begin + 1;
      while (end < count && offset(end) == start + static_cast<LongType>(end - begin)) ++end;
      snapshotCudaCheck(cudaMemcpyAsync(dst + begin * t.width, src + start * t.width,
                                        (end - begin) * t.width, cudaMemcpyDeviceToHost, cudaStream));
      begin = end;
    }
  }
  pending.ready = true;
#endif
}

void DspDiagnostics::drainTensorSnapshots(void* stream, bool finished) {
#if defined(SD_CUDA)
  std::lock_guard<std::mutex> lock(tensorSnapshotMutex_);
  auto* state = tensorSnapshots_;
  if (!state) return;
  if (state->owner != std::this_thread::get_id() || state->stream != stream)
    throw std::runtime_error("DSP tensor snapshot drain owner/stream mismatch");
  for (auto& batch : state->calls) {
    if (!batch.ready) continue;
    const std::string path = state->prefix + ".tensor-" + std::to_string(batch.index) + ".dspt";
    std::ostringstream header(std::ios::out | std::ios::binary);
    header.write("DSPTS001", 8);
    snapshotInteger(header, batch.index, 4);
    snapshotInteger(header, batch.position, 8);
    // Existing source field carries bounded host-only provenance; DSPTS001 stays
    // readable by old readers. Phase semantics also live in the unique tensor names.
    const std::string source = state->source + ";stream=" +
        std::to_string(reinterpret_cast<uintptr_t>(state->stream)) +
        ";enqueuePhases=" + std::to_string(batch.phases) +
        ";tensors=" + std::to_string(batch.tensors.size());
    if (source.size() > 1024) throw std::runtime_error("DSP tensor snapshot source metadata exceeds bound");
    snapshotString(header, source);
    snapshotInteger(header, batch.tensors.size(), 4);
    const uint16_t endian = 1;
    snapshotInteger(header, *reinterpret_cast<const unsigned char*>(&endian), 1);
    for (const auto& t : batch.tensors) {
      snapshotString(header, t.name);
      snapshotInteger(header, t.dtype, 4); snapshotInteger(header, t.width, 4);
      snapshotInteger(header, t.shape.size(), 4); snapshotInteger(header, t.order, 1);
      snapshotInteger(header, t.originalOffset, 8);
      for (size_t d = 0; d < t.shape.size(); ++d) {
        snapshotInteger(header, t.shape[d], 8); snapshotInteger(header, t.strides[d], 8);
      }
      snapshotInteger(header, t.bytes, 8);
      snapshotInteger(header, snapshotChecksum(t.bytes ? static_cast<const unsigned char*>(batch.pinned) + t.offset : nullptr, t.bytes), 8);
    }
    const auto metadata = header.str();
    if (!header.good() || metadata.size() + 8 > tensorSnapshotMetadataBudget)
      throw std::runtime_error("DSP tensor snapshot metadata exceeds reserved budget");
    // Never overwrite evidence from an earlier process using the same prefix.
    FILE* file = std::fopen(path.c_str(), "wbx");
    if (!file) throw std::runtime_error("Cannot exclusively create DSP tensor artifact: " + path);
    bool ok = std::fwrite(metadata.data(), 1, metadata.size(), file) == metadata.size();
    if (batch.bytes) ok = (std::fwrite(batch.pinned, 1, batch.bytes, file) == batch.bytes) && ok;
    // Footer makes an interrupted/failed write unambiguously incomplete.
    ok = (std::fwrite("DSPTDONE", 1, 8, file) == 8) && ok;
    ok = (std::fclose(file) == 0) && ok;
    if (!ok) throw std::runtime_error("Incomplete DSP tensor artifact: " + path);
    batch.pinned = nullptr; batch.ready = false;
  }
  if ((finished || state->calls.size() == 3) && state->arena) {
    snapshotCudaCheck(cudaFreeHost(state->arena));
    state->arena = nullptr;
    state->finished = true;
  }
#endif
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
    if (token == "TENSOR_SNAPSHOT") mask |= DSP_DIAG_TENSOR_SNAPSHOT;
    if (token == "ALL" || token == "*") mask |= DSP_DIAG_ALL;

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
