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

#ifdef SD_TPU

#include <graph/tpu/TpuReplayHandle.h>
#include <graph/tpu/PjrtClientManager.h>
#include <graph/DspDiagnostics.h>

#include <cstring>
#include <functional>

namespace sd {
namespace graph {

// ── Constructor / Destructor ────────────────────────────────────────────────

TpuReplayHandle::TpuReplayHandle(int deviceId)
    : state_(ReplayState::EMPTY),
      deviceId_(deviceId) {
}

TpuReplayHandle::~TpuReplayHandle() {
  freeHostPointers();
  cleanup();
}

// ── Capture Lifecycle ───────────────────────────────────────────────────────

bool TpuReplayHandle::beginCapture(void* stream) {
  if (state_ != ReplayState::EMPTY && state_ != ReplayState::ERRORED) {
    DSP_DIAG(EXECUTE, "TpuReplayHandle::beginCapture: invalid state %d",
             static_cast<int>(state_));
    return false;
  }

  // For TPU, capture is a logical phase only — XLA does not use stream capture.
  // We transition state to allow HLO module collection.
  state_ = ReplayState::CAPTURING;

  DSP_DIAG(EXECUTE, "TpuReplayHandle::beginCapture: entered capture phase "
           "on device %d", deviceId_);
  return true;
}

bool TpuReplayHandle::endCapture(void* stream) {
  if (state_ != ReplayState::CAPTURING) {
    DSP_DIAG(EXECUTE, "TpuReplayHandle::endCapture: not capturing (state %d)",
             static_cast<int>(state_));
    return false;
  }

  state_ = ReplayState::CAPTURED;

  DSP_DIAG(EXECUTE, "TpuReplayHandle::endCapture: capture phase complete, "
           "HLO module size=%zu bytes", hloModuleText_.size());
  return true;
}

bool TpuReplayHandle::finalize() {
  if (state_ == ReplayState::READY) return true;  // Already finalized

  if (state_ != ReplayState::CAPTURED) {
    DSP_DIAG(EXECUTE, "TpuReplayHandle::finalize: not captured (state %d)",
             static_cast<int>(state_));
    return false;
  }

  if (hloModuleText_.empty()) {
    DSP_DIAG(EXECUTE, "TpuReplayHandle::finalize: no HLO module set");
    state_ = ReplayState::ERRORED;
    return false;
  }

  // Hash the HLO module and check the compilation cache
  uint64_t hloHash = hashHloModule();

  auto cacheIt = compilationCache_.find(hloHash);
  if (cacheIt != compilationCache_.end()) {
    // Cache hit — reuse previously compiled executable
    compiledExecutable_ = cacheIt->second;
    DSP_DIAG(COMPILE, "TpuReplayHandle::finalize: compilation cache hit "
             "for hash 0x%016llx", static_cast<unsigned long long>(hloHash));
  } else {
    // Cache miss — compile via PjrtClientManager
    DSP_DIAG(COMPILE, "TpuReplayHandle::finalize: compilation cache miss "
             "for hash 0x%016llx, compiling %zu bytes HLO",
             static_cast<unsigned long long>(hloHash), hloModuleText_.size());

    auto& mgr = PjrtClientManager::getInstance();
    compiledExecutable_ = mgr.compile(hloModuleText_.data(),
                                       hloModuleText_.size());

    if (compiledExecutable_ == nullptr) {
      DSP_DIAG(COMPILE, "TpuReplayHandle::finalize: compilation failed: %s",
               mgr.getLastError().c_str());
      state_ = ReplayState::ERRORED;
      return false;
    }

    // Store in cache for future reuse
    compilationCache_[hloHash] = compiledExecutable_;
    DSP_DIAG(COMPILE, "TpuReplayHandle::finalize: compiled and cached "
             "executable %p", compiledExecutable_);
  }

  state_ = ReplayState::READY;

  DSP_DIAG(EXECUTE, "TpuReplayHandle::finalize: ready for replay "
           "(cache size=%d)", static_cast<int>(compilationCache_.size()));
  return true;
}

bool TpuReplayHandle::replay(void* stream) {
  if (state_ != ReplayState::READY) {
    DSP_DIAG(EXECUTE, "TpuReplayHandle::replay: not ready (state %d)",
             static_cast<int>(state_));
    return false;
  }

  if (compiledExecutable_ == nullptr) {
    DSP_DIAG(EXECUTE, "TpuReplayHandle::replay: no compiled executable");
    state_ = ReplayState::ERRORED;
    return false;
  }

  if (boundInputBuffers_.empty()) {
    DSP_DIAG(EXECUTE, "TpuReplayHandle::replay: no input buffers bound");
    return false;
  }

  // Execute the compiled module via PjrtClientManager
  auto& mgr = PjrtClientManager::getInstance();

  void** outputPtrs = boundOutputBuffers_.empty() ? nullptr : boundOutputBuffers_.data();
  int numOutputs = static_cast<int>(boundOutputBuffers_.size());

  bool success = mgr.execute(
      compiledExecutable_,
      boundInputBuffers_.data(),
      static_cast<int>(boundInputBuffers_.size()),
      outputPtrs,
      &numOutputs);

  if (!success) {
    DSP_DIAG(EXECUTE, "TpuReplayHandle::replay: execution failed: %s",
             mgr.getLastError().c_str());
    return false;
  }

  replayCount_++;

  DSP_DIAG(EXECUTE, "TpuReplayHandle::replay: execution %d complete "
           "(%d inputs, %d outputs)",
           replayCount_,
           static_cast<int>(boundInputBuffers_.size()),
           numOutputs);
  return true;
}

// ── State & Statistics ──────────────────────────────────────────────────────

ReplayState TpuReplayHandle::getState() const {
  return state_;
}

ReplayStatistics TpuReplayHandle::getStatistics() const {
  ReplayStatistics stats;
  stats.numOperations = 0;  // Not tracked at this level for TPU
  stats.numMemoryOps = 0;
  stats.estimatedMemory = captureWorkspaceBytes_;
  stats.captureTimeMs = 0.0;
  stats.lastReplayTimeMs = 0.0;
  stats.replayCount = replayCount_;
  return stats;
}

// ── Workspace Management ────────────────────────────────────────────────────

bool TpuReplayHandle::allocateWorkspace(size_t bytes, int deviceId,
                                         void* registryPtr, int segIdx) {
  if (captureWorkspacePtr_ != nullptr) return true;  // Already allocated

  // TPU workspace allocation goes through PjrtClientManager
  auto& mgr = PjrtClientManager::getInstance();
  if (!mgr.isAvailable()) {
    DSP_DIAG(MEMORY, "TpuReplayHandle: PJRT not available for workspace alloc");
    return false;
  }

  // Allocate a zero-filled host buffer as workspace staging
  // (PJRT manages device memory internally)
  void* ptr = calloc(1, bytes);
  if (ptr == nullptr) {
    DSP_DIAG(MEMORY, "TpuReplayHandle: host workspace alloc failed for %zu bytes",
             bytes);
    return false;
  }

  captureWorkspacePtr_ = ptr;
  captureWorkspaceBytes_ = bytes;

  DSP_DIAG(MEMORY, "TpuReplayHandle: allocated %zuMB workspace on device %d",
           bytes / (1024 * 1024), deviceId);
  return true;
}

void TpuReplayHandle::releaseWorkspace(void* registryPtr, int segIdx) {
  if (captureWorkspacePtr_ == nullptr) return;

  free(captureWorkspacePtr_);
  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;

  DSP_DIAG(MEMORY, "TpuReplayHandle: released workspace");
}

void TpuReplayHandle::freeHostPointers() {
  for (auto* ptr : capturedHostPtrs_) {
    if (ptr != nullptr) {
      free(ptr);
    }
  }
  capturedHostPtrs_.clear();
}

// ── HLO Module Binding ─────────────────────────────────────────────────────

void TpuReplayHandle::setHloModule(const std::string& hloText) {
  hloModuleText_ = hloText;
  DSP_DIAG(COMPILE, "TpuReplayHandle::setHloModule: %zu bytes HLO text",
           hloText.size());
}

void TpuReplayHandle::setHloModule(const void* hloBytes, size_t hloSize) {
  if (hloBytes != nullptr && hloSize > 0) {
    hloModuleText_.assign(static_cast<const char*>(hloBytes), hloSize);
  } else {
    hloModuleText_.clear();
  }
  DSP_DIAG(COMPILE, "TpuReplayHandle::setHloModule: %zu bytes HLO data",
           hloSize);
}

void TpuReplayHandle::bindBuffers(void** inputBuffers, int numInputs,
                                   void** outputBuffers, int numOutputs) {
  boundInputBuffers_.clear();
  boundOutputBuffers_.clear();

  if (inputBuffers != nullptr && numInputs > 0) {
    boundInputBuffers_.assign(inputBuffers, inputBuffers + numInputs);
  }
  if (outputBuffers != nullptr && numOutputs > 0) {
    boundOutputBuffers_.assign(outputBuffers, outputBuffers + numOutputs);
  }

  DSP_DIAG(EXECUTE, "TpuReplayHandle::bindBuffers: %d inputs, %d outputs",
           numInputs, numOutputs);
}

// ── Internal Helpers ────────────────────────────────────────────────────────

uint64_t TpuReplayHandle::hashHloModule() const {
  // Use std::hash<std::string> for a fast, non-cryptographic hash.
  // Sufficient for compilation cache keying.
  return std::hash<std::string>{}(hloModuleText_);
}

void TpuReplayHandle::cleanup() {
  // Destroy all cached compiled executables
  auto& mgr = PjrtClientManager::getInstance();
  for (auto& pair : compilationCache_) {
    if (pair.second != nullptr) {
      mgr.destroyExecutable(pair.second);
    }
  }
  compilationCache_.clear();

  compiledExecutable_ = nullptr;
  boundInputBuffers_.clear();
  boundOutputBuffers_.clear();

  state_ = ReplayState::EMPTY;
  replayCount_ = 0;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
