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

#ifdef HAVE_HEXAGON_MLIR

#include <graph/hexagon/HexagonReplayHandle.h>
#include <graph/hexagon/HexagonRuntimeManager.h>
#include <graph/DspDiagnostics.h>

#include <cstring>

namespace sd {
namespace graph {

// ── Constructor / Destructor ────────────────────────────────────────────────

HexagonReplayHandle::HexagonReplayHandle(int deviceId)
    : state_(ReplayState::EMPTY),
      deviceId_(deviceId) {
}

HexagonReplayHandle::~HexagonReplayHandle() {
  // Release workspace if still allocated
  if (captureWorkspacePtr_ != nullptr && npuContext_ != nullptr) {
    HexagonRuntimeManager::getInstance().freeTcm(npuContext_, captureWorkspacePtr_);
    captureWorkspacePtr_ = nullptr;
    captureWorkspaceBytes_ = 0;
  }

  // Free pinned host pointers
  freeHostPointers();

  // Destroy all NPU handles
  cleanup();
}

// ── Device Initialization ───────────────────────────────────────────────────

bool HexagonReplayHandle::initDevice() {
  if (npuContext_ != nullptr) return true;  // Already initialized

  auto& runtime = HexagonRuntimeManager::getInstance();
  if (!runtime.isAvailable()) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::initDevice: runtime not available");
    return false;
  }

  int deviceCount = runtime.getDeviceCount();
  if (deviceId_ >= deviceCount) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::initDevice: deviceId %d >= deviceCount %d",
             deviceId_, deviceCount);
    return false;
  }

  npuContext_ = runtime.initDevice(deviceId_);
  if (npuContext_ == nullptr) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::initDevice: failed to init device %d",
             deviceId_);
    return false;
  }

  DSP_DIAG(EXECUTE, "HexagonReplayHandle::initDevice: device %d initialized, ctx=%p",
           deviceId_, npuContext_);
  return true;
}

// ── Capture Lifecycle ───────────────────────────────────────────────────────

bool HexagonReplayHandle::beginCapture(void* stream) {
  if (state_ != ReplayState::EMPTY && state_ != ReplayState::ERRORED) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::beginCapture: invalid state %d",
             static_cast<int>(state_));
    return false;
  }

  // Lazy initialization of NPU device
  if (!initDevice()) {
    state_ = ReplayState::ERRORED;
    return false;
  }

  // Clear any previously recorded commands
  recordedCommands_.clear();
  numCommands_ = 0;

  state_ = ReplayState::CAPTURING;

  DSP_DIAG(EXECUTE, "HexagonReplayHandle::beginCapture: command list recording "
           "started on device %d", deviceId_);
  return true;
}

bool HexagonReplayHandle::endCapture(void* stream) {
  if (state_ != ReplayState::CAPTURING) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::endCapture: not capturing (state %d)",
             static_cast<int>(state_));
    return false;
  }

  if (recordedCommands_.empty()) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::endCapture: no commands recorded");
    state_ = ReplayState::ERRORED;
    return false;
  }

  // Seal the command list — no more appends allowed
  state_ = ReplayState::CAPTURED;
  DSP_DIAG(EXECUTE, "HexagonReplayHandle::endCapture: command list sealed with "
           "%d commands", numCommands_);
  return true;
}

bool HexagonReplayHandle::finalize() {
  if (state_ == ReplayState::READY) return true;  // Already finalized

  if (state_ != ReplayState::CAPTURED) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::finalize: not captured (state %d)",
             static_cast<int>(state_));
    return false;
  }

  // Validate all recorded kernel handles are still valid
  for (const auto& cmd : recordedCommands_) {
    if (cmd.type == RecordedCommand::KERNEL_DISPATCH && cmd.kernelHandle == nullptr) {
      DSP_DIAG(EXECUTE, "HexagonReplayHandle::finalize: null kernel handle in "
               "recorded command");
      state_ = ReplayState::ERRORED;
      return false;
    }
  }

  state_ = ReplayState::READY;

  DSP_DIAG(EXECUTE, "HexagonReplayHandle::finalize: ready for replay "
           "(%d commands)", numCommands_);
  return true;
}

bool HexagonReplayHandle::replay(void* stream) {
  if (state_ != ReplayState::READY) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::replay: not ready (state %d)",
             static_cast<int>(state_));
    return false;
  }

  if (npuContext_ == nullptr) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::replay: null NPU context");
    state_ = ReplayState::ERRORED;
    return false;
  }

  auto& runtime = HexagonRuntimeManager::getInstance();

  // Execute all recorded commands in order
  for (const auto& cmd : recordedCommands_) {
    bool ok = false;
    switch (cmd.type) {
      case RecordedCommand::DMA_HOST_TO_TCM:
        ok = runtime.dmaHostToTcm(npuContext_, cmd.dst,
                                   cmd.src, cmd.bytes);
        break;

      case RecordedCommand::DMA_TCM_TO_HOST:
        ok = runtime.dmaTcmToHost(npuContext_, cmd.dst,
                                   cmd.src, cmd.bytes);
        break;

      case RecordedCommand::KERNEL_DISPATCH: {
        // Build mutable arg pointer array from recorded args
        std::vector<void*> argsCopy(cmd.args);
        ok = runtime.dispatchKernel(npuContext_, cmd.kernelHandle,
                                     argsCopy.data(),
                                     static_cast<int>(argsCopy.size()));
        break;
      }
    }

    if (!ok) {
      DSP_DIAG(EXECUTE, "HexagonReplayHandle::replay: command failed (type=%d)",
               static_cast<int>(cmd.type));
      state_ = ReplayState::ERRORED;
      return false;
    }
  }

  // Wait for all NPU operations to complete
  if (!runtime.waitForCompletion(npuContext_)) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::replay: waitForCompletion failed");
    state_ = ReplayState::ERRORED;
    return false;
  }

  replayCount_++;

  DSP_DIAG(EXECUTE, "HexagonReplayHandle::replay: execution %d complete",
           replayCount_);
  return true;
}

// ── State & Statistics ──────────────────────────────────────────────────────

ReplayState HexagonReplayHandle::getState() const {
  return state_;
}

ReplayStatistics HexagonReplayHandle::getStatistics() const {
  ReplayStatistics stats;
  stats.numOperations = numCommands_;
  stats.numMemoryOps = 0;  // Not separately tracked
  stats.estimatedMemory = captureWorkspaceBytes_;
  stats.captureTimeMs = 0.0;
  stats.lastReplayTimeMs = 0.0;
  stats.replayCount = replayCount_;
  return stats;
}

// ── Command Recording ───────────────────────────────────────────────────────

bool HexagonReplayHandle::recordDmaHostToTcm(void* tcmDst, const void* hostSrc,
                                              size_t bytes) {
  if (state_ != ReplayState::CAPTURING) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::recordDmaHostToTcm: not capturing");
    return false;
  }

  RecordedCommand cmd;
  cmd.type = RecordedCommand::DMA_HOST_TO_TCM;
  cmd.dst = tcmDst;
  cmd.src = hostSrc;
  cmd.bytes = bytes;
  recordedCommands_.push_back(std::move(cmd));
  numCommands_++;

  DSP_DIAG(EXECUTE, "HexagonReplayHandle::recordDmaHostToTcm: %zu bytes, "
           "host=%p -> tcm=%p", bytes, hostSrc, tcmDst);
  return true;
}

bool HexagonReplayHandle::recordDmaTcmToHost(void* hostDst, const void* tcmSrc,
                                              size_t bytes) {
  if (state_ != ReplayState::CAPTURING) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::recordDmaTcmToHost: not capturing");
    return false;
  }

  RecordedCommand cmd;
  cmd.type = RecordedCommand::DMA_TCM_TO_HOST;
  cmd.dst = hostDst;
  cmd.src = tcmSrc;
  cmd.bytes = bytes;
  recordedCommands_.push_back(std::move(cmd));
  numCommands_++;

  DSP_DIAG(EXECUTE, "HexagonReplayHandle::recordDmaTcmToHost: %zu bytes, "
           "tcm=%p -> host=%p", bytes, tcmSrc, hostDst);
  return true;
}

bool HexagonReplayHandle::recordKernelDispatch(void* kernelHandle,
                                                void** args, int numArgs) {
  if (state_ != ReplayState::CAPTURING) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::recordKernelDispatch: not capturing");
    return false;
  }

  if (kernelHandle == nullptr) {
    DSP_DIAG(EXECUTE, "HexagonReplayHandle::recordKernelDispatch: null kernel handle");
    return false;
  }

  RecordedCommand cmd;
  cmd.type = RecordedCommand::KERNEL_DISPATCH;
  cmd.kernelHandle = kernelHandle;
  if (args != nullptr && numArgs > 0) {
    cmd.args.assign(args, args + numArgs);
  }
  recordedCommands_.push_back(std::move(cmd));
  numCommands_++;

  DSP_DIAG(EXECUTE, "HexagonReplayHandle::recordKernelDispatch: kernel=%p, "
           "%d args, command #%d", kernelHandle, numArgs, numCommands_);
  return true;
}

// ── Workspace Management ────────────────────────────────────────────────────

bool HexagonReplayHandle::allocateWorkspace(size_t bytes, int deviceId,
                                             void* registryPtr, int segIdx) {
  if (captureWorkspacePtr_ != nullptr) return true;  // Already allocated

  // Ensure device is initialized
  if (!initDevice()) return false;

  auto& runtime = HexagonRuntimeManager::getInstance();

  // Allocate TCM for workspace staging
  void* ptr = runtime.allocateTcm(npuContext_, bytes);
  if (ptr == nullptr) {
    // Fall back to DDR if TCM is full
    ptr = runtime.allocateDdr(npuContext_, bytes);
    if (ptr == nullptr) {
      DSP_DIAG(MEMORY, "HexagonReplayHandle: workspace alloc failed for %zu bytes",
               bytes);
      return false;
    }
    ddrAllocations_.push_back(ptr);
    DSP_DIAG(MEMORY, "HexagonReplayHandle: allocated %zuKB DDR workspace (TCM full)",
             bytes / 1024);
  } else {
    tcmAllocations_.push_back(ptr);
    DSP_DIAG(MEMORY, "HexagonReplayHandle: allocated %zuKB TCM workspace",
             bytes / 1024);
  }

  captureWorkspacePtr_ = ptr;
  captureWorkspaceBytes_ = bytes;
  return true;
}

void HexagonReplayHandle::releaseWorkspace(void* registryPtr, int segIdx) {
  if (captureWorkspacePtr_ == nullptr) return;

  auto& runtime = HexagonRuntimeManager::getInstance();

  if (npuContext_ != nullptr) {
    // Check if it's in TCM or DDR allocations and free accordingly
    auto tcmIt = std::find(tcmAllocations_.begin(), tcmAllocations_.end(),
                           captureWorkspacePtr_);
    if (tcmIt != tcmAllocations_.end()) {
      runtime.freeTcm(npuContext_, captureWorkspacePtr_);
      tcmAllocations_.erase(tcmIt);
    } else {
      auto ddrIt = std::find(ddrAllocations_.begin(), ddrAllocations_.end(),
                             captureWorkspacePtr_);
      if (ddrIt != ddrAllocations_.end()) {
        runtime.freeDdr(npuContext_, captureWorkspacePtr_);
        ddrAllocations_.erase(ddrIt);
      }
    }

    DSP_DIAG(MEMORY, "HexagonReplayHandle: released workspace");
  }

  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
}

void HexagonReplayHandle::freeHostPointers() {
  // Host pointers are standard allocations — free with delete[]
  for (auto* ptr : capturedHostPtrs_) {
    if (ptr != nullptr) {
      delete[] static_cast<uint8_t*>(ptr);
    }
  }
  capturedHostPtrs_.clear();
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

void HexagonReplayHandle::cleanup() {
  auto& runtime = HexagonRuntimeManager::getInstance();

  // Free TCM allocations
  if (npuContext_ != nullptr) {
    for (auto* ptr : tcmAllocations_) {
      if (ptr != nullptr) {
        runtime.freeTcm(npuContext_, ptr);
      }
    }
    tcmAllocations_.clear();

    // Free DDR allocations
    for (auto* ptr : ddrAllocations_) {
      if (ptr != nullptr) {
        runtime.freeDdr(npuContext_, ptr);
      }
    }
    ddrAllocations_.clear();
  }

  // Clear recorded commands
  recordedCommands_.clear();

  // Release NPU context last — all allocations must be freed first
  if (npuContext_ != nullptr) {
    runtime.releaseDevice(npuContext_);
    npuContext_ = nullptr;
  }

  commandList_ = nullptr;
  state_ = ReplayState::EMPTY;
  replayCount_ = 0;
  numCommands_ = 0;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_HEXAGON_MLIR
