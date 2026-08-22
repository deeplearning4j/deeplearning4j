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

#include <graph/DspDiagnostics.h>
#include <graph/tpu/PjrtClientManager.h>

#include <cstdlib>
#include <vector>

namespace sd {
namespace graph {

TpuReplayHandle::TpuReplayHandle(int deviceId) : deviceId_(deviceId) {}

TpuReplayHandle::~TpuReplayHandle() {
  clearBindings();
  cleanupExecutable();
  releaseWorkspace();
  freeHostPointers();
}

bool TpuReplayHandle::beginCapture(void* stream) {
  (void)stream;
  if (state_ != ReplayState::EMPTY && state_ != ReplayState::ERRORED) {
    DSP_DIAG(EXECUTE, "TpuReplayHandle::beginCapture invalid state=%d",
             static_cast<int>(state_));
    return false;
  }
  if (program_.empty()) {
    state_ = ReplayState::ERRORED;
    DSP_DIAG(COMPILE, "TpuReplayHandle::beginCapture has no PJRT program");
    return false;
  }
  state_ = ReplayState::CAPTURING;
  return true;
}

bool TpuReplayHandle::endCapture(void* stream) {
  (void)stream;
  if (state_ != ReplayState::CAPTURING) return false;
  state_ = ReplayState::CAPTURED;
  return true;
}

bool TpuReplayHandle::finalize() {
  if (state_ != ReplayState::CAPTURED || program_.empty()) {
    state_ = ReplayState::ERRORED;
    return false;
  }

  cleanupExecutable();
  auto& manager = PjrtClientManager::getInstance();
  compiledExecutable_ = manager.compile(program_.data(), program_.size(),
                                        programFormat_.c_str(), deviceId_);
  if (compiledExecutable_ == nullptr) {
    state_ = ReplayState::ERRORED;
    DSP_DIAG(COMPILE, "TpuReplayHandle::finalize failed: %s",
             manager.getLastError().c_str());
    return false;
  }
  state_ = ReplayState::READY;
  return true;
}

bool TpuReplayHandle::replay(void* stream) {
  (void)stream;
  if (state_ != ReplayState::READY || compiledExecutable_ == nullptr) return false;
  if (boundInputArrays_.size() != inputSourceIndices_.size() ||
      boundOutputArrays_.size() != outputSlotIndices_.size()) {
    DSP_DIAG(EXECUTE,
             "TpuReplayHandle::replay binding mismatch inputs=%d/%d outputs=%d/%d",
             static_cast<int>(boundInputArrays_.size()),
             static_cast<int>(inputSourceIndices_.size()),
             static_cast<int>(boundOutputArrays_.size()),
             static_cast<int>(outputSlotIndices_.size()));
    return false;
  }

  auto& manager = PjrtClientManager::getInstance();
  std::vector<void*> inputBuffers;
  std::vector<void*> outputBuffers;
  inputBuffers.reserve(boundInputArrays_.size());
  bool success = true;
  for (auto* input : boundInputArrays_) {
    void* buffer = manager.createBuffer(input, deviceId_);
    if (buffer == nullptr) {
      success = false;
      break;
    }
    inputBuffers.push_back(buffer);
  }

  if (success) {
    success = manager.execute(
        compiledExecutable_, inputBuffers.empty() ? nullptr : inputBuffers.data(),
        static_cast<int>(inputBuffers.size()), deviceId_, outputBuffers);
  }
  if (success && outputBuffers.size() != boundOutputArrays_.size()) {
    DSP_DIAG(EXECUTE,
             "TpuReplayHandle::replay output count mismatch runtime=%d expected=%d",
             static_cast<int>(outputBuffers.size()),
             static_cast<int>(boundOutputArrays_.size()));
    success = false;
  }
  if (success) {
    for (size_t i = 0; i < outputBuffers.size(); ++i) {
      if (!manager.bufferToArray(outputBuffers[i], boundOutputArrays_[i])) {
        success = false;
        break;
      }
    }
  }

  for (auto* output : outputBuffers) manager.destroyBuffer(output);
  for (auto* input : inputBuffers) manager.destroyBuffer(input);

  if (!success) {
    DSP_DIAG(EXECUTE, "TpuReplayHandle::replay failed: %s",
             manager.getLastError().c_str());
    return false;
  }
  ++replayCount_;
  return true;
}

ReplayState TpuReplayHandle::getState() const { return state_; }

ReplayStatistics TpuReplayHandle::getStatistics() const {
  ReplayStatistics statistics;
  statistics.numOperations = numOperations_;
  statistics.numMemoryOps = static_cast<int>(inputSourceIndices_.size() +
                                              outputSlotIndices_.size());
  statistics.estimatedMemory = captureWorkspaceBytes_;
  statistics.captureTimeMs = 0.0;
  statistics.lastReplayTimeMs = 0.0;
  statistics.replayCount = replayCount_;
  return statistics;
}

bool TpuReplayHandle::allocateWorkspace(size_t bytes, int deviceId,
                                        void* registryPtr, int segIdx) {
  (void)deviceId;
  (void)registryPtr;
  (void)segIdx;
  if (captureWorkspacePtr_ != nullptr || bytes == 0) return true;
  void* allocation = std::calloc(1, bytes);
  if (allocation == nullptr) return false;
  captureWorkspacePtr_ = allocation;
  captureWorkspaceBytes_ = bytes;
  workspaceIsExternal_ = false;
  return true;
}

void TpuReplayHandle::releaseWorkspace(void* registryPtr, int segIdx) {
  (void)registryPtr;
  (void)segIdx;
  if (captureWorkspacePtr_ == nullptr) return;
  if (!workspaceIsExternal_) std::free(captureWorkspacePtr_);
  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
  workspaceIsExternal_ = false;
}

void TpuReplayHandle::freeHostPointers() {
  GraphReplayHandle::freeHostPointers();
}

void TpuReplayHandle::setProgram(
    const std::string& program, const std::string& format,
    const std::vector<int>& inputSourceIndices,
    const std::vector<int>& outputSlotIndices,
    int numOperations) {
  cleanupExecutable();
  clearBindings();
  program_ = program;
  programFormat_ = format;
  inputSourceIndices_ = inputSourceIndices;
  outputSlotIndices_ = outputSlotIndices;
  numOperations_ = numOperations;
  state_ = ReplayState::EMPTY;
}

void TpuReplayHandle::bindArrays(NDArray** inputArrays, int numInputs,
                                 NDArray** outputArrays, int numOutputs) {
  clearBindings();
  if (inputArrays != nullptr && numInputs > 0) {
    boundInputArrays_.assign(inputArrays, inputArrays + numInputs);
  }
  if (outputArrays != nullptr && numOutputs > 0) {
    boundOutputArrays_.assign(outputArrays, outputArrays + numOutputs);
  }
}

void TpuReplayHandle::cleanupExecutable() {
  if (compiledExecutable_ != nullptr) {
    PjrtClientManager::getInstance().destroyExecutable(compiledExecutable_);
    compiledExecutable_ = nullptr;
  }
}

void TpuReplayHandle::clearBindings() {
  boundInputArrays_.clear();
  boundOutputArrays_.clear();
}

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
