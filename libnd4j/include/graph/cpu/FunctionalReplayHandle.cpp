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

#include <graph/cpu/FunctionalReplayHandle.h>

namespace sd {
namespace graph {

FunctionalReplayHandle::FunctionalReplayHandle()
    : state_(ReplayState::EMPTY),
      replayCount_(0),
      lastReplayTimeMs_(0.0) {
}

FunctionalReplayHandle::~FunctionalReplayHandle() {
  // recordedOps_ contains non-owning pointers — nothing to free.
}

bool FunctionalReplayHandle::beginCapture(void* stream) {
  if (state_ != ReplayState::EMPTY && state_ != ReplayState::READY) {
    return false;
  }
  recordedOps_.clear();
  state_ = ReplayState::CAPTURING;
  return true;
}

bool FunctionalReplayHandle::endCapture(void* stream) {
  if (state_ != ReplayState::CAPTURING) {
    return false;
  }
  state_ = ReplayState::CAPTURED;
  return true;
}

bool FunctionalReplayHandle::finalize() {
  if (state_ != ReplayState::CAPTURED) {
    return false;
  }
  // For CPU functional replay, finalization is trivial — the recorded
  // op list is already ready. No instantiation step needed.
  state_ = ReplayState::READY;
  return true;
}

bool FunctionalReplayHandle::replay(void* stream) {
  if (state_ != ReplayState::READY) {
    return false;
  }
  // CPU functional replay doesn't execute ops here — the caller
  // (NativeDynamicShapePlan) still drives the slot-by-slot execution
  // but uses the cached shapes and frozen contexts from the warmup pass.
  // This handle tracks the replay state and provides statistics.
  replayCount_++;
  return true;
}

ReplayState FunctionalReplayHandle::getState() const {
  return state_;
}

ReplayStatistics FunctionalReplayHandle::getStatistics() const {
  ReplayStatistics stats;
  stats.numOperations = static_cast<int>(recordedOps_.size());
  stats.replayCount = replayCount_;
  stats.lastReplayTimeMs = lastReplayTimeMs_;
  return stats;
}

void FunctionalReplayHandle::recordOp(sd::ops::DeclarableOp* op, int slotIndex) {
  if (state_ == ReplayState::CAPTURING) {
    recordedOps_.push_back({op, slotIndex});
  }
}

}  // namespace graph
}  // namespace sd
