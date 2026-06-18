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

#ifndef LIBND4J_DSP_THREAD_STATE_H
#define LIBND4J_DSP_THREAD_STATE_H

#include <system/common.h>
#include <array/DataBuffer.h>  // for tl_graphExecutionActive, tl_dspReplayActive, tl_dspExecutionStream externs

#ifdef SD_CUDA
#include <cuda_runtime.h>

// tl_dspGapStream is defined in NativeDynamicShapePlan_gpubackend.cu;
// we need the extern declaration here for the DspThreadState RAII guard.
extern thread_local cudaStream_t tl_dspGapStream;

namespace sd {
namespace graph {

/**
 * RAII guard for DSP replay thread-local state.
 *
 * Saves tl_dspReplayActive on construction, sets it to a new value,
 * and restores the previous value on destruction. Exception-safe:
 * all early-return paths automatically restore the flag.
 *
 * This replaces the manual save/restore pattern:
 *   bool prev = tl_dspReplayActive;
 *   tl_dspReplayActive = true;
 *   ... (6 error return paths that each need `tl_dspReplayActive = prev;`)
 *   tl_dspReplayActive = prev;
 *
 * Usage:
 *   {
 *     DspReplayGuard guard(true);
 *     // ... all code in scope sees tl_dspReplayActive == true ...
 *     if (error) return status;  // automatically restored
 *   } // restored here too
 */
class DspReplayGuard {
 public:
  explicit DspReplayGuard(bool replayActive)
      : previous_(tl_dspReplayActive) {
    tl_dspReplayActive = replayActive;
  }

  ~DspReplayGuard() {
    tl_dspReplayActive = previous_;
  }

  // Non-copyable, non-movable
  DspReplayGuard(const DspReplayGuard&) = delete;
  DspReplayGuard& operator=(const DspReplayGuard&) = delete;
  DspReplayGuard(DspReplayGuard&&) = delete;
  DspReplayGuard& operator=(DspReplayGuard&&) = delete;

  bool previous() const { return previous_; }

 private:
  bool previous_;
};

/**
 * RAII guard for tl_graphExecutionActive.
 *
 * Saves and restores the thread-local graph execution flag.
 * Used around gap-slot execution within a captured graph scope
 * where the flag needs to be temporarily cleared.
 *
 * Usage:
 *   {
 *     DspGraphActiveGuard guard(false);  // temporarily clear
 *     executeGapSlot(...);
 *   } // flag restored
 */
class DspGraphActiveGuard {
 public:
  explicit DspGraphActiveGuard(bool graphActive)
      : previous_(tl_graphExecutionActive) {
    tl_graphExecutionActive = graphActive;
  }

  ~DspGraphActiveGuard() {
    tl_graphExecutionActive = previous_;
  }

  // Non-copyable, non-movable
  DspGraphActiveGuard(const DspGraphActiveGuard&) = delete;
  DspGraphActiveGuard& operator=(const DspGraphActiveGuard&) = delete;
  DspGraphActiveGuard(DspGraphActiveGuard&&) = delete;
  DspGraphActiveGuard& operator=(DspGraphActiveGuard&&) = delete;

  bool previous() const { return previous_; }

 private:
  bool previous_;
};

/**
 * Combined RAII guard for the full DSP thread-local state during replay.
 *
 * Manages tl_graphExecutionActive, tl_dspReplayActive, tl_dspExecutionStream,
 * and tl_dspGapStream atomically. On destruction, all four are restored to
 * their previous values regardless of how the scope exits.
 *
 * For cases where only one or two flags need toggling, prefer the lighter
 * single-purpose guards (DspReplayGuard, DspGraphActiveGuard, DspStreamGuard).
 */
class DspThreadState {
 public:
  DspThreadState(cudaStream_t execStream, cudaStream_t gapStream,
                 bool graphActive, bool replayActive)
      : prevGraphActive_(tl_graphExecutionActive),
        prevReplayActive_(tl_dspReplayActive),
        prevExecStream_(reinterpret_cast<cudaStream_t>(tl_dspExecutionStream)),
        prevGapStream_(reinterpret_cast<cudaStream_t>(tl_dspGapStream)) {
    tl_graphExecutionActive = graphActive;
    tl_dspReplayActive = replayActive;
    tl_dspExecutionStream = execStream;
    tl_dspGapStream = gapStream;
  }

  // Overload for callers using void* streams (e.g. via dspGetExecutionStream()).
  DspThreadState(void* execStream, void* gapStream,
                 bool graphActive, bool replayActive)
      : DspThreadState(reinterpret_cast<cudaStream_t>(execStream),
                       reinterpret_cast<cudaStream_t>(gapStream),
                       graphActive, replayActive) {}

  ~DspThreadState() {
    tl_graphExecutionActive = prevGraphActive_;
    tl_dspReplayActive = prevReplayActive_;
    tl_dspExecutionStream = prevExecStream_;
    tl_dspGapStream = prevGapStream_;
  }

  // Non-copyable, non-movable
  DspThreadState(const DspThreadState&) = delete;
  DspThreadState& operator=(const DspThreadState&) = delete;
  DspThreadState(DspThreadState&&) = delete;
  DspThreadState& operator=(DspThreadState&&) = delete;

  bool previousGraphActive() const { return prevGraphActive_; }
  bool previousReplayActive() const { return prevReplayActive_; }
  cudaStream_t previousExecStream() const { return prevExecStream_; }
  cudaStream_t previousGapStream() const { return prevGapStream_; }

 private:
  bool prevGraphActive_;
  bool prevReplayActive_;
  cudaStream_t prevExecStream_;
  cudaStream_t prevGapStream_;
};

}  // namespace graph
}  // namespace sd

#else  // !SD_CUDA — CPU build: no-op guards

namespace sd {
namespace graph {

class DspReplayGuard {
 public:
  explicit DspReplayGuard(bool) {}
  ~DspReplayGuard() = default;
  DspReplayGuard(const DspReplayGuard&) = delete;
  DspReplayGuard& operator=(const DspReplayGuard&) = delete;
  bool previous() const { return false; }
};

class DspGraphActiveGuard {
 public:
  explicit DspGraphActiveGuard(bool) {}
  ~DspGraphActiveGuard() = default;
  DspGraphActiveGuard(const DspGraphActiveGuard&) = delete;
  DspGraphActiveGuard& operator=(const DspGraphActiveGuard&) = delete;
  bool previous() const { return false; }
};

class DspThreadState {
 public:
  DspThreadState(void* = nullptr, void* = nullptr, bool = false, bool = false) {}
  ~DspThreadState() = default;
  DspThreadState(const DspThreadState&) = delete;
  DspThreadState& operator=(const DspThreadState&) = delete;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA

#endif  // LIBND4J_DSP_THREAD_STATE_H
