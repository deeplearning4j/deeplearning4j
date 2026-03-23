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

#ifndef LIBND4J_TPU_REPLAY_HANDLE_H
#define LIBND4J_TPU_REPLAY_HANDLE_H

#include <system/common.h>

#ifdef SD_TPU

#include <graph/GraphReplayHandle.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

/**
 * TPU implementation of GraphReplayHandle.
 *
 * TPU "replay" = re-executing a cached XLA compiled binary via PJRT.
 * Unlike CUDA graphs or Level Zero mutable command lists, XLA/PJRT does not
 * use stream-based capture. Instead:
 *   1. beginCapture() / endCapture(): state transitions only (no stream capture)
 *   2. finalize(): hash HLO module -> check compilation cache -> compile if miss
 *   3. replay(): execute compiled module with bound input/output PJRT buffers
 *
 * The compilation cache maps HLO module hashes to compiled executables,
 * avoiding redundant XLA compilations when the same graph structure is
 * re-encountered with different shapes (after re-compilation).
 *
 * Address snapshot tracks PJRT buffer pointers (opaque void*) to detect
 * when input/output bindings have changed and re-binding is needed.
 */
class SD_LIB_EXPORT TpuReplayHandle : public GraphReplayHandle {
 public:
  explicit TpuReplayHandle(int deviceId = 0);
  ~TpuReplayHandle() override;

  // Non-copyable
  TpuReplayHandle(const TpuReplayHandle&) = delete;
  TpuReplayHandle& operator=(const TpuReplayHandle&) = delete;

  // ── GraphReplayHandle interface ───────────────────────────────────────

  /**
   * Begin capture phase. For TPU, this is a state transition only.
   * XLA does not use stream-based capture.
   * @param stream Ignored for TPU
   * @return true if state transition succeeded
   */
  bool beginCapture(void* stream) override;

  /**
   * End capture phase. For TPU, this is a state transition only.
   * @param stream Ignored for TPU
   * @return true if state transition succeeded
   */
  bool endCapture(void* stream) override;

  /**
   * Finalize the captured operations for replay.
   * Hashes the HLO module, checks the compilation cache, and compiles
   * via PjrtClientManager if cache miss.
   * @return true if a compiled executable is ready for replay
   */
  bool finalize() override;

  /**
   * Replay the compiled XLA executable.
   * Binds input PJRT buffers, executes the compiled module, and copies
   * output buffers back.
   * @param stream Ignored for TPU (PJRT manages execution scheduling)
   * @return true if execution completed successfully
   */
  bool replay(void* stream) override;

  ReplayState getState() const override;
  ReplayStatistics getStatistics() const override;
  const char* backendName() const override { return "TPU (PJRT)"; }

  // ── Workspace management ──────────────────────────────────────────────

  bool allocateWorkspace(size_t bytes, int deviceId = 0,
                         void* registryPtr = nullptr, int segIdx = 0) override;
  void releaseWorkspace(void* registryPtr = nullptr, int segIdx = 0) override;
  void freeHostPointers() override;

  // ── TPU-specific: HLO module binding ──────────────────────────────────

  /**
   * Set the serialized HLO module text for this replay handle.
   * Must be called before finalize().
   *
   * @param hloText  Serialized HLO module (text format)
   */
  void setHloModule(const std::string& hloText);

  /**
   * Set the serialized HLO module from raw bytes.
   *
   * @param hloBytes  HLO module data
   * @param hloSize   Size in bytes
   */
  void setHloModule(const void* hloBytes, size_t hloSize);

  /**
   * Bind PJRT buffer pointers for input/output arrays.
   * Must be called before replay().
   *
   * @param inputBuffers   Array of opaque PJRT_Buffer* input handles
   * @param numInputs      Number of input buffers
   * @param outputBuffers  Array of opaque PJRT_Buffer* output handles
   * @param numOutputs     Number of output buffers
   */
  void bindBuffers(void** inputBuffers, int numInputs,
                   void** outputBuffers, int numOutputs);

  /** Get the device ID this handle targets. */
  int getDeviceId() const { return deviceId_; }

  /** Get the number of replay executions. */
  int getReplayCount() const { return replayCount_; }

 private:
  ReplayState state_;
  int deviceId_;

  // Compiled XLA executable (opaque PJRT_LoadedExecutable*)
  void* compiledExecutable_ = nullptr;

  // Serialized HLO module
  std::string hloModuleText_;

  // Compilation cache: HLO hash -> compiled executable
  // Avoids redundant compilations when the same graph structure is
  // re-encountered (e.g., after shape change triggers re-compilation).
  std::unordered_map<uint64_t, void*> compilationCache_;

  // Bound input/output PJRT buffer pointers for replay
  std::vector<void*> boundInputBuffers_;
  std::vector<void*> boundOutputBuffers_;

  // Execution statistics
  int replayCount_ = 0;

  /**
   * Compute a hash of the current HLO module for cache lookup.
   * @return 64-bit hash of hloModuleText_
   */
  uint64_t hashHloModule() const;

  /**
   * Release the compiled executable and clear related state.
   */
  void cleanup();
};

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
#endif  // LIBND4J_TPU_REPLAY_HANDLE_H
