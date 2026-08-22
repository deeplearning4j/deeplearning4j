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

#include <array/NDArray.h>
#include <graph/GraphReplayHandle.h>

#include <cstdint>
#include <string>
#include <vector>

namespace sd {
namespace graph {

/**
 * PJRT replay handle for one compiled StableHLO segment.
 *
 * The handle owns the loaded executable. NDArray bindings are borrowed only for
 * the duration of segment execution; every replay uploads exact boundary inputs,
 * executes PJRT, downloads exact boundary outputs, and destroys transient PJRT
 * buffers and completion events.
 */
class SD_LIB_EXPORT TpuReplayHandle : public GraphReplayHandle {
 public:
  explicit TpuReplayHandle(int deviceId = 0);
  ~TpuReplayHandle() override;

  TpuReplayHandle(const TpuReplayHandle&) = delete;
  TpuReplayHandle& operator=(const TpuReplayHandle&) = delete;

  bool beginCapture(void* stream) override;
  bool endCapture(void* stream) override;
  bool finalize() override;
  bool replay(void* stream) override;

  ReplayState getState() const override;
  ReplayStatistics getStatistics() const override;
  const char* backendName() const override { return "TPU (PJRT)"; }

  bool allocateWorkspace(size_t bytes, int deviceId = 0,
                         void* registryPtr = nullptr, int segIdx = 0) override;
  void releaseWorkspace(void* registryPtr = nullptr, int segIdx = 0) override;
  void freeHostPointers() override;

  void setProgram(const std::string& program, const std::string& format,
                  const std::vector<int>& inputSourceIndices,
                  const std::vector<int>& outputSlotIndices,
                  int numOperations);

  void bindArrays(NDArray** inputArrays, int numInputs,
                  NDArray** outputArrays, int numOutputs);

  const std::vector<int>& inputSourceIndices() const {
    return inputSourceIndices_;
  }
  const std::vector<int>& outputSlotIndices() const {
    return outputSlotIndices_;
  }
  int getDeviceId() const { return deviceId_; }
  int getReplayCount() const { return replayCount_; }

 private:
  void cleanupExecutable();
  void clearBindings();

  ReplayState state_ = ReplayState::EMPTY;
  int deviceId_ = 0;
  void* compiledExecutable_ = nullptr;

  std::string program_;
  std::string programFormat_ = "mlir";
  std::vector<int> inputSourceIndices_;
  std::vector<int> outputSlotIndices_;
  std::vector<NDArray*> boundInputArrays_;
  std::vector<NDArray*> boundOutputArrays_;

  int numOperations_ = 0;
  int replayCount_ = 0;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
#endif  // LIBND4J_TPU_REPLAY_HANDLE_H
