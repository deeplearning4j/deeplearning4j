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

// CUDA-specific ExecutionState methods.
// Separated from ExecutionState.cpp because these require nvcc compilation
// for CUDA headers and tl_dspExecutionStream access.

#include <graph/ExecutionState.h>
#include <graph/DspDiagnostics.h>
#include <array/DataBuffer.h>  // for tl_dspExecutionStream extern

#include <cuda_runtime.h>

namespace sd {
namespace graph {

void* ExecutionState::bindSegmentDevice(int segIdx) {
  if (segIdx < 0 || segIdx >= numSegments_) return nullptr;

  auto& seg = segmentStates_[segIdx];

  // Save current tl_dspExecutionStream (will be restored by restoreSegmentContext)
  cudaStream_t previousStream = tl_dspExecutionStream;

  // Set device if segment has a specific device
  if (seg.deviceId >= 0) {
    int currentDevice;
    cudaGetDevice(&currentDevice);
    if (currentDevice != seg.deviceId) {
      cudaSetDevice(seg.deviceId);
    }
  }

  // Set the DSP execution stream for this segment.
  // This allows syncToSpecial() and CudaMemoryPool to use the segment's stream
  // instead of stream 0, avoiding cross-stream synchronization.
  if (seg.stream != nullptr) {
    tl_dspExecutionStream = static_cast<cudaStream_t>(seg.stream);
  }

  DSP_DIAG(EXECUTE, "ExecutionState: bound segment %d (device=%d)", segIdx, seg.deviceId);
  return static_cast<void*>(previousStream);
}

void ExecutionState::restoreSegmentContext(void* previousStream, int previousDevice) {
  // Restore tl_dspExecutionStream
  tl_dspExecutionStream = static_cast<cudaStream_t>(previousStream);

  // Restore device if needed
  if (previousDevice >= 0) {
    int currentDevice;
    cudaGetDevice(&currentDevice);
    if (currentDevice != previousDevice) {
      cudaSetDevice(previousDevice);
    }
  }
}

}  // namespace graph
}  // namespace sd
