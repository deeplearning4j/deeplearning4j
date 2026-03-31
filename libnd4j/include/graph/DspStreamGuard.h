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

#ifndef LIBND4J_DSP_STREAM_GUARD_H
#define LIBND4J_DSP_STREAM_GUARD_H

#include <system/common.h>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#include <array/DataBuffer.h>  // for tl_dspExecutionStream extern

namespace sd {
namespace graph {

/**
 * RAII guard for the thread-local DSP execution stream.
 *
 * Sets tl_dspExecutionStream on construction, restores the previous
 * value on destruction. Exception-safe: if any code between construction
 * and destruction throws, the stream is still restored.
 *
 * Optionally saves/restores the CUDA device context for multi-device support.
 *
 * Usage:
 *   {
 *     DspStreamGuard guard(myStream);
 *     // ... all H2D copies in this scope use myStream ...
 *   } // stream restored automatically
 */
class DspStreamGuard {
 public:
  /**
   * Save current tl_dspExecutionStream and set to the given stream.
   *
   * @param stream     The CUDA stream to set as the DSP execution stream
   * @param deviceId   If >= 0, also save/restore cudaSetDevice context
   */
  explicit DspStreamGuard(cudaStream_t stream, int deviceId = -1)
      : previousStream_(tl_dspExecutionStream),
        previousDevice_(-1),
        restoreDevice_(deviceId >= 0) {
    if (restoreDevice_) {
      cudaGetDevice(&previousDevice_);
      if (previousDevice_ != deviceId) {
        cudaSetDevice(deviceId);
      }
    }
    tl_dspExecutionStream = stream;
  }

  ~DspStreamGuard() {
    tl_dspExecutionStream = previousStream_;
    if (restoreDevice_ && previousDevice_ >= 0) {
      int currentDevice;
      cudaGetDevice(&currentDevice);
      if (currentDevice != previousDevice_) {
        cudaSetDevice(previousDevice_);
      }
    }
  }

  // Non-copyable, non-movable
  DspStreamGuard(const DspStreamGuard&) = delete;
  DspStreamGuard& operator=(const DspStreamGuard&) = delete;
  DspStreamGuard(DspStreamGuard&&) = delete;
  DspStreamGuard& operator=(DspStreamGuard&&) = delete;

  /**
   * Get the stream that was active before this guard was created.
   */
  cudaStream_t previousStream() const { return previousStream_; }

  /**
   * Get the device that was active before this guard was created.
   * Returns -1 if device save/restore was not requested.
   */
  int previousDevice() const { return previousDevice_; }

 private:
  cudaStream_t previousStream_;
  int previousDevice_;
  bool restoreDevice_;
};

}  // namespace graph
}  // namespace sd

#else  // !SD_CUDA — CPU build: empty no-op guard

namespace sd {
namespace graph {

/**
 * CPU build: DspStreamGuard is a no-op. Zero cost.
 */
class DspStreamGuard {
 public:
  explicit DspStreamGuard(void* stream = nullptr, int deviceId = -1) {
    (void)stream;
    (void)deviceId;
  }
  ~DspStreamGuard() = default;

  DspStreamGuard(const DspStreamGuard&) = delete;
  DspStreamGuard& operator=(const DspStreamGuard&) = delete;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA

#endif  // LIBND4J_DSP_STREAM_GUARD_H
