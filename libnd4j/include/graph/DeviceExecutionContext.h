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

#ifndef LIBND4J_DEVICE_EXECUTION_CONTEXT_H
#define LIBND4J_DEVICE_EXECUTION_CONTEXT_H

#include <system/common.h>
#include <cstddef>

namespace sd {
namespace graph {

/**
 * Explicit per-segment device/stream context for multi-device DSP execution.
 *
 * This replaces implicit thread-local device management with an explicit context
 * that can be passed to segment execution methods. Introduced as an optional
 * parallel API — existing code continues using thread-locals via fromThreadLocals().
 *
 * Transition strategy:
 *   1. New multi-device code passes explicit DeviceExecutionContext
 *   2. Existing single-device code uses fromThreadLocals() / applyToThreadLocals()
 *   3. Eventually, thread-local usage is eliminated entirely
 */
struct DeviceExecutionContext {
  int deviceId = 0;                     // Target GPU device ID
  void* stream = nullptr;               // CUDA stream (cudaStream_t) or nullptr for CPU
  void* captureWorkspace = nullptr;     // Pre-allocated capture workspace buffer
  size_t workspaceSize = 0;             // Total workspace size in bytes
  size_t workspaceOffset = 0;           // Current offset into workspace
  bool isCapturing = false;             // True during CUDA graph capture

  /**
   * Create a DeviceExecutionContext from current thread-local state.
   * Reads the current CUDA device, DSP execution stream, etc.
   * For backward compatibility with code that uses thread-locals.
   */
  static DeviceExecutionContext fromThreadLocals();

  /**
   * Apply this context's settings to thread-locals.
   * Sets the current CUDA device and DSP execution stream.
   * For backward compatibility with code that reads thread-locals.
   */
  void applyToThreadLocals() const;

  /**
   * Allocate from the capture workspace.
   * Returns nullptr if not enough space or no workspace configured.
   *
   * @param bytes  Number of bytes to allocate
   * @param align  Alignment (default 256 for GPU buffers)
   * @return Pointer to allocated memory, or nullptr
   */
  void* workspaceAlloc(size_t bytes, size_t align = 256);
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DEVICE_EXECUTION_CONTEXT_H
