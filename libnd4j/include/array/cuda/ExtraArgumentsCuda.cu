/******************************************************************************
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

//
// Capture-safe helper functions for ExtraArguments.
//
// ExtraArguments.cpp is compiled by GCC but needs access to NVCC-defined
// thread_local variables (tl_graphExecutionActive, tl_captureWorkspace, etc.).
// GCC and NVCC have incompatible TLS models on Linux, so direct extern access
// from .cpp files causes linker errors. These thin wrappers live in a .cu file
// (compiled by NVCC) and provide the same functionality via function calls.
//

#include <cuda_runtime.h>
#include <array/DataBuffer.h>
#include <execution/LaunchContext.h>

namespace sd {

bool extraArgsCaptureActive() {
  return tl_graphExecutionActive;
}

void* extraArgsCaptureDevAlloc(size_t bytes) {
  if (!tl_graphExecutionActive || tl_captureWorkspace == nullptr) return nullptr;
  size_t aligned = (bytes + 255) & ~255ULL;
  if (tl_captureWorkspaceOffset + aligned > tl_captureWorkspaceSize) return nullptr;
  void* ptr = static_cast<char*>(tl_captureWorkspace) + tl_captureWorkspaceOffset;
  tl_captureWorkspaceOffset += aligned;
  return ptr;
}

void* extraArgsCaptureHostAlloc(size_t bytes) {
  if (!tl_graphExecutionActive || tl_captureHostWorkspace == nullptr) return nullptr;
  size_t aligned = (bytes + 255) & ~255ULL;
  if (tl_captureHostWorkspaceOffset + aligned > tl_captureHostWorkspaceSize) return nullptr;
  void* ptr = static_cast<char*>(tl_captureHostWorkspace) + tl_captureHostWorkspaceOffset;
  tl_captureHostWorkspaceOffset += aligned;
  return ptr;
}

void extraArgsCaptureH2D(void* dst, const void* src, size_t bytes) {
  cudaStream_t stream = (tl_graphCaptureStream != nullptr)
      ? tl_graphCaptureStream
      : *LaunchContext::defaultContext()->getCudaStream();
  cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
}

}  // namespace sd
