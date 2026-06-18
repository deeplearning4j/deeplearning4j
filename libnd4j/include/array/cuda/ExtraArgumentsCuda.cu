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

#include <array/ExtraArguments_cuda.h>
#include <cuda_runtime.h>
#include <array/DataBuffer.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <system/op_boilerplate.h>

namespace sd {

// Forward declarations — definitions follow after extra_args_detail.
bool extraArgsCaptureActive();
void extraArgsCaptureH2D(void* dst, const void* src, size_t bytes);

namespace extra_args_detail {

void* extraArgsAllocDevice(size_t bytes) {
  int deviceId = sd::AffinityManager::currentDeviceId();
  auto ptr = sd::memory::CudaMemoryPool::getInstance().allocate(bytes, deviceId);
  if (!ptr) THROW_EXCEPTION("ExtraArguments: CudaMemoryPool::allocate failed");
  return ptr;
}

void extraArgsFreeDevice(void* ptr) {
  if (ptr == nullptr) return;
  int deviceId = sd::AffinityManager::currentDeviceId();
  sd::memory::CudaMemoryPool::getInstance().free(ptr, deviceId);
}

void extraArgsCopyH2DDispatch(void* dst, const void* src, size_t bytes) {
  // extraArgsCaptureActive and extraArgsCaptureH2D are defined in namespace sd
  // (below, in the same TU), not in extra_args_detail.
  if (::sd::extraArgsCaptureActive()) {
    // During CUDA graph capture, synchronous cudaMemcpy on the legacy stream
    // poisons the capture stream with error 901. Use the async capture path.
    ::sd::extraArgsCaptureH2D(dst, src, bytes);
  } else {
    // Outside capture: use cudaMemcpyAsync on cudaStreamPerThread instead of
    // synchronous cudaMemcpy on the legacy stream (stream 0). Stream 0 causes
    // error 906 when another thread on the same device is mid-capture.
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);
  }
}

}  // namespace extra_args_detail


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
      ? reinterpret_cast<cudaStream_t>(tl_graphCaptureStream)
      : *LaunchContext::defaultContext()->getCudaStream();
  cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
}

}  // namespace sd
