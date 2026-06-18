/* ******************************************************************************
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
// CUDA implementations for ResourceBinder operations.
// All CUDA runtime API calls for ResourceBinder live here exclusively.
//

#ifdef SD_CUDA

#include <cuda_runtime.h>
#include <graph/gpu/ResourceBinder_cuda.h>
#include <array/DataBuffer.h>  // for tl_dspExecutionStream
#include <memory/cuda/CudaMemoryPool.h>

namespace sd {
namespace graph {

// ── TLS stream management ────────────────────────────────────────────────

void* ResourceBinder_getDspExecutionStream() {
  return tl_dspExecutionStream;
}

void ResourceBinder_setDspExecutionStream(void* stream) {
  tl_dspExecutionStream = stream;
}

// ── Event lifecycle ──────────────────────────────────────────────────────

void* ResourceBinder_createCompletionEvent() {
  cudaEvent_t event;
  cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  return reinterpret_cast<void*>(event);
}

void ResourceBinder_destroyCompletionEvent(void* event) {
  if (event != nullptr) {
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
  }
}

void ResourceBinder_recordEvent(void* event, void* stream) {
  if (event != nullptr && stream != nullptr) {
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                    reinterpret_cast<cudaStream_t>(stream));
  }
}

void ResourceBinder_streamWaitEvent(void* stream, void* event) {
  if (event != nullptr && stream != nullptr) {
    cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream),
                        reinterpret_cast<cudaEvent_t>(event), 0);
  }
}

// ── Stream lifecycle ─────────────────────────────────────────────────────

void* ResourceBinder_createStream() {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  return reinterpret_cast<void*>(stream);
}

void ResourceBinder_destroyStream(void* stream, int deviceId) {
  if (stream == nullptr) return;
  // Remove from the pool's dirty-stream tracking BEFORE destroying the stream.
  // CudaMemoryPool::free() records each stream that received a cudaFreeAsync on it.
  // trimPool() later syncs those streams to safely reclaim memory. If the stream
  // is destroyed first, that sync hits a dangling handle -> cudaError 700
  // (illegal memory access) which corrupts the entire CUDA context.
  if (deviceId >= 0) {
    memory::CudaMemoryPool::getInstance().removeDirtyStream(
        deviceId, reinterpret_cast<cudaStream_t>(stream));
  }
  cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
}

// ── Device memory (workspaces, arg tables) ───────────────────────────────

void* ResourceBinder_deviceAlloc(size_t bytes) {
  void* ptr = nullptr;
  cudaMalloc(&ptr, bytes);
  return ptr;
}

void ResourceBinder_deviceFree(void* ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

// ── Pinned host memory (staging buffers) ─────────────────────────────────

void* ResourceBinder_pinnedAlloc(size_t bytes) {
  void* ptr = nullptr;
  cudaMallocHost(&ptr, bytes);
  return ptr;
}

void ResourceBinder_pinnedFree(void* ptr) {
  if (ptr != nullptr) {
    cudaFreeHost(ptr);
  }
}

// ── Async memcpy ─────────────────────────────────────────────────────────

void ResourceBinder_memcpyD2HAsync(void* dst, const void* src, size_t bytes, void* stream) {
  cudaMemcpyAsync(dst, src, bytes,
                  cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream));
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
