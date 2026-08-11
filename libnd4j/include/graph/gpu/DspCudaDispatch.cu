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
// CUDA implementations for DspCudaDispatch.
// All CUDA runtime/driver API calls used by DSP/graph .cpp files live here.
//

#include <graph/gpu/DspCudaDispatch.h>
#include <graph/GraphBackend.h>
#include <array/NDArray.h>
#ifdef SD_CUDA
#include <memory/cuda/CudaMemoryPool.h>
#include <unordered_map>
#include <mutex>
#endif
#include <config.h>
#if HAVE_TRITON
#include <graph/gpu/TritonGraphBackend.h>
#endif

#ifdef SD_CUDA

namespace sd {
namespace graph {
namespace cuda {

void* dspBuffer(NDArray* arr) { return arr->specialBuffer(); }

const void* dspBufferConst(const NDArray* arr) {
  return const_cast<NDArray*>(arr)->specialBuffer();
}

}  // namespace cuda
}  // namespace graph
}  // namespace sd

#include <cuda.h>
#include <cuda_runtime.h>
#include <array/DataBuffer.h>    // for tl_dspExecutionStream, tl_dspReplayActive
#include <execution/LaunchContext.h>
#include <helpers/logger.h>
#include <graph/gpu/NvrtcGraphBackend.h>
#include <graph/gpu/PtxGraphBackend.h>

// TLS stream externs — defined in NativeDynamicShapePlan_gpubackend.cu
extern thread_local cudaStream_t tl_dspGapStream;
// tl_graphCaptureStream is a macro from DataBuffer.h (expands to tl_dataBufferState.graphCaptureStream)

namespace sd {

// Forward declaration — defined in NativeDynamicShapePlan_cuda.cu or DataBuffer.cu
void dspPublishThreadCompletionEvent(void* streamPtr);

namespace graph {

// Defined by the CUDA NativeDynamicShapePlan implementation. Keep this state in
// the plan's namespace; the backend-specific dispatch namespace only queries it.
extern std::unordered_map<int, void*> g_globalCaptureWorkspaceByDevice;
extern std::mutex g_globalCaptureWorkspaceMtx;

namespace cuda {

// ═══════════════════════════════════════════════════════════════════════════════
// Error management
// ═══════════════════════════════════════════════════════════════════════════════

int dspClearLastCudaError() {
  return static_cast<int>(cudaGetLastError());
}

int dspPeekLastCudaError() {
  return static_cast<int>(cudaPeekAtLastError());
}

const char* dspCudaErrorString(int errorCode) {
  return cudaGetErrorString(static_cast<cudaError_t>(errorCode));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Device queries
// ═══════════════════════════════════════════════════════════════════════════════

int dspGetCurrentDevice() {
  int dev = -1;
  cudaError_t err = cudaGetDevice(&dev);
  return (err == cudaSuccess) ? dev : -1;
}

void dspSetCurrentDevice(int deviceId) {
  cudaSetDevice(deviceId);
}

int dspGetDeviceCount() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return (err == cudaSuccess) ? count : 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stream capture queries
// ═══════════════════════════════════════════════════════════════════════════════

bool dspStreamIsCapturing(void* stream) {
  if (stream == nullptr) return false;
  cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(stream), &capStat);
  return capStat != cudaStreamCaptureStatusNone;
}

bool dspEndStaleCapture(void* stream, const char* label) {
  if (stream == nullptr) return false;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  cudaStreamCaptureStatus capStatus = cudaStreamCaptureStatusNone;
  auto capErr = cudaStreamGetCaptureInfo(s, &capStatus, nullptr);
  if (capErr == cudaSuccess && capStatus != cudaStreamCaptureStatusNone) {
    cudaGraph_t staleGraph = nullptr;
    cudaStreamEndCapture(s, &staleGraph);
    if (staleGraph != nullptr) cudaGraphDestroy(staleGraph);
    cudaGetLastError();  // clear any sticky error from the stale capture
    return true;
  }
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Memory operations
// ═══════════════════════════════════════════════════════════════════════════════

void dspDeviceFree(void* ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

int dspMemcpyH2DAsync(void* dst, const void* src, size_t bytes, void* stream) {
  return static_cast<int>(
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream)));
}

void dspMemcpyD2DDefaultStream(void* dst, const void* src, size_t bytes) {
  if (dst != nullptr && src != nullptr && bytes > 0) {
    auto stream = sd::LaunchContext::defaultContext()->getCudaStream();
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, *stream);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Memory pool management
// ═══════════════════════════════════════════════════════════════════════════════

bool dspMemPoolTrim(int deviceId, size_t minBytes) {
  cudaMemPool_t pool = nullptr;
  if (cudaDeviceGetMemPool(&pool, deviceId) == cudaSuccess && pool != nullptr) {
    return cudaMemPoolTrimTo(pool, minBytes) == cudaSuccess;
  }
  return false;
}

size_t dspGetDeviceTotalMemory() {
  size_t freeMem = 0, totalMem = 0;
  cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
  return (err == cudaSuccess) ? totalMem : 0;
}

bool dspHasDeviceMemory() { return true; }

bool dspIsCudaBuild() { return true; }

bool dspIsHostBuild() { return false; }

void dspFreeWorkspaceOnPool(void* ptr) {
  if (ptr != nullptr) {
    memory::CudaMemoryPool::getInstance().unregisterCaptureWorkspace(ptr);
    cudaFree(ptr);
  }
}

// Multi-GPU: the capture workspace is per-device (NativeDynamicShapePlan_gpubackend.cu). A pointer
// is a "global capture workspace" if it matches ANY device's entry.
bool dspIsGlobalCaptureWorkspace(void* ptr) {
  if (ptr == nullptr) return false;
  std::lock_guard<std::mutex> lk(::sd::graph::g_globalCaptureWorkspaceMtx);
  for (const auto& kv : ::sd::graph::g_globalCaptureWorkspaceByDevice) {
    if (kv.second == ptr) return true;
  }
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Event management
// ═══════════════════════════════════════════════════════════════════════════════

void* dspCreateEvent() {
  cudaEvent_t evt;
  cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
  return reinterpret_cast<void*>(evt);
}

void dspDestroyEvent(void* event) {
  if (event != nullptr) {
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
  }
}

void dspEventRecord(void* event, void* stream) {
  if (event != nullptr) {
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                    reinterpret_cast<cudaStream_t>(stream));
  }
}

void dspStreamWaitEvent(void* stream, void* event) {
  if (event != nullptr) {
    cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream),
                        reinterpret_cast<cudaEvent_t>(event), 0);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TLS stream access
// ═══════════════════════════════════════════════════════════════════════════════

void* dspStreamPtrToValue(void* streamPtr) {
  // STREAM-POINTER (cudaStream_t*) -> STREAM-VALUE (cudaStream_t as void*). See the
  // convention block in DspCudaDispatch.h.
  return (streamPtr != nullptr)
             ? static_cast<void*>(*static_cast<cudaStream_t*>(streamPtr))
             : nullptr;
}

bool dspSynchronizeStream(void* stream) {
  if (stream == nullptr) return true;
  return cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess;
}

void* dspGetExecutionStream() {
  return tl_dspExecutionStream;
}

void dspSetExecutionStream(void* stream) {
  tl_dspExecutionStream = stream;
}

void* dspGetGapStream() {
  return static_cast<void*>(tl_dspGapStream);
}

void* dspGetGraphCaptureStream() {
  return tl_graphCaptureStream;
}

void dspSetGraphCaptureStream(void* stream) {
  tl_graphCaptureStream = stream;
}

void* dspGetLcDefaultStream() {
  cudaStream_t* ptr = sd::LaunchContext::defaultContext()->getCudaStream();
  return (ptr != nullptr) ? static_cast<void*>(*ptr) : nullptr;
}

void dspSyncDefaultStream() {
  cudaStream_t* ptr = sd::LaunchContext::defaultContext()->getCudaStream();
  if (ptr != nullptr) {
    cudaStreamSynchronize(*ptr);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Thread completion event (delegates to existing impl)
// ═══════════════════════════════════════════════════════════════════════════════

void dspPublishThreadCompletionEvent(void* streamPtr) {
  sd::dspPublishThreadCompletionEvent(streamPtr);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DSP replay active flag
// ═══════════════════════════════════════════════════════════════════════════════

bool dspGetReplayActive() {
  return tl_dspReplayActive;
}

void dspSetReplayActive(bool active) {
  tl_dspReplayActive = active;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA-only graph backend accessors
// ═══════════════════════════════════════════════════════════════════════════════

GraphBackend* dspGetNvrtcBackend() {
  return &NvrtcGraphBackend::getInstance();
}

GraphBackend* dspGetPtxBackend() {
  return &PtxGraphBackend::getInstance();
}

void dspCopyCublasHandle(LaunchContext* dst, LaunchContext* src) {
  dst->setCublasHandle(src->getCublasHandle());
}

void dspTritonClearFailedCache() {
#if HAVE_TRITON
  TritonGraphBackend::getInstance().clearFailedSegmentCache();
#endif
}

GraphBackend* dspGetTritonBackend() {
#if HAVE_TRITON
  return &TritonGraphBackend::getInstance();
#else
  return nullptr;
#endif
}

}  // namespace cuda
}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
