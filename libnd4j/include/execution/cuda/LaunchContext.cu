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

//
// @author raver119@gmail.com
//
#include <cuda_runtime.h>

#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <helpers/cublasHelper.h>
#include <helpers/logger.h>

#include <algorithm>

#include <thread>

// Per-device ContextBuffers array. Each device gets its own persistent buffers
// (streams, workspace) so device switches don't destroy/recreate buffers.
// This eliminates the "flip-flop" pattern where switching devices caused
// release+reinitialize cycles that fail when a device is out of memory.
//
//  ContextBuffers MUST live in thread-local storage (TLS), NOT on the
// heap. Known buffer overrun issues in C++ ops corrupt heap metadata, causing
// "free(): invalid pointer" crashes. TLS is in a separate memory segment that
// heap corruption cannot reach. Using `new ContextBuffers()` (heap) triggers
// this corruption; thread_local objects (TLS) are immune.
static constexpr int MAX_CUDA_DEVICES = 16;
thread_local sd::ContextBuffers contextBuffersStorage[MAX_CUDA_DEVICES];

// Returns the ContextBuffers for the current CUDA device.
// Exported for use by AffinityManager.cu.
sd::ContextBuffers& contextBuffersForCurrentDevice() {
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  if (deviceId < 0 || deviceId >= MAX_CUDA_DEVICES) deviceId = 0;
  // NOTE: contextBuffersStorage[N] for a secondary device N can be default-constructed with a
  // device-0 stream. The multi-GPU segment guard (platformBindSegmentDevice) routes secondary
  // ops onto cudaStreamPerThread (the driver-resolved current-device stream) instead of relying
  // on this per-device stream, so we do NOT re-home here — re-homing (release()+initialize())
  // inside this hot getter frees buffers still referenced by in-flight ops and does a cudaSetDevice
  // mid-operation, which corrupted shape/NDArray state. Reduction/scalar buffers for a sharded
  // secondary device are a separate follow-up (BGE reductions), not needed for the current path.
  return contextBuffersStorage[deviceId];
}

// Legacy single-buffer reference — redirects to per-device map.
// Kept as a macro-like reference for minimal code changes.
#define contextBuffers (contextBuffersForCurrentDevice())

// DSP gap-stream override for LaunchContext::getCudaStream().
// When set (non-null), getCudaStream() returns a pointer to this stream
// instead of the thread-local contextBuffers exec stream. This makes gap
// ops (matmul, reshape, etc.) in compositeReplay run on the DSP execution
// stream (cudaStr), eliminating all cross-stream event syncs between
// Triton islands and gap ops. Set/cleared by compositeReplay around gap
// unit execution. Same pattern as tl_dspExecutionStream for H2D copies.
thread_local cudaStream_t tl_dspGapStream = nullptr;

namespace sd {

// This avoids static destruction order crashes during JVM shutdown
std::vector<LaunchContext*>& LaunchContext::contexts() {
  static std::vector<LaunchContext*>* _contexts = new std::vector<LaunchContext*>();
  return *_contexts;
}

bool LaunchContext::isManagedContext(LaunchContext* contextPtr) {
  auto& ctxs = LaunchContext::contexts();
  return std::find(ctxs.begin(), ctxs.end(), contextPtr) != ctxs.end();
}

void LaunchContext::operator delete(void* ptr) noexcept {
  if (ptr == nullptr) return;
  auto* ctx = reinterpret_cast<LaunchContext*>(ptr);
  if (LaunchContext::isManagedContext(ctx)) return;

  ::operator delete(ptr);
}

std::mutex LaunchContext::_mutex;
SD_MAP_IMPL<int, std::mutex*> LaunchContext::_deviceMutexes;

////////////////////////////////////////////////////////////////////////
LaunchContext::LaunchContext(cudaStream_t* cudaStream, cudaStream_t& specialCudaStream, void* reductionPointer,
                             void* scalarPointer, int* allocationPointer) {

  _workspace = nullptr;
  _isAllocated = false;
  _deviceID = AffinityManager::currentDeviceId();

  // Store the provided stream and pointers
  _externalStream = cudaStream;
  _externalReductionPointer = reductionPointer;
  _externalScalarPointer = scalarPointer;
  _externalAllocationPointer = allocationPointer;
}

std::mutex* LaunchContext::deviceMutex() {
  auto deviceId = AffinityManager::currentDeviceId();

  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _deviceMutexes.find(deviceId);
  if (it == _deviceMutexes.end() || it->second == nullptr) {
    auto mutex = new std::mutex();
    _deviceMutexes[deviceId] = mutex;
    return mutex;
  }

  return it->second;
}

LaunchContext::~LaunchContext() {
  if (_isAllocated) {
  }
}

////////////////////////////////////////////////////////////////////////
LaunchContext::LaunchContext() {
  // default constructor, just to make clang/ranlib happy
  _workspace = nullptr;
  // Bind to the CURRENT device, not a hardcoded 0. defaultContext(deviceId) calls
  // setCurrentNativeDevice(deviceId) immediately before constructing, so the correct device is
  // already active here. A hardcoded 0 made defaultContext(1)->getDeviceID() return 0, which
  // routed device-1 CUDA-graph capture and CudaGraphScheduler onto device 0 (multi-GPU sharding).
  _deviceID = AffinityManager::currentDeviceId();

  _isAllocated = true;

  // No external pointers - will use thread_local contextBuffers
  _externalStream = nullptr;
  _externalReductionPointer = nullptr;
  _externalScalarPointer = nullptr;
  _externalAllocationPointer = nullptr;
}

LaunchContext::LaunchContext(Pointer cudaStream, Pointer reductionPointer, Pointer scalarPointer,
                             Pointer allocationPointer) {
  // Clear any stale CUDA errors from initialization before operations run
  cudaGetLastError();

  _isAllocated = false;
  _workspace = nullptr;
  _deviceID = AffinityManager::currentDeviceId();

  // Store externally provided pointers - these will be used instead of thread_local contextBuffers
  _externalStream = reinterpret_cast<cudaStream_t*>(cudaStream);
  _externalReductionPointer = reductionPointer;
  _externalScalarPointer = scalarPointer;
  _externalAllocationPointer = allocationPointer;
}

LaunchContext* LaunchContext::defaultContext() {
  /**
   * This method returns LaunchContext, that has multiple entities within:
   * 1) temporary buffers. they must be per-thread
   * 2) CUDA stream. it must be either per-thread or per-device
   * 3) cuBLAS handle. it must be per-device
   *
   * currentDeviceId() now always syncs with the native CUDA device (cudaGetDevice()),
   * so we use it directly. This ensures consistency across all code paths.
   */
  auto deviceId = AffinityManager::currentDeviceId();

  {
    // Initialize only the current device. Eagerly touching every physical GPU
    // makes startup fail when a non-selected GPU is temporarily out of memory.
    std::lock_guard<std::mutex> lock(_mutex);

    auto requiredSize = static_cast<size_t>(deviceId + 1);
    if (contexts().size() < requiredSize) {
      contexts().resize(requiredSize, nullptr);
    }

    auto mutexIt = _deviceMutexes.find(deviceId);
    if (mutexIt == _deviceMutexes.end() || mutexIt->second == nullptr) {
      _deviceMutexes[deviceId] = new std::mutex();
    }

    if (contexts().at(deviceId) == nullptr) {
      AffinityManager::setCurrentNativeDevice(deviceId);
      contexts().at(deviceId) = new LaunchContext();
    }
  }

  // return context for current device
  return contexts().at(deviceId);
}

void* LaunchContext::getReductionPointer() const {
  // IMPORTANT: Always use the thread-local contextBuffers reduction buffer instead of
  // external pointer. The external pointer passed from Java may point to freed memory if
  // contextBuffers was reinitialized between the time Java obtained the pointer and now.
  // Using the thread-local contextBuffers ensures we always have a valid buffer that
  // matches the current device and stream (getCudaStream() already does this).
  return contextBuffers.reductionBuffer();
};

void* LaunchContext::getScalarPointer() const {
  // Always use thread-local contextBuffers to match getCudaStream() behavior.
  // See getReductionPointer() comment for rationale.
  return contextBuffers.scalarBuffer();
};

LongType* LaunchContext::getAllocationPointer() const {
  // Always use thread-local contextBuffers to match getCudaStream() behavior.
  // See getReductionPointer() comment for rationale.
  return reinterpret_cast<LongType*>(contextBuffers.allocationBuffer());
};

void* LaunchContext::getCublasHandle() const { return CublasHelper::getInstance().handle(); };

void* LaunchContext::getCusolverHandle() const { return CublasHelper::getInstance().solver(); };

cudaStream_t* LaunchContext::getCudaStream() const {
  // DSP gap-stream override: when compositeReplay sets tl_dspGapStream,
  // gap ops (matmul, reshape, etc.) run on the DSP execution stream
  // instead of the default contextBuffers stream. This eliminates all
  // cross-stream event syncs between Triton islands and gap ops.
  if (tl_dspGapStream != nullptr) {
    return &tl_dspGapStream;
  }

  // IMPORTANT: Always use the thread-local contextBuffers stream instead of external stream
  // The external stream pointer passed from Java may point to freed memory if the
  // contextBuffers was reinitialized between the time Java obtained the pointer and now.
  // Using the thread-local contextBuffers ensures we always have a valid stream.
  auto stream = reinterpret_cast<cudaStream_t*>(contextBuffers.execStream());
  if (stream == nullptr || *stream == nullptr) {
    // Stream not initialized - this shouldn't happen if contextBuffers.execStream() works correctly
    fprintf(stderr, "WARNING: getCudaStream() returning null stream - context may not be initialized\n");
    fflush(stderr);
  }
  return stream;
};

cudaStream_t* LaunchContext::getCudaSpecialStream() const {
  return reinterpret_cast<cudaStream_t*>(contextBuffers.specialStream());
  ;
};

void LaunchContext::setReductionPointer(void* reductionPointer) {
  contextBuffers.setReductionBuffer(reductionPointer);
};

void LaunchContext::setScalarPointer(void* scalarPointer) { contextBuffers.setScalarBuffer(scalarPointer); };

void LaunchContext::setAllocationPointer(int* allocationPointer) {
  contextBuffers.setAllocationBuffer(allocationPointer);
};

void LaunchContext::setCudaStream(cudaStream_t* cudaStream){
};

void LaunchContext::setCudaSpecialStream(cudaStream_t* cudaStream){
};

void LaunchContext::setCublasHandle(void* handle) { _cublasHandle = handle; };

void LaunchContext::swapContextBuffers(ContextBuffers& buffers) {
  // Use move assignment to properly transfer ownership of GPU resources (streams,
  // reduction/allocation buffers) into the thread-local contextBuffers.
  // Copy assignment sets _allocated=false on the copy, leaving the thread-local
  // with pointers it doesn't own — if the original is destroyed, those become dangling.
  // Move assignment transfers ownership: the thread-local becomes the owner and the
  // source is left in a safe empty state.
  contextBuffers = std::move(buffers);
};

void LaunchContext::releaseBuffers() {
  contextBuffers.release();
}

bool LaunchContext::isInitialized() { return contextBuffers.isInitialized(); }

void* LaunchContext::getCuDnnHandle() const { return CublasHelper::getInstance().cudnn(); }

void* LaunchContext::getCusparseHandle() const { return CublasHelper::getInstance().sparseHandle(); }

ErrorReference* LaunchContext::errorReference() { return contextBuffers.errorReference(); }

void* LaunchContext::engine() { return _engine; }

// ============================================================================
// CUDA Graph Support Implementation
// ============================================================================

bool LaunchContext::beginGraphCapture(cudaStreamCaptureMode mode) {
  if (_graphCaptureActive) {
    sd_print("LaunchContext::beginGraphCapture - Already capturing\n");
    return false;
  }

  cudaStream_t* stream = getCudaStream();
  if (stream == nullptr) {
    sd_print("LaunchContext::beginGraphCapture - No stream available\n");
    return false;
  }

  cudaError_t err = cudaStreamBeginCapture(*stream, mode);
  if (err != cudaSuccess) {
    sd_printf("LaunchContext::beginGraphCapture failed: %s", cudaGetErrorString(err));
    return false;
  }

  _graphCaptureActive = true;
  _captureMode = mode;
  return true;
}

bool LaunchContext::endGraphCapture(cudaGraph_t* outGraph) {
  if (!_graphCaptureActive) {
    sd_print("LaunchContext::endGraphCapture - Not currently capturing\n");
    return false;
  }

  if (outGraph == nullptr) {
    sd_print("LaunchContext::endGraphCapture - Null output pointer\n");
    return false;
  }

  cudaStream_t* stream = getCudaStream();
  if (stream == nullptr) {
    return false;
  }

  cudaError_t err = cudaStreamEndCapture(*stream, outGraph);
  _graphCaptureActive = false;

  if (err != cudaSuccess) {
    sd_printf("LaunchContext::endGraphCapture failed: %s", cudaGetErrorString(err));
    return false;
  }

  return *outGraph != nullptr;
}

void LaunchContext::abortGraphCapture() {
  if (!_graphCaptureActive) {
    return;
  }

  cudaStream_t* stream = getCudaStream();
  if (stream != nullptr) {
    cudaGraph_t discardedGraph;
    cudaStreamEndCapture(*stream, &discardedGraph);
    if (discardedGraph != nullptr) {
      cudaGraphDestroy(discardedGraph);
    }
  }

  _graphCaptureActive = false;
}

bool LaunchContext::instantiateGraph(cudaGraph_t graph, cudaGraphExec_t* outGraphExec) {
  if (graph == nullptr || outGraphExec == nullptr) {
    return false;
  }

  cudaGraphNode_t errorNode;
  char logBuffer[1024] = {0};

  cudaError_t err = cudaGraphInstantiate(outGraphExec, graph, &errorNode, logBuffer, sizeof(logBuffer));

  if (err != cudaSuccess) {
    sd_printf("LaunchContext::instantiateGraph failed: %s", cudaGetErrorString(err));
    if (strlen(logBuffer) > 0) {
      sd_printf("Graph instantiation log: %s", logBuffer);
    }
    return false;
  }

  return true;
}

bool LaunchContext::launchGraph(cudaGraphExec_t graphExec) {
  if (graphExec == nullptr) {
    return false;
  }

  cudaStream_t* stream = getCudaStream();
  if (stream == nullptr) {
    return false;
  }

  cudaError_t err = cudaGraphLaunch(graphExec, *stream);
  if (err != cudaSuccess) {
    sd_printf("LaunchContext::launchGraph failed: %s", cudaGetErrorString(err));
    return false;
  }

  // Synchronous execution
  err = cudaStreamSynchronize(*stream);
  if (err != cudaSuccess) {
    sd_printf("LaunchContext::launchGraph sync failed: %s", cudaGetErrorString(err));
    return false;
  }

  return true;
}

bool LaunchContext::launchGraphAsync(cudaGraphExec_t graphExec, cudaEvent_t completionEvent) {
  if (graphExec == nullptr) {
    return false;
  }

  cudaStream_t* stream = getCudaStream();
  if (stream == nullptr) {
    return false;
  }

  cudaError_t err = cudaGraphLaunch(graphExec, *stream);
  if (err != cudaSuccess) {
    sd_printf("LaunchContext::launchGraphAsync failed: %s", cudaGetErrorString(err));
    return false;
  }

  if (completionEvent != nullptr) {
    err = cudaEventRecord(completionEvent, *stream);
    if (err != cudaSuccess) {
      sd_printf("LaunchContext::launchGraphAsync event record failed: %s", cudaGetErrorString(err));
      // Don't fail - graph was launched, just event recording failed
    }
  }

  return true;
}

}  // namespace sd
