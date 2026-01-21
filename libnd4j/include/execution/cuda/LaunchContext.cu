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
#include <exceptions/cuda_exception.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <helpers/cublasHelper.h>
#include <helpers/logger.h>

#include <algorithm>

#include <thread>

thread_local sd::ContextBuffers contextBuffers = sd::ContextBuffers();

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
  return _deviceMutexes[deviceId];
}

LaunchContext::~LaunchContext() {
  if (_isAllocated) {
  }
}

////////////////////////////////////////////////////////////////////////
LaunchContext::LaunchContext() {
  // default constructor, just to make clang/ranlib happy
  _workspace = nullptr;
  _deviceID = 0;

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
    // we need this block synchronous, to avoid double initialization etc
    std::lock_guard<std::mutex> lock(_mutex);
    if (contexts().empty()) {
      // create one context per device
      auto numDevices = AffinityManager::numberOfDevices();

      contexts().resize(numDevices);
      for (int e = 0; e < numDevices; e++) {
        _deviceMutexes[e] = new std::mutex();

        AffinityManager::setCurrentNativeDevice(e);

        contexts().at(e) = new LaunchContext();
      }

      // don't forget to restore device back again
      AffinityManager::setCurrentNativeDevice(deviceId);
    }
  }

  // return context for current device
  return contexts().at(deviceId);
}

void* LaunchContext::getReductionPointer() const {
  if (_externalReductionPointer != nullptr) return _externalReductionPointer;
  return contextBuffers.reductionBuffer();
};

void* LaunchContext::getScalarPointer() const {
  if (_externalScalarPointer != nullptr) return _externalScalarPointer;
  return contextBuffers.scalarBuffer();
};

LongType* LaunchContext::getAllocationPointer() const {
  if (_externalAllocationPointer != nullptr) return reinterpret_cast<LongType*>(_externalAllocationPointer);
  return reinterpret_cast<LongType*>(contextBuffers.allocationBuffer());
};

void* LaunchContext::getCublasHandle() const { return CublasHelper::getInstance().handle(); };

void* LaunchContext::getCusolverHandle() const { return CublasHelper::getInstance().solver(); };

cudaStream_t* LaunchContext::getCudaStream() const {
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

void LaunchContext::swapContextBuffers(ContextBuffers& buffers) { contextBuffers = buffers; };

void LaunchContext::releaseBuffers() {
  contextBuffers.release();
}

bool LaunchContext::isInitialized() { return contextBuffers.isInitialized(); }

void* LaunchContext::getCuDnnHandle() const { return CublasHelper::getInstance().cudnn(); }

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
