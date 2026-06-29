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
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_CUDACONTEXT_H
#define LIBND4J_CUDACONTEXT_H



#ifdef SD_CUDA
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "config.h"
#endif

// Forward declarations for CUDA graph support
#ifdef SD_CUDA
namespace sd {
namespace cuda {
    class CudaGraphHandle;
    class CudaGraphScheduler;
}  // namespace cuda
}  // namespace sd
#endif

// used for MKLDNN etc
#if !defined(__STANDALONE_BUILD__)
#include "config.h"
#endif

#include <execution/ContextBuffers.h>
#include <execution/ErrorReference.h>
#include <memory/Workspace.h>
#include <system/common.h>
#include <system/op_boilerplate.h>

#include <memory>
#include <mutex>
#include <vector>

namespace sd {

// Forward declaration needed for LaunchContextNoOpDeleter
class LaunchContext;

// Custom deleter for sd::LaunchContext instances that are intentionally leaked
// by the LaunchContext::defaultContext() mechanism.
// This prevents std::shared_ptr from attempting to delete these objects,
// avoiding double-free or invalid memory access issues during shutdown.
struct LaunchContextNoOpDeleter {
    void operator()(LaunchContext* lc) const {
        // Do nothing. The LaunchContext instance is managed by the static contexts() vector
        // and is intentionally leaked to avoid static destruction order problems.
        // Its memory is reclaimed by the operating system on process exit.
    }
};

class SD_LIB_EXPORT LaunchContext {

 private:
  // Previous implementation used heap-allocated pointer which caused crashes during JVM shutdown:
  // - The vector* was allocated with new and "intentionally leaked"
  // - But C++ static destruction still tried to destruct it, causing SIGSEGV
  // - This SIGSEGV triggered signal handler which called abort(), resulting in SIGABRT
  // Function-local static has well-defined destruction order and avoids this crash
  static std::vector<LaunchContext*>& contexts();
  static std::mutex _mutex;

  static SD_MAP_IMPL<int, std::mutex*> _deviceMutexes;

  // used for MKLDNN
  void* _engine = nullptr;

#ifdef SD_CUDA

#ifndef __JAVACPP_HACK__

  void* _cublasHandle = nullptr;
  void* _cusolverHandle = nullptr;

#endif  // JCPP

  bool _isAllocated = false;

  // CUDA graph capture state
  bool _graphCaptureActive = false;
  cudaStreamCaptureMode _captureMode = cudaStreamCaptureModeGlobal;

  // Externally provided stream pointer (when using the Pointer constructor)
  // If non-null, getCudaStream() returns this instead of thread_local contextBuffers
  cudaStream_t* _externalStream = nullptr;
  void* _externalReductionPointer = nullptr;
  void* _externalScalarPointer = nullptr;
  void* _externalAllocationPointer = nullptr;

#endif  // CUDA
  memory::Workspace* _workspace = nullptr;
  int _deviceID = 0;

 public:
#ifdef SD_CUDA

#ifndef __JAVACPP_HACK__
  LaunchContext(cudaStream_t* cudaStream, cudaStream_t& specialCudaStream, void* reductionPointer = nullptr,
                void* scalarPointer = nullptr, int* allocationPointer = nullptr);

  void* getReductionPointer() const;
  void* getScalarPointer() const;
  LongType* getAllocationPointer() const;
  void* getCublasHandle() const;
  void* getCusolverHandle() const;
  void* getCuDnnHandle() const;
  void* getCusparseHandle() const;
  cudaStream_t* getCudaStream() const;
  cudaStream_t* getCudaSpecialStream() const;

  void setReductionPointer(void* reductionPointer);
  void setScalarPointer(void* scalarPointer);
  void setAllocationPointer(int* allocationPointer);
  void setCudaStream(cudaStream_t* cudaStream);
  void setCudaSpecialStream(cudaStream_t* cudaStream);
  void setCublasHandle(void* handle);

  // =========================================================================
  // CUDA Graph Support
  // =========================================================================

  /**
   * Begin capturing operations to a CUDA graph on this context's stream
   * @param mode Capture mode (global, thread-local, or relaxed)
   * @return true if capture started successfully
   */
  bool beginGraphCapture(cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal);

  /**
   * End graph capture and return the captured graph
   * @param outGraph Pointer to store the captured graph
   * @return true if capture ended successfully
   */
  bool endGraphCapture(cudaGraph_t* outGraph);

  /**
   * Check if currently capturing a graph
   */
  bool isCapturingGraph() const { return _graphCaptureActive; }

  /**
   * Get the current capture mode
   */
  cudaStreamCaptureMode getCaptureMode() const { return _captureMode; }

  /**
   * Abort current graph capture
   */
  void abortGraphCapture();

  /**
   * Instantiate a captured graph for execution
   * @param graph The captured graph
   * @param outGraphExec Pointer to store the executable graph
   * @return true if instantiation succeeded
   */
  static bool instantiateGraph(cudaGraph_t graph, cudaGraphExec_t* outGraphExec);

  /**
   * Launch an executable graph on this context's stream
   * @param graphExec The executable graph
   * @return true if launch succeeded
   */
  bool launchGraph(cudaGraphExec_t graphExec);

  /**
   * Launch an executable graph asynchronously
   * @param graphExec The executable graph
   * @param completionEvent Optional event to signal completion
   * @return true if launch succeeded
   */
  bool launchGraphAsync(cudaGraphExec_t graphExec, cudaEvent_t completionEvent = nullptr);

#endif  // JCPP

#endif  // CUDA
  LaunchContext(Pointer cudaStream, Pointer reductionPointer = nullptr, Pointer scalarPointer = nullptr,
                Pointer allocationPointer = nullptr);
  LaunchContext();
  ~LaunchContext();
  static bool isManagedContext(LaunchContext* contextPtr);
  void operator delete(void* ptr) noexcept;
  memory::Workspace* getWorkspace() const {
    return _workspace;
  }
  void setWorkspace(memory::Workspace* theWorkspace) { _workspace = theWorkspace; }

  void* engine();

  int getDeviceID() const { return _deviceID; }
  void setDeviceID(int deviceID) { _deviceID = deviceID; }
  ErrorReference* errorReference();

#ifndef __JAVACPP_HACK__
  // this method returns mutex shared between all threads that use the same device
  static std::mutex* deviceMutex();

#endif

  static bool isInitialized();
  static void releaseBuffers();

  static LaunchContext* defaultContext();

  static void swapContextBuffers(ContextBuffers& buffers);
};

}  // namespace sd

#endif  // LIBND4J_CUDACONTEXT_H
