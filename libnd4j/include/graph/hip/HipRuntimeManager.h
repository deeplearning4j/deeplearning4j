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

#ifndef LIBND4J_HIP_RUNTIME_MANAGER_H
#define LIBND4J_HIP_RUNTIME_MANAGER_H

// Guard the entire file behind SD_HIP so it has zero footprint on CUDA/CPU builds.
#ifdef SD_HIP

#include <system/common.h>

#include <mutex>
#include <string>

// ── Opaque HIP function-pointer typedefs ─────────────────────────────────────
//
// All HIP API calls go through these function pointers resolved at runtime via
// dlopen("libamdhip64.so").  NO HIP headers are included here — this is the
// dlopen-opaque pattern that lets the file compile on any host even without an
// AMD GPU or ROCm installation.
//
// The signatures match the real HIP API signatures from
// /opt/rocm/include/hip/hip_runtime_api.h but use:
//   - void*         in place of hipStream_t, hipGraph_t, hipGraphExec_t (opaque handles)
//   - unsigned int  in place of hipStreamCaptureMode (enum value 0 = hipStreamCaptureModeGlobal)
//   - int           in place of hipError_t (0 = hipSuccess)
//
// At runtime the caller passes void* for all handle arguments; the HIP runtime
// sees the correctly-typed value because handles are pointer-sized types.

// hipStreamCreate(hipStream_t* pStream)
using HipStreamCreateFn  = int (*)(void**);

// hipStreamDestroy(hipStream_t stream)
using HipStreamDestroyFn = int (*)(void*);

// hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode)
//   mode: 0 = hipStreamCaptureModeGlobal (safe default)
using HipStreamBeginCaptureFn = int (*)(void*, unsigned int);

// hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph)
using HipStreamEndCaptureFn   = int (*)(void*, void**);

// hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
//                     hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize)
//   pErrorNode and pLogBuffer may be nullptr.
using HipGraphInstantiateFn   = int (*)(void**, void*, void*, char*, size_t);

// hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream)
using HipGraphLaunchFn        = int (*)(void*, void*);

// hipGraphDestroy(hipGraph_t graph)
using HipGraphDestroyFn       = int (*)(void*);

// hipGraphExecDestroy(hipGraphExec_t graphExec)
using HipGraphExecDestroyFn   = int (*)(void*);

// hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, ..., void** kernelParams,
//                       void** extra) — simplified: only the function pointer and stream
//   Full signature used for the syntax-check; resolved lazily in practice.
using HipModuleLaunchKernelFn = int (*)(void*, unsigned int, unsigned int, unsigned int,
                                         unsigned int, unsigned int, unsigned int,
                                         unsigned int, void*, void**, void**);

// hipStreamSynchronize(hipStream_t stream)
using HipStreamSynchronizeFn  = int (*)(void*);

// hipGetErrorString(hipError_t hipError) → const char*
using HipGetErrorStringFn     = const char* (*)(int);

namespace sd {
namespace graph {

/**
 * HipRuntimeManager — dlopen-opaque loader for libamdhip64.so.
 *
 * Responsibilities:
 *   1. Locate and dlopen the HIP runtime shared library.
 *   2. Resolve all required function pointers (see typedefs above).
 *   3. Expose isAvailable() so callers can gate hardware paths without
 *      crashing on non-AMD hosts.
 *   4. Provide thin inline wrappers around each resolved function so that
 *      call sites do not need to cast void(*) themselves.
 *
 * Thread safety: getInstance() and isAvailable() are safe to call from any
 * thread after the singleton is constructed.  The internal mutex_ protects
 * the lazy-init path.
 *
 * Usage (requires ROCm/HIP runtime — validated on an AMD box only):
 *   auto& mgr = HipRuntimeManager::getInstance();
 *   if (!mgr.isAvailable()) return; // not on AMD hardware
 *   void* stream = nullptr;
 *   mgr.streamCreate(&stream);
 *   mgr.streamBeginCapture(stream, 0);
 *   // ... kernel launches ...
 *   void* graph = nullptr;
 *   mgr.streamEndCapture(stream, &graph);
 *   void* exec  = nullptr;
 *   mgr.graphInstantiate(&exec, graph, nullptr, nullptr, 0);
 *   mgr.graphLaunch(exec, stream);
 *   mgr.streamSynchronize(stream);
 *   mgr.graphExecDestroy(exec);
 *   mgr.graphDestroy(graph);
 *   mgr.streamDestroy(stream);
 */
class SD_LIB_EXPORT HipRuntimeManager {
 public:
  /**
   * Get the singleton instance.
   * Construction is lazy and thread-safe (call_once).
   */
  static HipRuntimeManager& getInstance();

  /**
   * Returns true if libamdhip64.so was loaded and all required symbols
   * were resolved.  Always false on non-AMD / non-ROCm hosts.
   */
  bool isAvailable() const;

  /**
   * Return a human-readable description of why the library is unavailable,
   * or an empty string if it is available.
   */
  const std::string& getLastError() const;

  // ── Thin function-pointer wrappers ────────────────────────────────────────
  // These forward directly to the resolved function pointers.
  // Callers must check isAvailable() before calling any of these — behaviour
  // is undefined (likely segfault) if called when the library is not loaded.

  int streamCreate(void** pStream);
  int streamDestroy(void* stream);
  int streamBeginCapture(void* stream, unsigned int mode = 0 /*hipStreamCaptureModeGlobal*/);
  int streamEndCapture(void* stream, void** pGraph);
  int graphInstantiate(void** pExec, void* graph,
                       void* pErrorNode, char* pLogBuffer, size_t bufferSize);
  int graphLaunch(void* exec, void* stream);
  int graphDestroy(void* graph);
  int graphExecDestroy(void* exec);
  int streamSynchronize(void* stream);
  const char* getErrorString(int hipError);

 private:
  HipRuntimeManager();
  ~HipRuntimeManager();

  // Non-copyable
  HipRuntimeManager(const HipRuntimeManager&) = delete;
  HipRuntimeManager& operator=(const HipRuntimeManager&) = delete;

  bool loadLibrary();

  // dlopen handle (nullptr if library not loaded)
  void* libHandle_ = nullptr;

  bool available_   = false;
  bool initFailed_  = false;
  std::string lastError_;

  // Resolved function pointers — all nullptr until loadLibrary() succeeds.
  HipStreamCreateFn         fnStreamCreate_         = nullptr;
  HipStreamDestroyFn        fnStreamDestroy_        = nullptr;
  HipStreamBeginCaptureFn   fnStreamBeginCapture_   = nullptr;
  HipStreamEndCaptureFn     fnStreamEndCapture_     = nullptr;
  HipGraphInstantiateFn     fnGraphInstantiate_     = nullptr;
  HipGraphLaunchFn          fnGraphLaunch_          = nullptr;
  HipGraphDestroyFn         fnGraphDestroy_         = nullptr;
  HipGraphExecDestroyFn     fnGraphExecDestroy_     = nullptr;
  HipModuleLaunchKernelFn   fnModuleLaunchKernel_   = nullptr;
  HipStreamSynchronizeFn    fnStreamSynchronize_    = nullptr;
  HipGetErrorStringFn       fnGetErrorString_       = nullptr;

  mutable std::mutex mutex_;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_HIP
#endif  // LIBND4J_HIP_RUNTIME_MANAGER_H
