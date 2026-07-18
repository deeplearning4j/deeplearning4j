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

#if defined(SD_HIP) || defined(ZLUDA_TARGET_AMD) || defined(HAVE_MIOPEN)

#include <graph/hip/HipRuntimeManager.h>
#include <graph/DspDiagnostics.h>

#include <dlfcn.h>
#include <mutex>
#include <string>

// Macro to resolve one symbol from the loaded library.
// sym must be a char* literal matching the real HIP function name.
// dest is the typed function-pointer field to fill.
#define HIP_RESOLVE(sym, dest, FnType) do { \
  (dest) = reinterpret_cast<FnType>(dlsym(libHandle_, (sym))); \
  if ((dest) == nullptr) { \
    lastError_ = std::string("HipRuntimeManager: failed to resolve '") + (sym) + \
                 "': " + dlerror(); \
    DSP_DIAG(BACKEND, "%s", lastError_.c_str()); \
    dlclose(libHandle_); \
    libHandle_ = nullptr; \
    return false; \
  } \
  DSP_DIAG(BACKEND, "HipRuntimeManager: resolved %s @ %p", (sym), (void*)(dest)); \
} while(0)

namespace sd {
namespace graph {

// ── Singleton ────────────────────────────────────────────────────────────────

HipRuntimeManager& HipRuntimeManager::getInstance() {
  static HipRuntimeManager* instance = nullptr;
  static std::once_flag flag;
  std::call_once(flag, []() {
    instance = new HipRuntimeManager();
  });
  return *instance;
}

// ── Constructor / Destructor ─────────────────────────────────────────────────

HipRuntimeManager::HipRuntimeManager() {
  // Lazy init — library is loaded on first isAvailable() or explicit call.
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initFailed_) {
    available_ = loadLibrary();
    if (!available_) initFailed_ = true;
  }
}

HipRuntimeManager::~HipRuntimeManager() {
  if (libHandle_ != nullptr) {
    DSP_DIAG(BACKEND, "HipRuntimeManager: unloading libamdhip64.so");
    dlclose(libHandle_);
    libHandle_ = nullptr;
  }
}

// ── Library Loading ──────────────────────────────────────────────────────────

bool HipRuntimeManager::loadLibrary() {
  // Candidate library paths, in preference order.
  // The first path that dlopen() succeeds on is used.
  static const char* kCandidates[] = {
    "libamdhip64.so",                    // rely on LD_LIBRARY_PATH / loader
    "libamdhip64.so.6",                  // ROCm 6.x versioned name
    "libamdhip64.so.5",                  // ROCm 5.x versioned name
    "/opt/rocm/lib/libamdhip64.so",      // default ROCm installation
    "/opt/rocm-6.0/lib/libamdhip64.so",  // versioned ROCm install
    "/opt/rocm-5.7/lib/libamdhip64.so",  // older ROCm install
    nullptr
  };

  for (int i = 0; kCandidates[i] != nullptr; ++i) {
    libHandle_ = dlopen(kCandidates[i], RTLD_NOW | RTLD_LOCAL);
    if (libHandle_ != nullptr) {
      DSP_DIAG(BACKEND, "HipRuntimeManager: loaded %s @ %p",
               kCandidates[i], libHandle_);
      break;
    }
    DSP_DIAG(BACKEND, "HipRuntimeManager: dlopen(%s) failed: %s",
             kCandidates[i], dlerror());
  }

  if (libHandle_ == nullptr) {
    lastError_ = std::string(
      "HipRuntimeManager: could not load libamdhip64.so from any known path. "
      "This is expected on non-AMD hosts. "
      "On AMD/ROCm hosts ensure ROCm is installed and LD_LIBRARY_PATH includes "
      "the ROCm lib directory (e.g. /opt/rocm/lib). "
      "Last dlerror: ") + dlerror();
    DSP_DIAG(BACKEND, "%s", lastError_.c_str());
    return false;
  }

  // ── Resolve required symbols ───────────────────────────────────────────────
  HIP_RESOLVE("hipStreamCreate",       fnStreamCreate_,       HipStreamCreateFn);
  HIP_RESOLVE("hipStreamDestroy",      fnStreamDestroy_,      HipStreamDestroyFn);
  HIP_RESOLVE("hipStreamBeginCapture", fnStreamBeginCapture_, HipStreamBeginCaptureFn);
  HIP_RESOLVE("hipStreamEndCapture",   fnStreamEndCapture_,   HipStreamEndCaptureFn);
  HIP_RESOLVE("hipGraphInstantiate",   fnGraphInstantiate_,   HipGraphInstantiateFn);
  HIP_RESOLVE("hipGraphLaunch",        fnGraphLaunch_,        HipGraphLaunchFn);
  HIP_RESOLVE("hipGraphDestroy",       fnGraphDestroy_,       HipGraphDestroyFn);
  HIP_RESOLVE("hipGraphExecDestroy",   fnGraphExecDestroy_,   HipGraphExecDestroyFn);
  HIP_RESOLVE("hipStreamSynchronize",  fnStreamSynchronize_,  HipStreamSynchronizeFn);
  HIP_RESOLVE("hipGetErrorString",     fnGetErrorString_,     HipGetErrorStringFn);

  // hipModuleLaunchKernel is optional (used for custom kernel dispatch).
  // If it is absent we log a diagnostic but do not fail.
  fnModuleLaunchKernel_ = reinterpret_cast<HipModuleLaunchKernelFn>(
      dlsym(libHandle_, "hipModuleLaunchKernel"));
  if (fnModuleLaunchKernel_ == nullptr) {
    DSP_DIAG(BACKEND,
             "HipRuntimeManager: hipModuleLaunchKernel not found (optional): %s",
             dlerror());
  }

  DSP_DIAG(BACKEND, "HipRuntimeManager: all required HIP symbols resolved");
  return true;
}

// ── Public interface ─────────────────────────────────────────────────────────

bool HipRuntimeManager::isAvailable() const {
  return available_;
}

const std::string& HipRuntimeManager::getLastError() const {
  return lastError_;
}

int HipRuntimeManager::streamCreate(void** pStream) {
  // Requires ROCm/HIP runtime — not validated on non-AMD hosts.
  return fnStreamCreate_(pStream);
}

int HipRuntimeManager::streamDestroy(void* stream) {
  return fnStreamDestroy_(stream);
}

int HipRuntimeManager::streamBeginCapture(void* stream, unsigned int mode) {
  // Requires ROCm/HIP runtime.
  // mode 0 = hipStreamCaptureModeGlobal (safe default matching CUDA default).
  return fnStreamBeginCapture_(stream, mode);
}

int HipRuntimeManager::streamEndCapture(void* stream, void** pGraph) {
  return fnStreamEndCapture_(stream, pGraph);
}

int HipRuntimeManager::graphInstantiate(void** pExec, void* graph,
                                         void* pErrorNode, char* pLogBuffer,
                                         size_t bufferSize) {
  return fnGraphInstantiate_(pExec, graph, pErrorNode, pLogBuffer, bufferSize);
}

int HipRuntimeManager::graphLaunch(void* exec, void* stream) {
  return fnGraphLaunch_(exec, stream);
}

int HipRuntimeManager::graphDestroy(void* graph) {
  return fnGraphDestroy_(graph);
}

int HipRuntimeManager::graphExecDestroy(void* exec) {
  return fnGraphExecDestroy_(exec);
}

int HipRuntimeManager::streamSynchronize(void* stream) {
  return fnStreamSynchronize_(stream);
}

const char* HipRuntimeManager::getErrorString(int hipError) {
  if (fnGetErrorString_ == nullptr) return "(HipRuntimeManager: not loaded)";
  return fnGetErrorString_(hipError);
}

}  // namespace graph
}  // namespace sd

#endif  // SD_HIP
