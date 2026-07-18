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

#ifdef HAVE_HEXAGON_MLIR

#include <graph/hexagon/HexagonRuntimeManager.h>
#include <graph/DspDiagnostics.h>

#include <cstring>
#include <dlfcn.h>
#include <mutex>

namespace sd {
namespace graph {

// ---- Helper macro for dlsym resolution ----

#define RESOLVE_SYM(handle, fnPtr, symName, fnType)                            \
  do {                                                                         \
    (fnPtr) = reinterpret_cast<fnType>(dlsym((handle), (symName)));            \
    if (!(fnPtr)) {                                                            \
      DSP_DIAG(BACKEND, "HexagonRuntimeManager: failed to resolve '%s': %s",  \
               (symName), dlerror());                                          \
      unloadRuntimeLocked();                                                   \
      return false;                                                            \
    }                                                                          \
  } while (0)

// ---- Singleton ----

HexagonRuntimeManager& HexagonRuntimeManager::getInstance() {
  static HexagonRuntimeManager* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new HexagonRuntimeManager();
  });
  return *instance;
}

// ---- Constructor / Destructor ----

HexagonRuntimeManager::HexagonRuntimeManager() {
  available_ = loadRuntime();
  if (available_) {
    DSP_DIAG(BACKEND, "HexagonRuntimeManager: runtime loaded, %d NPU device(s) found",
             getDeviceCount());
  } else {
    DSP_DIAG(BACKEND, "HexagonRuntimeManager: runtime not available");
  }
}

HexagonRuntimeManager::~HexagonRuntimeManager() {
  unloadRuntime();
}

// ---- Runtime Loading ----

bool HexagonRuntimeManager::loadRuntime() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (libHandle_ != nullptr) return true;  // Already loaded

  // Try loading the hexagon-mlir runtime shared library
  libHandle_ = dlopen("libhexagon_mlir_runtime.so", RTLD_NOW | RTLD_LOCAL);
  if (libHandle_ == nullptr) {
    DSP_DIAG(BACKEND, "HexagonRuntimeManager: dlopen failed: %s", dlerror());
    return false;
  }

  // Resolve all function pointers
  RESOLVE_SYM(libHandle_, fnGetDeviceCount_, "hexmlir_get_device_count", FnGetDeviceCount);
  RESOLVE_SYM(libHandle_, fnInitDevice_,     "hexmlir_init_device",      FnInitDevice);
  RESOLVE_SYM(libHandle_, fnReleaseDevice_,  "hexmlir_release_device",   FnReleaseDevice);

  RESOLVE_SYM(libHandle_, fnAllocateTcm_,    "hexmlir_allocate_tcm",     FnAllocateTcm);
  RESOLVE_SYM(libHandle_, fnFreeTcm_,        "hexmlir_free_tcm",         FnFreeTcm);
  RESOLVE_SYM(libHandle_, fnGetTcmCapacity_, "hexmlir_get_tcm_capacity", FnGetTcmCapacity);

  RESOLVE_SYM(libHandle_, fnAllocateDdr_,    "hexmlir_allocate_ddr",     FnAllocateDdr);
  RESOLVE_SYM(libHandle_, fnFreeDdr_,        "hexmlir_free_ddr",         FnFreeDdr);

  // A deployment may be AOT-only (load symbol), development/JIT-only
  // (compile symbol), or support both. At least one creation path is required.
  fnCompileKernel_ = reinterpret_cast<FnCompileKernel>(
      dlsym(libHandle_, "hexmlir_compile_kernel"));
  fnLoadKernel_ = reinterpret_cast<FnLoadKernel>(
      dlsym(libHandle_, "hexmlir_load_kernel"));
  if (fnCompileKernel_ == nullptr && fnLoadKernel_ == nullptr) {
    DSP_DIAG(BACKEND,
             "HexagonRuntimeManager: runtime exports neither "
             "hexmlir_compile_kernel nor hexmlir_load_kernel");
    unloadRuntimeLocked();
    return false;
  }
  RESOLVE_SYM(libHandle_, fnReleaseKernel_,    "hexmlir_release_kernel",     FnReleaseKernel);
  RESOLVE_SYM(libHandle_, fnDispatchKernel_,   "hexmlir_dispatch_kernel",    FnDispatchKernel);
  RESOLVE_SYM(libHandle_, fnWaitForCompletion_,"hexmlir_wait_for_completion", FnWaitForCompletion);

  RESOLVE_SYM(libHandle_, fnDmaHostToTcm_, "hexmlir_dma_host_to_tcm", FnDmaCopy);
  RESOLVE_SYM(libHandle_, fnDmaTcmToHost_, "hexmlir_dma_tcm_to_host", FnDmaCopy);
  RESOLVE_SYM(libHandle_, fnDmaCopy_,      "hexmlir_dma_copy",        FnDmaCopy);

  // Verify at least one NPU device is present
  int deviceCount = fnGetDeviceCount_();
  if (deviceCount <= 0) {
    DSP_DIAG(BACKEND, "HexagonRuntimeManager: runtime loaded but no NPU devices found");
    unloadRuntimeLocked();
    return false;
  }

  return true;
}

void HexagonRuntimeManager::unloadRuntime() {
  std::lock_guard<std::mutex> lock(mutex_);
  unloadRuntimeLocked();
}

void HexagonRuntimeManager::unloadRuntimeLocked() {
  // Null all function pointers
  fnGetDeviceCount_ = nullptr;
  fnInitDevice_ = nullptr;
  fnReleaseDevice_ = nullptr;
  fnAllocateTcm_ = nullptr;
  fnFreeTcm_ = nullptr;
  fnGetTcmCapacity_ = nullptr;
  fnAllocateDdr_ = nullptr;
  fnFreeDdr_ = nullptr;
  fnCompileKernel_ = nullptr;
  fnLoadKernel_ = nullptr;
  fnReleaseKernel_ = nullptr;
  fnDispatchKernel_ = nullptr;
  fnWaitForCompletion_ = nullptr;
  fnDmaHostToTcm_ = nullptr;
  fnDmaTcmToHost_ = nullptr;
  fnDmaCopy_ = nullptr;

  if (libHandle_ != nullptr) {
    dlclose(libHandle_);
    libHandle_ = nullptr;
  }

  available_ = false;
}

// ---- Availability ----

bool HexagonRuntimeManager::isAvailable() const {
  return available_;
}

// ---- Device Management ----

int HexagonRuntimeManager::getDeviceCount() const {
  if (!available_ || fnGetDeviceCount_ == nullptr) return 0;
  return fnGetDeviceCount_();
}

void* HexagonRuntimeManager::initDevice(int deviceId) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!available_ || fnInitDevice_ == nullptr) {
    DSP_DIAG(BACKEND, "HexagonRuntimeManager::initDevice: runtime not available");
    return nullptr;
  }

  void* ctx = fnInitDevice_(deviceId);
  if (ctx == nullptr) {
    DSP_DIAG(BACKEND, "HexagonRuntimeManager::initDevice: failed for device %d", deviceId);
  } else {
    DSP_DIAG(BACKEND, "HexagonRuntimeManager::initDevice: device %d initialized, ctx=%p",
             deviceId, ctx);
  }
  return ctx;
}

void HexagonRuntimeManager::releaseDevice(void* npuContext) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (npuContext == nullptr) return;
  if (fnReleaseDevice_ != nullptr) {
    fnReleaseDevice_(npuContext);
    DSP_DIAG(BACKEND, "HexagonRuntimeManager::releaseDevice: ctx=%p released", npuContext);
  }
}

// ---- TCM Management ----

void* HexagonRuntimeManager::allocateTcm(void* npuContext, size_t bytes) {
  if (!available_ || fnAllocateTcm_ == nullptr || npuContext == nullptr) {
    DSP_DIAG(MEMORY, "HexagonRuntimeManager::allocateTcm: not available");
    return nullptr;
  }

  void* ptr = fnAllocateTcm_(npuContext, bytes);
  if (ptr == nullptr) {
    DSP_DIAG(MEMORY, "HexagonRuntimeManager::allocateTcm: failed for %zu bytes", bytes);
  } else {
    DSP_DIAG(MEMORY, "HexagonRuntimeManager::allocateTcm: allocated %zu bytes at %p",
             bytes, ptr);
  }
  return ptr;
}

void HexagonRuntimeManager::freeTcm(void* npuContext, void* tcmPtr) {
  if (npuContext == nullptr || tcmPtr == nullptr) return;
  if (fnFreeTcm_ != nullptr) {
    fnFreeTcm_(npuContext, tcmPtr);
  }
}

size_t HexagonRuntimeManager::getTcmCapacity(void* npuContext) const {
  if (!available_ || fnGetTcmCapacity_ == nullptr || npuContext == nullptr) return 0;
  return fnGetTcmCapacity_(npuContext);
}

// ---- DDR Management ----

void* HexagonRuntimeManager::allocateDdr(void* npuContext, size_t bytes) {
  if (!available_ || fnAllocateDdr_ == nullptr || npuContext == nullptr) {
    DSP_DIAG(MEMORY, "HexagonRuntimeManager::allocateDdr: not available");
    return nullptr;
  }

  void* ptr = fnAllocateDdr_(npuContext, bytes);
  if (ptr == nullptr) {
    DSP_DIAG(MEMORY, "HexagonRuntimeManager::allocateDdr: failed for %zu bytes", bytes);
  } else {
    DSP_DIAG(MEMORY, "HexagonRuntimeManager::allocateDdr: allocated %zu bytes at %p",
             bytes, ptr);
  }
  return ptr;
}

void HexagonRuntimeManager::freeDdr(void* npuContext, void* ddrPtr) {
  if (npuContext == nullptr || ddrPtr == nullptr) return;
  if (fnFreeDdr_ != nullptr) {
    fnFreeDdr_(npuContext, ddrPtr);
  }
}

// ---- Kernel Compilation ----

void* HexagonRuntimeManager::compileKernel(void* npuContext,
                                            const uint8_t* mlirBytecode,
                                            size_t bytecodeSize) {
  if (!available_ || fnCompileKernel_ == nullptr || npuContext == nullptr) {
    DSP_DIAG(COMPILE, "HexagonRuntimeManager::compileKernel: not available");
    return nullptr;
  }

  if (mlirBytecode == nullptr || bytecodeSize == 0) {
    DSP_DIAG(COMPILE, "HexagonRuntimeManager::compileKernel: empty bytecode");
    return nullptr;
  }

  void* kernel = fnCompileKernel_(npuContext, mlirBytecode, bytecodeSize);
  if (kernel == nullptr) {
    DSP_DIAG(COMPILE, "HexagonRuntimeManager::compileKernel: compilation failed "
             "(%zu bytes bytecode)", bytecodeSize);
  } else {
    DSP_DIAG(COMPILE, "HexagonRuntimeManager::compileKernel: compiled %zu bytes -> %p",
             bytecodeSize, kernel);
  }
  return kernel;
}

bool HexagonRuntimeManager::supportsPrecompiledKernels() const {
  return available_ && fnLoadKernel_ != nullptr;
}

void* HexagonRuntimeManager::loadKernel(void* npuContext,
                                        const uint8_t* kernelBinary,
                                        size_t binarySize) {
  if (!available_ || fnLoadKernel_ == nullptr || npuContext == nullptr) {
    DSP_DIAG(COMPILE,
             "HexagonRuntimeManager::loadKernel: AOT import unavailable");
    return nullptr;
  }
  if (kernelBinary == nullptr || binarySize == 0) {
    DSP_DIAG(COMPILE, "HexagonRuntimeManager::loadKernel: empty binary");
    return nullptr;
  }

  void* kernel = fnLoadKernel_(npuContext, kernelBinary, binarySize);
  if (kernel == nullptr) {
    DSP_DIAG(COMPILE,
             "HexagonRuntimeManager::loadKernel: import failed (%zu bytes)",
             binarySize);
  } else {
    DSP_DIAG(COMPILE,
             "HexagonRuntimeManager::loadKernel: imported %zu AOT bytes -> %p",
             binarySize, kernel);
  }
  return kernel;
}

void HexagonRuntimeManager::releaseKernel(void* npuContext, void* kernelHandle) {
  if (npuContext == nullptr || kernelHandle == nullptr) return;
  if (fnReleaseKernel_ != nullptr) {
    fnReleaseKernel_(npuContext, kernelHandle);
  }
}

// ---- Kernel Dispatch ----

bool HexagonRuntimeManager::dispatchKernel(void* npuContext, void* kernelHandle,
                                            void** args, int numArgs) {
  if (!available_ || fnDispatchKernel_ == nullptr) {
    DSP_DIAG(EXECUTE, "HexagonRuntimeManager::dispatchKernel: not available");
    return false;
  }

  if (npuContext == nullptr || kernelHandle == nullptr) {
    DSP_DIAG(EXECUTE, "HexagonRuntimeManager::dispatchKernel: null context or kernel");
    return false;
  }

  int result = fnDispatchKernel_(npuContext, kernelHandle, args, numArgs);
  if (result != 0) {
    DSP_DIAG(EXECUTE, "HexagonRuntimeManager::dispatchKernel: dispatch failed with %d",
             result);
    return false;
  }

  return true;
}

bool HexagonRuntimeManager::waitForCompletion(void* npuContext) {
  if (!available_ || fnWaitForCompletion_ == nullptr || npuContext == nullptr) {
    return false;
  }

  int result = fnWaitForCompletion_(npuContext);
  if (result != 0) {
    DSP_DIAG(EXECUTE, "HexagonRuntimeManager::waitForCompletion: failed with %d",
             result);
    return false;
  }

  return true;
}

// ---- DMA Operations ----

bool HexagonRuntimeManager::dmaHostToTcm(void* npuContext, void* dst,
                                          const void* src, size_t bytes) {
  if (!available_ || fnDmaHostToTcm_ == nullptr || npuContext == nullptr) {
    return false;
  }

  int result = fnDmaHostToTcm_(npuContext, dst, src, bytes);
  if (result != 0) {
    DSP_DIAG(MEMORY, "HexagonRuntimeManager::dmaHostToTcm: failed (%zu bytes)", bytes);
    return false;
  }
  return true;
}

bool HexagonRuntimeManager::dmaTcmToHost(void* npuContext, void* dst,
                                          const void* src, size_t bytes) {
  if (!available_ || fnDmaTcmToHost_ == nullptr || npuContext == nullptr) {
    return false;
  }

  int result = fnDmaTcmToHost_(npuContext, dst, src, bytes);
  if (result != 0) {
    DSP_DIAG(MEMORY, "HexagonRuntimeManager::dmaTcmToHost: failed (%zu bytes)", bytes);
    return false;
  }
  return true;
}

bool HexagonRuntimeManager::dmaCopy(void* npuContext, void* dst,
                                     const void* src, size_t bytes) {
  if (!available_ || fnDmaCopy_ == nullptr || npuContext == nullptr) {
    return false;
  }

  int result = fnDmaCopy_(npuContext, dst, src, bytes);
  if (result != 0) {
    DSP_DIAG(MEMORY, "HexagonRuntimeManager::dmaCopy: failed (%zu bytes)", bytes);
    return false;
  }
  return true;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_HEXAGON_MLIR
