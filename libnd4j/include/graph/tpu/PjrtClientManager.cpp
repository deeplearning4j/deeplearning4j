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

#ifdef SD_TPU

#include <graph/tpu/PjrtClientManager.h>
#include <graph/DspDiagnostics.h>

#include <cstring>
#include <dlfcn.h>
#include <mutex>

namespace sd {
namespace graph {

// ============================================================
// PJRT/TPU Backend — STUB ONLY, not yet implemented.
// All methods return false/nullptr/error. Do not call in production.
// See PjrtClientManager.h and ADR 0072 for the implementation roadmap.
// ============================================================

// ── PJRT C API function pointer types ───────────────────────────────────────
// These mirror the PJRT C API signatures but are resolved at runtime via dlsym.
// We use void* for all opaque PJRT handles to avoid header dependency.

// GetPjrtApi returns a PJRT_Api* function table
using GetPjrtApiFn = void* (*)();

// ── Singleton ───────────────────────────────────────────────────────────────

PjrtClientManager& PjrtClientManager::getInstance() {
  static PjrtClientManager* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new PjrtClientManager();
  });
  return *instance;
}

// ── Constructor / Destructor ────────────────────────────────────────────────

PjrtClientManager::PjrtClientManager() {
  // Lazy init — don't load library until first use
}

PjrtClientManager::~PjrtClientManager() {
  // Destroy client before closing library
  if (client_ != nullptr) {
    // In a full implementation, we would call PJRT_Client_Destroy here
    // via the function table. For now, just null it out.
    DSP_DIAG(BACKEND, "PjrtClientManager: destroying PJRT client %p", client_);
    client_ = nullptr;
  }

  devices_.clear();

  if (libHandle_ != nullptr) {
    DSP_DIAG(BACKEND, "PjrtClientManager: closing libtpu.so");
    dlclose(libHandle_);
    libHandle_ = nullptr;
  }

  pjrtApi_ = nullptr;
  initialized_ = false;
}

// ── Library Loading ─────────────────────────────────────────────────────────

bool PjrtClientManager::loadLibrary() {
  // Try standard library name first, then fallback paths
  static const char* libPaths[] = {
    "libtpu.so",
    "/usr/lib/libtpu.so",
    "/usr/local/lib/libtpu.so",
    nullptr
  };

  for (int i = 0; libPaths[i] != nullptr; ++i) {
    libHandle_ = dlopen(libPaths[i], RTLD_NOW | RTLD_LOCAL);
    if (libHandle_ != nullptr) {
      DSP_DIAG(BACKEND, "PjrtClientManager: loaded %s", libPaths[i]);
      break;
    }
  }

  if (libHandle_ == nullptr) {
    lastError_ = std::string("Failed to load libtpu.so: ") + dlerror();
    DSP_DIAG(BACKEND, "PjrtClientManager: %s", lastError_.c_str());
    return false;
  }

  // Resolve GetPjrtApi symbol
  auto getPjrtApi = reinterpret_cast<GetPjrtApiFn>(dlsym(libHandle_, "GetPjrtApi"));
  if (getPjrtApi == nullptr) {
    lastError_ = std::string("Failed to resolve GetPjrtApi: ") + dlerror();
    DSP_DIAG(BACKEND, "PjrtClientManager: %s", lastError_.c_str());
    dlclose(libHandle_);
    libHandle_ = nullptr;
    return false;
  }

  // Get the PJRT API function table
  pjrtApi_ = getPjrtApi();
  if (pjrtApi_ == nullptr) {
    lastError_ = "GetPjrtApi() returned nullptr";
    DSP_DIAG(BACKEND, "PjrtClientManager: %s", lastError_.c_str());
    dlclose(libHandle_);
    libHandle_ = nullptr;
    return false;
  }

  DSP_DIAG(BACKEND, "PjrtClientManager: PJRT API function table resolved at %p",
           pjrtApi_);
  return true;
}

bool PjrtClientManager::initClient() {
  if (pjrtApi_ == nullptr) return false;

  // In a full implementation, this would:
  // 1. Cast pjrtApi_ to PJRT_Api*
  // 2. Call PJRT_Client_Create to create a client
  // 3. Call PJRT_Client_Devices to enumerate devices
  // 4. Store device handles in devices_
  //
  // For now, we set up the skeleton and log diagnostics.
  // The actual PJRT calls require the PJRT C API struct layout,
  // which would be resolved from the function table at runtime.

  DSP_DIAG(BACKEND, "PjrtClientManager: initializing PJRT client from API table %p",
           pjrtApi_);

  // TODO: Implement actual PJRT_Client_Create call via function table.
  // The PJRT C API uses a struct-of-function-pointers pattern:
  //   api->PJRT_Client_Create(&create_args);
  //   api->PJRT_Client_Devices(&devices_args);
  //
  // This requires knowing the exact struct layout, which we get from
  // the PJRT C API headers (pjrt_c_api.h). When building with actual
  // TPU support, those headers would be included and the void* casts
  // would be replaced with proper types.

  lastError_ = "PJRT client creation not yet implemented - requires pjrt_c_api.h struct layout";
  DSP_DIAG(BACKEND, "PjrtClientManager: %s", lastError_.c_str());

  // Return false until actual PJRT integration is wired up
  return false;
}

// ── Public Interface ────────────────────────────────────────────────────────

bool PjrtClientManager::isAvailable() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return initialized_ && client_ != nullptr;
}

int PjrtClientManager::getDeviceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || client_ == nullptr) return 0;
  return static_cast<int>(devices_.size());
}

std::vector<void*> PjrtClientManager::getDevices() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || client_ == nullptr) return {};
  return devices_;
}

void* PjrtClientManager::createBuffer(const void* hostData, size_t sizeBytes,
                                       int deviceIdx) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_ || client_ == nullptr) {
    // Attempt lazy init
    if (!initFailed_ && !initialized_) {
      if (loadLibrary() && initClient()) {
        initialized_ = true;
      } else {
        initFailed_ = true;
        return nullptr;
      }
    } else {
      return nullptr;
    }
  }

  if (deviceIdx < 0 || deviceIdx >= static_cast<int>(devices_.size())) {
    lastError_ = "Device index out of range";
    DSP_DIAG(MEMORY, "PjrtClientManager::createBuffer: %s (idx=%d, count=%d)",
             lastError_.c_str(), deviceIdx, static_cast<int>(devices_.size()));
    return nullptr;
  }

  // TODO: Implement via PJRT_Client_BufferFromHostBuffer
  // api->PJRT_Client_BufferFromHostBuffer(&args);
  DSP_DIAG(MEMORY, "PjrtClientManager::createBuffer: %zu bytes on device %d "
           "(not yet implemented)", sizeBytes, deviceIdx);
  return nullptr;
}

void PjrtClientManager::destroyBuffer(void* buffer) {
  if (buffer == nullptr) return;

  std::lock_guard<std::mutex> lock(mutex_);

  // TODO: Implement via PJRT_Buffer_Destroy
  // api->PJRT_Buffer_Destroy(&args);
  DSP_DIAG(MEMORY, "PjrtClientManager::destroyBuffer: %p (not yet implemented)",
           buffer);
}

bool PjrtClientManager::bufferToHost(void* buffer, void* hostDst,
                                      size_t sizeBytes) {
  if (buffer == nullptr || hostDst == nullptr) return false;

  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_ || client_ == nullptr) return false;

  // TODO: Implement via PJRT_Buffer_ToHostBuffer
  // api->PJRT_Buffer_ToHostBuffer(&args);
  DSP_DIAG(MEMORY, "PjrtClientManager::bufferToHost: %zu bytes from %p "
           "(not yet implemented)", sizeBytes, buffer);
  return false;
}

void* PjrtClientManager::compile(const void* hloBytes, size_t hloSize) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_ || client_ == nullptr) {
    // Attempt lazy init
    if (!initFailed_ && !initialized_) {
      if (loadLibrary() && initClient()) {
        initialized_ = true;
      } else {
        initFailed_ = true;
        return nullptr;
      }
    } else {
      return nullptr;
    }
  }

  if (hloBytes == nullptr || hloSize == 0) {
    lastError_ = "Empty HLO module";
    return nullptr;
  }

  DSP_DIAG(COMPILE, "PjrtClientManager::compile: %zu bytes HLO module",
           hloSize);

  // TODO: Implement via PJRT_Client_Compile
  // 1. Create PJRT_Program from hloBytes
  // 2. Call api->PJRT_Client_Compile(&compile_args)
  // 3. Return the PJRT_LoadedExecutable*

  lastError_ = "HLO compilation not yet implemented";
  DSP_DIAG(COMPILE, "PjrtClientManager::compile: %s", lastError_.c_str());
  return nullptr;
}

bool PjrtClientManager::execute(void* executable, void** inputBuffers,
                                 int numInputs, void** outputBuffers,
                                 int* numOutputs) {
  if (executable == nullptr) return false;

  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_ || client_ == nullptr) return false;

  DSP_DIAG(EXECUTE, "PjrtClientManager::execute: executable=%p, %d inputs",
           executable, numInputs);

  // TODO: Implement via PJRT_LoadedExecutable_Execute
  // api->PJRT_LoadedExecutable_Execute(&execute_args);

  lastError_ = "Executable execution not yet implemented";
  DSP_DIAG(EXECUTE, "PjrtClientManager::execute: %s", lastError_.c_str());
  return false;
}

void PjrtClientManager::destroyExecutable(void* executable) {
  if (executable == nullptr) return;

  std::lock_guard<std::mutex> lock(mutex_);

  // TODO: Implement via PJRT_LoadedExecutable_Destroy
  // api->PJRT_LoadedExecutable_Destroy(&args);
  DSP_DIAG(COMPILE, "PjrtClientManager::destroyExecutable: %p "
           "(not yet implemented)", executable);
}

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
