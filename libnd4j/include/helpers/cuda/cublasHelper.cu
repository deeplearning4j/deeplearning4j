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

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>
#include <execution/AffinityManager.h>
#include <execution/ZludaRuntime.h>
#include <string>
#include <helpers/logger.h>
#include <mutex>
#include <system/Environment.h>

#include "../cublasHelper.h"
#include "config.h"
#include <array/DataBuffer.h>
#include <atomic>
#include <graph/DspDiagnostics.h>

#if HAVE_CUDNN
#include <cudnn.h>
#endif

namespace sd {
std::mutex CublasHelper::_mutex;

static void* handle_() {
  auto _handle = new cublasHandle_t();
  auto status = cublasCreate_v2(_handle);  // initialize CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::string msg = "cuBLAS handle creation failed !; Error code: [" + std::to_string(status) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  // Enable TF32 math mode on sm_80+ (Ampere and later) when configured.
  // TF32 uses tensor cores for FP32 GEMMs with 10-bit mantissa precision,
  // providing significant speedup for compute-bound operations.
  if (sd::Environment::getInstance().cublasTf32Enabled()) {
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    if (prop.major >= 8) {
      cublasSetMathMode(*_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
  }

  return reinterpret_cast<void*>(_handle);
}

#if !defined(HAVE_ZLUDA)
static void* solver_() {
  auto cusolverH = new cusolverDnHandle_t();
  auto status = cusolverDnCreate(cusolverH);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    delete cusolverH;
    std::string msg = "cuSolver handle creation failed !; Error code: [" + std::to_string(status) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  return cusolverH;
}
#endif

static void* cudnn_() {
#if HAVE_CUDNN
  auto cudnnH = new cudnnHandle_t();
  auto status = cudnnCreate(cudnnH);
  if (status != CUDNN_STATUS_SUCCESS) {
    std::string msg = "cuDNN handle creation failed !; Error code: [" + std::to_string(status) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  return cudnnH;
#else
  return nullptr;
#endif
}

static void* sparse_() {
  auto sparseH = new cusparseHandle_t();
  auto status = cusparseCreate(sparseH);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::string msg = "cuSPARSE handle creation failed !; Error code: [" + std::to_string(status) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  return sparseH;
}

static void destroyHandle_(void* handle) {
  auto ch = reinterpret_cast<cublasHandle_t*>(handle);
  auto status = cublasDestroy_v2(*ch);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::string msg = "cuBLAS handle destruction failed !; Error code: [" + std::to_string(status) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  delete ch;
}

CublasHelper::CublasHelper() {
  auto numDevices = AffinityManager::numberOfDevices();
  auto currentDevice = AffinityManager::currentDeviceId();
  _cache.resize(numDevices);
  _solvers.resize(numDevices, nullptr);
  _cudnn.resize(numDevices);
  _sparse.resize(numDevices, nullptr);
  for (int e = 0; e < numDevices; e++) {
    AffinityManager::setCurrentNativeDevice(e);

    _cache[e] = handle_();
    _cudnn[e] = cudnn_();
    // _solvers[e] is created only when a solver-backed operation requests it.
    // ZLUDA supports the CUDA execution path without implementing cuSolver,
    // so ordinary array allocation must not probe that optional library.
    // _sparse[e] is lazily created on first use via sparseHandle(int deviceId)
  }

  // don't forget to restore back original device
  AffinityManager::setCurrentNativeDevice(currentDevice);
}

CublasHelper::~CublasHelper() {
  // The legacy cuBLAS cache and thread-local handles retain their existing
  // process-lifetime ownership; only optional per-device handles are owned here.

#if !defined(HAVE_ZLUDA)
  // Destroy only handles that were requested and successfully created.
  for (int e = 0; e < static_cast<int>(_solvers.size()); e++) {
    if (_solvers[e] != nullptr) {
      auto* solverHandle = reinterpret_cast<cusolverDnHandle_t*>(_solvers[e]);
      cusolverDnDestroy(*solverHandle);
      delete solverHandle;
      _solvers[e] = nullptr;
    }
  }
#endif

  // Destroy any lazily-created cuSPARSE handles
  for (int e = 0; e < static_cast<int>(_sparse.size()); e++) {
    if (_sparse[e] != nullptr) {
      auto* sp = reinterpret_cast<cusparseHandle_t*>(_sparse[e]);
      cusparseDestroy(*sp);
      delete sp;
      _sparse[e] = nullptr;
    }
  }
}

CublasHelper& CublasHelper::getInstance() {
  static CublasHelper* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new CublasHelper();
  });
  return *instance;
}

void CublasHelper::applyTf32Mode(bool enable) {
  // Thread-local handles get TF32 applied lazily in handle() via tl_tf32Applied.
  // This method is kept for any code that still references the legacy _cache handles.
  for (int e = 0; e < _cache.size(); e++) {
    auto handle = _cache[e];
    if (handle == nullptr) continue;
    auto ch = reinterpret_cast<cublasHandle_t*>(handle);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, e);
    if (prop.major >= 8) {
      cublasSetMathMode(*ch, enable ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
    }
  }
}

void* CublasHelper::cudnn() {
  auto deviceId = AffinityManager::currentDeviceId();
  if (deviceId < 0 || deviceId >= _cudnn.size()) {
    std::string msg = "requested deviceId doesn't look valid for cuDNN; Error code: [" + std::to_string(deviceId) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  auto handle = _cudnn[deviceId];
  if (handle == nullptr) {
    sd_printf("WARNING: cuDNN handle is null for device %d\n", deviceId);
  }
  return handle;
}

void* CublasHelper::sparseHandle() {
  auto deviceId = AffinityManager::currentDeviceId();
  return sparseHandle(deviceId);
}

void* CublasHelper::sparseHandle(int deviceId) {
  if (deviceId < 0 || deviceId >= static_cast<int>(_sparse.size())) {
    std::string msg = "requested deviceId doesn't look valid for cuSPARSE; Error code: [" + std::to_string(deviceId) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  // Lazy creation under mutex: double-checked locking to avoid unnecessary lock
  // on the hot path. cuSPARSE handles are NOT thread-safe so callers must
  // serialize their own handle usage; here we only guard the one-time creation.
  if (_sparse[deviceId] == nullptr) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_sparse[deviceId] == nullptr) {
      auto savedDevice = AffinityManager::currentDeviceId();
      AffinityManager::setCurrentNativeDevice(deviceId);
      _sparse[deviceId] = sparse_();
      AffinityManager::setCurrentNativeDevice(savedDevice);
    }
  }

  return _sparse[deviceId];
}

void* CublasHelper::handle() {
  auto deviceId = AffinityManager::currentDeviceId();
  return handle(deviceId);
}

void* CublasHelper::solver() {
#if defined(HAVE_ZLUDA)
  // ZLUDA does not expose the cuSolver ABI. Keep the public capability boundary
  // but compile every cuSolver symbol out of the AMD-only classifier.
  return nullptr;
#else
  // cuSolver is an optional backend capability. ZLUDA implements the CUDA ABI
  // used by ordinary ND4J operations but not cuSolver, so do not probe it while
  // constructing a launch context.
  if (!zluda::supportsCusolver()) {
    return nullptr;
  }

  auto deviceId = AffinityManager::currentDeviceId();
  if (deviceId < 0 || deviceId >= _solvers.size()) {
    std::string msg = "requested deviceId doesn't look valid for cuSolver; Error code: [" + std::to_string(deviceId) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  // cuSolver is optional for most CUDA operations and is not implemented by
  // every CUDA ABI provider (notably ZLUDA). Create its handle only when an
  // operation such as SVD or LUP explicitly asks for it. Always take the lock:
  // reading and writing a plain vector slot outside the lock would be a data
  // race during first use from concurrent threads.
  std::lock_guard<std::mutex> lock(_mutex);
  if (_solvers[deviceId] == nullptr) {
    // deviceId is the current device by construction, so solver_() binds the
    // handle to the same context without any process-global device switching.
    _solvers[deviceId] = solver_();
  }
  return _solvers[deviceId];
#endif
}

void* CublasHelper::handle(int deviceId) {
  if (deviceId < 0 || deviceId >= _cache.size()) {
    std::string msg = "requested deviceId doesn't look valid for cuBLAS; Error code: [" + std::to_string(deviceId) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  // Thread-local cuBLAS handle per (thread, device) pair.
  // cuBLAS handles carry mutable state (stream, math mode, workspace) and are
  // explicitly NOT thread-safe (NVIDIA docs). Sharing a single handle across
  // concurrent sd.output() threads causes races: thread A sets stream/workspace,
  // thread B overwrites them, thread A launches GEMM on the wrong stream →
  // CUDA error 906 (cudaErrorLaunchFailure).
  thread_local cublasHandle_t* tl_handle = nullptr;
  thread_local int tl_deviceId = -1;
  static thread_local int tl_smMajor = -1;

  if (tl_deviceId != deviceId || tl_handle == nullptr) {
    // Device changed or first call on this thread — create a new handle.
    if (tl_handle != nullptr) {
      cublasDestroy_v2(*tl_handle);
      delete tl_handle;
      tl_handle = nullptr;
    }

    tl_handle = new cublasHandle_t();
    auto status = cublasCreate_v2(tl_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      delete tl_handle;
      tl_handle = nullptr;
      std::string msg = "thread-local cuBLAS handle creation failed; Error code: [" + std::to_string(status) + "]";
      THROW_EXCEPTION(msg.c_str());
    }
    tl_deviceId = deviceId;
    tl_smMajor = -1;  // force re-query for the new device's SM major
  }

  // Lazily converge THIS thread's handle to the mode the process currently
  // requires. Three states, priority order:
  //   PEDANTIC — a deterministic window is open (DSP plan execution whose
  //              ModeContract requires bit-reproducible GEMMs). Handles are
  //              thread-local, so the window must be applied at ACQUISITION:
  //              the plan thread's platformBeginExecution cannot reach the
  //              handles of executor/pool threads that dispatch gap GEMMs —
  //              they inherited fresh DEFAULT/TF32 handles and drifted vs the
  //              PEDANTIC reference (batch-only, bit-identical ~1e-2 on
  //              norm-heavy graphs).
  //   caller-managed — tl_cublasLtDisabled=true: the calling thread set the
  //              mode explicitly (setCublasWorkspaceForCapture/Warmup); do
  //              not overwrite it here.
  //   TF32/DEFAULT — normal lazy TF32 policy (sm_80+, env-gated).
  static thread_local int tl_appliedMode = -1;  // CublasMathModeState
  constexpr int MODE_DEFAULT = 0, MODE_TF32 = 1, MODE_PEDANTIC = 2;
  bool wantTf32 = sd::Environment::getInstance().cublasTf32Enabled();
  if (tl_smMajor < 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    tl_smMajor = prop.major;
  }
  if (CublasHelper::inDeterministicWindow()) {
    if (tl_appliedMode != MODE_PEDANTIC) {
      cublasSetMathMode(*tl_handle, CUBLAS_PEDANTIC_MATH);
      DSP_DIAG(EXECUTE, "CUBLAS_HANDLE_MODE: thread handle %p -> PEDANTIC "
               "(deterministic window open, was mode=%d)",
               (void*)tl_handle, tl_appliedMode);
      tl_appliedMode = MODE_PEDANTIC;
    }
  } else if (!tl_cublasLtDisabled) {
    int want = (wantTf32 && tl_smMajor >= 8) ? MODE_TF32 : MODE_DEFAULT;
    if (want != tl_appliedMode) {
      cublasSetMathMode(*tl_handle, want == MODE_TF32 ? CUBLAS_TF32_TENSOR_OP_MATH
                                                      : CUBLAS_DEFAULT_MATH);
      DSP_DIAG(EXECUTE, "CUBLAS_HANDLE_MODE: thread handle %p -> %s "
               "(window closed, was mode=%d)",
               (void*)tl_handle, want == MODE_TF32 ? "TF32" : "DEFAULT", tl_appliedMode);
      tl_appliedMode = want;
    }
  } else {
    // Caller manages the mode explicitly; forget our tracking so the next
    // unmanaged acquisition re-applies the policy mode.
    tl_appliedMode = -1;
  }

  return reinterpret_cast<void*>(tl_handle);
}

// ── Process-global deterministic-math window ────────────────────────────
// Depth-counted so nested/overlapping plan executions (multi-threaded plan
// sharing) keep the window open until the LAST participant exits.
static std::atomic<int> g_deterministicCublasDepth{0};

void CublasHelper::enterDeterministicWindow() {
  int prev = g_deterministicCublasDepth.fetch_add(1, std::memory_order_relaxed);
  DSP_DIAG(EXECUTE, "CUBLAS_DETERMINISTIC_WINDOW: enter depth=%d -> %d "
           "(handles acquired on ANY thread converge to PEDANTIC)",
           prev, prev + 1);
}
void CublasHelper::exitDeterministicWindow() {
  int prev = g_deterministicCublasDepth.fetch_sub(1, std::memory_order_relaxed);
  DSP_DIAG(EXECUTE, "CUBLAS_DETERMINISTIC_WINDOW: exit depth=%d -> %d%s",
           prev, prev > 0 ? prev - 1 : 0,
           prev <= 0 ? " (UNBALANCED — clamped to 0)" : "");
  if (prev <= 0) {
    // Unbalanced exit — clamp to zero rather than going negative (a negative
    // depth would silently disable PEDANTIC for all future windows).
    g_deterministicCublasDepth.store(0, std::memory_order_relaxed);
  }
}
bool CublasHelper::inDeterministicWindow() {
  // The EXISTING explicit fast-math opt-in wins over the deterministic window:
  // cublasTf32Enabled is a deliberate, documented user choice for TF32 GEMMs
  // (speed over bitwise reproducibility — production decode tolerates the
  // logit drift). Accuracy/parity tests never set it, so they keep PEDANTIC.
  if (sd::Environment::getInstance().cublasTf32Enabled()) return false;
  return g_deterministicCublasDepth.load(std::memory_order_relaxed) > 0;
}

// cuBLAS Lt handle: created on-demand per thread/device
// Returns nullptr if Lt is not available (fallback to standard cuBLAS)
void* CublasHelper::ltHandle() {
  // Thread-local Lt handle cache - plain thread_local, not static thread_local
  // to avoid initialization order issues in shared libraries
  thread_local cublasLtHandle_t tl_ltHandle = nullptr;
  thread_local int tl_deviceId = -1;
  thread_local bool tl_available = true;
  
  // Check if already initialized for current device
  int currentDevice = AffinityManager::currentDeviceId();
  if (tl_deviceId == currentDevice && tl_ltHandle != nullptr) {
    return tl_available ? reinterpret_cast<void*>(&tl_ltHandle) : nullptr;
  }
  
  // Device changed or not initialized - create/replace Lt handle
  if (tl_ltHandle != nullptr) {
    cublasLtDestroy(tl_ltHandle);
    tl_ltHandle = nullptr;
  }
  
  if (!tl_available) {
    return nullptr;  // Lt not available on this system
  }
  
  tl_deviceId = currentDevice;
  
  cublasLtHandle_t ltHandle;
  auto status = cublasLtCreate(&ltHandle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    tl_available = false;
    return nullptr;
  }
  
  tl_ltHandle = ltHandle;
  return reinterpret_cast<void*>(&tl_ltHandle);
}

void* CublasHelper::ltHandle(int deviceId) {
  // For explicit device request, save and restore current device
  int savedDevice = AffinityManager::currentDeviceId();
  AffinityManager::setCurrentNativeDevice(deviceId);
  void* result = ltHandle();
  AffinityManager::setCurrentNativeDevice(savedDevice);
  return result;
}
}  // namespace sd
