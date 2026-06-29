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
#include <string>
#include <helpers/logger.h>
#include <mutex>
#include <system/Environment.h>

#include "../cublasHelper.h"
#include "config.h"
#include <array/DataBuffer.h>

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

static void* solver_() {
  auto cusolverH = new cusolverDnHandle_t();
  auto status = cusolverDnCreate(cusolverH);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    std::string msg = "cuSolver handle creation failed !; Error code: [" + std::to_string(status) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  return cusolverH;
}

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
  _solvers.resize(numDevices);
  _cudnn.resize(numDevices);
  _sparse.resize(numDevices, nullptr);
  for (int e = 0; e < numDevices; e++) {
    AffinityManager::setCurrentNativeDevice(e);

    _cache[e] = handle_();
    _solvers[e] = solver_();
    _cudnn[e] = cudnn_();
    // _sparse[e] is lazily created on first use via sparseHandle(int deviceId)
  }

  // don't forget to restore back original device
  AffinityManager::setCurrentNativeDevice(currentDevice);
}

CublasHelper::~CublasHelper() {
  auto numDevices = AffinityManager::numberOfDevices();

  // for (int e = 0; e < numDevices; e++) destroyHandle_(_cache[e]);

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
  auto deviceId = AffinityManager::currentDeviceId();
  if (deviceId < 0 || deviceId >= _solvers.size()) {
    std::string msg = "requested deviceId doesn't look valid for cuSolver; Error code: [" + std::to_string(deviceId) + "]";
    THROW_EXCEPTION(msg.c_str());
  }

  auto handle = _solvers[deviceId];
  if (handle == nullptr) {
    std::string msg = "cuSolver handle is null for device - initialization may have failed; Error code: [" + std::to_string(deviceId) + "]";
    THROW_EXCEPTION(msg.c_str());
  }
  return handle;
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
  }

  // Lazily apply/remove TF32 mode when the flag changes.
  // CRITICAL: When DSP has set deterministic cuBLAS (PEDANTIC_MATH via
  // tl_cublasLtDisabled=true), skip the lazy TF32 application entirely.
  static thread_local bool tl_tf32Applied = false;
  static thread_local int tl_smMajor = -1;
  bool wantTf32 = sd::Environment::getInstance().cublasTf32Enabled();
  if (!tl_cublasLtDisabled && wantTf32 != tl_tf32Applied) {
    if (tl_smMajor < 0) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, deviceId);
      tl_smMajor = prop.major;
    }
    if (tl_smMajor >= 8) {
      cublasSetMathMode(*tl_handle, wantTf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
    }
    tl_tf32Applied = wantTf32;
  }

  return reinterpret_cast<void*>(tl_handle);
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
