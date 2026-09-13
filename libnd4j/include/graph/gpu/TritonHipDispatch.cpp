/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>
#include "TritonHipDispatch.h"

#if HAVE_TRITON && defined(TRITON_HAS_HIP)

// AMD-native host API TU. Do not include CUDA, system/common.h, NDArray,
// DspDiagnostics or the CUDA-facing TritonTargetDispatch.h here: even HIP's
// runtime_api header declares vector types that conflict with CUDA's types.
#if defined(__HIP_PLATFORM_NVIDIA__)
#error "TritonHipDispatch.cpp must use the AMD HIP ABI"
#endif
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif

#include <hip/hip_runtime_api.h>

namespace sd {
namespace graph {
namespace triton_hip {

bool detectDevice(std::string& archName, std::string& deviceName) {
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount <= 0) return false;

  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) return false;
  archName = props.gcnArchName;
  deviceName = props.name;
  return true;
}

void* loadModule(const void* data, const char*& error) {
  error = nullptr;
  hipModule_t module = nullptr;
  hipError_t status = hipModuleLoadData(&module, data);
  if (status != hipSuccess) {
    error = hipGetErrorString(status);
    return nullptr;
  }
  return static_cast<void*>(module);
}

void* getKernelFunction(void* module, const char* name, const char*& error) {
  error = nullptr;
  hipFunction_t function = nullptr;
  hipError_t status = hipModuleGetFunction(&function, static_cast<hipModule_t>(module), name);
  if (status != hipSuccess) {
    error = hipGetErrorString(status);
    return nullptr;
  }
  return static_cast<void*>(function);
}

bool launchKernel(void* function,
                  unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                  unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                  unsigned int sharedMemBytes, void* stream, void** args,
                  const char*& error) {
  error = nullptr;
  hipError_t status = hipModuleLaunchKernel(
      static_cast<hipFunction_t>(function), gridX, gridY, gridZ,
      blockX, blockY, blockZ, sharedMemBytes, static_cast<hipStream_t>(stream),
      args, nullptr);
  if (status != hipSuccess) {
    error = hipGetErrorString(status);
    return false;
  }
  return true;
}

void unloadModule(void* module) {
  hipModuleUnload(static_cast<hipModule_t>(module));
}

}  // namespace triton_hip
}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON && TRITON_HAS_HIP
