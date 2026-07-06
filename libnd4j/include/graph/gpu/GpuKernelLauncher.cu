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


#include <graph/DspDiagnostics.h>
#include <graph/gpu/GpuKernelLauncher.h>
#include <system/common.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

namespace sd {
namespace graph {
namespace GpuKernelLauncher {

void* loadPtxModule(const char* ptxText, size_t ptxSize) {
  if (!ptxText || ptxSize == 0) return nullptr;

  // Ensure the primary context for the current device is bound as the active
  // driver context.  After cudaSetDevice() the CUDA runtime holds a retain on
  // the primary context, but the driver API thread-local context stack may
  // still be empty (e.g., on async compile threads that have never entered the
  // CUDA runtime).  If no context is current, retain + push the primary context
  // so cuModuleLoadDataEx operates on the correct device.
  int currentDev = 0;
  cudaGetDevice(&currentDev);

  CUdevice cuDev = 0;
  bool didPush = false;
  if (cuDeviceGet(&cuDev, currentDev) == CUDA_SUCCESS) {
    CUcontext existingCtx = nullptr;
    cuCtxGetCurrent(&existingCtx);
    if (!existingCtx) {
      CUcontext primaryCtx = nullptr;
      if (cuDevicePrimaryCtxRetain(&primaryCtx, cuDev) == CUDA_SUCCESS && primaryCtx) {
        if (cuCtxPushCurrent(primaryCtx) == CUDA_SUCCESS) {
          didPush = true;
        } else {
          cuDevicePrimaryCtxRelease(cuDev);
        }
      }
    }
  }

  CUmodule module = nullptr;
  char jitErr[8192];
  char jitInfo[8192];
  std::memset(jitErr, 0, sizeof(jitErr));
  std::memset(jitInfo, 0, sizeof(jitInfo));

  CUjit_option options[] = {
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
  };
  void* optionValues[] = {
      jitErr,
      reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(jitErr))),
      jitInfo,
      reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(jitInfo)))
  };

  CUresult res = cuModuleLoadDataEx(
      &module, ptxText, static_cast<unsigned int>(sizeof(options) / sizeof(options[0])),
      options, optionValues);

  // Pop the pushed context regardless of load outcome.
  if (didPush) {
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
    cuDevicePrimaryCtxRelease(cuDev);
  }

  if (res != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(res, &errStr);
    if (sd::Environment::getInstance().isDebug()) {
      fprintf(stderr,
              "GpuKernelLauncher::loadPtxModule: cuModuleLoadDataEx failed: %s\n"
              "  PTX size: %zu bytes\n"
              "  JIT error log: %s\n"
              "  JIT info log: %s\n",
              errStr ? errStr : "unknown",
              ptxSize,
              jitErr[0] ? jitErr : "<empty>",
              jitInfo[0] ? jitInfo : "<empty>");

      size_t previewLen = ptxSize < 512 ? ptxSize : 512;
      fprintf(stderr, "  PTX preview (%zu bytes):\n%.*s\n",
              previewLen, static_cast<int>(previewLen), ptxText);
    }
    return nullptr;
  }
  return static_cast<void*>(module);
}

void* getKernelFunc(void* module, const char* kernelName) {
  if (!module || !kernelName) return nullptr;

  CUfunction func = nullptr;
  CUresult res = cuModuleGetFunction(&func, static_cast<CUmodule>(module), kernelName);
  if (res != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(res, &errStr);
    if (sd::Environment::getInstance().isDebug()) {
      fprintf(stderr,"GpuKernelLauncher::getKernelFunc: cuModuleGetFunction('%s') failed: %s\n",
                kernelName, errStr ? errStr : "unknown");
    }
    return nullptr;
  }
  return static_cast<void*>(func);
}

bool launchKernel(void* func,
                  unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                  unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                  unsigned int sharedMem, void* stream,
                  void** args, int numArgs) {
  if (!func) return false;

  CUresult res = cuLaunchKernel(
      static_cast<CUfunction>(func),
      gridX, gridY, gridZ,
      blockX, blockY, blockZ,
      sharedMem,
      static_cast<CUstream>(stream),
      args, nullptr);

  if (res != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(res, &errStr);
    if (sd::Environment::getInstance().isDebug()) {
      fprintf(stderr,"GpuKernelLauncher::launchKernel: cuLaunchKernel failed: %s\n",
                errStr ? errStr : "unknown");
    }
    return false;
  }
  return true;
}

void unloadModule(void* module) {
  if (!module) return;
  DSP_DIAG(EXECUTE, "GpuKernelLauncher: cuModuleUnload module=%p", module);
  cuModuleUnload(static_cast<CUmodule>(module));
}

}  // namespace GpuKernelLauncher
}  // namespace graph
}  // namespace sd

