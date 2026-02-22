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

#ifdef SD_CUDA

#include <graph/gpu/GpuKernelLauncher.h>
#include <system/common.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace sd {
namespace graph {
namespace GpuKernelLauncher {

void* loadPtxModule(const char* ptxText, size_t ptxSize) {
  if (!ptxText || ptxSize == 0) return nullptr;

  CUmodule module = nullptr;
  CUresult res = cuModuleLoadDataEx(&module, ptxText, 0, nullptr, nullptr);
  if (res != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(res, &errStr);
    fprintf(stderr,"GpuKernelLauncher::loadPtxModule: cuModuleLoadDataEx failed: %s\n",
              errStr ? errStr : "unknown");
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
    fprintf(stderr,"GpuKernelLauncher::getKernelFunc: cuModuleGetFunction('%s') failed: %s\n",
              kernelName, errStr ? errStr : "unknown");
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
    fprintf(stderr,"GpuKernelLauncher::launchKernel: cuLaunchKernel failed: %s\n",
              errStr ? errStr : "unknown");
    return false;
  }
  return true;
}

void unloadModule(void* module) {
  if (!module) return;
  cuModuleUnload(static_cast<CUmodule>(module));
}

}  // namespace GpuKernelLauncher
}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
