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

#ifndef LIBND4J_GPU_KERNEL_LAUNCHER_H
#define LIBND4J_GPU_KERNEL_LAUNCHER_H

#include <system/common.h>

#include <cstddef>

namespace sd {
namespace graph {

/**
 * Shared CUDA Driver API utilities for loading PTX modules, getting kernel
 * functions, launching kernels, and unloading modules.
 *
 * Used by NvrtcGraphBackend, PtxGraphBackend, and TritonGraphBackend (NVIDIA path).
 * All functions are CUDA-only (require SD_CUDA).
 */
namespace GpuKernelLauncher {

  /**
   * Load PTX text into a CUmodule via cuModuleLoadDataEx.
   * @param ptxText  Null-terminated PTX source string
   * @param ptxSize  Length of ptxText (excluding null terminator)
   * @return Opaque CUmodule handle, or nullptr on failure
   */
  void* loadPtxModule(const char* ptxText, size_t ptxSize);

  /**
   * Get a kernel function handle from a loaded module.
   * @param module      CUmodule handle from loadPtxModule()
   * @param kernelName  Name of the kernel entry point
   * @return Opaque CUfunction handle, or nullptr on failure
   */
  void* getKernelFunc(void* module, const char* kernelName);

  /**
   * Launch a kernel on the GPU via cuLaunchKernel.
   * @param func         CUfunction handle
   * @param gridX/Y/Z    Grid dimensions
   * @param blockX/Y/Z   Block dimensions
   * @param sharedMem    Shared memory bytes per block
   * @param stream       cudaStream_t (or nullptr for default stream)
   * @param args         Array of kernel argument pointers
   * @param numArgs      Number of arguments
   * @return true on success
   */
  bool launchKernel(void* func,
                    unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                    unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                    unsigned int sharedMem, void* stream,
                    void** args, int numArgs);

  /**
   * Unload a previously loaded CUmodule.
   * @param module  CUmodule handle from loadPtxModule()
   */
  void unloadModule(void* module);

}  // namespace GpuKernelLauncher

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_GPU_KERNEL_LAUNCHER_H
