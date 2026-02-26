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

#ifndef LIBND4J_TRITON_TARGET_DISPATCH_H
#define LIBND4J_TRITON_TARGET_DISPATCH_H

#include <config.h>

#if HAVE_TRITON

#include <system/common.h>

#include <cstddef>
#include <string>

namespace sd {
namespace graph {

/**
 * GPU target type detected at runtime.
 */
enum class TritonGpuTarget {
  NVIDIA,   // CUDA / PTX
  AMD,      // ROCm / AMDGCN
  INTEL,    // Level Zero / SPIR-V
  UNKNOWN
};

/**
 * Compiled GPU binary produced by the Triton compiler pipeline.
 */
struct TritonCompiledBinary {
  void* data;           // Raw binary (PTX text, AMDGCN ELF, SPIR-V)
  size_t size;
  TritonGpuTarget target;
  std::string targetArch;   // e.g. "sm_89", "gfx1100"
  int numWarps;
  int sharedMemBytes;
};

/**
 * Multi-target GPU dispatch layer for the Triton compiler backend.
 *
 * Responsibilities:
 * 1. Detect which GPU target is available (NVIDIA/AMD/Intel)
 * 2. Compile MLIR modules through the Triton pipeline (TTIR -> TTGIR -> target ISA)
 * 3. Load compiled binaries into GPU driver modules
 * 4. Launch kernels via the appropriate driver API
 * 5. Unload modules on cache invalidation
 *
 * Uses CUDA Driver API (cuModule*) for NVIDIA, HIP (hipModule*) for AMD,
 * and Level Zero (zeModule*) for Intel.
 */
class TritonTargetDispatch {
 public:
  /**
   * Check if any supported GPU target is available and the Triton
   * compiler libraries are loaded.
   */
  static bool isReady();

  /**
   * Detect the current GPU target by querying the driver.
   * Caches the result after first call.
   */
  static TritonGpuTarget detectTarget();

  /**
   * Get the compute architecture string for the current device.
   * Examples: "sm_89" (NVIDIA), "gfx1100" (AMD), "pvc" (Intel)
   */
  static std::string getTargetArch();

  /**
   * Compile an MLIR module (Triton IR) to a GPU binary.
   *
   * Runs the full Triton compilation pipeline:
   *   TTIR -> TTGIR -> LLVM IR -> target ISA (PTX/AMDGCN/SPIR-V)
   *
   * @param mlirModule  Opaque handle to a Triton MLIR module (tt::ModuleOp)
   * @param numWarps    Number of warps per block (default 4)
   * @param numStages   Number of pipeline stages (default 3)
   * @return Compiled binary, or {nullptr, 0} on failure
   */
  static TritonCompiledBinary compile(void* mlirModule, int numWarps = 4, int numStages = 3);

  /**
   * Load a compiled binary into a GPU driver module.
   *
   * @param binary  The compiled GPU binary
   * @return Opaque handle to the loaded module, or nullptr on failure.
   *         For NVIDIA this is CUmodule, for AMD hipModule_t, for Intel ze_module_handle_t.
   */
  static void* loadModule(const TritonCompiledBinary& binary);

  /**
   * Get a kernel function handle from a loaded module.
   *
   * @param gpuModule   Module handle from loadModule()
   * @param kernelName  Name of the kernel function
   * @return Opaque kernel function handle, or nullptr on failure.
   */
  static void* getKernelFunction(void* gpuModule, const std::string& kernelName);

  /**
   * Launch a kernel on the GPU.
   *
   * @param kernelFunc      Kernel function handle from getKernelFunction()
   * @param gridX/Y/Z       Grid dimensions
   * @param blockX/Y/Z      Block dimensions
   * @param sharedMemBytes  Shared memory per block
   * @param stream          GPU stream (cudaStream_t/hipStream_t/ze_command_list_handle_t)
   * @param args            Array of kernel argument pointers
   * @param numArgs         Number of arguments
   * @return true on success
   */
  static bool launchKernel(void* kernelFunc,
                           unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                           unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                           unsigned int sharedMemBytes,
                           void* stream,
                           void** args, int numArgs);

  /**
   * Launch a compiled kernel using cooperative launch (cudaLaunchCooperativeKernel).
   * All blocks must be co-resident on the GPU simultaneously, enabling grid-wide
   * synchronization barriers between kernel sections.
   *
   * Same parameters as launchKernel, but uses cudaLaunchCooperativeKernel instead
   * of cuLaunchKernel. Only supported on NVIDIA GPUs (CUDA 11.0+).
   * Falls back to standard launch on AMD/Intel targets.
   */
  static bool launchCooperativeKernel(void* kernelFunc,
                                      unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                                      unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                                      unsigned int sharedMemBytes,
                                      void* stream,
                                      void** args, int numArgs);

  /**
   * Unload a previously loaded GPU module and free its resources.
   *
   * @param gpuModule  Module handle from loadModule()
   */
  static void unloadModule(void* gpuModule);

 private:
  static TritonGpuTarget cachedTarget_;
  static std::string cachedArch_;
  static bool targetDetected_;
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
#endif  // LIBND4J_TRITON_TARGET_DISPATCH_H
