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

#ifndef LIBND4J_NVRTC_KERNEL_CACHE_H
#define LIBND4J_NVRTC_KERNEL_CACHE_H

#ifdef SD_CUDA

#include <cuda.h>
#include <graph/gpu/NvrtcKernelBuilder.h>
#include <system/common.h>

#include <string>
#include <vector>

namespace sd {
namespace graph {

/**
 * Compiled NVRTC kernel handle.
 * Owns the CUmodule and provides launch functionality.
 */
struct NvrtcKernelHandle {
  CUmodule module;
  CUfunction function;
  std::string kernelName;
  int blockSize;

  // Parameter binding metadata (copied from JitKernelSource at compile time)
  std::vector<JitParamBinding> paramBindings;

  // Whether this handle has a valid compiled kernel
  bool valid;

  NvrtcKernelHandle()
      : module(nullptr), function(nullptr), blockSize(256), valid(false) {}

  ~NvrtcKernelHandle();

  // No copy
  NvrtcKernelHandle(const NvrtcKernelHandle&) = delete;
  NvrtcKernelHandle& operator=(const NvrtcKernelHandle&) = delete;

  // Move OK
  NvrtcKernelHandle(NvrtcKernelHandle&& other) noexcept;
  NvrtcKernelHandle& operator=(NvrtcKernelHandle&& other) noexcept;
};

/**
 * Compile a CUDA C++ source string via NVRTC into a ready-to-launch kernel.
 *
 * Flow: nvrtcCreateProgram -> nvrtcCompileProgram -> nvrtcGetPTX
 *       -> cuModuleLoadDataEx -> cuModuleGetFunction
 *
 * @param source        JitKernelSource with CUDA C++ code and metadata
 * @param deviceId      CUDA device ID (for compute capability detection)
 * @return Compiled kernel handle (check .valid for success)
 */
NvrtcKernelHandle* compileKernel(const JitKernelSource& source, int deviceId);

/**
 * Launch a compiled NVRTC kernel.
 *
 * Resolves each ParamBinding to current specialBuffer() pointers,
 * packs into void* args[], and calls cuLaunchKernel on the given stream.
 *
 * @param handle          Compiled kernel handle
 * @param elementCount    Number of elements to process
 * @param externalInputs  External input NDArrays
 * @param numExternalInputs  Number of external inputs
 * @param outputSlots     Current outputSlots_ array
 * @param totalOutputSlots   Size of outputSlots array
 * @param stream          CUDA stream to launch on
 * @return Status::OK on success, Status::KERNEL_FAILURE on error
 */
Status launchKernel(NvrtcKernelHandle* handle,
                    int64_t elementCount,
                    NDArray** externalInputs, int numExternalInputs,
                    NDArray** outputSlots, int totalOutputSlots,
                    void* stream);

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
#endif  // LIBND4J_NVRTC_KERNEL_CACHE_H
