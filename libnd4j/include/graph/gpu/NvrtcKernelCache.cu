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


#include <graph/gpu/NvrtcKernelCache.h>
#include <graph/DspDiagnostics.h>
#include <array/NDArray.h>
#include <system/common.h>
#include <helpers/logger.h>

#include <cuda.h>
#include <nvrtc.h>

#include <cstring>
#include <string>
#include <vector>

namespace sd {
namespace graph {

// ── NvrtcKernelHandle lifecycle ────────────────────────────────────────────

NvrtcKernelHandle::~NvrtcKernelHandle() {
  if (module != nullptr) {
    DSP_DIAG(EXECUTE, "NvrtcKernelCache: cuModuleUnload module=%p", (void*)module);
    cuModuleUnload(module);
    module = nullptr;
  }
  function = nullptr;
  valid = false;
}

NvrtcKernelHandle::NvrtcKernelHandle(NvrtcKernelHandle&& other) noexcept
    : module(other.module), function(other.function),
      kernelName(std::move(other.kernelName)), blockSize(other.blockSize),
      paramBindings(std::move(other.paramBindings)), valid(other.valid) {
  other.module = nullptr;
  other.function = nullptr;
  other.valid = false;
}

NvrtcKernelHandle& NvrtcKernelHandle::operator=(NvrtcKernelHandle&& other) noexcept {
  if (this != &other) {
    if (module != nullptr) {
      DSP_DIAG(EXECUTE, "NvrtcKernelCache(dtor): cuModuleUnload module=%p", (void*)module);
      cuModuleUnload(module);
    }
    module = other.module;
    function = other.function;
    kernelName = std::move(other.kernelName);
    blockSize = other.blockSize;
    paramBindings = std::move(other.paramBindings);
    valid = other.valid;
    other.module = nullptr;
    other.function = nullptr;
    other.valid = false;
  }
  return *this;
}

// ── compileKernel ──────────────────────────────────────────────────────────

NvrtcKernelHandle* compileKernel(const JitKernelSource& source, int deviceId) {
  auto* handle = new NvrtcKernelHandle();
  handle->kernelName = source.kernelName;
  handle->paramBindings = source.paramBindings;

  // Get compute capability for this device
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceId);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceId);
  std::string archOpt = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);

  // Create NVRTC program
  nvrtcProgram prog;
  nvrtcResult nvrtcRes = nvrtcCreateProgram(
      &prog,
      source.sourceCode.c_str(),
      (source.kernelName + ".cu").c_str(),
      0, nullptr, nullptr);

  if (nvrtcRes != NVRTC_SUCCESS) {
    DSP_DIAG(COMPILE, "NVRTC: nvrtcCreateProgram failed: %s", nvrtcGetErrorString(nvrtcRes));
    delete handle;
    return nullptr;
  }

  // Compile
  const char* opts[] = {archOpt.c_str(), "--std=c++17"};
  nvrtcRes = nvrtcCompileProgram(prog, 2, opts);

  if (nvrtcRes != NVRTC_SUCCESS) {
    // Get compilation log for diagnostics
    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
      std::string log(logSize, '\0');
      nvrtcGetProgramLog(prog, &log[0]);
      DSP_DIAG(COMPILE, "NVRTC compilation failed for %s:\n%s",
               source.kernelName.c_str(), log.c_str());
    }
    DSP_DIAG(COMPILE, "NVRTC: nvrtcCompileProgram failed: %s", nvrtcGetErrorString(nvrtcRes));
    nvrtcDestroyProgram(&prog);
    delete handle;
    return nullptr;
  }

  // Get PTX
  size_t ptxSize = 0;
  nvrtcGetPTXSize(prog, &ptxSize);
  std::string ptx(ptxSize, '\0');
  nvrtcGetPTX(prog, &ptx[0]);
  nvrtcDestroyProgram(&prog);

  // Ensure CUDA driver API is initialized
  CUresult cuRes = cuInit(0);
  if (cuRes != CUDA_SUCCESS) {
    DSP_DIAG(COMPILE, "NVRTC: cuInit failed: %d", static_cast<int>(cuRes));
    delete handle;
    return nullptr;
  }

  // Load module from PTX
  cuRes = cuModuleLoadDataEx(&handle->module, ptx.c_str(), 0, nullptr, nullptr);
  if (cuRes != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(cuRes, &errStr);
    DSP_DIAG(COMPILE, "NVRTC: cuModuleLoadDataEx failed: %d (%s)",
             static_cast<int>(cuRes), errStr ? errStr : "unknown");
    delete handle;
    return nullptr;
  }

  // Get kernel function
  cuRes = cuModuleGetFunction(&handle->function, handle->module, source.kernelName.c_str());
  if (cuRes != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(cuRes, &errStr);
    DSP_DIAG(COMPILE, "NVRTC: cuModuleGetFunction failed for '%s': %d (%s)",
             source.kernelName.c_str(), static_cast<int>(cuRes),
             errStr ? errStr : "unknown");
    delete handle;
    return nullptr;
  }

  handle->valid = true;
  DSP_DIAG(COMPILE, "NVRTC: compiled kernel '%s' (arch=sm_%d%d, PTX=%zu bytes)",
           source.kernelName.c_str(), major, minor, ptxSize);
  return handle;
}

// ── launchKernel ───────────────────────────────────────────────────────────

Status launchKernel(NvrtcKernelHandle* handle,
                    int64_t elementCount,
                    NDArray** externalInputs, int numExternalInputs,
                    NDArray** outputSlots, int totalOutputSlots,
                    void* stream) {
  if (handle == nullptr || !handle->valid) {
    return Status::KERNEL_FAILURE;
  }

  // Build the argument array — each entry is a pointer to the argument value
  int numParams = static_cast<int>(handle->paramBindings.size());

  // Allocate storage for argument values (pointers + one int64)
  // We need to keep the values alive until cuLaunchKernel returns
  std::vector<void*> ptrValues(numParams, nullptr);
  int64_t lengthValue = elementCount;
  float clipMinValue = 0.0f, clipMaxValue = 0.0f;

  // Array of pointers to argument values (what cuLaunchKernel expects)
  std::vector<void*> args(numParams);

  for (int i = 0; i < numParams; i++) {
    const auto& binding = handle->paramBindings[i];
    switch (binding.type) {
      case JitParamBinding::LENGTH:
        args[i] = &lengthValue;
        break;

      case JitParamBinding::INPUT_PTR: {
        int extIdx = binding.externalInputIdx;
        if (extIdx < 0 || extIdx >= numExternalInputs || externalInputs[extIdx] == nullptr) {
          DSP_DIAG(EXECUTE, "NVRTC launch: invalid external input index %d", extIdx);
          return Status::KERNEL_FAILURE;
        }
        ptrValues[i] = externalInputs[extIdx]->specialBuffer();
        args[i] = &ptrValues[i];
        break;
      }

      case JitParamBinding::CROSS_SEG_PTR: {
        int slotIdx = binding.slotIdx;
        if (slotIdx < 0 || slotIdx >= totalOutputSlots || outputSlots[slotIdx] == nullptr) {
          DSP_DIAG(EXECUTE, "NVRTC launch: invalid cross-segment slot index %d", slotIdx);
          return Status::KERNEL_FAILURE;
        }
        ptrValues[i] = outputSlots[slotIdx]->specialBuffer();
        args[i] = &ptrValues[i];
        break;
      }

      case JitParamBinding::OUTPUT_PTR: {
        int slotIdx = binding.slotIdx;
        if (slotIdx < 0 || slotIdx >= totalOutputSlots || outputSlots[slotIdx] == nullptr) {
          DSP_DIAG(EXECUTE, "NVRTC launch: invalid output slot index %d", slotIdx);
          return Status::KERNEL_FAILURE;
        }
        ptrValues[i] = outputSlots[slotIdx]->specialBuffer();
        args[i] = &ptrValues[i];
        break;
      }
    }
  }

  // Calculate grid dimensions
  int blockSize = handle->blockSize;
  int gridSize = static_cast<int>((elementCount + blockSize - 1) / blockSize);

  // stream is a cudaStream_t* (pointer to cudaStream_t) — dereference and cast
  // cudaStream_t is interchangeable with CUstream in the CUDA driver API
  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;
  CUstream cuStream = static_cast<CUstream>(cudaStr);

  CUresult cuRes = cuLaunchKernel(
      handle->function,
      gridSize, 1, 1,       // grid dimensions
      blockSize, 1, 1,      // block dimensions
      0,                    // shared memory bytes
      cuStream,             // stream
      args.data(),          // kernel arguments
      nullptr               // extra (unused)
  );

  if (cuRes != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(cuRes, &errStr);
    DSP_DIAG(EXECUTE, "NVRTC: cuLaunchKernel failed for '%s': %d (%s)",
             handle->kernelName.c_str(), static_cast<int>(cuRes),
             errStr ? errStr : "unknown");
    return Status::KERNEL_FAILURE;
  }

  return Status::OK;
}

}  // namespace graph
}  // namespace sd

