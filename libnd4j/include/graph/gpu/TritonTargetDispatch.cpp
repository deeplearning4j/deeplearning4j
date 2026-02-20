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

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonTargetDispatch.h>
#include <system/common.h>

#include <cstring>

// ─── Platform GPU headers ───────────────────────────────────────────────────
//
// CRITICAL: Under ZLUDA builds, SD_CUDA is defined even though the real GPU
// may be AMD (HIP) or Intel (Level Zero). ZLUDA intercepts CUDA API calls
// and translates them, but it expects PTX for cuModuleLoadDataEx.
//
// Triton produces NATIVE binaries for each target:
//   NVIDIA → PTX text   (compatible with cuModuleLoadDataEx)
//   AMD    → AMDGCN ELF (requires hipModuleLoadData, NOT cuModuleLoadDataEx)
//   Intel  → SPIR-V     (requires zeModuleCreate, NOT cuModuleLoadDataEx)
//
// Therefore:
//   - NVIDIA target: always use CUDA Driver API
//   - AMD target:    always use HIP directly (bypass ZLUDA for Triton kernels)
//   - Intel target:  always use Level Zero directly (bypass ZLUDA for Triton kernels)
//
// The switch is on binary.target / detectTarget() result, not on build flags.
// Build flags only gate which headers and APIs are available.

#ifdef SD_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// HIP headers: available in native ROCm builds AND ZLUDA+AMD builds.
// ZLUDA+AMD sets HAVE_MIOPEN=1 and includes ROCm in the build.
#if defined(ZLUDA_TARGET_AMD) || defined(HAVE_MIOPEN) || defined(SD_HIP)
#define TRITON_HAS_HIP 1
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#endif

// Level Zero headers: available in native Intel builds and ZLUDA+Intel builds.
#if defined(ZLUDA_TARGET_INTEL) || defined(SD_LEVEL_ZERO)
#define TRITON_HAS_LEVEL_ZERO 1
#include <level_zero/ze_api.h>
#endif

// Triton compiler C++ API
#include <triton/Compiler/Compiler.h>
#include <triton/Target/LLVMIR/Passes.h>

namespace sd {
namespace graph {

// Static member initialization
TritonGpuTarget TritonTargetDispatch::cachedTarget_ = TritonGpuTarget::UNKNOWN;
std::string TritonTargetDispatch::cachedArch_;
bool TritonTargetDispatch::targetDetected_ = false;

// ─── Target detection ───────────────────────────────────────────────────────

bool TritonTargetDispatch::isReady() {
  auto target = detectTarget();
  return target != TritonGpuTarget::UNKNOWN;
}

TritonGpuTarget TritonTargetDispatch::detectTarget() {
  if (targetDetected_) return cachedTarget_;
  targetDetected_ = true;

  // Detection priority:
  //   1. HIP (most accurate for AMD — gives gcnArchName)
  //   2. Level Zero (most accurate for Intel — gives device properties)
  //   3. CUDA (for native NVIDIA, or ZLUDA fallback)
  //
  // Under ZLUDA+AMD: both SD_CUDA and HAVE_MIOPEN are defined.
  // HIP gives us the real gcnArchName (e.g., "gfx1100"), while CUDA
  // would require guessing the arch from the device name string.
  // So we try HIP first.

#if TRITON_HAS_HIP
  // Try HIP detection first — preferred for AMD because gcnArchName is exact
  {
    int deviceCount = 0;
    auto err = hipGetDeviceCount(&deviceCount);
    if (err == hipSuccess && deviceCount > 0) {
      hipDeviceProp_t props;
      hipGetDeviceProperties(&props, 0);

      // gcnArchName is the canonical arch string (e.g., "gfx1100", "gfx90a")
      std::string archName = props.gcnArchName;
      if (!archName.empty() && archName.find("gfx") != std::string::npos) {
        cachedArch_ = archName;
        cachedTarget_ = TritonGpuTarget::AMD;
        sd_printf("TritonTargetDispatch: detected AMD GPU '%s' via HIP, arch=%s\n",
                  props.name, cachedArch_.c_str());
        return cachedTarget_;
      }
    }
  }
#endif

#if TRITON_HAS_LEVEL_ZERO
  // Try Level Zero detection — preferred for Intel
  {
    ze_result_t res = zeInit(0);
    if (res == ZE_RESULT_SUCCESS) {
      uint32_t driverCount = 0;
      zeDriverGet(&driverCount, nullptr);
      if (driverCount > 0) {
        std::vector<ze_driver_handle_t> drivers(driverCount);
        zeDriverGet(&driverCount, drivers.data());

        uint32_t deviceCount = 0;
        zeDeviceGet(drivers[0], &deviceCount, nullptr);
        if (deviceCount > 0) {
          std::vector<ze_device_handle_t> devices(deviceCount);
          zeDeviceGet(drivers[0], &deviceCount, devices.data());

          ze_device_properties_t deviceProps = {};
          deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
          zeDeviceGetProperties(devices[0], &deviceProps);

          if (deviceProps.type == ZE_DEVICE_TYPE_GPU) {
            cachedArch_ = "pvc";  // Default; refine based on deviceId
            std::string devName(deviceProps.name);
            if (devName.find("Arc") != std::string::npos) {
              cachedArch_ = "xehpg";  // Alchemist (Arc A-series)
            }
            if (devName.find("Max") != std::string::npos ||
                devName.find("Ponte Vecchio") != std::string::npos) {
              cachedArch_ = "pvc";  // Data Center Max
            }
            cachedTarget_ = TritonGpuTarget::INTEL;
            sd_printf("TritonTargetDispatch: detected Intel GPU '%s' via Level Zero, arch=%s\n",
                      deviceProps.name, cachedArch_.c_str());
            return cachedTarget_;
          }
        }
      }
    }
  }
#endif

#ifdef SD_CUDA
  // CUDA detection — for native NVIDIA GPUs (or ZLUDA fallback)
  {
    int deviceCount = 0;
    auto err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess && deviceCount > 0) {
      cudaDeviceProp props;
      cudaGetDeviceProperties(&props, 0);

      std::string deviceName(props.name);

      // If we get here under ZLUDA, the HIP/Level Zero detection above
      // didn't match (unusual). Check the device name as a fallback.
#ifdef HAVE_ZLUDA
      // AMD GPU behind ZLUDA but HIP detection didn't work
      if (deviceName.find("AMD") != std::string::npos ||
          deviceName.find("Radeon") != std::string::npos ||
          deviceName.find("gfx") != std::string::npos) {
        // Best-effort arch from compute capability (imprecise)
        cachedArch_ = "gfx" + std::to_string(props.major * 100 + props.minor * 10);
        cachedTarget_ = TritonGpuTarget::AMD;
        sd_printf("TritonTargetDispatch: detected AMD GPU '%s' via CUDA (ZLUDA fallback), arch=%s\n",
                  props.name, cachedArch_.c_str());
        return cachedTarget_;
      }

      // Intel GPU behind ZLUDA but Level Zero detection didn't work
      if (deviceName.find("Intel") != std::string::npos ||
          deviceName.find("Arc") != std::string::npos) {
        cachedArch_ = "xehp";
        if (deviceName.find("Arc") != std::string::npos) cachedArch_ = "xehpg";
        if (deviceName.find("Max") != std::string::npos) cachedArch_ = "pvc";
        cachedTarget_ = TritonGpuTarget::INTEL;
        sd_printf("TritonTargetDispatch: detected Intel GPU '%s' via CUDA (ZLUDA fallback), arch=%s\n",
                  props.name, cachedArch_.c_str());
        return cachedTarget_;
      }
#endif

      // Native NVIDIA GPU
      cachedArch_ = "sm_" + std::to_string(props.major * 10 + props.minor);
      cachedTarget_ = TritonGpuTarget::NVIDIA;
      sd_printf("TritonTargetDispatch: detected NVIDIA GPU '%s', arch=%s\n",
                props.name, cachedArch_.c_str());
      return cachedTarget_;
    }
  }
#endif

  sd_printf("TritonTargetDispatch: no supported GPU target detected\n", "");
  cachedTarget_ = TritonGpuTarget::UNKNOWN;
  return cachedTarget_;
}

std::string TritonTargetDispatch::getTargetArch() {
  detectTarget();
  return cachedArch_;
}

// ─── Compilation ────────────────────────────────────────────────────────────

TritonCompiledBinary TritonTargetDispatch::compile(void* mlirModule, int numWarps, int numStages) {
  TritonCompiledBinary result = {nullptr, 0, TritonGpuTarget::UNKNOWN, "", 0, 0};

  auto target = detectTarget();
  if (target == TritonGpuTarget::UNKNOWN) {
    sd_printf("TritonTargetDispatch::compile: no GPU target available\n", "");
    return result;
  }

  result.target = target;
  result.targetArch = cachedArch_;
  result.numWarps = numWarps;

  // Use Triton's C++ compiler API to lower the MLIR module.
  // The pipeline is: TTIR -> TTGIR -> LLVM IR -> target ISA
  //
  // Each target uses a different Triton backend:
  //   NVIDIA: mlir::triton::nvidia_gpu -> PTX -> cubin
  //   AMD:    mlir::triton::amd        -> AMDGCN -> HSACO
  //   Intel:  mlir::triton::intel      -> SPIR-V -> native binary

  auto moduleOp = static_cast<mlir::ModuleOp*>(mlirModule);

  switch (target) {
    case TritonGpuTarget::NVIDIA: {
      // Parse compute capability from arch string (e.g., "sm_89" -> 89)
      int computeCapability = 0;
      if (cachedArch_.size() > 3) {
        computeCapability = std::stoi(cachedArch_.substr(3));
      }

      mlir::triton::nvidia_gpu::ClusterInfo clusterInfo;
      clusterInfo.clusterDimX = 1;
      clusterInfo.clusterDimY = 1;
      clusterInfo.clusterDimZ = 1;

      // Run the Triton compilation pipeline: TTIR -> TTGIR -> LLVM IR -> PTX
      auto ptxOrErr = mlir::triton::translateTritonGPUToLLVMIR(
          *moduleOp, computeCapability, /*compilation*/ {});

      if (!ptxOrErr) {
        sd_printf("TritonTargetDispatch::compile: NVIDIA PTX compilation failed\n", "");
        return result;
      }

      auto& ptx = *ptxOrErr;
      result.size = ptx.size();
      result.data = new char[result.size];
      std::memcpy(result.data, ptx.data(), result.size);
      break;
    }

    case TritonGpuTarget::AMD: {
      // Parse GFX version from arch string (e.g., "gfx1100" -> 1100)
      int gfxVersion = 0;
      if (cachedArch_.size() > 3) {
        gfxVersion = std::stoi(cachedArch_.substr(3));
      }

      // Run the Triton AMD compilation pipeline: TTIR -> TTGIR -> LLVM IR -> AMDGCN
      auto amdgcnOrErr = mlir::triton::translateTritonGPUToLLVMIR(
          *moduleOp, gfxVersion, /*compilation*/ {});

      if (!amdgcnOrErr) {
        sd_printf("TritonTargetDispatch::compile: AMD AMDGCN compilation failed for %s\n",
                  cachedArch_.c_str());
        return result;
      }

      auto& amdgcn = *amdgcnOrErr;
      result.size = amdgcn.size();
      result.data = new char[result.size];
      std::memcpy(result.data, amdgcn.data(), result.size);
      break;
    }

    case TritonGpuTarget::INTEL: {
      // Run the Triton Intel compilation pipeline: TTIR -> TTGIR -> LLVM IR -> SPIR-V
      auto spirvOrErr = mlir::triton::translateTritonGPUToLLVMIR(
          *moduleOp, 0 /*not used for Intel*/, /*compilation*/ {});

      if (!spirvOrErr) {
        sd_printf("TritonTargetDispatch::compile: Intel SPIR-V compilation failed for %s\n",
                  cachedArch_.c_str());
        return result;
      }

      auto& spirv = *spirvOrErr;
      result.size = spirv.size();
      result.data = new char[result.size];
      std::memcpy(result.data, spirv.data(), result.size);
      break;
    }

    default:
      return result;
  }

  return result;
}

// ─── Module loading ─────────────────────────────────────────────────────────
//
// CRITICAL DESIGN: Each target loads its native binary through the correct API.
// Under ZLUDA+AMD, we use hipModuleLoadData (NOT cuModuleLoadDataEx) because:
//   - Triton compiles to AMDGCN ELF for AMD targets
//   - ZLUDA's cuModuleLoadDataEx expects PTX, not AMDGCN
//   - HIP is always available in ZLUDA+AMD builds (ROCm is installed)

void* TritonTargetDispatch::loadModule(const TritonCompiledBinary& binary) {
  if (!binary.data || binary.size == 0) return nullptr;

  switch (binary.target) {

    case TritonGpuTarget::NVIDIA: {
#ifdef SD_CUDA
      // NVIDIA: CUDA Driver API for PTX loading.
      // Works for both native CUDA and ZLUDA (if target was somehow NVIDIA).
      CUmodule module = nullptr;
      CUresult res = cuModuleLoadDataEx(&module, binary.data, 0, nullptr, nullptr);
      if (res != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(res, &errStr);
        sd_printf("TritonTargetDispatch::loadModule: cuModuleLoadDataEx failed: %s\n",
                  errStr ? errStr : "unknown");
        return nullptr;
      }
      return static_cast<void*>(module);
#else
      sd_printf("TritonTargetDispatch::loadModule: NVIDIA target requires SD_CUDA\n", "");
      return nullptr;
#endif
    }

    case TritonGpuTarget::AMD: {
#if TRITON_HAS_HIP
      // AMD: HIP for AMDGCN/HSACO loading.
      // This code path is active for BOTH:
      //   - Native ROCm/HIP builds (SD_HIP defined, SD_CUDA not defined)
      //   - ZLUDA+AMD builds (SD_CUDA defined, ZLUDA_TARGET_AMD defined, HAVE_MIOPEN defined)
      // In ZLUDA+AMD builds, we bypass ZLUDA's CUDA interception and use HIP directly.
      hipModule_t module = nullptr;
      hipError_t res = hipModuleLoadData(&module, binary.data);
      if (res != hipSuccess) {
        sd_printf("TritonTargetDispatch::loadModule: hipModuleLoadData failed: %s\n",
                  hipGetErrorString(res));
        return nullptr;
      }
      return static_cast<void*>(module);
#else
      sd_printf("TritonTargetDispatch::loadModule: AMD target requires HIP (HAVE_MIOPEN/SD_HIP/ZLUDA_TARGET_AMD)\n", "");
      return nullptr;
#endif
    }

    case TritonGpuTarget::INTEL: {
#if TRITON_HAS_LEVEL_ZERO
      // Intel: Level Zero for SPIR-V module loading.
      // Active for both native Level Zero builds and ZLUDA+Intel builds.
      uint32_t driverCount = 1;
      ze_driver_handle_t driver;
      zeDriverGet(&driverCount, &driver);

      uint32_t deviceCount = 1;
      ze_device_handle_t device;
      zeDeviceGet(driver, &deviceCount, &device);

      // Create context
      ze_context_desc_t ctxDesc = {};
      ctxDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
      ze_context_handle_t context;
      zeContextCreate(driver, &ctxDesc, &context);

      // Create module from SPIR-V binary
      ze_module_desc_t moduleDesc = {};
      moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
      moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
      moduleDesc.inputSize = binary.size;
      moduleDesc.pInputModule = static_cast<const uint8_t*>(binary.data);

      ze_module_handle_t module = nullptr;
      ze_module_build_log_handle_t buildLog = nullptr;
      ze_result_t res = zeModuleCreate(context, device, &moduleDesc, &module, &buildLog);

      if (res != ZE_RESULT_SUCCESS) {
        if (buildLog) {
          size_t logSize = 0;
          zeModuleBuildLogGetString(buildLog, &logSize, nullptr);
          if (logSize > 0) {
            std::string logStr(logSize, '\0');
            zeModuleBuildLogGetString(buildLog, &logSize, &logStr[0]);
            sd_printf("TritonTargetDispatch::loadModule: zeModuleCreate failed: %s\n", logStr.c_str());
          }
          zeModuleBuildLogDestroy(buildLog);
        }
        return nullptr;
      }
      if (buildLog) zeModuleBuildLogDestroy(buildLog);
      return static_cast<void*>(module);
#else
      sd_printf("TritonTargetDispatch::loadModule: Intel target requires Level Zero (SD_LEVEL_ZERO/ZLUDA_TARGET_INTEL)\n", "");
      return nullptr;
#endif
    }

    default:
      sd_printf("TritonTargetDispatch::loadModule: unsupported target %d\n",
                static_cast<int>(binary.target));
      return nullptr;
  }
}

// ─── Kernel function lookup ─────────────────────────────────────────────────

void* TritonTargetDispatch::getKernelFunction(void* gpuModule, const std::string& kernelName) {
  if (!gpuModule) return nullptr;

  auto target = detectTarget();
  switch (target) {

    case TritonGpuTarget::NVIDIA: {
#ifdef SD_CUDA
      CUfunction func = nullptr;
      CUresult res = cuModuleGetFunction(&func, static_cast<CUmodule>(gpuModule), kernelName.c_str());
      if (res != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(res, &errStr);
        sd_printf("TritonTargetDispatch::getKernelFunction: cuModuleGetFunction failed: %s\n",
                  errStr ? errStr : "unknown");
        return nullptr;
      }
      return static_cast<void*>(func);
#else
      return nullptr;
#endif
    }

    case TritonGpuTarget::AMD: {
#if TRITON_HAS_HIP
      hipFunction_t func = nullptr;
      hipError_t res = hipModuleGetFunction(&func, static_cast<hipModule_t>(gpuModule), kernelName.c_str());
      if (res != hipSuccess) {
        sd_printf("TritonTargetDispatch::getKernelFunction: hipModuleGetFunction failed: %s\n",
                  hipGetErrorString(res));
        return nullptr;
      }
      return static_cast<void*>(func);
#else
      return nullptr;
#endif
    }

    case TritonGpuTarget::INTEL: {
#if TRITON_HAS_LEVEL_ZERO
      ze_kernel_desc_t kernelDesc = {};
      kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
      kernelDesc.pKernelName = kernelName.c_str();

      ze_kernel_handle_t kernel = nullptr;
      ze_result_t res = zeKernelCreate(static_cast<ze_module_handle_t>(gpuModule), &kernelDesc, &kernel);
      if (res != ZE_RESULT_SUCCESS) {
        sd_printf("TritonTargetDispatch::getKernelFunction: zeKernelCreate failed for '%s'\n",
                  kernelName.c_str());
        return nullptr;
      }
      return static_cast<void*>(kernel);
#else
      return nullptr;
#endif
    }

    default:
      return nullptr;
  }
}

// ─── Kernel launch ──────────────────────────────────────────────────────────

bool TritonTargetDispatch::launchKernel(void* kernelFunc,
                                        unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                                        unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                                        unsigned int sharedMemBytes,
                                        void* stream,
                                        void** args, int numArgs) {
  if (!kernelFunc) return false;

  auto target = detectTarget();
  switch (target) {

    case TritonGpuTarget::NVIDIA: {
#ifdef SD_CUDA
      CUresult res = cuLaunchKernel(
          static_cast<CUfunction>(kernelFunc),
          gridX, gridY, gridZ,
          blockX, blockY, blockZ,
          sharedMemBytes,
          static_cast<CUstream>(stream),
          args, nullptr);
      if (res != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(res, &errStr);
        sd_printf("TritonTargetDispatch::launchKernel: cuLaunchKernel failed: %s\n",
                  errStr ? errStr : "unknown");
        return false;
      }
      return true;
#else
      return false;
#endif
    }

    case TritonGpuTarget::AMD: {
#if TRITON_HAS_HIP
      // Launch via HIP directly. Under ZLUDA+AMD this bypasses ZLUDA's
      // CUDA interception and uses the real HIP runtime.
      hipError_t res = hipModuleLaunchKernel(
          static_cast<hipFunction_t>(kernelFunc),
          gridX, gridY, gridZ,
          blockX, blockY, blockZ,
          sharedMemBytes,
          static_cast<hipStream_t>(stream),
          args, nullptr);
      if (res != hipSuccess) {
        sd_printf("TritonTargetDispatch::launchKernel: hipModuleLaunchKernel failed: %s\n",
                  hipGetErrorString(res));
        return false;
      }
      return true;
#else
      return false;
#endif
    }

    case TritonGpuTarget::INTEL: {
#if TRITON_HAS_LEVEL_ZERO
      auto kernel = static_cast<ze_kernel_handle_t>(kernelFunc);

      // Set group size
      zeKernelSetGroupSize(kernel, blockX, blockY, blockZ);

      // Set kernel arguments
      for (int i = 0; i < numArgs; i++) {
        zeKernelSetArgumentValue(kernel, i, sizeof(void*), args[i]);
      }

      // Launch on the command list (passed as stream)
      ze_group_count_t groupCount = {gridX, gridY, gridZ};
      auto cmdList = static_cast<ze_command_list_handle_t>(stream);
      ze_result_t res = zeCommandListAppendLaunchKernel(
          cmdList, kernel, &groupCount, nullptr, 0, nullptr);
      if (res != ZE_RESULT_SUCCESS) {
        sd_printf("TritonTargetDispatch::launchKernel: zeCommandListAppendLaunchKernel failed\n", "");
        return false;
      }
      return true;
#else
      return false;
#endif
    }

    default:
      sd_printf("TritonTargetDispatch::launchKernel: unsupported target %d\n",
                static_cast<int>(target));
      return false;
  }
}

// ─── Module unload ──────────────────────────────────────────────────────────

void TritonTargetDispatch::unloadModule(void* gpuModule) {
  if (!gpuModule) return;

  auto target = detectTarget();
  switch (target) {

    case TritonGpuTarget::NVIDIA: {
#ifdef SD_CUDA
      cuModuleUnload(static_cast<CUmodule>(gpuModule));
#endif
      break;
    }

    case TritonGpuTarget::AMD: {
#if TRITON_HAS_HIP
      hipModuleUnload(static_cast<hipModule_t>(gpuModule));
#endif
      break;
    }

    case TritonGpuTarget::INTEL: {
#if TRITON_HAS_LEVEL_ZERO
      zeModuleDestroy(static_cast<ze_module_handle_t>(gpuModule));
#endif
      break;
    }

    default:
      break;
  }
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
