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
#include <helpers/logger.h>
#include <system/common.h>

#include <cstring>
#include <csignal>
#include <setjmp.h>

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

// MLIR core infrastructure
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>

// LLVM backend for PTX generation
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/raw_ostream.h>


// Triton dialect passes — TTIR -> TTGIR -> LLVM
#include <triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h>
#include <triton/Conversion/TritonGPUToLLVM/Passes.h>
#include <triton/Dialect/TritonGPU/Transforms/Passes.h>
#include <triton/Target/LLVMIR/Passes.h>

// NVIDIA backend passes (when building with NVIDIA codegen backend)
#if __has_include(<TritonNVIDIAGPUToLLVM/Passes.h>)
#include <TritonNVIDIAGPUToLLVM/Passes.h>
#include <NVGPUToLLVM/NVGPUToLLVMPass.h>
#define HAVE_TRITON_NVIDIA_PASSES 1
#endif

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

  auto moduleOp = static_cast<mlir::ModuleOp*>(mlirModule);
  if (!moduleOp || !*moduleOp) {
    sd_printf("TritonTargetDispatch::compile: null MLIR module\n", "");
    return result;
  }

  // ── Pass pipeline: TTIR -> TTGIR -> LLVM dialect -> LLVM IR -> target ISA ──
  //
  // Phase 1: TTIR optimizations (inliner, canonicalizer, CSE)
  // Phase 2: TTIR -> TTGIR conversion (adds GPU-specific tensor encoding)
  // Phase 3: TTGIR optimizations (coalesce)
  // Phase 4: TTGIR -> LLVM MLIR dialect (backend-specific lowering)
  // Phase 5: LLVM MLIR -> LLVM IR module
  // Phase 6: LLVM IR -> target ISA (PTX/AMDGCN/SPIR-V)

  // Determine target-specific parameters
  std::string targetStr;
  int computeCapability = 0;

  switch (target) {
    case TritonGpuTarget::NVIDIA: {
      if (cachedArch_.size() > 3) {
        computeCapability = std::stoi(cachedArch_.substr(3));
      }
      targetStr = "cuda:" + std::to_string(computeCapability);
      break;
    }
    case TritonGpuTarget::AMD: {
      targetStr = "hip:" + cachedArch_;
      break;
    }
    case TritonGpuTarget::INTEL: {
      targetStr = "xpu:" + cachedArch_;
      break;
    }
    default:
      return result;
  }

  // Phase 1-2: TTIR -> TTGIR
  {
    mlir::PassManager pm(moduleOp->getContext());

    // Phase 1: TTIR optimizations
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // Phase 2: TTIR -> TTGIR
    pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(
        targetStr, numWarps, /*threadsPerWarp=*/32, /*numCTAs=*/1));

    // Phase 3: TTGIR optimizations
    pm.addPass(mlir::triton::gpu::createTritonGPUCoalesce());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (mlir::failed(pm.run(*moduleOp))) {
      sd_printf("TritonTargetDispatch::compile: TTIR->TTGIR pass pipeline failed\n", "");
      return result;
    }
  }

  // Phase 4: TTGIR -> LLVM MLIR dialect
  {
    mlir::PassManager pm(moduleOp->getContext());

    // Shared memory allocation (required before LLVM lowering)
    pm.addPass(mlir::triton::gpu::createAllocateSharedMemoryPass());

    // Backend-specific TTGIR -> LLVM conversion
    //
    // In Triton v3.2.0, TTGIR -> LLVM lowering is handled by backend-specific
    // third-party libraries (e.g., TritonNVIDIAGPUToLLVM for NVIDIA).
    // The core Triton library does NOT provide a generic createConvertTritonGPUToLLVMPass.
    bool hasBackendLowering = false;
    switch (target) {
      case TritonGpuTarget::NVIDIA: {
#ifdef HAVE_TRITON_NVIDIA_PASSES
        pm.addPass(mlir::triton::NVIDIA::createDecomposeUnsupportedConversionsPass());
        pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(computeCapability));
        pm.addPass(mlir::triton::createConvertNVGPUToLLVMPass());
        hasBackendLowering = true;
#else
        sd_printf("TritonTargetDispatch::compile: NVIDIA backend passes not available "
                  "(TritonNVIDIAGPUToLLVM not found at build time)\n", "");
#endif
        break;
      }
      case TritonGpuTarget::AMD:
        sd_printf("TritonTargetDispatch::compile: AMD backend TTGIR->LLVM lowering "
                  "not yet integrated (requires TritonAMDGPUToLLVM)\n", "");
        break;
      case TritonGpuTarget::INTEL:
        sd_printf("TritonTargetDispatch::compile: Intel backend TTGIR->LLVM lowering "
                  "not yet integrated (requires TritonIntelGPUToLLVM)\n", "");
        break;
      default:
        break;
    }

    if (!hasBackendLowering) {
      sd_printf("TritonTargetDispatch::compile: no backend lowering passes available for target\n", "");
      return result;
    }

    // Standard MLIR -> LLVM lowering passes
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());

    if (mlir::failed(pm.run(*moduleOp))) {
      sd_printf("TritonTargetDispatch::compile: TTGIR->LLVM pass pipeline failed\n", "");
      return result;
    }
  }

  // Phase 5: MLIR LLVM dialect -> LLVM IR module
  // Verify the MLIR module before attempting LLVM translation
  if (mlir::failed(mlir::verify(*moduleOp))) {
    sd_printf("TritonTargetDispatch::compile: MLIR module verification failed after lowering\n", "");
    return result;
  }

  // Register ALL dialect translation interfaces — builtin.module, NVVM, GPU, etc.
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  moduleOp->getContext()->appendDialectRegistry(registry);
  llvm::LLVMContext llvmCtx;

  // LLVM debug builds call abort() on assertion failures (e.g., invalid cast).
  // We install a SIGABRT handler with setjmp/longjmp to recover gracefully.
  // fork() cannot be used because CUDA contexts break on fork.
  static thread_local jmp_buf tritonJmpBuf;
  static thread_local bool tritonInProtectedRegion = false;

  struct sigaction oldSigabrt;
  struct sigaction newSigabrt;
  memset(&newSigabrt, 0, sizeof(newSigabrt));
  newSigabrt.sa_handler = [](int) {
    if (tritonInProtectedRegion) {
      longjmp(tritonJmpBuf, 1);
    }
  };
  sigemptyset(&newSigabrt.sa_mask);
  newSigabrt.sa_flags = 0;
  sigaction(SIGABRT, &newSigabrt, &oldSigabrt);

  tritonInProtectedRegion = true;
  std::unique_ptr<llvm::Module> llvmModule;

  if (setjmp(tritonJmpBuf) != 0) {
    // longjmp from SIGABRT handler — LLVM assertion failed
    tritonInProtectedRegion = false;
    sigaction(SIGABRT, &oldSigabrt, nullptr);
    sd_printf("TritonTargetDispatch::compile: MLIR->LLVM translation hit assertion failure "
              "(recovered via SIGABRT handler)\n", "");
    return result;
  }

  llvmModule = mlir::translateModuleToLLVMIR(*moduleOp, llvmCtx);
  tritonInProtectedRegion = false;
  sigaction(SIGABRT, &oldSigabrt, nullptr);
  if (!llvmModule) {
    sd_printf("TritonTargetDispatch::compile: MLIR -> LLVM IR translation failed (parent retry)\n", "");
    return result;
  }

  // Verify the LLVM module
  std::string verifyErr;
  llvm::raw_string_ostream verifyOS(verifyErr);
  if (llvm::verifyModule(*llvmModule, &verifyOS)) {
    sd_printf("TritonTargetDispatch::compile: LLVM module verification failed: %s\n", verifyErr.c_str());
    return result;
  }

  // Phase 6: LLVM IR -> target ISA
  // Initialize LLVM targets
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  std::string triple;
  std::string proc;
  std::string features;

  switch (target) {
    case TritonGpuTarget::NVIDIA:
      triple = "nvptx64-nvidia-cuda";
      proc = (computeCapability == 90) ? "sm_90a" : ("sm_" + std::to_string(computeCapability));
      break;
    case TritonGpuTarget::AMD:
      triple = "amdgcn-amd-amdhsa";
      proc = cachedArch_;
      break;
    case TritonGpuTarget::INTEL:
      triple = "spir64-unknown-unknown";
      proc = "";
      break;
    default:
      return result;
  }

  llvmModule->setTargetTriple(triple);

  std::string lookupError;
  auto* llvmTarget = llvm::TargetRegistry::lookupTarget(triple, lookupError);
  if (!llvmTarget) {
    sd_printf("TritonTargetDispatch::compile: LLVM target lookup failed for '%s': %s\n",
              triple.c_str(), lookupError.c_str());
    return result;
  }

  llvm::TargetOptions targetOptions;
  auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
      llvmTarget->createTargetMachine(triple, proc, features,
                                       targetOptions, llvm::Reloc::PIC_));
  if (!targetMachine) {
    sd_printf("TritonTargetDispatch::compile: failed to create TargetMachine for %s/%s\n",
              triple.c_str(), proc.c_str());
    return result;
  }

  llvmModule->setDataLayout(targetMachine->createDataLayout());

  // Emit assembly (PTX text for NVIDIA, assembly for AMD)
  llvm::SmallString<0> asmBuffer;
  llvm::raw_svector_ostream asmStream(asmBuffer);
  llvm::legacy::PassManager codegenPM;

  if (targetMachine->addPassesToEmitFile(codegenPM, asmStream, nullptr,
                                          llvm::CodeGenFileType::AssemblyFile)) {
    sd_printf("TritonTargetDispatch::compile: TargetMachine can't emit assembly for %s\n",
              triple.c_str());
    return result;
  }

  codegenPM.run(*llvmModule);
  std::string asmOutput(asmBuffer.begin(), asmBuffer.end());

  if (asmOutput.empty()) {
    sd_printf("TritonTargetDispatch::compile: empty output for %s\n", cachedArch_.c_str());
    return result;
  }

  result.size = asmOutput.size();
  result.data = new char[result.size + 1];
  std::memcpy(result.data, asmOutput.data(), result.size);
  static_cast<char*>(result.data)[result.size] = '\0';

  sd_printf("TritonTargetDispatch::compile: generated %zu bytes for %s (%s)\n",
            result.size, cachedArch_.c_str(), triple.c_str());

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
