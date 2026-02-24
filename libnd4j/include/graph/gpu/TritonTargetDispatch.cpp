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
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>

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
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Bitcode/BitcodeReader.h>


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

  // ── SIGABRT protection for ALL compilation phases ──
  // Triton/LLVM passes can call abort() on assertion failures (e.g., unsupported ops,
  // invalid tensor encodings). We install a SIGABRT handler with setjmp/longjmp to
  // recover gracefully instead of crashing the JVM.
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

  // Capture TTIR module text before any passes (for diagnostics on compilation failure)
  std::string preDump;
  {
    llvm::raw_string_ostream os(preDump);
    moduleOp->print(os);
  }

  tritonInProtectedRegion = true;
  if (setjmp(tritonJmpBuf) != 0) {
    // longjmp from SIGABRT handler — assertion failed in some compilation phase
    tritonInProtectedRegion = false;
    sigaction(SIGABRT, &oldSigabrt, nullptr);
    // Clear any sticky CUDA errors that may have been set during the failed compilation.
    // Without this, ALL subsequent CUDA runtime calls fail (e.g., cudaMemGetInfo returns total=0).
#ifdef SD_CUDA
    cudaGetLastError();
#endif
    sd_printf("TritonTargetDispatch::compile: compilation hit assertion failure "
              "(recovered via SIGABRT handler). TTIR before passes:\n%.2000s\n", preDump.c_str());
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

    // Phase 3: Full TTGIR optimization pipeline (Triton NVIDIA backend)
    pm.addPass(mlir::triton::gpu::createTritonGPUCoalesce());
    pm.addPass(mlir::triton::gpu::createTritonGPUF32DotTC());
    pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeThreadLocality());
    pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmul());
    pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
    {
      mlir::triton::gpu::TritonGPUOptimizeDotOperandsOptions dotOpts;
      dotOpts.hoistLayoutConversion = true;
      pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands(dotOpts));
    }
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeAccumulatorInit());
    {
      mlir::triton::gpu::TritonGPUPipelineOptions pipeOpts;
      pipeOpts.numStages = numStages;
      pm.addPass(mlir::triton::gpu::createTritonGPUPipeline(pipeOpts));
    }
    pm.addPass(mlir::triton::gpu::createTritonGPUPrefetch());
    {
      mlir::triton::gpu::TritonGPUOptimizeDotOperandsOptions dotOpts2;
      dotOpts2.hoistLayoutConversion = true;
      pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands(dotOpts2));
    }
    pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(mlir::triton::gpu::createTritonGPUReduceDataDuplication());
    pm.addPass(mlir::triton::gpu::createTritonGPUReorderInstructions());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pm.run(*moduleOp))) {
      tritonInProtectedRegion = false;
      sigaction(SIGABRT, &oldSigabrt, nullptr);
#ifdef SD_CUDA
      cudaGetLastError();  // Clear sticky CUDA errors from failed pass pipeline
#endif
      // Dump full TTIR to file for diagnosis
      FILE* diagFile = fopen("/tmp/triton_ttir_dump.txt", "w");
      if (diagFile) {
        fprintf(diagFile, "%s", preDump.c_str());
        fclose(diagFile);
      }
      sd_printf("TritonTargetDispatch::compile: TTIR->TTGIR pass pipeline failed. "
                "TTIR dumped to /tmp/triton_ttir_dump.txt (%d bytes)\n",
                static_cast<int>(preDump.size()));
      return result;
    }

  }

  // Phase 4: TTGIR -> LLVM MLIR dialect
  // Pass order matches Triton NVIDIA backend (compiler.py lines 265-283)
  {
    mlir::PassManager pm(moduleOp->getContext());

    bool hasBackendLowering = false;
    switch (target) {
      case TritonGpuTarget::NVIDIA: {
#ifdef HAVE_TRITON_NVIDIA_PASSES
        // 1. Decompose unsupported layout conversions
        pm.addPass(mlir::triton::NVIDIA::createDecomposeUnsupportedConversionsPass());
        // 1b. Combine tensor select and if (matches Triton compiler.py line 274)
        pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
        // 2. SCF -> CF (must come BEFORE AllocateSharedMemory — membar needs cf dialect)
        pm.addPass(mlir::createConvertSCFToCFPass());
        // 3. Index -> LLVM
        pm.addPass(mlir::createConvertIndexToLLVMPass());
        // 4. Shared memory allocation (after SCF lowering)
        pm.addPass(mlir::triton::gpu::createAllocateSharedMemoryPass());
        // 5. TritonGPU -> LLVM
        pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(computeCapability));
        // 6. NVGPU -> LLVM
        pm.addPass(mlir::triton::createConvertNVGPUToLLVMPass());
        // 7. Arith -> LLVM
        pm.addPass(mlir::createArithToLLVMConversionPass());
        // 8. Math -> LLVM (lowering remaining math ops like exp, log, etc.)
        pm.addPass(mlir::createConvertMathToLLVMPass());
        // 9. ControlFlow -> LLVM (lowering remaining cf.br, cf.cond_br, etc.)
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        // 10. Func -> LLVM (lowering remaining func.func, func.call, etc.)
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        // 11. Reconcile unrealized casts (resolves leftover builtin.unrealized_conversion_cast)
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        // 12. Final cleanup
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createSymbolDCEPass());
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
      tritonInProtectedRegion = false;
      sigaction(SIGABRT, &oldSigabrt, nullptr);
#ifdef SD_CUDA
      cudaGetLastError();
#endif
      sd_printf("TritonTargetDispatch::compile: no backend lowering passes available for target\n", "");
      return result;
    }

    if (mlir::failed(pm.run(*moduleOp))) {
      tritonInProtectedRegion = false;
      sigaction(SIGABRT, &oldSigabrt, nullptr);
#ifdef SD_CUDA
      cudaGetLastError();  // Clear sticky CUDA errors from failed TTGIR->LLVM lowering
#endif
      std::string mlirDump;
      llvm::raw_string_ostream mlirOS(mlirDump);
      moduleOp->print(mlirOS);
      FILE* diagFile = fopen("/tmp/triton_mlir_dump.txt", "a");
      if (diagFile) {
        fprintf(diagFile, "=== PHASE 4 FAILED ===\n%s\n=== END ===\n", mlirDump.c_str());
        fclose(diagFile);
      }
      fprintf(stderr, "TritonTargetDispatch::compile: TTGIR->LLVM pass pipeline failed. "
              "Module dumped to /tmp/triton_mlir_dump.txt\n");
      return result;
    }
  }

  // Phase 5: MLIR LLVM dialect -> LLVM IR module
  // Verify the MLIR module before attempting LLVM translation
  if (mlir::failed(mlir::verify(*moduleOp))) {
    tritonInProtectedRegion = false;
    sigaction(SIGABRT, &oldSigabrt, nullptr);
#ifdef SD_CUDA
    cudaGetLastError();
#endif
    sd_printf("TritonTargetDispatch::compile: MLIR module verification failed after lowering\n", "");
    return result;
  }

  // Register ALL dialect translation interfaces — builtin.module, NVVM, GPU, etc.
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  moduleOp->getContext()->appendDialectRegistry(registry);
  llvm::LLVMContext llvmCtx;

  std::unique_ptr<llvm::Module> llvmModule;

  llvmModule = mlir::translateModuleToLLVMIR(*moduleOp, llvmCtx);
  tritonInProtectedRegion = false;
  sigaction(SIGABRT, &oldSigabrt, nullptr);
  if (!llvmModule) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors from failed translation
#endif
    std::string postLowerDump;
    llvm::raw_string_ostream postOS(postLowerDump);
    moduleOp->print(postOS);
    // Write to file since sd_printf may be swallowed by surefire
    FILE* diagFile = fopen("/tmp/triton_mlir_dump.txt", "a");
    if (diagFile) {
      fprintf(diagFile, "=== TRANSLATION FAILED ===\n%s\n=== END ===\n", postLowerDump.c_str());
      fclose(diagFile);
    }
    fprintf(stderr, "TritonTargetDispatch::compile: MLIR -> LLVM IR translation failed. "
            "Post-lowering module dumped to /tmp/triton_mlir_dump.txt\n");
    return result;
  }

  // Phase 5b: Link libdevice for NVIDIA math intrinsics (__nv_sqrtf, __nv_expf, etc.)
  // The math-to-LLVM pass lowers math.sqrt/exp/log/etc. to calls to __nv_* functions
  // which are defined in NVIDIA's libdevice bitcode library.
  if (target == TritonGpuTarget::NVIDIA) {
    // Search for libdevice.10.bc in common locations
    std::vector<std::string> libdevicePaths = {
      "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-12.9/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-12.6/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-12.4/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-12.2/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-12.0/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-11.8/nvvm/libdevice/libdevice.10.bc",
    };

    // Also check CUDA_PATH environment variable
    if (const char* cudaPath = std::getenv("CUDA_PATH")) {
      libdevicePaths.insert(libdevicePaths.begin(),
          std::string(cudaPath) + "/nvvm/libdevice/libdevice.10.bc");
    }

    bool linked = false;
    for (const auto& path : libdevicePaths) {
      auto bufOrErr = llvm::MemoryBuffer::getFile(path);
      if (!bufOrErr) continue;

      auto libdeviceModOrErr = llvm::parseBitcodeFile(
          bufOrErr.get()->getMemBufferRef(), llvmCtx);
      if (!libdeviceModOrErr) {
        llvm::consumeError(libdeviceModOrErr.takeError());
        continue;
      }

      // Set target triple to match the main module
      (*libdeviceModOrErr)->setTargetTriple(llvmModule->getTargetTriple());
      (*libdeviceModOrErr)->setDataLayout(llvmModule->getDataLayout());

      if (llvm::Linker::linkModules(*llvmModule, std::move(*libdeviceModOrErr),
                                     llvm::Linker::Flags::LinkOnlyNeeded)) {
        sd_printf("TritonTargetDispatch::compile: failed to link libdevice from %s\n", path.c_str());
        continue;
      }

      sd_printf("TritonTargetDispatch::compile: linked libdevice from %s\n", path.c_str());
      linked = true;
      break;
    }

    if (!linked) {
      sd_printf("TritonTargetDispatch::compile: WARNING — libdevice.10.bc not found, "
                "math intrinsics (__nv_sqrtf etc.) will be unresolved\n", "");
    }
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
      // NVIDIA: CUDA Driver API for PTX loading with JIT error logging.
      CUmodule module = nullptr;
      char jitErrorLog[4096] = {0};
      char jitInfoLog[4096] = {0};
      CUjit_option jitOptions[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
      };
      void* jitOptionValues[] = {
        jitErrorLog,
        reinterpret_cast<void*>(sizeof(jitErrorLog)),
        jitInfoLog,
        reinterpret_cast<void*>(sizeof(jitInfoLog))
      };
      CUresult res = cuModuleLoadDataEx(&module, binary.data,
                                         4, jitOptions, jitOptionValues);
      if (res != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(res, &errStr);
        sd_printf("TritonTargetDispatch::loadModule: cuModuleLoadDataEx failed: %s\n"
                  "  JIT error: %s\n  JIT info: %s\n  PTX (first 2000 chars): %.2000s\n",
                  errStr ? errStr : "unknown", jitErrorLog, jitInfoLog,
                  static_cast<const char*>(binary.data));
        FILE* df = fopen("/tmp/triton_launch_diag.txt", "a");
        if (df) {
          fprintf(df, "cuModuleLoadDataEx_FAIL: res=%d err=%s jitErr=%s jitInfo=%s binarySize=%zu\n",
                  (int)res, errStr ? errStr : "unknown", jitErrorLog, jitInfoLog, binary.size);
          fflush(df);
          fclose(df);
        }
        // Dump full PTX for debugging
        FILE* ptxDump = fopen("/tmp/triton_ptx_dump.ptx", "w");
        if (ptxDump) {
          fwrite(binary.data, 1, binary.size, ptxDump);
          fclose(ptxDump);
        }
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
        sd_printf("TritonTargetDispatch::launchKernel: cuLaunchKernel failed: %s (code=%d) "
                  "grid=(%u,%u,%u) block=(%u,%u,%u) sharedMem=%u\n",
                  errStr ? errStr : "unknown", (int)res,
                  gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes);
        FILE* df = fopen("/tmp/triton_launch_diag.txt", "a");
        if (df) {
          fprintf(df, "STD_LAUNCH_FAIL: %s (code=%d) grid=(%u,%u,%u) block=(%u,%u,%u) sharedMem=%u\n",
                  errStr ? errStr : "unknown", (int)res,
                  gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes);
          fflush(df); fclose(df);
        }
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

// ─── Cooperative kernel launch ───────────────────────────────────────────────

bool TritonTargetDispatch::launchCooperativeKernel(void* kernelFunc,
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
      // cudaLaunchCooperativeKernel requires a void* function pointer (host symbol).
      // We have a CUfunction (driver API handle). Use the driver API equivalent:
      // cuLaunchCooperativeKernel (CUDA 9.0+).
      CUresult res = cuLaunchCooperativeKernel(
          static_cast<CUfunction>(kernelFunc),
          gridX, gridY, gridZ,
          blockX, blockY, blockZ,
          sharedMemBytes,
          static_cast<CUstream>(stream),
          args);
      if (res != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(res, &errStr);
        sd_printf("TritonTargetDispatch::launchCooperativeKernel: cuLaunchCooperativeKernel failed: %s (code=%d) "
                  "grid=(%u,%u,%u) block=(%u,%u,%u) sharedMem=%u\n",
                  errStr ? errStr : "unknown", (int)res,
                  gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes);
        FILE* df = fopen("/tmp/triton_launch_diag.txt", "a");
        if (df) {
          fprintf(df, "COOP_LAUNCH_FAIL: %s (code=%d) grid=(%u,%u,%u) block=(%u,%u,%u) sharedMem=%u\n",
                  errStr ? errStr : "unknown", (int)res,
                  gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes);
          fflush(df); fclose(df);
        }
        return false;
      }
      return true;
#else
      return false;
#endif
    }

    case TritonGpuTarget::AMD:
    case TritonGpuTarget::INTEL:
      // Cooperative launch not supported on AMD/Intel via this path.
      // Fall back to standard launch (no grid sync barriers).
      sd_printf("TritonTargetDispatch::launchCooperativeKernel: cooperative launch not supported on "
                "target %d, falling back to standard launch\n", static_cast<int>(target));
      return launchKernel(kernelFunc, gridX, gridY, gridZ, blockX, blockY, blockZ,
                          sharedMemBytes, stream, args, numArgs);

    default:
      sd_printf("TritonTargetDispatch::launchCooperativeKernel: unsupported target %d\n",
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
