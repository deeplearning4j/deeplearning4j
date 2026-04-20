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
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>
#include <system/Environment.h>
#include <system/common.h>

#include <cstring>
#include <csignal>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <setjmp.h>
#include <thread>

// ─── Platform GPU headers ───────────────────────────────────────────────────
//
//  Under ZLUDA builds, SD_CUDA is defined even though the real GPU
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
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#if __has_include(<mlir/Pass/PassInstrumentation.h>)
#include <mlir/Pass/PassInstrumentation.h>
#define SD_TRITON_HAS_PASS_INSTRUMENTATION 1
#endif
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
#include <mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h>
#include <mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h>

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
#include <llvm/TargetParser/Triple.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Bitcode/BitcodeReader.h>


// Triton dialect passes — TTIR -> TTGIR -> LLVM
#include <triton/Conversion/TritonToTritonGPU/Passes.h>
#include <triton/Conversion/TritonGPUToLLVM/Passes.h>
#include <triton/Dialect/TritonGPU/Transforms/Passes.h>
#include <triton/Target/LLVMIR/Passes.h>

// Triton NVIDIA GPU dialect transforms (tensor memory, proxy fence, etc.)
#if __has_include(<triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h>)
#include <triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h>
#define HAVE_TRITON_NVIDIA_GPU_DIALECT 1
#endif

// Gluon dialect (inliner pass, new in Triton 3.6.0)
#if __has_include(<triton/Dialect/Gluon/Transforms/Passes.h>)
#include <triton/Dialect/Gluon/Transforms/Passes.h>
#define HAVE_TRITON_GLUON 1
#endif

// NVIDIA backend passes (when building with NVIDIA codegen backend)
#if __has_include(<TritonNVIDIAGPUToLLVM/Passes.h>)
#include <TritonNVIDIAGPUToLLVM/Passes.h>
#define HAVE_TRITON_NVIDIA_PASSES 1
#endif

// NVGPU to LLVM pass
#if __has_include(<NVGPUToLLVM/Passes.h>)
#include <NVGPUToLLVM/Passes.h>
#endif

// AMD backend passes (when building with AMD codegen backend)
#if __has_include(<TritonAMDGPUToLLVM/Passes.h>)
#include <TritonAMDGPUToLLVM/Passes.h>
#define HAVE_TRITON_AMD_PASSES 1
#endif

namespace sd {
namespace graph {

namespace {

std::string canonicalizeAmdArch(const std::string& arch) {
  if (arch.empty()) return arch;
  size_t suffixPos = arch.find(':');
  if (suffixPos == std::string::npos) return arch;
  return arch.substr(0, suffixPos);
}

int getModuleGlobalScratchMemoryBytes(mlir::ModuleOp* moduleOp) {
  if (moduleOp == nullptr || !*moduleOp) return 0;
  auto scratchAttr =
      moduleOp->getOperation()->getAttrOfType<mlir::IntegerAttr>("ttg.global_scratch_memory_size");
  if (!scratchAttr) return 0;
  int64_t scratchBytes = scratchAttr.getInt();
  if (scratchBytes <= 0) return 0;
  return static_cast<int>(std::min(scratchBytes, static_cast<int64_t>(std::numeric_limits<int>::max())));
}

int getModuleGlobalScratchAlignment(mlir::ModuleOp* moduleOp) {
  if (moduleOp == nullptr || !*moduleOp) return 128;
  auto alignAttr =
      moduleOp->getOperation()->getAttrOfType<mlir::IntegerAttr>("ttg.global_scratch_memory_alignment");
  if (!alignAttr) return 128;
  int64_t align = alignAttr.getInt();
  return align > 0 ? static_cast<int>(align) : 128;
}

int getModuleSharedMemoryBytes(mlir::ModuleOp* moduleOp) {
  if (moduleOp == nullptr || !*moduleOp) return 0;

  // Triton 3.6.0 uses "ttg.shared", older versions used "triton_gpu.shared"
  auto sharedAttr =
      moduleOp->getOperation()->getAttrOfType<mlir::IntegerAttr>("ttg.shared");
  if (!sharedAttr) {
    sharedAttr = moduleOp->getOperation()->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared");
  }
  if (!sharedAttr) return 0;

  int64_t sharedBytes = sharedAttr.getInt();
  if (sharedBytes <= 0) return 0;
  if (sharedBytes > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(sharedBytes);
}

const char* tritonTargetName(TritonGpuTarget target) {
  switch (target) {
    case TritonGpuTarget::NVIDIA:
      return "NVIDIA";
    case TritonGpuTarget::AMD:
      return "AMD";
    case TritonGpuTarget::INTEL:
      return "INTEL";
    default:
      return "UNKNOWN";
  }
}

long long nextTritonCompileId() {
  static std::atomic<long long> nextId{0};
  return nextId.fetch_add(1) + 1;
}

long long elapsedMsSince(const std::chrono::steady_clock::time_point& start) {
  return static_cast<long long>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start).count());
}

// RAII guard that emits DSP_DIAG START on construction and DONE (with elapsed time)
// on destruction. Consolidates the 4 compile-phase START/DONE pairs.
struct DspCompilePhaseGuard {
  long long compileId_;
  const char* phaseName_;
  std::chrono::steady_clock::time_point start_;
  uint32_t category_;

  DspCompilePhaseGuard(long long id, const char* phase, uint32_t cat)
      : compileId_(id), phaseName_(phase),
        start_(std::chrono::steady_clock::now()), category_(cat) {
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(category_)) {
      sd::graph::DspDiagnostics::getInstance().recordEvent(
          category_, -1, -1, -1, nullptr, 0,
          "TritonTargetDispatch::compile[%lld]: phase=%s START",
          compileId_, phaseName_);
    }
  }

  ~DspCompilePhaseGuard() {
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(category_)) {
      sd::graph::DspDiagnostics::getInstance().recordEvent(
          category_, -1, -1, -1, nullptr, 0,
          "TritonTargetDispatch::compile[%lld]: phase=%s DONE elapsedMs=%lld",
          compileId_, phaseName_, elapsedMsSince(start_));
    }
  }
};

#ifdef SD_TRITON_HAS_PASS_INSTRUMENTATION
class TritonPassProgressInstrumentation final : public mlir::PassInstrumentation {
 public:
  TritonPassProgressInstrumentation(long long compileId, const char* pipelineTag)
      : compileId_(compileId), pipelineTag_(pipelineTag) {
    heartbeatThread_ = std::thread([this]() {
      std::unique_lock<std::mutex> lock(heartbeatMutex_);
      while (!stopHeartbeat_) {
        heartbeatCv_.wait_for(lock, std::chrono::seconds(heartbeatIntervalSec_));
        if (stopHeartbeat_ || !passRunning_) continue;
        const long long passCounter = passCounter_;
        const std::string passName = currentPassName_;
        const auto passStart = currentPassStart_;
        lock.unlock();
        DSP_DIAG(COMPILE, "TritonTargetDispatch::compile[%lld]: %s pass#%lld HEARTBEAT %s elapsedMs=%lld",
                  compileId_, pipelineTag_, passCounter, passName.c_str(),
                  elapsedMsSince(passStart));
        lock.lock();
      }
    });
  }

  ~TritonPassProgressInstrumentation() override {
    {
      std::lock_guard<std::mutex> lock(heartbeatMutex_);
      stopHeartbeat_ = true;
    }
    heartbeatCv_.notify_all();
    if (heartbeatThread_.joinable()) {
      heartbeatThread_.join();
    }
  }

  void runBeforePass(mlir::Pass* pass, mlir::Operation*) override {
    if (pass == nullptr) return;
    long long passCounter = 0;
    std::string passName;
    {
      std::lock_guard<std::mutex> lock(heartbeatMutex_);
      passCounter_++;
      currentPassName_ = pass->getName().str();
      currentPassStart_ = std::chrono::steady_clock::now();
      passRunning_ = true;
      passCounter = passCounter_;
      passName = currentPassName_;
    }
    DSP_DIAG(COMPILE, "TritonTargetDispatch::compile[%lld]: %s pass#%lld START %s",
              compileId_, pipelineTag_, passCounter, passName.c_str());
  }

  void runAfterPass(mlir::Pass* pass, mlir::Operation*) override {
    if (pass == nullptr) return;
    long long passCounter = 0;
    std::string passName;
    std::chrono::steady_clock::time_point passStart;
    {
      std::lock_guard<std::mutex> lock(heartbeatMutex_);
      passCounter = passCounter_;
      passName = currentPassName_;
      passStart = currentPassStart_;
      passRunning_ = false;
    }
    heartbeatCv_.notify_all();
    DSP_DIAG(COMPILE, "TritonTargetDispatch::compile[%lld]: %s pass#%lld DONE %s elapsedMs=%lld",
              compileId_, pipelineTag_, passCounter, passName.c_str(),
              elapsedMsSince(passStart));
  }

  void runAfterPassFailed(mlir::Pass* pass, mlir::Operation*) override {
    if (pass == nullptr) return;
    long long passCounter = 0;
    std::string passName;
    std::chrono::steady_clock::time_point passStart;
    {
      std::lock_guard<std::mutex> lock(heartbeatMutex_);
      passCounter = passCounter_;
      passName = currentPassName_;
      passStart = currentPassStart_;
      passRunning_ = false;
    }
    heartbeatCv_.notify_all();
    DSP_DIAG(COMPILE, "TritonTargetDispatch::compile[%lld]: %s pass#%lld FAILED %s elapsedMs=%lld",
              compileId_, pipelineTag_, passCounter, passName.c_str(),
              elapsedMsSince(passStart));
  }

 private:
  long long compileId_;
  const char* pipelineTag_;
  static constexpr int heartbeatIntervalSec_ = 30;
  std::thread heartbeatThread_;
  std::mutex heartbeatMutex_;
  std::condition_variable heartbeatCv_;
  bool stopHeartbeat_ = false;
  bool passRunning_ = false;
  long long passCounter_ = 0;
  std::string currentPassName_;
  std::chrono::steady_clock::time_point currentPassStart_;
};
#endif

static thread_local jmp_buf tritonJmpBuf;
static thread_local bool tritonInProtectedRegion = false;

void tritonSigabrtHandler(int) {
  if (tritonInProtectedRegion) {
    longjmp(tritonJmpBuf, 1);
  }
}

std::mutex& tritonSigabrtMutex() {
  static std::mutex mtx;
  return mtx;
}

int& tritonSigabrtRefCount() {
  static int refCount = 0;
  return refCount;
}

struct sigaction& tritonSigabrtPreviousAction() {
  static struct sigaction action {};
  return action;
}

class TritonSigabrtInstallGuard {
 public:
  TritonSigabrtInstallGuard() {
    std::lock_guard<std::mutex> lock(tritonSigabrtMutex());
    int& refCount = tritonSigabrtRefCount();
    if (refCount == 0) {
      struct sigaction newSigabrt;
      std::memset(&newSigabrt, 0, sizeof(newSigabrt));
      newSigabrt.sa_handler = tritonSigabrtHandler;
      sigemptyset(&newSigabrt.sa_mask);
      newSigabrt.sa_flags = 0;
      sigaction(SIGABRT, &newSigabrt, &tritonSigabrtPreviousAction());
    }
    refCount++;
  }

  ~TritonSigabrtInstallGuard() {
    std::lock_guard<std::mutex> lock(tritonSigabrtMutex());
    int& refCount = tritonSigabrtRefCount();
    if (refCount <= 0) return;
    refCount--;
    if (refCount == 0) {
      sigaction(SIGABRT, &tritonSigabrtPreviousAction(), nullptr);
    }
  }

  TritonSigabrtInstallGuard(const TritonSigabrtInstallGuard&) = delete;
  TritonSigabrtInstallGuard& operator=(const TritonSigabrtInstallGuard&) = delete;
};

}  // namespace

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
  static std::mutex detectTargetMutex;
  std::lock_guard<std::mutex> lock(detectTargetMutex);
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
      std::string archName = canonicalizeAmdArch(props.gcnArchName);
      if (!archName.empty() && archName.find("gfx") != std::string::npos) {
        cachedArch_ = archName;
        cachedTarget_ = TritonGpuTarget::AMD;
        DSP_DIAG(BACKEND, "TritonTargetDispatch: detected AMD GPU '%s' via HIP, arch=%s",
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
            DSP_DIAG(BACKEND, "TritonTargetDispatch: detected Intel GPU '%s' via Level Zero, arch=%s",
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
        DSP_DIAG(BACKEND, "TritonTargetDispatch: detected AMD GPU '%s' via CUDA (ZLUDA fallback), arch=%s",
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
        DSP_DIAG(BACKEND, "TritonTargetDispatch: detected Intel GPU '%s' via CUDA (ZLUDA fallback), arch=%s",
                  props.name, cachedArch_.c_str());
        return cachedTarget_;
      }
#endif

      // Native NVIDIA GPU
      cachedArch_ = "sm_" + std::to_string(props.major * 10 + props.minor);
      cachedTarget_ = TritonGpuTarget::NVIDIA;
      DSP_DIAG(BACKEND, "TritonTargetDispatch: detected NVIDIA GPU '%s', arch=%s",
                props.name, cachedArch_.c_str());
      return cachedTarget_;
    }
  }
#endif

  DSP_DIAG(BACKEND, "TritonTargetDispatch: no supported GPU target detected");
  cachedTarget_ = TritonGpuTarget::UNKNOWN;
  return cachedTarget_;
}

std::string TritonTargetDispatch::getTargetArch() {
  detectTarget();
#ifdef SD_CUDA
  // Return the arch for the CURRENT device, not just device 0.
  // Multi-GPU systems may have different SM versions (e.g., sm_89 + sm_75).
  // cachedArch_ is always set from device 0; using it for device 1 causes
  // PTX JIT failures ("SM version specified by .target is higher than default").
  if (cachedTarget_ == TritonGpuTarget::NVIDIA) {
    int currentDevice = 0;
    cudaGetDevice(&currentDevice);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, currentDevice);
    return "sm_" + std::to_string(props.major * 10 + props.minor);
  }
#endif
  return cachedArch_;
}

// ─── Compilation ────────────────────────────────────────────────────────────

// TTIR -> PTX compilation pipeline.
//
// Thread safety: Each call uses its own stack-local state (LLVMContext,
// TargetMachine, PassManager) and operates on a per-sub-segment MLIRContext
// created in TritonIRBuilder::buildModule(). Shared global state is protected
// by targeted mutexes:
//   - LLVM target init:           std::once_flag (llvmInitFlag)
//   - MLIR dialect registry:      mlirRegistryMtx (phase 4)
//   - MLIR translation registry:  mlirTranslationMtx (phase 5)
//   - MLIR context creation:      getMlirContextMutex() (TritonIRBuilder)
//   - cuModuleLoadDataEx:         loadModuleMtx (loadModule())
//
// Multiple compile() calls can safely run in parallel from the worker
// thread pool in TritonGraphBackend_compile.cu.
TritonCompiledBinary TritonTargetDispatch::compile(void* mlirModule, int numWarps, int numStages) {
  // Global compile mutex: LLVM's code generation pipeline (TargetMachine creation,
  // PassManager, MCCodeEmitter) has thread-unsafe global state that causes heap
  // corruption ("malloc_consolidate(): invalid chunk size") under concurrent access.
  // Per-thread LLVMContext isolation is insufficient — LLVM's target registry,
  // pass registry, and MCInst pools have shared mutable state.
  // Serializing compile() is the correct fix. The outer loop in
  // NativeDynamicShapePlan_cuda.cu still runs 8 parallel threads for segment
  // dispatch, but actual LLVM codegen is serialized here.
  static std::mutex compileMtx;
  std::lock_guard<std::mutex> compileLock(compileMtx);

  TritonCompiledBinary result = {nullptr, 0, TritonGpuTarget::UNKNOWN, "", 0, 0, 0, 128};
  auto& env = sd::Environment::getInstance();
  const bool tritonVerbose = env.tritonVerbose();
  const long long compileId = nextTritonCompileId();
  const auto compileStart = std::chrono::steady_clock::now();

  auto target = detectTarget();
  if (target == TritonGpuTarget::UNKNOWN) {
    DSP_DIAG(COMPILE, "TritonTargetDispatch::compile[%lld]: no GPU target available", compileId);
    return result;
  }

  const std::string archOverride = env.tritonOverrideArch();
  std::string targetArch = getTargetArch();  // per-device arch, not cached device 0
  if (!archOverride.empty()) {
    targetArch = archOverride;
    DSP_DIAG(COMPILE, "TritonTargetDispatch::compile: using architecture override '%s'",
              targetArch.c_str());
  }
  if (target == TritonGpuTarget::AMD) {
    targetArch = canonicalizeAmdArch(targetArch);
  }

  int numCTAs = std::max(1, env.tritonNumCTAs());
  int maxNreg = std::max(0, env.tritonMaxNreg());

  DSP_DIAG(COMPILE, "TritonTargetDispatch::compile[%lld]: START target=%s arch=%s "
            "warps=%d stages=%d numCTAs=%d maxNreg=%d verbose=%d",
            compileId, tritonTargetName(target), targetArch.c_str(),
            numWarps, numStages, numCTAs, maxNreg, tritonVerbose ? 1 : 0);

  result.target = target;
  result.targetArch = targetArch;
  result.numWarps = numWarps;

  auto moduleOp = static_cast<mlir::ModuleOp*>(mlirModule);
  if (!moduleOp || !*moduleOp) {
    DSP_DIAG(COMPILE, "TritonTargetDispatch::compile: null MLIR module");
    return result;
  }
  if (maxNreg > 0) {
    auto i32Type = mlir::IntegerType::get(moduleOp->getContext(), 32);
    moduleOp->getOperation()->setAttr("ttg.maxnreg",
                                      mlir::IntegerAttr::get(i32Type, maxNreg));
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
  int ptxVersion = 0;  // PTX ISA version, derived from compute capability

  switch (target) {
    case TritonGpuTarget::NVIDIA: {
      std::string digits;
      for (char c : targetArch) {
        if (std::isdigit(static_cast<unsigned char>(c))) {
          digits.push_back(c);
        }
      }
      if (digits.empty()) {
        for (char c : cachedArch_) {
          if (std::isdigit(static_cast<unsigned char>(c))) {
            digits.push_back(c);
          }
        }
      }
      if (!digits.empty()) {
        computeCapability = std::stoi(digits);
      }
      if (computeCapability <= 0) {
        DSP_DIAG(COMPILE, "TritonTargetDispatch::compile: failed to derive NVIDIA compute capability "
                  "from targetArch='%s'",
                  targetArch.c_str());
        return result;
      }
      if (numCTAs > 1 && computeCapability < 90) {
        DSP_DIAG(COMPILE, "TritonTargetDispatch::compile: numCTAs=%d requested on sm_%d; "
                  "clamping to 1 (multi-CTA requires SM90+)",
                  numCTAs, computeCapability);
        numCTAs = 1;
      }
      // Derive PTX ISA version from compute capability.
      // This must match the minimum PTX version that supports the target SM.
      if (computeCapability >= 100) ptxVersion = 86;
      else if (computeCapability >= 90) ptxVersion = 80;
      else if (computeCapability >= 89) ptxVersion = 78;
      else if (computeCapability >= 86) ptxVersion = 71;
      else if (computeCapability >= 80) ptxVersion = 70;
      else if (computeCapability >= 75) ptxVersion = 63;
      else if (computeCapability >= 70) ptxVersion = 60;
      else ptxVersion = 50;  // fallback for older GPUs
      targetStr = "cuda:" + std::to_string(computeCapability);
      break;
    }
    case TritonGpuTarget::AMD: {
      targetStr = "hip:" + targetArch;
      break;
    }
    case TritonGpuTarget::INTEL: {
      targetStr = "xpu:" + targetArch;
      break;
    }
    default:
      return result;
  }

  // ── SIGABRT protection for ALL compilation phases ──
  // Triton/LLVM passes can call abort() on assertion failures (e.g., unsupported ops,
  // invalid tensor encodings). Install a process-level handler while compiles are active
  // so multiple compile workers can run in parallel without restoring handlers out of order.
  TritonSigabrtInstallGuard sigabrtGuard;

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
    // Clear any sticky CUDA errors that may have been set during the failed compilation.
    // Without this, ALL subsequent CUDA runtime calls fail (e.g., cudaMemGetInfo returns total=0).
#ifdef SD_CUDA
    cudaGetLastError();
#endif
    DSP_DIAG(FALLBACK, "TritonTargetDispatch::compile[%lld]: compilation hit assertion failure "
              "(recovered via SIGABRT handler). TTIR before passes:\n%.2000s",
              compileId, preDump.c_str());
    return result;
  }

  // Pre-pass SplatOp type validation (diagnostic for 'tt.splat' type mismatch errors)
  {
    bool splatTypeError = false;
    moduleOp->walk([&](mlir::triton::SplatOp op) {
      auto scalarType = op.getSrc().getType();
      auto resultTensorType = mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
      if (resultTensorType) {
        auto elemType = resultTensorType.getElementType();
        if (scalarType != elemType) {
          std::string scalarStr, elemStr, opStr;
          llvm::raw_string_ostream scalarOS(scalarStr), elemOS(elemStr), opOS(opStr);
          scalarType.print(scalarOS);
          elemType.print(elemOS);
          op->print(opOS);
          DSP_DIAG(COMPILE, "SPLAT TYPE MISMATCH: scalar=%s, tensor_elem=%s  op: %s",
                    scalarStr.c_str(), elemStr.c_str(), opStr.c_str());
          // Also dump to file for full diagnostics
          FILE* f = fopen("/tmp/triton_splat_mismatch.txt", "a");
          if (f) {
            fprintf(f, "MISMATCH: scalar=%s tensor_elem=%s\n  %s\n",
                    scalarStr.c_str(), elemStr.c_str(), opStr.c_str());
            fclose(f);
          }
          splatTypeError = true;
        }
      }
    });
    if (splatTypeError) {
      // Dump full TTIR for diagnosis
      FILE* diagFile = fopen("/tmp/triton_ttir_splat_error.txt", "w");
      if (diagFile) {
        fprintf(diagFile, "%s", preDump.c_str());
        fclose(diagFile);
      }
      DSP_DIAG(COMPILE, "TritonTargetDispatch::compile[%lld]: SplatOp type mismatch detected! "
                "Full TTIR dumped to /tmp/triton_ttir_splat_error.txt", compileId);
    }
  }

  // Phase 1-2: TTIR -> TTGIR
  { // DspCompilePhaseGuard scope: emits TTIR_TO_TTGIR START/DONE automatically
    DspCompilePhaseGuard phase12Guard(compileId, "TTIR_TO_TTGIR", sd::graph::DSP_DIAG_COMPILE);
    {
    mlir::PassManager pm(moduleOp->getContext());
    if (tritonVerbose) {
#ifdef SD_TRITON_HAS_PASS_INSTRUMENTATION
      pm.addInstrumentation(
          std::make_unique<TritonPassProgressInstrumentation>(compileId, "TTIR_TO_TTGIR"));
#endif
    }

    // Phase 1: TTIR optimizations
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // Phase 2: TTIR -> TTGIR
    {
      mlir::triton::ConvertTritonToTritonGPUOptions ttirOpts;
      ttirOpts.target = targetStr;
      ttirOpts.numWarps = numWarps;
      ttirOpts.threadsPerWarp = 32;
      ttirOpts.numCTAs = numCTAs;
      pm.addPass(mlir::triton::createConvertTritonToTritonGPU(ttirOpts));
    }

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
#ifdef SD_CUDA
      cudaGetLastError();  // Clear sticky CUDA errors from failed pass pipeline
#endif
      // Dump full TTIR to file for diagnosis
      FILE* diagFile = fopen("/tmp/triton_ttir_dump.txt", "w");
      if (diagFile) {
        fprintf(diagFile, "%s", preDump.c_str());
        fclose(diagFile);
      }
      DSP_DIAG(COMPILE, "TritonTargetDispatch::compile: TTIR->TTGIR pass pipeline failed. "
                "TTIR dumped to /tmp/triton_ttir_dump.txt (%d bytes)",
                static_cast<int>(preDump.size()));
      return result;
    }

    }
  } // phase12Guard destructor fires here, emitting TTIR_TO_TTGIR DONE

  // Phase 4: TTGIR -> LLVM MLIR dialect
  // Pass order matches Triton 3.6.0 NVIDIA backend (compiler.py make_llir)
  { // DspCompilePhaseGuard scope: emits TTGIR_TO_LLVM_DIALECT START/DONE automatically
    DspCompilePhaseGuard phase4Guard(compileId, "TTGIR_TO_LLVM_DIALECT", sd::graph::DSP_DIAG_COMPILE);
    {
    // Register LLVM dialect inliner interface (required by GluonInline pass in 3.6.0)
    // Thread-safety: appendDialectRegistry modifies MLIR context global state.
    // Must be serialized across concurrent compilation threads.
    {
      static std::mutex mlirRegistryMtx;
      std::lock_guard<std::mutex> lock(mlirRegistryMtx);
      mlir::DialectRegistry phase4Registry;
      mlir::LLVM::registerInlinerInterface(phase4Registry);
      moduleOp->getContext()->appendDialectRegistry(phase4Registry);
    }
    mlir::PassManager pm(moduleOp->getContext());
    if (tritonVerbose) {
#ifdef SD_TRITON_HAS_PASS_INSTRUMENTATION
      pm.addInstrumentation(
          std::make_unique<TritonPassProgressInstrumentation>(compileId, "TTGIR_TO_LLVM_DIALECT"));
#endif
    }

    bool hasBackendLowering = false;
    switch (target) {
      case TritonGpuTarget::NVIDIA: {
#ifdef HAVE_TRITON_NVIDIA_PASSES
        // Triton 3.6.0 NVIDIA backend pass pipeline (matches compiler.py make_llir)
        // 1. Combine tensor select and if
        pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
        // 2. Allocate warp groups (new in 3.6.0)
        pm.addPass(mlir::triton::gpu::createTritonGPUAllocateWarpGroups());
        // 3. SCF -> CF (must come BEFORE AllocateSharedMemory — membar needs cf dialect)
        pm.addPass(mlir::createSCFToControlFlowPass());
        // 4. Gluon inliner (new in 3.6.0)
#ifdef HAVE_TRITON_GLUON
        pm.addPass(mlir::triton::gluon::createGluonInline());
#else
        pm.addPass(mlir::createInlinerPass());
#endif
        // 5. Allocate shared memory NV (renamed in 3.6.0, takes capability + PTX version)
        pm.addPass(mlir::triton::createAllocateSharedMemoryNvPass(computeCapability, ptxVersion));
        // 6. Allocate tensor memory (new in 3.6.0)
#ifdef HAVE_TRITON_NVIDIA_GPU_DIALECT
        pm.addPass(mlir::triton::nvidia_gpu::createTritonTensorMemoryAllocationPass());
#endif
        // 7. Allocate global scratch memory (new in 3.6.0)
        pm.addPass(mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass());
        // 8. Proxy fence insertion (new in 3.6.0)
#ifdef HAVE_TRITON_NVIDIA_GPU_DIALECT
        {
          mlir::triton::nvidia_gpu::TritonGPUProxyFenceInsertionOptions fenceOpts;
          fenceOpts.computeCapability = computeCapability;
          pm.addPass(mlir::triton::nvidia_gpu::createTritonGPUProxyFenceInsertion(fenceOpts));
        }
#endif
        // 9. TritonGPU -> LLVM
        pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(computeCapability, ptxVersion));
        // 10. Canonicalize + CSE
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        // 11. NVGPU -> LLVM
        pm.addPass(mlir::triton::createConvertNVGPUToLLVM());
        // 12. Warp specialize to LLVM (new in 3.6.0)
        pm.addPass(mlir::triton::createConvertWarpSpecializeToLLVM());
        // 13. Final cleanup
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createSymbolDCEPass());
        // 14. NVVM -> LLVM (replaces arith+math+cf to LLVM in 3.6.0)
        pm.addPass(mlir::createConvertNVVMToLLVMPass());
        hasBackendLowering = true;
#else
        DSP_DIAG(COMPILE, "TritonTargetDispatch::compile: NVIDIA backend passes not available "
                  "(TritonNVIDIAGPUToLLVM not found at build time)");
#endif
        break;
      }
      case TritonGpuTarget::AMD: {
#ifdef HAVE_TRITON_AMD_PASSES
        // Phase order mirrors Triton's AMD backend make_llir() flow.
        const bool hipFtz = true;
        pm.addPass(mlir::triton::AMD::createDecomposeUnsupportedConversionsPass(targetArch));
        pm.addPass(mlir::triton::AMD::createOptimizeLDSUsagePass(targetArch, 0));
        pm.addPass(mlir::createSCFToControlFlowPass());
        pm.addPass(mlir::createConvertIndexToLLVMPass());
        pm.addPass(mlir::triton::gpu::createAllocateSharedMemory());
        pm.addPass(mlir::triton::createConvertTritonAMDGPUToLLVMPass(targetArch, hipFtz));
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createSymbolDCEPass());
        pm.addPass(mlir::triton::createConvertBuiltinFuncToLLVMPass(hipFtz));
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        hasBackendLowering = true;
#else
        DSP_DIAG(COMPILE, "TritonTargetDispatch::compile: AMD backend TTGIR->LLVM lowering "
                  "not available (TritonAMDGPUToLLVM headers not found)");
#endif
        break;
      }
      case TritonGpuTarget::INTEL:
        DSP_DIAG(COMPILE, "TritonTargetDispatch::compile: Intel backend TTGIR->LLVM lowering "
                  "not yet integrated (requires TritonIntelGPUToLLVM)");
        break;
      default:
        break;
    }

    if (!hasBackendLowering) {
      tritonInProtectedRegion = false;
#ifdef SD_CUDA
      cudaGetLastError();
#endif
      DSP_DIAG(COMPILE, "TritonTargetDispatch::compile: no backend lowering passes available for target");
      return result;
    }

    if (mlir::failed(pm.run(*moduleOp))) {
      tritonInProtectedRegion = false;
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
    } // closes inner brace from line 876
  } // phase4Guard destructor fires here, emitting TTGIR_TO_LLVM_DIALECT DONE

  // Triton's AllocateSharedMemory pass stores kernel shared memory usage
  // in the module attribute "triton_gpu.shared".
  result.sharedMemBytes = getModuleSharedMemoryBytes(moduleOp);
  result.globalScratchBytes = getModuleGlobalScratchMemoryBytes(moduleOp);
  result.globalScratchAlignment = getModuleGlobalScratchAlignment(moduleOp);

  // Phase 5: MLIR LLVM dialect -> LLVM IR module
  // Verify the MLIR module before attempting LLVM translation
  if (mlir::failed(mlir::verify(*moduleOp))) {
    tritonInProtectedRegion = false;
#ifdef SD_CUDA
    cudaGetLastError();
#endif
    DSP_DIAG(COMPILE, "TritonTargetDispatch::compile: MLIR module verification failed after lowering");
    return result;
  }

  // Register ALL dialect translation interfaces — builtin.module, NVVM, GPU, etc.
  // Thread-safety: registerAllToLLVMIRTranslations and appendDialectRegistry
  // modify global MLIR state. Must be serialized across concurrent threads.
  {
    static std::mutex mlirTranslationMtx;
    std::lock_guard<std::mutex> lock(mlirTranslationMtx);
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    moduleOp->getContext()->appendDialectRegistry(registry);
  }
  llvm::LLVMContext llvmCtx;

  std::unique_ptr<llvm::Module> llvmModule;

  { // DspCompilePhaseGuard scope: emits MLIR_TO_LLVM_IR START/DONE automatically
    DspCompilePhaseGuard phase5Guard(compileId, "MLIR_TO_LLVM_IR", sd::graph::DSP_DIAG_COMPILE);
    llvmModule = mlir::translateModuleToLLVMIR(*moduleOp, llvmCtx);
    tritonInProtectedRegion = false;
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
  } // phase5Guard destructor fires here, emitting MLIR_TO_LLVM_IR DONE

  // Phase 5b: Link libdevice for NVIDIA math intrinsics (__nv_sqrtf, __nv_expf, etc.)
  // The math-to-LLVM pass lowers math.sqrt/exp/log/etc. to calls to __nv_* functions
  // which are defined in NVIDIA's libdevice bitcode library.
  if (target == TritonGpuTarget::NVIDIA) {
    // Search for libdevice.10.bc in common locations
    std::vector<std::string> libdevicePaths = {
      "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-13.1/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-12.9/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-12.6/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-12.4/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-12.2/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-12.0/nvvm/libdevice/libdevice.10.bc",
      "/usr/local/cuda-11.8/nvvm/libdevice/libdevice.10.bc",
    };

    // Also check CUDA toolkit path from Environment.
    const std::string cudaPath = sd::Environment::getInstance().cudaToolkitPath();
    if (!cudaPath.empty()) {
      libdevicePaths.insert(libdevicePaths.begin(),
          cudaPath + "/nvvm/libdevice/libdevice.10.bc");
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
        DSP_DIAG(JIT, "TritonTargetDispatch::compile: failed to link libdevice from %s", path.c_str());
        continue;
      }

      DSP_DIAG(JIT, "TritonTargetDispatch::compile: linked libdevice from %s", path.c_str());
      linked = true;
      break;
    }

    if (!linked) {
      DSP_DIAG(JIT, "TritonTargetDispatch::compile: WARNING — libdevice.10.bc not found, "
                "math intrinsics (__nv_sqrtf etc.) will be unresolved");
    }
  }

  // Verify the LLVM module
  std::string verifyErr;
  llvm::raw_string_ostream verifyOS(verifyErr);
  if (llvm::verifyModule(*llvmModule, &verifyOS)) {
    DSP_DIAG(JIT, "TritonTargetDispatch::compile: LLVM module verification failed: %s", verifyErr.c_str());
    return result;
  }

  // Phase 6: LLVM IR -> target ISA
  // Initialize LLVM targets
  // Thread-safety: LLVM target initialization modifies global state
  // (TargetRegistry linked list). Must only be called once across all threads.
  static std::once_flag llvmInitFlag;
  std::call_once(llvmInitFlag, []() {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
  });

  // Declare outside guard scope so triple is accessible after the phase for the
  // post-phase DSP_DIAG "generated N bytes" log line.
  std::string triple;
  std::string proc;
  std::string features;

  { // DspCompilePhaseGuard scope: emits LLVM_IR_TO_ASM START/DONE automatically
    DspCompilePhaseGuard phase6Guard(compileId, "LLVM_IR_TO_ASM", sd::graph::DSP_DIAG_JIT);

    switch (target) {
      case TritonGpuTarget::NVIDIA:
        triple = "nvptx64-nvidia-cuda";
        proc = (computeCapability == 90) ? "sm_90a" : ("sm_" + std::to_string(computeCapability));
        break;
      case TritonGpuTarget::AMD:
        triple = "amdgcn-amd-amdhsa";
        proc = result.targetArch;
        break;
      case TritonGpuTarget::INTEL:
        triple = "spir64-unknown-unknown";
        proc = "";
        break;
      default:
        return result;
    }

    llvmModule->setTargetTriple(llvm::Triple(triple));

    std::string lookupError;
    auto* llvmTarget = llvm::TargetRegistry::lookupTarget(triple, lookupError);
    if (!llvmTarget) {
      DSP_DIAG(JIT, "TritonTargetDispatch::compile: LLVM target lookup failed for '%s': %s",
                triple.c_str(), lookupError.c_str());
      return result;
    }

    llvm::TargetOptions targetOptions;
    targetOptions.AllowFPOpFusion =
        env.tritonEnableFpFusion() ? llvm::FPOpFusion::Fast : llvm::FPOpFusion::Strict;
    auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
        llvmTarget->createTargetMachine(triple, proc, features,
                                         targetOptions, llvm::Reloc::PIC_));
    if (!targetMachine) {
      DSP_DIAG(JIT, "TritonTargetDispatch::compile: failed to create TargetMachine for %s/%s",
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
      DSP_DIAG(JIT, "TritonTargetDispatch::compile: TargetMachine can't emit assembly for %s",
                triple.c_str());
      return result;
    }

    codegenPM.run(*llvmModule);
    std::string asmOutput(asmBuffer.begin(), asmBuffer.end());

    if (asmOutput.empty()) {
      DSP_DIAG(JIT, "TritonTargetDispatch::compile: empty output for %s", result.targetArch.c_str());
      return result;
    }

    result.size = asmOutput.size();
    result.data = new char[result.size + 1];
    std::memcpy(result.data, asmOutput.data(), result.size);
    static_cast<char*>(result.data)[result.size] = '\0';
  } // phase6Guard destructor fires here, emitting LLVM_IR_TO_ASM DONE

  DSP_DIAG(JIT, "TritonTargetDispatch::compile: generated %zu bytes for %s (%s)",
            result.size, result.targetArch.c_str(), triple.c_str());
  DSP_DIAG(COMPILE, "TritonTargetDispatch::compile[%lld]: DONE totalElapsedMs=%lld",
            compileId, elapsedMsSince(compileStart));

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

  // Serialize module loading. cuModuleLoadDataEx performs PTX JIT compilation
  // using internal CUDA driver allocators that are not thread-safe under
  // concurrent calls. This is the ONLY serialization point in the Triton
  // compilation pipeline — all MLIR/LLVM compilation runs in parallel with
  // per-thread isolated contexts.
  static std::mutex loadModuleMtx;
  std::lock_guard<std::mutex> lock(loadModuleMtx);

  switch (binary.target) {

    case TritonGpuTarget::NVIDIA: {
#ifdef SD_CUDA
      // cuModuleLoadDataEx requires a current CUDA context on this thread.
      // Worker threads used by TritonGraphBackend may not have bound one yet.
      // We must use the Driver API to ensure the primary context is pushed.
      int currentDevice = 0;
      cudaError_t getDeviceErr = cudaGetDevice(&currentDevice);
      if (getDeviceErr != cudaSuccess) {
        DSP_DIAG(COMPILE, "TritonTargetDispatch::loadModule: cudaGetDevice failed before "
                  "cuModuleLoadDataEx: %s",
                  cudaGetErrorString(getDeviceErr));
        cudaGetLastError();
        return nullptr;
      }
      // Ensure a CUDA driver context is active on this thread.
      // cudaSetDevice alone may not push a driver context visible to cuModuleLoadDataEx.
      CUcontext currentCtx = nullptr;
      cuCtxGetCurrent(&currentCtx);
      CUcontext pushedCtx = nullptr;
      bool didPushCtx = false;
      if (!currentCtx) {
        // No driver context — retain and push the primary context for this device
        CUdevice cuDev;
        cuDeviceGet(&cuDev, currentDevice);
        cuDevicePrimaryCtxRetain(&pushedCtx, cuDev);
        cuCtxPushCurrent(pushedCtx);
        didPushCtx = true;
      }

      // NVIDIA: CUDA Driver API for PTX loading with JIT error logging.
      CUmodule module = nullptr;
      char jitErrorLog[4096] = {0};
      char jitInfoLog[4096] = {0};
      int generateLineInfo =
          sd::Environment::getInstance().tritonDisableLineInfo() ? 0 : 1;
      CUjit_option jitOptions[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_GENERATE_LINE_INFO
      };
      void* jitOptionValues[] = {
        jitErrorLog,
        reinterpret_cast<void*>(sizeof(jitErrorLog)),
        jitInfoLog,
        reinterpret_cast<void*>(sizeof(jitInfoLog)),
        reinterpret_cast<void*>(static_cast<uintptr_t>(generateLineInfo))
      };
      CUresult res = cuModuleLoadDataEx(&module, binary.data,
                                         5, jitOptions, jitOptionValues);
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
        if (didPushCtx) { CUcontext dummy; cuCtxPopCurrent(&dummy); }
        return nullptr;
      }
      if (didPushCtx) { CUcontext dummy; cuCtxPopCurrent(&dummy); }
      return static_cast<void*>(module);
#else
      DSP_DIAG(COMPILE, "TritonTargetDispatch::loadModule: NVIDIA target requires SD_CUDA");
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
      DSP_DIAG(COMPILE, "TritonTargetDispatch::loadModule: AMD target requires HIP (HAVE_MIOPEN/SD_HIP/ZLUDA_TARGET_AMD)");
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
      DSP_DIAG(COMPILE, "TritonTargetDispatch::loadModule: Intel target requires Level Zero (SD_LEVEL_ZERO/ZLUDA_TARGET_INTEL)");
      return nullptr;
#endif
    }

    default:
      DSP_DIAG(COMPILE, "TritonTargetDispatch::loadModule: unsupported target %d",
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
        DSP_DIAG(EXECUTE, "TritonTargetDispatch::getKernelFunction: cuModuleGetFunction failed: %s",
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
        DSP_DIAG(EXECUTE, "TritonTargetDispatch::getKernelFunction: hipModuleGetFunction failed: %s",
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
        DSP_DIAG(EXECUTE, "TritonTargetDispatch::getKernelFunction: zeKernelCreate failed for '%s'",
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
      DSP_DIAG(EXECUTE, "TritonTargetDispatch::launchKernel: unsupported target %d",
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
      DSP_DIAG(FALLBACK, "TritonTargetDispatch::launchCooperativeKernel: cooperative launch not supported on "
                "target %d, falling back to standard launch", static_cast<int>(target));
      return launchKernel(kernelFunc, gridX, gridY, gridZ, blockX, blockY, blockZ,
                          sharedMemBytes, stream, args, numArgs);

    default:
      DSP_DIAG(EXECUTE, "TritonTargetDispatch::launchCooperativeKernel: unsupported target %d",
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
