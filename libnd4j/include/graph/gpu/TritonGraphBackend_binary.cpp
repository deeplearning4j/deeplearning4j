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

#include <graph/gpu/TritonGraphBackend.h>
#include <graph/gpu/TritonGraphBackend_internal.h>
#include <graph/gpu/TritonIRBuilder.h>
#include <graph/gpu/TritonTargetDispatch.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspConstants.h>
#include <system/Environment.h>
#include <helpers/logger.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <llvm/Support/raw_ostream.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace sd {
namespace graph {

using namespace triton_internal;

// Utility: get platform-independent temp directory
static std::string getTempDir() {
  // Try environment variables first (works on Linux, macOS, Windows)
  const char* tempEnv = std::getenv(dsp::ENV_TMPDIR);
  if (!tempEnv) tempEnv = std::getenv(dsp::ENV_TMP);
  if (!tempEnv) tempEnv = std::getenv(dsp::ENV_TEMP);
  if (!tempEnv) tempEnv = std::getenv(dsp::ENV_USERPROFILE);  // Windows fallback
  
  if (tempEnv && strlen(tempEnv) > 0) {
    std::string path(tempEnv);
    // Ensure path ends with separator
    if (path.back() != '/' && path.back() != '\\') {
#ifdef _WIN32
      path += '\\';
#else
      path += '/';
#endif
    }
    return path;
  }
  
  // Default fallbacks
#ifdef _WIN32
  return "C:\\temp\\";
#else
  return "/tmp/";
#endif
}

// Utility: report MLIR verification failure with error message
static void reportMLIRVerificationFailure(mlir::ModuleOp mod, int startSlot, int endSlot, const std::string& context = "") {
  std::string irDump;
  llvm::raw_string_ostream os(irDump);
  mod->print(os, mlir::OpPrintingFlags().enableDebugInfo());
  
  // Build platform-independent temp file path
  std::string tempDir = getTempDir();
  std::string failPath = tempDir + "triton_ir_verify_fail_" + 
                         std::to_string(startSlot) + "_" + std::to_string(endSlot) + ".mlir";
  
  FILE* f = fopen(failPath.c_str(), "w");
  if (f) { 
    fprintf(f, "%s", irDump.c_str()); 
    fclose(f);
    DSP_DIAG(COMPILE, "TritonGraphBackend: MLIR verification FAILED for [%d-%d]. IR dumped to %s (%d bytes)%s",
              startSlot, endSlot, failPath.c_str(), static_cast<int>(irDump.size()),
              context.empty() ? "" : (" " + context).c_str());
  } else {
    DSP_DIAG(COMPILE, "TritonGraphBackend: MLIR verification FAILED for [%d-%d]%s (could not write IR dump)",
              startSlot, endSlot, 
              context.empty() ? "" : (" " + context).c_str());
  }
}

TritonGraphBackend::CompiledKernel TritonGraphBackend::compileToGpuBinary(
    NativeSlot* slots, int startSlot, int endSlot,
    int totalSlots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {
  CompiledKernel result;
  auto now = []() { return std::chrono::steady_clock::now(); };
  auto elapsedMs = [&](const std::chrono::steady_clock::time_point& t0) -> long long {
    return static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now() - t0).count());
  };
  const auto tCompileStart = now();
  DSP_DIAG(COMPILE, "TritonGraphBackend: compileToGpuBinary START [%d-%d]", startSlot, endSlot);

  // Build Triton IR
  const auto tIrStart = now();
  TritonIRBuilder localBuilder;
  auto irModule = localBuilder.buildModule(slots, startSlot, endSlot,
                                           totalSlots,
                                           externalInputs, numExternalInputs,
                                           outputSlots, totalOutputSlots);
  const long long irBuildMs = elapsedMs(tIrStart);
  auto cleanupModule = [&irModule]() {
    if (irModule.mlirModule) {
      auto* mod = static_cast<mlir::ModuleOp*>(irModule.mlirModule);
      mod->erase();
      delete mod;
      irModule.mlirModule = nullptr;
    }
    // Free the MLIRContext that owns all MLIR memory for this compilation.
    // Each sub-segment creates a new MLIRContext (~10-100MB for large kernels);
    // failing to free it causes unbounded memory growth during multi-sub-segment
    // compilation of VLM-scale graphs (3840 ops → many sub-segments).
    if (irModule.mlirContext) {
      delete static_cast<mlir::MLIRContext*>(irModule.mlirContext);
      irModule.mlirContext = nullptr;
    }
  };

  if (!irModule.valid) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors from failed IR build
#endif
    // Build op list for the failed range
    std::string failedOps;
    for (int s = startSlot; s <= endSlot; s++) {
      if (!failedOps.empty()) failedOps += ",";
      failedOps += slots[s].ident.opName;
    }
    DSP_DIAG(COMPILE, "TritonGraphBackend: IR build FAILED for [%d-%d] after %lld ms "
             "(numExtInputs=%d, numOutputSlots=%d, ops: %s)",
             startSlot, endSlot, irBuildMs, numExternalInputs, totalOutputSlots, failedOps.c_str());
    cleanupModule();
    return result;
  }
  DSP_DIAG(COMPILE, "TritonGraphBackend: IR build OK [%d-%d] in %lld ms "
           "(args=%d, indirect=%d, cooperative=%d, multiPhase=%d(%d phases), grid=%ux%ux%u, block=%ux%ux%u)",
           startSlot, endSlot, irBuildMs,
           static_cast<int>(irModule.args.size()),
           irModule.useIndirectArgs ? 1 : 0, irModule.useCooperativeLaunch ? 1 : 0,
           irModule.useMultiPhaseLaunch ? 1 : 0,
           static_cast<int>(irModule.launchPhases.size()),
           irModule.gridX, irModule.gridY, irModule.gridZ,
           irModule.blockX, irModule.blockY, irModule.blockZ);

  // Early MLIR verification to catch type mismatches before expensive compilation
  {
    auto* mod = static_cast<mlir::ModuleOp*>(irModule.mlirModule);
    if (mlir::failed(mlir::verify(*mod))) {
      reportMLIRVerificationFailure(*mod, startSlot, endSlot);
      cleanupModule();
      return result;
    }

    // Dump MLIR IR of first few compiled kernels for debugging
    static std::atomic<int> irDumpCount{0};
    if (irDumpCount.load(std::memory_order_relaxed) < 10) {
      std::string irDump;
      llvm::raw_string_ostream os(irDump);
      mod->print(os);
      char fname[256];
      int dumpIdx = irDumpCount.fetch_add(1, std::memory_order_relaxed);
      snprintf(fname, sizeof(fname), "/tmp/triton_ir_dump_%03d_slots_%d_%d.mlir",
               dumpIdx, startSlot, endSlot);
      FILE* f = fopen(fname, "w");
      if (f) {
        fprintf(f, "// Kernel: %s\n", irModule.kernelName.c_str());
        fprintf(f, "// Slots: [%d-%d]\n", startSlot, endSlot);
        fprintf(f, "// Args: %d (indirect=%d)\n",
                static_cast<int>(irModule.args.size()), irModule.useIndirectArgs ? 1 : 0);
        fprintf(f, "// Grid: %ux%ux%u Block: %ux%ux%u\n",
                irModule.gridX, irModule.gridY, irModule.gridZ,
                irModule.blockX, irModule.blockY, irModule.blockZ);
        fprintf(f, "// Args detail:\n");
        for (int a = 0; a < static_cast<int>(irModule.args.size()); a++) {
          auto& arg = irModule.args[a];
          fprintf(f, "//   [%d] slot=%d output=%d dtype=%d shape=[",
                  a, arg.slotIndex, arg.isOutput ? 1 : 0, static_cast<int>(arg.dtype));
          for (size_t d = 0; d < arg.shape.size(); d++) {
            if (d > 0) fprintf(f, ",");
            fprintf(f, "%lld", (long long)arg.shape[d]);
          }
          fprintf(f, "]\n");
        }
        fprintf(f, "\n%s", irDump.c_str());
        fclose(f);
        DSP_DIAG(COMPILE, "TritonGraphBackend: dumped MLIR IR to %s (%d bytes)",
                 fname, static_cast<int>(irDump.size()));
      }
      // irDumpCount already incremented via fetch_add above
    }
  }

#ifdef SD_CUDA
  // ── Early cooperative launch capacity check ──
  // Reject BEFORE the expensive TTIR→PTX compilation (which can take 30+ minutes
  // for large fused kernels) if the required grid clearly exceeds what the GPU
  // can support for cooperative launch. We estimate blocks/SM from both thread
  // occupancy (maxThreadsPerSM / threadsPerBlock) and shared memory occupancy
  // (maxSharedPerSM / estimatedSharedMemBytes). The estimate is conservative
  // (may allow some cases that will fail post-compile) but catches the common
  // case of 400+ blocks on 128 SMs with large shared memory per block.
  if (irModule.useCooperativeLaunch) {
    unsigned long long requiredBlocks =
        static_cast<unsigned long long>(std::max(1u, irModule.gridX)) *
        static_cast<unsigned long long>(std::max(1u, irModule.gridY)) *
        static_cast<unsigned long long>(std::max(1u, irModule.gridZ));
    if (irModule.requiredGrid > 0) {
      requiredBlocks = std::max(requiredBlocks,
                                static_cast<unsigned long long>(std::max(1, irModule.requiredGrid)));
    }

    int currentDevice = 0;
    cudaError_t devErr = cudaGetDevice(&currentDevice);
    if (devErr == cudaSuccess) {
      CUdevice cuDevice = 0;
      CUresult cuDevErr = cuDeviceGet(&cuDevice, currentDevice);
      if (cuDevErr == CUDA_SUCCESS) {
        int smCount = 0;
        int maxThreadsPerSM = 0;
        int maxSharedPerSM = 0;
        cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice);
        cuDeviceGetAttribute(&maxThreadsPerSM, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cuDevice);
        cuDeviceGetAttribute(&maxSharedPerSM,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, cuDevice);

        // Compute blocks/SM upper bound from BOTH thread and shared memory occupancy.
        // The actual occupancy is min(thread limit, shared memory limit).
        int threadsPerBlock = std::max(1, irModule.numWarps) * 32;
        int blocksPerSmByThreads = (maxThreadsPerSM > 0 && threadsPerBlock > 0)
            ? (maxThreadsPerSM / threadsPerBlock)
            : 16;

        int blocksPerSmBySmem = 16;  // default if no estimate
        if (irModule.estimatedSharedMemBytes > 0 && maxSharedPerSM > 0) {
          blocksPerSmBySmem = maxSharedPerSM / irModule.estimatedSharedMemBytes;
        }

        int blocksPerSmEstimate = std::max(1, std::min(blocksPerSmByThreads, blocksPerSmBySmem));

        unsigned long long maxPossibleBlocks =
            static_cast<unsigned long long>(smCount) * blocksPerSmEstimate;
        if (smCount > 0 && requiredBlocks > maxPossibleBlocks) {
          DSP_DIAG(COMPILE, "TritonGraphBackend: EARLY REJECT cooperative launch for [%d-%d]: "
                   "requiredBlocks=%llu exceeds max=%llu "
                   "(smCount=%d, blocksPerSm<=%d [threads: %d/%d=%d, smem: %d/%d=%d]). "
                   "Skipping expensive compilation.",
                   startSlot, endSlot,
                   requiredBlocks, maxPossibleBlocks,
                   smCount, blocksPerSmEstimate,
                   maxThreadsPerSM, threadsPerBlock, blocksPerSmByThreads,
                   maxSharedPerSM, irModule.estimatedSharedMemBytes, blocksPerSmBySmem);
          cleanupModule();
          return result;
        }
        DSP_DIAG(COMPILE, "TritonGraphBackend: cooperative launch pre-check OK for [%d-%d]: "
                 "requiredBlocks=%llu, maxPossible=%llu (smCount=%d, blocksPerSm<=%d)",
                 startSlot, endSlot, requiredBlocks, maxPossibleBlocks,
                 smCount, blocksPerSmEstimate);
      }
    }
  }
#endif

  // Build compilation audit
  for (int i = startSlot; i <= endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].ident.opName;
    entry.wasCompiled = TritonIRBuilder::isTritonMappable(slots[i].ident.opName);
    if (!entry.wasCompiled) {
      entry.reason = "unmappable op (not in Triton op table)";
    }
    result.audit.push_back(entry);
  }

  // Capture TTIR text for deterministic cache-key generation.
  std::string ttirText;
  {
    auto* mod = static_cast<mlir::ModuleOp*>(irModule.mlirModule);
    llvm::raw_string_ostream os(ttirText);
    mod->print(os);
  }

  auto& env = sd::Environment::getInstance();
  int compileNumWarps = irModule.numWarps;
  int compileNumStages = irModule.numStages;
  if (env.tritonNumWarps() > 0) {
    compileNumWarps = std::max(1, std::min(env.tritonNumWarps(), 32));
  }
  if (env.tritonNumStages() > 0) {
    compileNumStages = std::max(1, std::min(env.tritonNumStages(), 16));
  }
  if (compileNumWarps != irModule.numWarps || compileNumStages != irModule.numStages) {
    DSP_DIAG(COMPILE, "TritonGraphBackend: compile option overrides for [%d-%d]: warps %d->%d, stages %d->%d",
             startSlot, endSlot,
             irModule.numWarps, compileNumWarps,
             irModule.numStages, compileNumStages);
  }

  const std::string cacheHash = computeDiskCacheHash(ttirText,
                                                      compileNumWarps, compileNumStages);

  TritonCompiledBinary binary = {nullptr, 0, TritonGpuTarget::UNKNOWN, "", compileNumWarps, 0};
  const std::string archOverride = env.tritonOverrideArch();
  auto loadBinaryFromBasePath = [&](const std::string& basePath,
                                    const char* sourceLabel,
                                    TritonCompiledBinary& out) -> bool {
    const std::string ptxPath = basePath + ".ptx";
    const std::string metaPath = basePath + ".meta";

    std::ifstream ptxFile(ptxPath, std::ios::binary);
    if (!ptxFile.good()) return false;

    std::string ptxText((std::istreambuf_iterator<char>(ptxFile)),
                        std::istreambuf_iterator<char>());
    if (ptxText.empty()) return false;
    if (ptxText.back() != '\0') ptxText.push_back('\0');

    int metaNumWarps = compileNumWarps;
    int metaSharedMem = 0;
    bool metaSharedMemPresent = false;
    int metaGlobalScratchBytes = 0;
    int metaGlobalScratchAlignment = 128;
    std::string metaKernelName;

    std::ifstream metaFile(metaPath);
    if (metaFile.good()) {
      std::string line;
      while (std::getline(metaFile, line)) {
        size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) continue;
        const std::string key = line.substr(0, eqPos);
        const std::string value = line.substr(eqPos + 1);
        if (key == "numWarps") {
          parseIntValue(value, metaNumWarps);
        } else if (key == "sharedMemBytes") {
          parseIntValue(value, metaSharedMem);
          metaSharedMemPresent = true;
        } else if (key == "globalScratchBytes") {
          parseIntValue(value, metaGlobalScratchBytes);
        } else if (key == "globalScratchAlignment") {
          parseIntValue(value, metaGlobalScratchAlignment);
        } else if (key == "kernelName") {
          metaKernelName = value;
        }
      }
    }

    if (!metaKernelName.empty() && metaKernelName != irModule.kernelName) {
      return false;
    }

    // Only reject entries where sharedMemBytes metadata was missing entirely
    // (pre-metadata era). sharedMemBytes=0 is valid for element-wise kernels
    // that declare extern .shared (Triton convention) but don't use it.
    if (!metaSharedMemPresent && metaSharedMem == 0 && ptxUsesExternSharedMemory(ptxText)) {
      DSP_DIAG(JIT, "TritonGraphBackend: %s entry for [%d-%d] is stale "
             "(extern shared PTX with no sharedMemBytes metadata); ignoring",
             sourceLabel, startSlot, endSlot);
      return false;
    }

    out.data = new char[ptxText.size()];
    std::memcpy(out.data, ptxText.data(), ptxText.size());
    out.size = ptxText.size() - 1;  // Excludes null terminator
    out.target = TritonTargetDispatch::detectTarget();
    out.targetArch = TritonTargetDispatch::getTargetArch();
    if (!archOverride.empty()) {
      out.targetArch = archOverride;
    }
    out.numWarps = metaNumWarps;
    out.sharedMemBytes = metaSharedMem;
    out.globalScratchBytes = metaGlobalScratchBytes;
    out.globalScratchAlignment = metaGlobalScratchAlignment;
    DSP_DIAG(JIT, "TritonGraphBackend: %s HIT for sub-segment [%d-%d] (%zu bytes)",
             sourceLabel, startSlot, endSlot, out.size);
    return true;
  };

  auto dumpKernelArtifacts = [&](const TritonCompiledBinary& dumpBinary) {
    if (!env.tritonKernelDump() || dumpBinary.data == nullptr || dumpBinary.size == 0) return;
    const std::string dumpDir = configuredOrDefaultTritonDir(
        env.tritonDumpDir(), env.homeDirectory(), "triton_dump");
    if (!ensureDiskCacheDir(dumpDir)) return;

    const std::string basePath = dumpDir + "/ttir_" + cacheHash;
    {
      std::ofstream ttirOut(basePath + ".ttir", std::ios::trunc);
      if (ttirOut.good()) {
        ttirOut << ttirText;
      }
    }
    {
      std::ofstream ptxOut(basePath + ".ptx", std::ios::binary | std::ios::trunc);
      if (ptxOut.good()) {
        ptxOut.write(static_cast<const char*>(dumpBinary.data),
                     static_cast<std::streamsize>(dumpBinary.size));
      }
    }
    {
      std::ofstream metaOut(basePath + ".meta", std::ios::trunc);
      if (metaOut.good()) {
        metaOut << "numWarps=" << dumpBinary.numWarps << "\n";
        metaOut << "sharedMemBytes=" << dumpBinary.sharedMemBytes << "\n";
        metaOut << "globalScratchBytes=" << dumpBinary.globalScratchBytes << "\n";
        metaOut << "globalScratchAlignment=" << dumpBinary.globalScratchAlignment << "\n";
        metaOut << "kernelName=" << irModule.kernelName << "\n";
        metaOut << "numStages=" << compileNumStages << "\n";
        metaOut << "numCTAs=" << std::max(1, env.tritonNumCTAs()) << "\n";
        metaOut << "maxNreg=" << std::max(0, env.tritonMaxNreg()) << "\n";
      }
    }
  };

  const auto tBinaryStageStart = now();
  bool loadedFromOverride = false;
  if (env.tritonKernelOverride()) {
    const std::string overrideDir = configuredOrDefaultTritonDir(
        env.tritonOverrideDir(), env.homeDirectory(), "triton_override");
    const std::string basePath = overrideDir + "/ttir_" + cacheHash;
    loadedFromOverride = loadBinaryFromBasePath(basePath, "override", binary);
  }

  const bool alwaysCompile = env.tritonAlwaysCompile();
  bool loadedFromDiskCache = false;
  if (!loadedFromOverride && !alwaysCompile) {
    loadedFromDiskCache = loadBinaryFromDiskCache(startSlot, endSlot, cacheHash, irModule, binary);
  }

  if (!loadedFromOverride && !loadedFromDiskCache) {
    DSP_DIAG(COMPILE, "TritonGraphBackend: TTIR->PTX compile START [%d-%d]", startSlot, endSlot);
    const auto tCompileStageStart = now();
    binary = TritonTargetDispatch::compile(irModule.mlirModule, compileNumWarps, compileNumStages);
    DSP_DIAG(COMPILE, "TritonGraphBackend: TTIR->PTX compile %s [%d-%d] in %lld ms "
             "(ptxBytes=%zu, warps=%d, smem=%d)",
             binary.data != nullptr ? "DONE" : "FAILED",
             startSlot, endSlot, elapsedMs(tCompileStageStart),
             binary.size, binary.numWarps, binary.sharedMemBytes);
    if (binary.data && !alwaysCompile) {
      writeBinaryToDiskCache(startSlot, endSlot, cacheHash, irModule, binary);
    }
  } else if (loadedFromOverride) {
    DSP_DIAG(JIT, "TritonGraphBackend: override load DONE [%d-%d] in %lld ms "
             "(ptxBytes=%zu, warps=%d, smem=%d)",
             startSlot, endSlot, elapsedMs(tBinaryStageStart),
             binary.size, binary.numWarps, binary.sharedMemBytes);
  } else {
    DSP_DIAG(JIT, "TritonGraphBackend: PTX cache load DONE [%d-%d] in %lld ms "
             "(ptxBytes=%zu, warps=%d, smem=%d)",
             startSlot, endSlot, elapsedMs(tBinaryStageStart),
             binary.size, binary.numWarps, binary.sharedMemBytes);
  }

  dumpKernelArtifacts(binary);

  // Debug: dump PTX for problematic sub-kernels
  if (binary.data && startSlot == 347) {
    std::string ptxPath = "/tmp/triton_ptx_" + std::to_string(startSlot) + "_" + std::to_string(endSlot) + ".ptx";
    std::ofstream ptxFile(ptxPath, std::ios::binary | std::ios::trunc);
    if (ptxFile.good()) {
      ptxFile.write(static_cast<const char*>(binary.data), static_cast<std::streamsize>(binary.size));
      DSP_DIAG(COMPILE, "TritonGraphBackend: PTX dumped to %s (%zu bytes)", ptxPath.c_str(), binary.size);
    }
  }

  if (!binary.data) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors from failed compilation
#endif
    DSP_DIAG(COMPILE, "TritonGraphBackend: Triton compilation FAILED for segment [%d-%d] "
             "(totalElapsed=%lld ms)",
             startSlot, endSlot, elapsedMs(tCompileStart));
    cleanupModule();
    return result;
  }

  // Load binary into driver module
  result.gpuModule = TritonTargetDispatch::loadModule(binary);
  if (!result.gpuModule) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors from failed module load
#endif
    DSP_DIAG(COMPILE, "TritonGraphBackend: module load failed for segment [%d-%d]", startSlot, endSlot);
    delete[] static_cast<char*>(binary.data);
    cleanupModule();
    return result;
  }

  // Track estimated GPU memory for the loaded module (binary size as proxy)
  result.estimatedModuleBytes = binary.size;
  // Stash residency-cache reload metadata so the launch path can re-load this
  // kernel from the disk cache after an LRU eviction.  ModuleResidencyCache
  // registration happens once the kernel has been moved into cache_[key]
  // (its address is unstable until then).
  result.diskCacheHash = cacheHash;
  result.kernelName = irModule.kernelName;
  {
    int currentDevice = 0;
#ifdef SD_CUDA
    cudaGetDevice(&currentDevice);
#endif
    recordModuleAlloc(currentDevice, binary.size);
    result.loadedDeviceId = currentDevice;
  }

  // Get kernel function
  result.kernelFunction = TritonTargetDispatch::getKernelFunction(result.gpuModule, irModule.kernelName);
  if (!result.kernelFunction) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors
#endif
    DSP_DIAG(COMPILE, "TritonGraphBackend: kernel function '%s' not found in module", irModule.kernelName.c_str());
    TritonTargetDispatch::unloadModule(result.gpuModule);
    result.gpuModule = nullptr;
    delete[] static_cast<char*>(binary.data);
    cleanupModule();
    return result;
  }

#ifdef SD_CUDA
  unsigned int requestedSharedMem =
      binary.sharedMemBytes > 0 ? static_cast<unsigned int>(binary.sharedMemBytes) : 0u;

  if (binary.target == TritonGpuTarget::NVIDIA) {
    if (!configureCudaKernelSharedMemory(result.kernelFunction, requestedSharedMem)) {
      DSP_DIAG(COMPILE, "TritonGraphBackend: shared memory setup failed for segment [%d-%d] "
                "(requested=%u bytes)",
                startSlot, endSlot, requestedSharedMem);
      TritonTargetDispatch::unloadModule(result.gpuModule);
      result.gpuModule = nullptr;
      result.kernelFunction = nullptr;
      delete[] static_cast<char*>(binary.data);
      cleanupModule();
      return result;
    }
  }

  if (binary.target == TritonGpuTarget::NVIDIA && irModule.useCooperativeLaunch) {
    const unsigned int launchBlockX = static_cast<unsigned int>(std::max(1, binary.numWarps) * 32);
    const unsigned int launchBlockY = std::max(1u, irModule.blockY);
    const unsigned int launchBlockZ = std::max(1u, irModule.blockZ);
    unsigned long long requiredBlocks = static_cast<unsigned long long>(std::max(1u, irModule.gridX)) *
                                        static_cast<unsigned long long>(std::max(1u, irModule.gridY)) *
                                        static_cast<unsigned long long>(std::max(1u, irModule.gridZ));
    if (irModule.requiredGrid > 0) {
      requiredBlocks = std::max(requiredBlocks,
                                static_cast<unsigned long long>(std::max(1, irModule.requiredGrid)));
    }

    bool coopSupported = false;
    long long maxCoopBlocks = 0;
    int blocksPerSm = 0;
    int smCount = 0;
    const bool capacityKnown = queryCudaCooperativeLaunchCapacity(
        result.kernelFunction,
        launchBlockX, launchBlockY, launchBlockZ,
        requestedSharedMem,
        &coopSupported, &maxCoopBlocks, &blocksPerSm, &smCount);

    if (!capacityKnown) {
      DSP_DIAG(COMPILE, "TritonGraphBackend: cooperative launch capacity check unavailable for [%d-%d]; "
               "continuing with runtime launch validation",
               startSlot, endSlot);
    } else if (!coopSupported) {
      DSP_DIAG(COMPILE, "TritonGraphBackend: cooperative launch required for [%d-%d], "
               "but current CUDA device does not support cooperative launch",
               startSlot, endSlot);
      TritonTargetDispatch::unloadModule(result.gpuModule);
      result.gpuModule = nullptr;
      result.kernelFunction = nullptr;
      delete[] static_cast<char*>(binary.data);
      cleanupModule();
      return result;
    } else if (maxCoopBlocks <= 0 ||
               requiredBlocks > static_cast<unsigned long long>(maxCoopBlocks)) {
      DSP_DIAG(COMPILE, "TritonGraphBackend: cooperative launch capacity exceeded for [%d-%d] "
               "(requiredBlocks=%llu, maxBlocks=%lld, smCount=%d, blocksPerSm=%d, "
               "grid=%ux%ux%u, block=%ux%ux%u, sharedMem=%u). "
               "Rejecting this fused range so adaptive splitting can retry.",
               startSlot, endSlot,
               static_cast<unsigned long long>(requiredBlocks), maxCoopBlocks,
               smCount, blocksPerSm,
               irModule.gridX, irModule.gridY, irModule.gridZ,
               launchBlockX, launchBlockY, launchBlockZ,
               requestedSharedMem);
      TritonTargetDispatch::unloadModule(result.gpuModule);
      result.gpuModule = nullptr;
      result.kernelFunction = nullptr;
      delete[] static_cast<char*>(binary.data);
      cleanupModule();
      return result;
    }
  }
#endif

  // Set launch config
  result.gridX = irModule.gridX;
  result.gridY = irModule.gridY;
  result.gridZ = irModule.gridZ;
  // Triton 3.6.0's AllocateWarpGroups pass may change the warp count during compilation.
  // blockX MUST match the actual compiled warp count, not the pre-compilation IR builder value.
  result.blockX = binary.numWarps * 32;
  result.blockY = irModule.blockY;
  result.blockZ = irModule.blockZ;
  result.sharedMemBytes = binary.sharedMemBytes;
  result.globalScratchBytes = binary.globalScratchBytes > 0
      ? static_cast<unsigned int>(binary.globalScratchBytes) : 0u;
  result.globalScratchAlignment = binary.globalScratchAlignment > 0
      ? static_cast<unsigned int>(binary.globalScratchAlignment) : 128u;
  result.numWarps = binary.numWarps;
  result.argSlotMapping = irModule.args;
  result.useCooperativeLaunch = irModule.useCooperativeLaunch;
  result.useDynamicGrid = irModule.useDynamicGrid;
  result.useIndirectArgs = irModule.useIndirectArgs;
  result.useMultiPhaseLaunch = irModule.useMultiPhaseLaunch;
  result.launchPhases = irModule.launchPhases;

  // Clean up
  delete[] static_cast<char*>(binary.data);
  cleanupModule();
  DSP_DIAG(COMPILE, "TritonGraphBackend: compileToGpuBinary DONE [%d-%d] total=%lld ms",
           startSlot, endSlot, elapsedMs(tCompileStart));

  return result;
}

}  // namespace graph
}  // namespace sd

#endif // HAVE_TRITON
