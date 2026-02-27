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

#ifdef HAVE_MLIR

#include <graph/cpu/ArmHybridGraphBackend.h>
#include <graph/gpu/OpCategoryTable.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include <algorithm>
#include <cctype>

// Vulkan availability probe (platform-specific)
#if defined(__ANDROID__)
#include <dlfcn.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

namespace sd {
namespace graph {

ArmHybridGraphBackend::ArmHybridGraphBackend() {
  // Probe for Vulkan availability at construction time
  vulkanAvailable_ = probeVulkanDevice();
  if (vulkanAvailable_) {
    sd_printf("ArmHybridGraphBackend: Vulkan compute available\n", "");
  } else {
    sd_printf("ArmHybridGraphBackend: Vulkan not available, CPU-only mode\n", "");
  }
}

ArmHybridGraphBackend::~ArmHybridGraphBackend() = default;

ArmHybridGraphBackend& ArmHybridGraphBackend::getInstance() {
  static ArmHybridGraphBackend instance;
  return instance;
}

bool ArmHybridGraphBackend::isAvailable() const {
  // Available if MLIR engine can initialize (always true on ARM with MLIR)
  return sd::mlir_runtime::MLIREngine::getInstance().initialize();
}

bool ArmHybridGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (end < start) return false;
  int segSize = end - start + 1;
  if (segSize < 2) return false;

  // Check all ops are MLIR-mappable
  for (int i = start; i <= end; i++) {
    if (!CpuIRBuilder::isMlirMappable(slots[i].opName)) {
      return false;
    }
  }

  return true;
}

bool ArmHybridGraphBackend::probeVulkanDevice() {
#if defined(__ANDROID__) || defined(__linux__)
  // Try to load libvulkan.so — if it loads, Vulkan is available
  void* vulkanLib = dlopen("libvulkan.so", RTLD_LAZY | RTLD_LOCAL);
  if (vulkanLib) {
    // Check if we can get vkCreateInstance — confirms a working Vulkan loader
    void* createInstance = dlsym(vulkanLib, "vkCreateInstance");
    dlclose(vulkanLib);
    return createInstance != nullptr;
  }

  // On Android, also try libvulkan.so.1
  vulkanLib = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (vulkanLib) {
    void* createInstance = dlsym(vulkanLib, "vkCreateInstance");
    dlclose(vulkanLib);
    return createInstance != nullptr;
  }
#endif
  return false;
}

sd::mlir_runtime::MLIRCompileOptions ArmHybridGraphBackend::getArmCompileOptions() const {
  auto opts = sd::mlir_runtime::MLIREngine::getArmAndroidDefaults();
  // Override AOT mode — for JIT execution we don't want AOT
  opts.aotMode = false;
  opts.aotTarget = sd::mlir_runtime::AOTTarget::HOST;
  return opts;
}

ArmHybridGraphBackend::ExecPath ArmHybridGraphBackend::selectExecPath(
    NativeSlot* slots, int start, int end,
    NDArray** outputSlots, int totalOutputSlots) {

  // If Vulkan isn't available or enabled, always use CPU
  if (!vulkanAvailable_ || !vulkanEnabled_) {
    return ExecPath::ARM_CPU;
  }

  // Count GPU-suitable ops and total tensor size
  int gpuCandidateOps = 0;
  int64_t maxTensorSize = 0;

  for (int i = start; i <= end; i++) {
    auto category = OpCategoryTable::getCategory(slots[i].opName);

    // MATMUL, CONVOLUTION, and large REDUCTION are GPU candidates
    if (category == TritonOpCategory::MATMUL ||
        category == TritonOpCategory::CONVOLUTION ||
        category == TritonOpCategory::FUSED_ATTENTION) {
      gpuCandidateOps++;
    }

    // Check tensor sizes
    for (int o = 0; o < slots[i].numOutputs; o++) {
      int outIdx = slots[i].outputSlotIndices[o];
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots && outputSlots[outIdx]) {
        int64_t len = outputSlots[outIdx]->lengthOf();
        if (len > maxTensorSize) maxTensorSize = len;
      }
    }
  }

  // Use GPU if we have GPU-candidate ops AND tensors are large enough
  // to amortize the CPU→GPU transfer overhead
  if (gpuCandidateOps > 0 && maxTensorSize >= gpuOffloadThreshold_) {
    return ExecPath::VULKAN_GPU;
  }

  return ExecPath::ARM_CPU;
}

bool ArmHybridGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                            NDArray** externalInputs, int numExternalInputs,
                                            NDArray** outputSlots, int totalOutputSlots,
                                            LongType shapeKey,
                                            int totalSlots,
                                            int* requestedOutputSlotIndices,
                                            int numRequestedOutputs) {
  int startSlot = seg.startSlot;
  int endSlot = seg.endSlot;

  // Check cache first
  SegmentCacheKey key{startSlot, endSlot, shapeKey};
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it != cache_.end() && it->second.valid) {
      lastCompilationAudit_ = it->second.compilationAudit;
      return true;
    }
  }

  // Determine execution path (CPU vs GPU)
  ExecPath execPath = selectExecPath(slots, startSlot, endSlot,
                                      outputSlots, totalOutputSlots);

  // Analyze the segment
  auto analysis = CpuIRBuilder::analyzeSegment(slots, startSlot, endSlot, totalSlots,
                                                externalInputs, numExternalInputs,
                                                outputSlots, totalOutputSlots);

  if (!analysis.canCompile) {
    sd_printf("ArmHybridGraphBackend: cannot compile segment [%d-%d]: %s\n",
              startSlot, endSlot, analysis.failureReason.c_str());
    return false;
  }

  // Build MLIR module
  auto& engine = sd::mlir_runtime::MLIREngine::getInstance();
  if (!engine.isInitialized() && !engine.initialize()) {
    sd_printf("ArmHybridGraphBackend: failed to initialize MLIREngine\n", "");
    return false;
  }

  auto module = irBuilder_.buildModule(engine.getContext(), slots, startSlot, endSlot,
                                       totalSlots, externalInputs, numExternalInputs,
                                       outputSlots, totalOutputSlots);
  if (!module) {
    sd_printf("ArmHybridGraphBackend: failed to build MLIR module for segment [%d-%d]\n",
              startSlot, endSlot);
    return false;
  }

  // Compile with ARM-tuned options
  auto compileOpts = getArmCompileOptions();

  std::shared_ptr<sd::mlir_runtime::CompiledKernel> kernel;
  if (execPath == ExecPath::VULKAN_GPU) {
    // Vulkan path: enable Vulkan in options
    compileOpts.enableVulkan = true;
    compileOpts.mobileGPU = sd::mlir_runtime::MobileGPUTarget::VULKAN_GENERIC;
    kernel = engine.compileModule(std::move(module), "fused_kernel", compileOpts);

    if (!kernel || !kernel->isValid()) {
      // Fall back to CPU path on Vulkan compile failure
      sd_printf("ArmHybridGraphBackend: Vulkan compile failed for segment [%d-%d], falling back to CPU\n",
                startSlot, endSlot);
      execPath = ExecPath::ARM_CPU;

      // Rebuild module for CPU path
      module = irBuilder_.buildModule(engine.getContext(), slots, startSlot, endSlot,
                                       totalSlots, externalInputs, numExternalInputs,
                                       outputSlots, totalOutputSlots);
      if (!module) return false;

      compileOpts = getArmCompileOptions();
      kernel = engine.compileModule(std::move(module), "fused_kernel", compileOpts);
    }
  } else {
    // CPU path with ARM optimizations
    kernel = engine.compileModule(std::move(module), "fused_kernel", compileOpts);
  }

  if (!kernel || !kernel->isValid()) {
    sd_printf("ArmHybridGraphBackend: failed to compile segment [%d-%d]\n",
              startSlot, endSlot);
    return false;
  }

  // Build argument mapping (same logic as MlirCpuGraphBackend)
  CompiledSegment compiled;
  compiled.kernel = kernel;
  compiled.shapeKey = shapeKey;
  compiled.valid = true;
  compiled.execPath = execPath;

  // Reconstruct arg ordering matching buildModule():
  // 1. Input args (external + pre-segment)
  // 2. Output args (externally visible)
  // 3. Internal intermediate outputs
  std::unordered_set<int> internalOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      internalOutputs.insert(slots[i].outputSlotIndices[o]);
    }
  }

  std::unordered_set<int> seenSrc;

  // Phase 1: Input args
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
      if (seenSrc.count(srcIdx)) continue;

      bool isExternal = (srcIdx < 0);
      bool isPreSegment = (!isExternal && !internalOutputs.count(srcIdx));

      if (isExternal || isPreSegment) {
        NDArray* arr = nullptr;
        if (isExternal) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExternalInputs && externalInputs) arr = externalInputs[extIdx];
        } else {
          if (srcIdx < totalOutputSlots && outputSlots) arr = outputSlots[srcIdx];
        }
        if (!arr) continue;

        seenSrc.insert(srcIdx);
        compiled.argMappings.push_back({srcIdx, false});
      }
    }
  }

  // Phase 2: Output args (externally visible)
  auto externalOutputSet = CpuIRBuilder::computeExternallyVisibleOutputs(
      slots, startSlot, endSlot, totalSlots);
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      int outIdx = slots[i].outputSlotIndices[o];
      if (!externalOutputSet.count(outIdx)) continue;
      if (seenSrc.count(outIdx)) continue;
      seenSrc.insert(outIdx);

      NDArray* arr = nullptr;
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots) arr = outputSlots[outIdx];
      if (!arr) continue;

      compiled.argMappings.push_back({outIdx, true});
    }
  }

  // Phase 3: Internal intermediate outputs
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      int outIdx = slots[i].outputSlotIndices[o];
      if (seenSrc.count(outIdx)) continue;

      NDArray* arr = nullptr;
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots) arr = outputSlots[outIdx];
      if (!arr) continue;

      seenSrc.insert(outIdx);
      compiled.argMappings.push_back({outIdx, true});
    }
  }

  // Build compilation audit
  for (int i = startSlot; i <= endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].opName;
    entry.wasCompiled = CpuIRBuilder::isMlirMappable(slots[i].opName);
    if (!entry.wasCompiled) {
      entry.reason = "unsupported op category";
    }
    compiled.compilationAudit.push_back(entry);
  }

  lastCompilationAudit_ = compiled.compilationAudit;

  const char* pathName = (execPath == ExecPath::VULKAN_GPU) ? "Vulkan GPU" : "ARM CPU";
  sd_printf("ArmHybridGraphBackend: compiled segment [%d-%d] via %s with %d buffer args (%d ops fused)\n",
            startSlot, endSlot, pathName,
            static_cast<int>(compiled.argMappings.size()),
            endSlot - startSlot + 1);

  // Cache the result
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_[key] = std::move(compiled);
  }

  return true;
}

Status ArmHybridGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                              NDArray** externalInputs, int numExternalInputs,
                                              NDArray** outputSlots, int totalOutputSlots,
                                              void* stream) {
  int startSlot = seg.startSlot;
  int endSlot = seg.endSlot;
  SegmentCacheKey key{startSlot, endSlot, seg.shapeKey};

  // Look up cached kernel
  CompiledSegment* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end() || !it->second.valid) {
      return Status::KERNEL_FAILURE;
    }
    compiled = &it->second;
  }

  if (!compiled->kernel || !compiled->kernel->isValid()) {
    return Status::KERNEL_FAILURE;
  }

  // Wire NDArray buffers to kernel arguments in the same order as compilation
  std::vector<NDArray*> inputArrays;
  std::vector<NDArray*> outputArrays;

  for (auto& mapping : compiled->argMappings) {
    NDArray* arr = nullptr;
    if (mapping.sourceIndex < 0) {
      int extIdx = -(mapping.sourceIndex + 1);
      if (extIdx < numExternalInputs && externalInputs) arr = externalInputs[extIdx];
    } else {
      if (mapping.sourceIndex < totalOutputSlots && outputSlots)
        arr = outputSlots[mapping.sourceIndex];
    }

    if (!arr) {
      sd_printf("ArmHybridGraphBackend: null array for arg sourceIndex=%d in segment [%d-%d]\n",
                mapping.sourceIndex, startSlot, endSlot);
      return Status::KERNEL_FAILURE;
    }

    if (mapping.isOutput) {
      outputArrays.push_back(arr);
    } else {
      inputArrays.push_back(arr);
    }
  }

  // Execute the kernel
  bool ok = compiled->kernel->execute(inputArrays, outputArrays);
  if (!ok) {
    sd_printf("ArmHybridGraphBackend: execution failed for segment [%d-%d] (path=%s)\n",
              startSlot, endSlot,
              compiled->execPath == ExecPath::VULKAN_GPU ? "Vulkan" : "ARM CPU");
    return Status::KERNEL_FAILURE;
  }

  return Status::OK;
}

void ArmHybridGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  cache_.clear();
}

std::vector<CompilationAuditEntry> ArmHybridGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLIR
