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

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <graph/vulkan/VulkanPipelineCache.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanSpirvDiskCache.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>

#if defined(HAVE_MLIR) && HAVE_MLIR
// ── VulkanOpLowerings patterns (matmul, rms_norm, rope → GPU/SPIR-V) ────────
// Capability macros come from the selected MLIR::SPIRV package target.
#include <graph/vulkan/VulkanOpLowerings.h>

// ── MLIR public headers ────────────────────────────────────────────────────
// Use only the stable public MLIR API headers — no internal "Detail" headers.
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/DialectConversion.h>

// Dialects loaded by the new lowering patterns
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#ifdef SD_MLIR_HAS_MATH_DIALECT
#include "mlir/Dialect/Math/IR/Math.h"
#endif

#ifdef SD_MLIR_HAS_GPU_TO_SPIRV_PASS_HEADER
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h>
#elif defined(SD_MLIR_HAS_GPU_TO_SPIRV_LEGACY_HEADER)
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#endif
#ifdef SD_MLIR_HAS_MATH_TO_SPIRV_PASS_HEADER
#include <mlir/Conversion/MathToSPIRV/MathToSPIRVPass.h>
#define SD_VULKAN_HAS_MATH_TO_SPIRV 1
#elif defined(SD_MLIR_HAS_MATH_TO_SPIRV_LEGACY_HEADER)
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#define SD_VULKAN_HAS_MATH_TO_SPIRV 1
#endif
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/Target/SPIRV/Serialization.h>
#include <mlir/Support/LogicalResult.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Config/llvm-config.h>
#endif  // HAVE_MLIR

#include <algorithm>
#include <cstring>
#include <limits>

namespace sd {
namespace graph {

std::mutex VulkanPipelineCache::registryMutex_;
std::unordered_set<VulkanPipelineCache*> VulkanPipelineCache::instances_;
std::atomic<uint64_t> VulkanPipelineCache::totalKernelLaunches_{0};
std::atomic<uint64_t> VulkanPipelineCache::totalCacheHits_{0};

namespace {

#if defined(HAVE_MLIR) && HAVE_MLIR
class VulkanAttachSPIRVEntryPointABIPass final
    : public mlir::PassWrapper<VulkanAttachSPIRVEntryPointABIPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VulkanAttachSPIRVEntryPointABIPass)

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<int32_t, 3> workgroupSize{1, 1, 1};
    auto entryPointABI = mlir::spirv::getEntryPointABIAttr(context, workgroupSize);
    llvm::StringRef entryPointAttrName =
        mlir::spirv::getEntryPointABIAttrName();
    llvm::StringRef interfaceAttrName =
        mlir::spirv::getInterfaceVarABIAttrName();
    bool valid = true;

    // Kernel outlining captures only values used by the kernel, in first-use
    // order. Recover each captured operand's original func.func argument index
    // so the reflected descriptor binding remains a stable index into the
    // runtime signature. Unused function arguments intentionally have no shader
    // binding.
    module.walk([&](mlir::gpu::LaunchFuncOp launch) {
      mlir::Operation* kernelOperation =
          module.lookupSymbol(launch.getKernelAttr());
      auto gpuFunc =
          llvm::dyn_cast_or_null<mlir::gpu::GPUFuncOp>(kernelOperation);
      auto parentFunc = launch->getParentOfType<mlir::func::FuncOp>();
      mlir::OperandRange kernelOperands = launch.getKernelOperands();
      if (!gpuFunc || !parentFunc ||
          !mlir::gpu::GPUDialect::isKernel(gpuFunc) ||
          gpuFunc.getNumArguments() != kernelOperands.size()) {
        launch.emitOpError(
            "cannot derive the Vulkan descriptor ABI from the outlined kernel");
        valid = false;
        return;
      }
      llvm::SmallVector<unsigned, 8> descriptorBindings;
      descriptorBindings.reserve(kernelOperands.size());
      for (auto item : llvm::enumerate(kernelOperands)) {
        auto functionArgument =
            llvm::dyn_cast<mlir::BlockArgument>(item.value());
        if (!functionArgument ||
            functionArgument.getOwner() != &parentFunc.getBody().front()) {
          launch.emitOpError(
              "outlined Vulkan kernel operand is not an entry-block function argument");
          valid = false;
          return;
        }

        unsigned binding = functionArgument.getArgNumber();
        if (std::find(descriptorBindings.begin(), descriptorBindings.end(),
                      binding) != descriptorBindings.end()) {
          launch.emitOpError(
              "outlined Vulkan kernel aliases one function argument twice");
          valid = false;
          return;
        }
        descriptorBindings.push_back(binding);

        std::optional<mlir::spirv::StorageClass> storageClass;
        if (gpuFunc.getArgument(item.index())
                .getType()
                .isIntOrIndexOrFloat()) {
          storageClass = mlir::spirv::StorageClass::StorageBuffer;
        }
        gpuFunc.setArgAttr(
            item.index(), interfaceAttrName,
            mlir::spirv::getInterfaceVarABIAttr(
                /*descriptorSet=*/0, binding, storageClass, context));
      }
      gpuFunc->setAttr(entryPointAttrName, entryPointABI);
    });

    module.walk([&](mlir::gpu::GPUFuncOp gpuFunc) {
      if (!mlir::gpu::GPUDialect::isKernel(gpuFunc)) return;
      if (!gpuFunc->getAttr(entryPointAttrName)) {
        gpuFunc.emitOpError("has no launch-derived Vulkan descriptor ABI");
        valid = false;
        return;
      }
      for (unsigned index = 0; index < gpuFunc.getNumArguments(); ++index) {
        if (!gpuFunc.getArgAttr(index, interfaceAttrName)) {
          gpuFunc.emitOpError()
              << "has no Vulkan descriptor binding for argument " << index;
          valid = false;
        }
      }
    });

    if (!valid) signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createVulkanAttachSPIRVEntryPointABIPass() {
  return std::make_unique<VulkanAttachSPIRVEntryPointABIPass>();
}
#endif  // HAVE_MLIR

std::string pipelineCacheKey(const std::string& opName,
                             const std::string& mlirModuleStr,
                             uint32_t pushConstantBytes) {
  return opName + "|push=" + std::to_string(pushConstantBytes) + "|" +
         mlirModuleStr;
}

}  // namespace

// ── Constructor / Destructor ─────────────────────────────────────────────────

VulkanPipelineCache::VulkanPipelineCache(VkDevice device, VkPhysicalDevice physDevice,
                                         const VulkanDeviceCaps& caps,
                                         VkPipelineCache driverPipelineCache)
    : device_(device),
      physDevice_(physDevice),
      driverPipelineCache_(driverPipelineCache),
      apiVersion_(caps.apiVersion),
      fp16_(caps.fp16),
      storage16_(caps.storage16),
      fp64_(caps.fp64),
      int64_(caps.int64),
      int8_(caps.int8) {
  std::lock_guard<std::mutex> guard(registryMutex_);
  instances_.insert(this);
}

VulkanPipelineCache::~VulkanPipelineCache() {
  {
    std::lock_guard<std::mutex> guard(registryMutex_);
    instances_.erase(this);
  }

  // The owning replay handle/device context proves its device work complete
  // before destroying the cache. Only that lifetime boundary is strong enough
  // to reclaim pipelines referenced by previously recorded command buffers.
  std::lock_guard<std::mutex> guard(mutex_);
  for (auto& entry : pipelineCache_) {
    destroyCachedPipeline(entry.second);
  }
  pipelineCache_.clear();
  for (auto& pipeline : retiredPipelines_) {
    destroyCachedPipeline(pipeline);
  }
  retiredPipelines_.clear();
}

// ── Public API ───────────────────────────────────────────────────────────────

VkPipeline VulkanPipelineCache::getOrCompile(
    const std::string& opName, const std::string& mlirModuleStr,
    VkDevice device, uint32_t pushConstantBytes) {
  return getOrCompile(opName, mlirModuleStr, device, GraphCompilationPolicy{},
                      pushConstantBytes);
}

VkPipeline VulkanPipelineCache::getOrCompile(
    const std::string& opName, const std::string& mlirModuleStr,
    VkDevice device, const GraphCompilationPolicy& compilationPolicy,
    uint32_t pushConstantBytes) {
  std::lock_guard<std::mutex> guard(mutex_);

  const VulkanSpirvDiskCache::DeviceCapsKey capsKey{
      apiVersion_, fp16_, storage16_, fp64_, int64_, int8_};
  const bool allowRuntimeCompilation =
      compilationPolicy.runtimeCompilationAllowed;
  const std::string& artifactDirectory =
      compilationPolicy.runtimeArtifactDirectory;
  const bool diskCacheActive =
      allowRuntimeCompilation && VulkanSpirvDiskCache::active();

  std::string diskKey;
  SpirvModule spirv;
  bool loadedFromDisk = false;

  // A precompiled-only caller must prove a device-compatible bundle artifact
  // exists even when another context already populated the in-memory cache.
  if (!allowRuntimeCompilation) {
    if (artifactDirectory.empty()) {
      DSP_DIAG(COMPILE,
               "VulkanPipelineCache: precompiled-only MISS for op '%s' "
               "(bundle artifact directory is empty)",
               opName.c_str());
      return VK_NULL_HANDLE;
    }
    loadedFromDisk = VulkanSpirvDiskCache::loadCompatibleFromDirectory(
        artifactDirectory, mlirModuleStr, pushConstantBytes, capsKey, opName,
        spirv.bytecode, spirv.descriptorBindings, &diskKey);
    if (!loadedFromDisk) {
      diskKey = VulkanSpirvDiskCache::computeKey(
          mlirModuleStr, pushConstantBytes, capsKey);
      DSP_DIAG(COMPILE,
               "VulkanPipelineCache: precompiled-only MISS for op '%s' "
               "exactKey=%s directory=%s (compatible targets also checked)",
               opName.c_str(), diskKey.c_str(), artifactDirectory.c_str());
      return VK_NULL_HANDLE;
    }
  }

  // The Vulkan pipeline layout is part of pipeline identity. Keep the
  // push-constant interface in the key without special-casing operation names.
  const std::string cacheKey =
      pipelineCacheKey(opName, mlirModuleStr, pushConstantBytes);
  auto it = pipelineCache_.find(cacheKey);
  if (it != pipelineCache_.end()) {
    totalCacheHits_.fetch_add(1, std::memory_order_relaxed);
    DSP_DIAG(COMPILE, "VulkanPipelineCache: cache HIT for op '%s'",
             opName.c_str());
    return it->second.pipeline;
  }

  DSP_DIAG(COMPILE, "VulkanPipelineCache: cache MISS for op '%s'",
           opName.c_str());

  // Prefer bundle-owned artifacts. Development/JVM callers may then consult the
  // process cache and finally lower MLIR on a miss.
  if (allowRuntimeCompilation && !artifactDirectory.empty()) {
    loadedFromDisk = VulkanSpirvDiskCache::loadCompatibleFromDirectory(
        artifactDirectory, mlirModuleStr, pushConstantBytes, capsKey, opName,
        spirv.bytecode, spirv.descriptorBindings, &diskKey);
    if (loadedFromDisk) {
      DSP_DIAG(COMPILE,
               "VulkanPipelineCache: bundle artifact HIT for op '%s'",
               opName.c_str());
    }
  }
  if (allowRuntimeCompilation && !loadedFromDisk && diskCacheActive) {
    if (diskKey.empty()) {
      diskKey = VulkanSpirvDiskCache::computeKey(
          mlirModuleStr, pushConstantBytes, capsKey);
    }
    loadedFromDisk = VulkanSpirvDiskCache::load(
        diskKey, opName, spirv.bytecode, spirv.descriptorBindings);
    if (loadedFromDisk) {
      DSP_DIAG(COMPILE,
               "VulkanPipelineCache: Tier-1 disk HIT for op '%s' — "
               "MLIR pipeline skipped",
               opName.c_str());
    }
  }
  if (!loadedFromDisk) {
    // This branch is unreachable for precompiled-only callers: they returned on
    // the validated bundle miss above.
    spirv = mlirToSpirv(mlirModuleStr);
    if (spirv.empty()) {
      sd_printf("VulkanPipelineCache: mlirToSpirv failed for op '%s'\n",
                opName.c_str());
      return VK_NULL_HANDLE;
    }
  }

  CachedPipeline cp;
  cp.descriptorBindings = spirv.descriptorBindings;
  cp.pushConstantBytes = pushConstantBytes;
  cp.pipeline = createComputePipeline(
      spirv.bytecode, cp.descriptorBindings, device, pushConstantBytes,
      cp.layout, cp.descSetLayout);
  if (cp.pipeline == VK_NULL_HANDLE) {
    sd_printf("VulkanPipelineCache: createComputePipeline failed for op '%s'\n",
              opName.c_str());
    return VK_NULL_HANDLE;
  }

  // Persist only artifacts generated by a caller that explicitly allowed
  // runtime compilation.
  if (diskCacheActive && !loadedFromDisk) {
    if (diskKey.empty()) {
      diskKey = VulkanSpirvDiskCache::computeKey(
          mlirModuleStr, pushConstantBytes, capsKey);
    }
    VulkanSpirvDiskCache::store(
        diskKey, opName, spirv.bytecode, spirv.descriptorBindings,
        pushConstantBytes, capsKey, mlirModuleStr);
  }

  pipelineCache_[cacheKey] = cp;
  DSP_DIAG(COMPILE,
           "VulkanPipelineCache: cached new pipeline for op '%s' (total=%zu)",
           opName.c_str(), pipelineCache_.size());
  return cp.pipeline;
}

void VulkanPipelineCache::clear() {
  std::lock_guard<std::mutex> guard(mutex_);
  retiredPipelines_.reserve(retiredPipelines_.size() + pipelineCache_.size());
  for (auto& entry : pipelineCache_) {
    // Vulkan command buffers retain raw pipeline/layout handles. Logical cache
    // invalidation removes lookup entries immediately, but physical reclamation
    // is deferred to cache destruction after the owner proves device completion.
    retiredPipelines_.push_back(entry.second);
  }
  pipelineCache_.clear();
}

size_t VulkanPipelineCache::size() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return pipelineCache_.size();
}

VkPipelineLayout VulkanPipelineCache::getPipelineLayout(
    const std::string& opName, const std::string& mlirModuleStr,
    uint32_t pushConstantBytes) const {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto it = pipelineCache_.find(
      pipelineCacheKey(opName, mlirModuleStr, pushConstantBytes));
  return it == pipelineCache_.end() ? VK_NULL_HANDLE : it->second.layout;
}

VkDescriptorSetLayout VulkanPipelineCache::getDescriptorSetLayout(
    const std::string& opName, const std::string& mlirModuleStr,
    uint32_t pushConstantBytes) const {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto it = pipelineCache_.find(
      pipelineCacheKey(opName, mlirModuleStr, pushConstantBytes));
  return it == pipelineCache_.end() ? VK_NULL_HANDLE
                                    : it->second.descSetLayout;
}

std::vector<uint32_t> VulkanPipelineCache::getDescriptorBindings(
    const std::string& opName, const std::string& mlirModuleStr,
    uint32_t pushConstantBytes) const {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto it = pipelineCache_.find(
      pipelineCacheKey(opName, mlirModuleStr, pushConstantBytes));
  return it == pipelineCache_.end() ? std::vector<uint32_t>{}
                                    : it->second.descriptorBindings;
}

uint64_t VulkanPipelineCache::totalKernelLaunches() {
  return totalKernelLaunches_.load(std::memory_order_relaxed);
}

uint64_t VulkanPipelineCache::totalCacheHits() {
  return totalCacheHits_.load(std::memory_order_relaxed);
}

void VulkanPipelineCache::recordKernelLaunches(uint64_t launches) {
  totalKernelLaunches_.fetch_add(launches, std::memory_order_relaxed);
}

void VulkanPipelineCache::resetCounters() {
  totalKernelLaunches_.store(0, std::memory_order_relaxed);
  totalCacheHits_.store(0, std::memory_order_relaxed);
}

void VulkanPipelineCache::invalidateAll() {
  std::lock_guard<std::mutex> registryGuard(registryMutex_);
  for (auto* cache : instances_) {
    if (cache != nullptr) cache->clear();
  }
}

// ── Private helpers ──────────────────────────────────────────────────────────

VulkanPipelineCache::SpirvModule VulkanPipelineCache::mlirToSpirv(
    const std::string& mlirModuleStr) {
#if defined(HAVE_MLIR) && HAVE_MLIR
  // Create a fresh MLIR context and load ALL dialects required by the combined
  // lowering pipeline:
  //   • Linalg / Arith / MemRef / SCF / Func — used by VulkanOpLowerings patterns
  //   • GPU / SPIR-V                          — used by the existing GPU→SPIR-V passes
  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::linalg::LinalgDialect>();
  ctx.loadDialect<mlir::arith::ArithDialect>();
  ctx.loadDialect<mlir::memref::MemRefDialect>();
  ctx.loadDialect<mlir::scf::SCFDialect>();
  ctx.loadDialect<mlir::func::FuncDialect>();
  ctx.loadDialect<mlir::gpu::GPUDialect>();
  ctx.loadDialect<mlir::spirv::SPIRVDialect>();
#ifdef SD_MLIR_HAS_MATH_DIALECT
  ctx.loadDialect<mlir::math::MathDialect>();
#endif

  // Parse the textual MLIR module.
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
      mlir::parseSourceString<mlir::ModuleOp>(mlirModuleStr, &ctx);
  if (!moduleOp) {
    sd_printf("VulkanPipelineCache: failed to parse MLIR module:\n%s\n",
              mlirModuleStr.c_str());
    return {};
  }

  // Describe the exact contract enabled on this logical device. Vulkan 1.0,
  // 1.1, and 1.2 admit SPIR-V 1.0, 1.3, and 1.5 respectively. Optional numeric
  // capabilities are present only when VulkanDeviceContext enabled the matching
  // feature in VkDeviceCreateInfo; the compiler may not assume more than that.
  mlir::spirv::Version targetVersion = mlir::spirv::Version::V_1_0;
  if (apiVersion_ >= VK_API_VERSION_1_2) {
    targetVersion = mlir::spirv::Version::V_1_5;
  } else if (apiVersion_ >= VK_API_VERSION_1_1) {
    targetVersion = mlir::spirv::Version::V_1_3;
  }

  llvm::SmallVector<mlir::spirv::Capability, 6> targetCapabilities{
      mlir::spirv::Capability::Shader};
  llvm::SmallVector<mlir::spirv::Extension, 2> targetExtensions;

  // StorageBuffer became core in SPIR-V 1.3. The Vulkan 1.0 target needs its
  // KHR extension explicitly in the target environment.
  if (apiVersion_ < VK_API_VERSION_1_1) {
    targetExtensions.push_back(
        mlir::spirv::Extension::SPV_KHR_storage_buffer_storage_class);
  }
  if (fp16_) {
    targetCapabilities.push_back(mlir::spirv::Capability::Float16);
  }
  if (storage16_) {
    targetCapabilities.push_back(
        mlir::spirv::Capability::StorageBuffer16BitAccess);
    if (apiVersion_ < VK_API_VERSION_1_1) {
      targetExtensions.push_back(mlir::spirv::Extension::SPV_KHR_16bit_storage);
    }
  }
  if (fp64_) {
    targetCapabilities.push_back(mlir::spirv::Capability::Float64);
  }
  if (int64_) {
    targetCapabilities.push_back(mlir::spirv::Capability::Int64);
  }
  if (int8_) {
    targetCapabilities.push_back(mlir::spirv::Capability::Int8);
  }

  auto targetTriple = mlir::spirv::VerCapExtAttr::get(
      targetVersion, targetCapabilities, targetExtensions, &ctx);
  auto targetEnv = mlir::spirv::TargetEnvAttr::get(
      targetTriple, mlir::spirv::getDefaultResourceLimits(&ctx),
      mlir::spirv::ClientAPI::Vulkan);
  (*moduleOp)->setAttr(mlir::spirv::getTargetEnvAttrName(), targetEnv);

  // ── Pass pipeline ──────────────────────────────────────────────────────────
  //
  // Stage 1 — nd4j VulkanOpLowerings (matmul, rms_norm, rope → SCF/Arith).
  //   Applied as a partial conversion so that ops we do NOT own are left
  //   untouched for Stage 2 and later passes.
  //
  // Stage 2 — Standard high-level to SPIR-V conversions.
  //   Arith, MemRef, SCF, Func, Math → SPIR-V dialect ops.
  //   GPU dialect → SPIR-V entry-point structure.
  //
  // Stage 3 — SPIR-V cleanup.
  //   LowerABIAttributes + UpdateVCE ensure the SPIR-V module declares the
  //   correct capabilities and extensions for the target Vulkan version.
  //
  // Stage 4 — Final canonicalisation.
  //
  // FindMLIR validates this complete GPU-to-SPIR-V pipeline at configure time;
  // an incomplete package is rejected before any backend source is compiled.

  mlir::PassManager pm(&ctx);

  // ── Stage 1: VulkanOpLowerings (our custom patterns) ─────────────────────
  // createVulkanOpLoweringPass() wraps populateVulkanLoweringPatterns() in an
  // OperationPass<ModuleOp> that uses applyPartialConversion.
  pm.addPass(sd::graph::createVulkanOpLoweringPass());

  // ── Intermediate cleanup after Stage 1 ───────────────────────────────────
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // ── Stage 2: outline the GPU kernel and lower it to SPIR-V ───────────────
  // VulkanOpLowerings emits gpu.launch for recordable operations. The official
  // outlining pass captures its operands in gpu.func. GPUToSPIRV requires an
  // explicit entry-point ABI, then performs the arith/memref/SCF conversions
  // inside that kernel while creating spirv.module. Mapping memory spaces is
  // required for Vulkan storage-buffer descriptor bindings.
  pm.addPass(mlir::createGpuLaunchSinkIndexComputationsPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(createVulkanAttachSPIRVEntryPointABIPass());
#ifdef SD_VULKAN_HAS_MATH_TO_SPIRV
  // GPUToSPIRV does not own Math dialect operations. Run MLIR's standard
  // MathToSPIRV conversion first, matching the shared Vulkan MLIR pipeline.
  pm.addPass(mlir::createConvertMathToSPIRVPass());
#endif
  pm.addPass(mlir::createConvertGPUToSPIRVPass(/*mapMemorySpace=*/true));

  // ── Stage 3: SPIR-V cleanup ───────────────────────────────────────────────
  // MLIR 20 removed buildSPIRVModulePassPipeline + PassPipelineOptions.
  // Replace with the two constituent passes it used to call:
  //   1. LowerABIAttributes — lowers spirv ABI attributes to spec constants / variables
  //   2. UpdateVCE         — updates spirv module capabilities/extensions for target
  pm.addNestedPass<mlir::spirv::ModuleOp>(
      mlir::spirv::createSPIRVLowerABIAttributesPass());
  pm.addNestedPass<mlir::spirv::ModuleOp>(
      mlir::spirv::createSPIRVUpdateVCEPass());

  // ── Stage 4: Final canonicalisation ──────────────────────────────────────
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (mlir::failed(pm.run(*moduleOp))) {
    sd_printf("VulkanPipelineCache: MLIR pass pipeline failed "
              "(op=mlirToSpirv, stages=VulkanOpLowerings+GPU2SPIRV)\n", 0);
    moduleOp->dump();
    return {};
  }

  // Read the descriptor ABI from the lowered SPIR-V module itself. The same
  // GlobalVariable operations are serialized into the shader, so the Vulkan
  // layout cannot drift from the compiled entry-point interface.
  SpirvModule result;
  bool foundModule = false;
  bool interfaceValid = true;
  moduleOp->walk([&](mlir::spirv::ModuleOp spvModule) {
    if (foundModule) return;  // Exactly one kernel module is expected.
    foundModule = true;

    spvModule.walk([&](mlir::spirv::GlobalVariableOp global) {
      auto bindingAttr = global->getAttrOfType<mlir::IntegerAttr>("binding");
      auto setAttr = global->getAttrOfType<mlir::IntegerAttr>("descriptor_set");
      if (!bindingAttr && !setAttr) return;

      if (!bindingAttr || !setAttr || setAttr.getInt() != 0 ||
          bindingAttr.getInt() < 0 ||
          static_cast<uint64_t>(bindingAttr.getInt()) >
              std::numeric_limits<uint32_t>::max() ||
          global.storageClass() != mlir::spirv::StorageClass::StorageBuffer) {
        interfaceValid = false;
        return;
      }
      result.descriptorBindings.push_back(
          static_cast<uint32_t>(bindingAttr.getInt()));
    });

    std::sort(result.descriptorBindings.begin(), result.descriptorBindings.end());
    if (result.descriptorBindings.empty() ||
        std::adjacent_find(result.descriptorBindings.begin(),
                           result.descriptorBindings.end()) !=
            result.descriptorBindings.end()) {
      interfaceValid = false;
      return;
    }

    llvm::SmallVector<uint32_t> spirvBinary;
    if (mlir::failed(mlir::spirv::serialize(spvModule, spirvBinary))) {
      sd_printf("VulkanPipelineCache: SPIR-V serialisation failed "
                "(op=mlirToSpirv)\n", 0);
      interfaceValid = false;
      return;
    }
    result.bytecode.assign(spirvBinary.begin(), spirvBinary.end());
  });

  if (!foundModule || !interfaceValid || result.empty()) {
    sd_printf("VulkanPipelineCache: lowered SPIR-V has no valid set-zero "
              "storage-buffer interface (op=mlirToSpirv)\n", 0);
    return {};
  }

  DSP_DIAG(COMPILE, "VulkanPipelineCache: MLIR→SPIR-V OK (%zu words, %zu bindings)",
           result.bytecode.size(), result.descriptorBindings.size());
  return result;
#else
  (void)mlirModuleStr;
  DSP_DIAG(COMPILE,
           "VulkanPipelineCache: runtime MLIR compiler unavailable; "
           "a precompiled SPIR-V artifact is required");
  return {};
#endif
}

VkPipeline VulkanPipelineCache::createComputePipeline(
    const std::vector<uint32_t>& spirvBytecode,
    const std::vector<uint32_t>& descriptorBindings, VkDevice device,
    uint32_t pushConstantBytes, VkPipelineLayout& outLayout,
    VkDescriptorSetLayout& outDescLayout) {
  outLayout = VK_NULL_HANDLE;
  outDescLayout = VK_NULL_HANDLE;

  if (device == VK_NULL_HANDLE || device != device_ ||
      (pushConstantBytes & 3u) != 0u) {
    sd_printf("VulkanPipelineCache: invalid device or push-constant alignment "
              "(bytes=%u)\n",
              pushConstantBytes);
    return VK_NULL_HANDLE;
  }

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physDevice_, &properties);
  if (pushConstantBytes > properties.limits.maxPushConstantsSize) {
    sd_printf("VulkanPipelineCache: push-constant interface exceeds device "
              "limit (requested=%u limit=%u)\n",
              pushConstantBytes, properties.limits.maxPushConstantsSize);
    return VK_NULL_HANDLE;
  }

  // --- 1. Create VkShaderModule ---
  VkShaderModuleCreateInfo shaderInfo = {};
  shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderInfo.codeSize = spirvBytecode.size() * sizeof(uint32_t);
  shaderInfo.pCode = spirvBytecode.data();

  VkShaderModule shaderModule = VK_NULL_HANDLE;
  VkResult result = vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanPipelineCache: vkCreateShaderModule failed (result=%d)\n",
              static_cast<int>(result));
    return VK_NULL_HANDLE;
  }

  // --- 2. Create VkDescriptorSetLayout ---
  // The bindings come from the lowered SPIR-V GlobalVariable interface. This
  // makes the Vulkan layout identical to the shader ABI for every kernel.
  std::vector<VkDescriptorSetLayoutBinding> bindings(descriptorBindings.size());
  for (size_t i = 0; i < descriptorBindings.size(); ++i) {
    bindings[i].binding = descriptorBindings[i];
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[i].pImmutableSamplers = nullptr;
  }

  VkDescriptorSetLayoutCreateInfo descLayoutInfo = {};
  descLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  descLayoutInfo.pBindings = bindings.data();

  result = vkCreateDescriptorSetLayout(device, &descLayoutInfo, nullptr, &outDescLayout);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanPipelineCache: vkCreateDescriptorSetLayout failed (result=%d)\n",
              static_cast<int>(result));
    vkDestroyShaderModule(device, shaderModule, nullptr);
    outDescLayout = VK_NULL_HANDLE;
    return VK_NULL_HANDLE;
  }

  // --- 3. Create VkPipelineLayout ---
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = pushConstantBytes;

  VkPipelineLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &outDescLayout;
  layoutInfo.pushConstantRangeCount = pushConstantBytes == 0 ? 0u : 1u;
  layoutInfo.pPushConstantRanges =
      pushConstantBytes == 0 ? nullptr : &pushConstantRange;

  result = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &outLayout);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanPipelineCache: vkCreatePipelineLayout failed (result=%d)\n",
              static_cast<int>(result));
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyDescriptorSetLayout(device, outDescLayout, nullptr);
    outDescLayout = VK_NULL_HANDLE;
    outLayout = VK_NULL_HANDLE;
    return VK_NULL_HANDLE;
  }

  // --- 4. Create VkComputePipeline ---
  VkComputePipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = shaderModule;
  pipelineInfo.stage.pName = "main";
  pipelineInfo.layout = outLayout;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  pipelineInfo.basePipelineIndex = -1;

  VkPipeline pipeline = VK_NULL_HANDLE;
  // Tier 2 (ADR 0115): route creation through the device-lifetime
  // VkPipelineCache so the driver reuses compiled shader state and the blob
  // accumulates for disk persistence. VK_NULL_HANDLE when unset/disabled.
  result = vkCreateComputePipelines(device, driverPipelineCache_, 1, &pipelineInfo,
                                    nullptr, &pipeline);

  // --- 5. Always destroy the shader module ---
  // The compiled shader code is baked into the VkPipeline; the VkShaderModule
  // object is no longer needed after pipeline creation.
  vkDestroyShaderModule(device, shaderModule, nullptr);

  if (result != VK_SUCCESS) {
    sd_printf("VulkanPipelineCache: vkCreateComputePipelines failed (result=%d)\n",
              static_cast<int>(result));
    vkDestroyPipelineLayout(device, outLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, outDescLayout, nullptr);
    outLayout = VK_NULL_HANDLE;
    outDescLayout = VK_NULL_HANDLE;
    return VK_NULL_HANDLE;
  }

  return pipeline;
}

void VulkanPipelineCache::destroyCachedPipeline(CachedPipeline& cp) {
  if (device_ == VK_NULL_HANDLE) return;

  // Destroy in reverse creation order.
  if (cp.pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(device_, cp.pipeline, nullptr);
    cp.pipeline = VK_NULL_HANDLE;
  }
  if (cp.layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device_, cp.layout, nullptr);
    cp.layout = VK_NULL_HANDLE;
  }
  if (cp.descSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device_, cp.descSetLayout, nullptr);
    cp.descSetLayout = VK_NULL_HANDLE;
  }
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
