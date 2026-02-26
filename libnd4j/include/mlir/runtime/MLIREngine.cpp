/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "MLIREngine.h"

#ifdef HAVE_MLIR

#if __has_include("mlir/Dialect/Affine/IR/AffineOps.h")
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#define SD_MLIR_HAS_AFFINE_DIALECT 1
#endif
#if __has_include("mlir/Dialect/Affine/Passes.h")
#include "mlir/Dialect/Affine/Passes.h"
#define SD_MLIR_HAS_AFFINE_PASSES 1
#endif

#include "mlir/Dialect/Arith/IR/Arith.h"
#if __has_include("mlir/Dialect/ArmNeon/ArmNeonDialect.h")
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#define SD_MLIR_HAS_ARMNEON_DIALECT 1
#endif
#if __has_include("mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h")
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#define SD_MLIR_HAS_ARMSVE_DIALECT 1
#endif
#if __has_include("mlir/Dialect/ArmSME/IR/ArmSME.h")
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#define SD_MLIR_HAS_ARMSME_DIALECT 1
#endif
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#if __has_include("mlir/Dialect/X86Vector/X86VectorDialect.h")
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#define SD_MLIR_HAS_X86VECTOR_DIALECT 1
#endif
#if __has_include("mlir/Dialect/AMX/AMXDialect.h")
#include "mlir/Dialect/AMX/AMXDialect.h"
#define SD_MLIR_HAS_AMX_DIALECT 1
#endif
#if __has_include("mlir/Dialect/AMX/Transforms.h")
#include "mlir/Dialect/AMX/Transforms.h"
#define SD_MLIR_HAS_AMX_TRANSFORMS 1
#endif

#if __has_include("mlir/Conversion/AffineToStandard/AffineToStandard.h")
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#define SD_MLIR_HAS_AFFINE_TO_STANDARD 1
#endif
#if __has_include("mlir/Conversion/ArmNeon2dToIntr/ArmNeon2dToIntr.h")
#include "mlir/Conversion/ArmNeon2dToIntr/ArmNeon2dToIntr.h"
#define SD_MLIR_HAS_ARMNEON2D_TO_INTR_PASS 1
#endif
#if __has_include("mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h")
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#define SD_MLIR_HAS_ARMSME_TO_LLVM_PASS 1
#endif
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#if __has_include("mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h")
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#define SD_MLIR_HAS_VECTOR_TO_LLVM_PASS 1
#endif
#if __has_include("mlir/Conversion/VectorToArmSME/VectorToArmSME.h")
#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"
#define SD_MLIR_HAS_VECTOR_TO_ARMSME_PASS 1
#endif
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#if __has_include("mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h")
#include "mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h"
#define SD_MLIR_HAS_X86VECTOR_TRANSLATION 1
#endif
#if __has_include("mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h")
#include "mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h"
#define SD_MLIR_HAS_ARMNEON_TRANSLATION 1
#endif
#if __has_include("mlir/Target/LLVMIR/Dialect/ArmSVE/ArmSVEToLLVMIRTranslation.h")
#include "mlir/Target/LLVMIR/Dialect/ArmSVE/ArmSVEToLLVMIRTranslation.h"
#define SD_MLIR_HAS_ARMSVE_TRANSLATION 1
#endif
#if __has_include("mlir/Target/LLVMIR/Dialect/ArmSME/ArmSMEToLLVMIRTranslation.h")
#include "mlir/Target/LLVMIR/Dialect/ArmSME/ArmSMEToLLVMIRTranslation.h"
#define SD_MLIR_HAS_ARMSME_TRANSLATION 1
#endif
#if __has_include("mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h")
#include "mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h"
#define SD_MLIR_HAS_AMX_TRANSLATION 1
#endif
#include "mlir/Transforms/Passes.h"

#ifdef MLIR_ENABLE_GPU
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#endif

#include "llvm/Support/TargetSelect.h"

#include <sstream>

namespace sd {
namespace mlir_runtime {

namespace {

bool isArmHostCompilationTarget() {
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
    return true;
#else
    return false;
#endif
}

}  // namespace

//===----------------------------------------------------------------------===//
// CompiledKernel Implementation
//===----------------------------------------------------------------------===//

CompiledKernel::CompiledKernel(std::unique_ptr<mlir::ExecutionEngine> engine,
                               const std::string& entryPoint)
    : _engine(std::move(engine)), _entryPoint(entryPoint) {}

CompiledKernel::~CompiledKernel() = default;

CompiledKernel::CompiledKernel(CompiledKernel&& other) noexcept
    : _engine(std::move(other._engine)), _entryPoint(std::move(other._entryPoint)) {}

CompiledKernel& CompiledKernel::operator=(CompiledKernel&& other) noexcept {
    if (this != &other) {
        _engine = std::move(other._engine);
        _entryPoint = std::move(other._entryPoint);
    }
    return *this;
}

bool CompiledKernel::execute(const std::vector<NDArray*>& inputs,
                             const std::vector<NDArray*>& outputs) {
    if (!isValid()) {
        return false;
    }

    // TODO: Implement actual execution via ExecutionEngine
    // This will involve:
    // 1. Converting NDArray buffers to MLIR MemRef descriptors
    // 2. Calling the JIT-compiled function via the execution engine
    // 3. Handling results

    // Placeholder implementation
    auto invokeFn = _engine->lookup(_entryPoint);
    if (!invokeFn) {
        return false;
    }

    // For now, return true as placeholder
    return true;
}

//===----------------------------------------------------------------------===//
// MLIREngine Implementation
//===----------------------------------------------------------------------===//

MLIREngine& MLIREngine::getInstance() {
    static MLIREngine instance;
    return instance;
}

MLIREngine::MLIREngine() {
    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
}

MLIREngine::~MLIREngine() = default;

bool MLIREngine::initialize() {
    if (_initialized) {
        return true;
    }

    // Create MLIR context with all required dialects
    _context = std::make_unique<mlir::MLIRContext>();

#ifdef SD_MLIR_HAS_AMX_TRANSFORMS
    // Register AMX conversion interface so generic LLVM conversion can lower AMX ops.
    mlir::DialectRegistry amxRegistry;
    mlir::registerConvertAMXToLLVMInterface(amxRegistry);
    _context->appendDialectRegistry(amxRegistry);
#endif

    // Load required dialects
    _context->loadDialect<mlir::arith::ArithDialect>();
    _context->loadDialect<mlir::func::FuncDialect>();
    _context->loadDialect<mlir::linalg::LinalgDialect>();
    _context->loadDialect<mlir::memref::MemRefDialect>();
    _context->loadDialect<mlir::scf::SCFDialect>();
    _context->loadDialect<mlir::tensor::TensorDialect>();
    _context->loadDialect<mlir::vector::VectorDialect>();
    _context->loadDialect<mlir::LLVM::LLVMDialect>();
    _context->loadDialect<mlir::bufferization::BufferizationDialect>();
#ifdef SD_MLIR_HAS_AFFINE_DIALECT
    _context->loadDialect<mlir::affine::AffineDialect>();
#endif
#ifdef SD_MLIR_HAS_X86VECTOR_DIALECT
    _context->loadDialect<mlir::x86vector::X86VectorDialect>();
#endif
#ifdef SD_MLIR_HAS_AMX_DIALECT
    _context->loadDialect<mlir::amx::AMXDialect>();
#endif
#ifdef SD_MLIR_HAS_ARMNEON_DIALECT
    _context->loadDialect<mlir::arm_neon::ArmNeonDialect>();
#endif
#ifdef SD_MLIR_HAS_ARMSVE_DIALECT
    _context->loadDialect<mlir::arm_sve::ArmSVEDialect>();
#endif
#ifdef SD_MLIR_HAS_ARMSME_DIALECT
    _context->loadDialect<mlir::arm_sme::ArmSMEDialect>();
#endif

#ifdef MLIR_ENABLE_GPU
    _context->loadDialect<mlir::gpu::GPUDialect>();
#endif

    mlir::registerLLVMDialectTranslation(*_context);
#ifdef SD_MLIR_HAS_X86VECTOR_TRANSLATION
    mlir::registerX86VectorDialectTranslation(*_context);
#endif
#ifdef SD_MLIR_HAS_ARMNEON_TRANSLATION
    mlir::registerArmNeonDialectTranslation(*_context);
#endif
#ifdef SD_MLIR_HAS_ARMSVE_TRANSLATION
    mlir::registerArmSVEDialectTranslation(*_context);
#endif
#ifdef SD_MLIR_HAS_ARMSME_TRANSLATION
    mlir::registerArmSMEDialectTranslation(*_context);
#endif
#ifdef SD_MLIR_HAS_AMX_TRANSLATION
    mlir::registerAMXDialectTranslation(*_context);
#endif

    _initialized = true;
    return true;
}

std::shared_ptr<CompiledKernel> MLIREngine::compile(
    const std::string& opName,
    const std::vector<std::vector<int64_t>>& inputShapes,
    const std::vector<int>& inputTypes,
    const MLIRCompileOptions& options) {

    if (!_initialized && !initialize()) {
        return nullptr;
    }

    // Create MLIR module for the operation
    auto module = createModuleForOp(opName, inputShapes, inputTypes);
    if (!module) {
        return nullptr;
    }

    // Build lowering pipeline
    mlir::PassManager pm(_context.get());

    if (options.enableGPU) {
        buildGPUPipeline(pm, options);
    } else {
        buildCPUPipeline(pm, options);
    }

    // Run the lowering pipeline
    if (mlir::failed(pm.run(*module))) {
        return nullptr;
    }

#ifdef MLIR_DEBUG_DUMPS
    if (options.debugDumps) {
        module->dump();
    }
#endif

    // Create execution engine
    llvm::SmallVector<llvm::StringRef, 0> sharedLibPaths;
    auto optPipeline = mlir::makeOptimizingTransformer(
        options.optLevel, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

    auto maybeEngine = mlir::ExecutionEngine::create(
        *module, /*llvmModuleBuilder=*/nullptr,
        optPipeline, llvm::CodeGenOptLevel::Default,
        sharedLibPaths);

    if (!maybeEngine) {
        return nullptr;
    }

    auto kernel = std::make_shared<CompiledKernel>(
        std::move(*maybeEngine), opName + "_kernel");

    return kernel;
}

std::shared_ptr<CompiledKernel> MLIREngine::getOrCompile(
    const std::string& opName,
    const std::vector<std::vector<int64_t>>& inputShapes,
    const std::vector<int>& inputTypes,
    const MLIRCompileOptions& options) {

    if (!_cachingEnabled) {
        return compile(opName, inputShapes, inputTypes, options);
    }

    std::string cacheKey = generateCacheKey(opName, inputShapes, inputTypes, options);

    {
        std::lock_guard<std::mutex> lock(_cacheMutex);
        auto it = _cache.find(cacheKey);
        if (it != _cache.end()) {
            ++_cacheHits;
            return it->second;
        }
        ++_cacheMisses;
    }

    // Compile outside the lock
    auto kernel = compile(opName, inputShapes, inputTypes, options);

    if (kernel) {
        std::lock_guard<std::mutex> lock(_cacheMutex);
        _cache[cacheKey] = kernel;
    }

    return kernel;
}

void MLIREngine::clearCache() {
    std::lock_guard<std::mutex> lock(_cacheMutex);
    _cache.clear();
    _cacheHits = 0;
    _cacheMisses = 0;
}

size_t MLIREngine::getCacheSize() const {
    std::lock_guard<std::mutex> lock(_cacheMutex);
    return _cache.size();
}

void MLIREngine::setDefaultOptions(const MLIRCompileOptions& options) {
    _defaultOptions = options;
}

void MLIREngine::buildCPUPipeline(mlir::PassManager& pm,
                                  const MLIRCompileOptions& options) {
    // Canonicalization and CSE
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // Linalg optimizations
    if (options.enableOptimizations) {
#ifdef SD_MLIR_HAS_AFFINE_PASSES
        if (options.enableAffineOptimizations) {
            pm.addNestedPass<mlir::func::FuncOp>(
                mlir::affine::createSimplifyAffineStructuresPass());
            pm.addPass(mlir::affine::createLoopFusionPass());
            pm.addNestedPass<mlir::func::FuncOp>(
                mlir::affine::createAffineLoopNormalizePass());
            pm.addNestedPass<mlir::func::FuncOp>(
                mlir::affine::createAffineLoopInvariantCodeMotionPass());
        }
#endif
    }

    // Bufferization (tensor -> memref)
    pm.addPass(mlir::bufferization::createOneShotBufferizePass());

#ifdef SD_MLIR_HAS_AFFINE_TO_STANDARD
    if (options.enableAffineOptimizations) {
        pm.addPass(mlir::createLowerAffinePass());
    }
#endif

    // Lower to LLVM
    pm.addPass(mlir::createConvertSCFToCFPass());
#ifdef SD_MLIR_HAS_ARMNEON2D_TO_INTR_PASS
    if (options.enableVectorization && options.enableArmNeon && isArmHostCompilationTarget()) {
        pm.addPass(mlir::createConvertArmNeon2dToIntrPass());
    }
#endif
#if defined(SD_MLIR_HAS_VECTOR_TO_ARMSME_PASS) && defined(SD_MLIR_HAS_ARMSME_TO_LLVM_PASS)
    if (options.enableVectorization && options.enableArmSME && isArmHostCompilationTarget()) {
        pm.addPass(mlir::createConvertVectorToArmSMEPass());
        pm.addPass(mlir::createConvertArmSMEToLLVMPass(/*dumpTileLiveRanges=*/false));
    }
#endif
    pm.addPass(mlir::createConvertLinalgToLLVMPass());
#ifdef SD_MLIR_HAS_VECTOR_TO_LLVM_PASS
    if (options.enableVectorization && options.enableX86Vector) {
        pm.addPass(mlir::createConvertVectorToLLVMPass());
    }
#endif
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());

    // Final cleanup
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}

void MLIREngine::buildGPUPipeline(mlir::PassManager& pm,
                                  const MLIRCompileOptions& options) {
#ifdef MLIR_ENABLE_GPU
    // Similar to CPU but with GPU-specific passes
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // GPU mapping and lowering
    // pm.addPass(mlir::createGpuMapParallelLoopsPass());
    // pm.addPass(mlir::createConvertGpuLaunchFuncToGpuRuntimeCallsPass());

    // Lower to NVVM
    pm.addPass(mlir::createConvertGPUToNVVMPass());

    // Final cleanup
    pm.addPass(mlir::createCanonicalizerPass());
#else
    // Fall back to CPU pipeline if GPU not enabled
    buildCPUPipeline(pm, options);
#endif
}

std::string MLIREngine::generateCacheKey(
    const std::string& opName,
    const std::vector<std::vector<int64_t>>& inputShapes,
    const std::vector<int>& inputTypes,
    const MLIRCompileOptions& options) {

    std::ostringstream ss;
    ss << opName << "_";

    // Add shapes
    for (const auto& shape : inputShapes) {
        for (int64_t dim : shape) {
            ss << dim << "x";
        }
        ss << "_";
    }

    // Add types
    for (int type : inputTypes) {
        ss << type << "_";
    }

    // Add relevant options
    ss << options.optLevel << "_"
       << options.enableVectorization << "_"
       << options.enableAffineOptimizations << "_"
       << options.enableX86Vector << "_"
       << options.enableAMX << "_"
       << options.enableArmNeon << "_"
       << options.enableArmSVE << "_"
       << options.enableArmSME << "_"
       << options.enableGPU;

    return ss.str();
}

mlir::OwningOpRef<mlir::ModuleOp> MLIREngine::createModuleForOp(
    const std::string& opName,
    const std::vector<std::vector<int64_t>>& inputShapes,
    const std::vector<int>& inputTypes) {

    // Create module builder
    mlir::OpBuilder builder(_context.get());
    auto loc = builder.getUnknownLoc();

    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // TODO: Generate operation-specific MLIR code
    // This would dispatch to specific code generators based on opName
    // For now, create a placeholder function

    // Create function type based on input shapes and types
    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        auto elemType = mlir::FloatType::getF32(_context.get());  // TODO: Use inputTypes
        auto tensorType = mlir::RankedTensorType::get(inputShapes[i], elemType);
        argTypes.push_back(tensorType);
    }

    // Output type (placeholder - should be computed based on op semantics)
    auto outputType = argTypes.empty() ?
        mlir::RankedTensorType::get({1}, mlir::FloatType::getF32(_context.get())) :
        argTypes[0];

    auto funcType = builder.getFunctionType(argTypes, {outputType});
    auto func = builder.create<mlir::func::FuncOp>(loc, opName + "_kernel", funcType);

    // Create entry block
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);

    // TODO: Generate actual operation implementation
    // For now, just return the first input
    if (!entryBlock->getArguments().empty()) {
        builder.create<mlir::func::ReturnOp>(loc, entryBlock->getArgument(0));
    } else {
        auto constant = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getFloatAttr(builder.getF32Type(), 0.0));
        auto tensor = builder.create<mlir::tensor::FromElementsOp>(
            loc, mlir::ValueRange{constant});
        builder.create<mlir::func::ReturnOp>(loc, tensor.getResult());
    }

    return module;
}

} // namespace mlir_runtime
} // namespace sd

#else // !HAVE_MLIR

// Stub implementation when MLIR is not available
namespace sd {
namespace mlir_runtime {

CompiledKernel::CompiledKernel(std::unique_ptr<mlir::ExecutionEngine>,
                               const std::string&) {}
CompiledKernel::~CompiledKernel() = default;
CompiledKernel::CompiledKernel(CompiledKernel&&) noexcept = default;
CompiledKernel& CompiledKernel::operator=(CompiledKernel&&) noexcept = default;

bool CompiledKernel::execute(const std::vector<NDArray*>&,
                             const std::vector<NDArray*>&) {
    return false;
}

MLIREngine& MLIREngine::getInstance() {
    static MLIREngine instance;
    return instance;
}

MLIREngine::MLIREngine() = default;
MLIREngine::~MLIREngine() = default;

bool MLIREngine::initialize() { return false; }

std::shared_ptr<CompiledKernel> MLIREngine::compile(
    const std::string&,
    const std::vector<std::vector<int64_t>>&,
    const std::vector<int>&,
    const MLIRCompileOptions&) {
    return nullptr;
}

std::shared_ptr<CompiledKernel> MLIREngine::getOrCompile(
    const std::string&,
    const std::vector<std::vector<int64_t>>&,
    const std::vector<int>&,
    const MLIRCompileOptions&) {
    return nullptr;
}

void MLIREngine::clearCache() {}
size_t MLIREngine::getCacheSize() const { return 0; }
void MLIREngine::setDefaultOptions(const MLIRCompileOptions&) {}

void MLIREngine::buildCPUPipeline(mlir::PassManager&, const MLIRCompileOptions&) {}
void MLIREngine::buildGPUPipeline(mlir::PassManager&, const MLIRCompileOptions&) {}

std::string MLIREngine::generateCacheKey(
    const std::string&,
    const std::vector<std::vector<int64_t>>&,
    const std::vector<int>&,
    const MLIRCompileOptions&) {
    return "";
}

mlir::OwningOpRef<mlir::ModuleOp> MLIREngine::createModuleForOp(
    const std::string&,
    const std::vector<std::vector<int64_t>>&,
    const std::vector<int>&) {
    return mlir::OwningOpRef<mlir::ModuleOp>();
}

} // namespace mlir_runtime
} // namespace sd

#endif // HAVE_MLIR
