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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
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

#ifdef MLIR_ENABLE_GPU
    _context->loadDialect<mlir::gpu::GPUDialect>();
#endif

    // Register LLVM IR translation
    mlir::registerLLVMDialectTranslation(*_context);

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
        // Tiling
        // pm.addNestedPass<mlir::func::FuncOp>(
        //     mlir::linalg::createLinalgTilingPass(
        //         mlir::linalg::LinalgTilingOptions()
        //             .setTileSizes({options.tileSize, options.tileSize})));

        // Vectorization
        if (options.enableVectorization) {
            // pm.addNestedPass<mlir::func::FuncOp>(
            //     mlir::linalg::createLinalgVectorizePass());
        }
    }

    // Bufferization (tensor -> memref)
    pm.addPass(mlir::bufferization::createOneShotBufferizePass());

    // Lower to LLVM
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertLinalgToLLVMPass());
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
