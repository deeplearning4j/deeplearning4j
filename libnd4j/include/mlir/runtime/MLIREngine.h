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

#ifndef SD_MLIR_RUNTIME_MLIRENGINE_H
#define SD_MLIR_RUNTIME_MLIRENGINE_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

// Forward declarations for MLIR types
namespace mlir {
class MLIRContext;
class ModuleOp;
class PassManager;
class ExecutionEngine;
template <typename T> class OwningOpRef;
}  // namespace mlir

namespace sd {

// Forward declaration
class NDArray;

namespace mlir_runtime {

/// Extended op parameters for MLIR compilation
/// Carries output shapes, integer/float/bool args from platform helpers
struct MlirOpParams {
    std::vector<std::vector<int64_t>> outputShapes;
    std::vector<int64_t> iArgs;
    std::vector<double> tArgs;
    std::vector<bool> bArgs;
    int numOutputs = 1;
};

/// Represents a compiled kernel that can be executed
class CompiledKernel {
public:
    CompiledKernel() = default;
    CompiledKernel(std::unique_ptr<mlir::ExecutionEngine> engine,
                   const std::string& entryPoint);
    ~CompiledKernel();

    CompiledKernel(CompiledKernel&& other) noexcept;
    CompiledKernel& operator=(CompiledKernel&& other) noexcept;

    // Non-copyable
    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;

    /// Execute the compiled kernel with given inputs and outputs
    /// @param inputs Vector of input NDArray pointers
    /// @param outputs Vector of output NDArray pointers
    /// @return true if execution succeeded
    bool execute(const std::vector<NDArray*>& inputs,
                 const std::vector<NDArray*>& outputs);

    /// Check if kernel is valid and ready for execution
    bool isValid() const { return _engine != nullptr; }

    /// Get the entry point function name
    const std::string& getEntryPoint() const { return _entryPoint; }

private:
    std::unique_ptr<mlir::ExecutionEngine> _engine;
    std::string _entryPoint;
};

/// Target architecture for AOT compilation
enum class AOTTarget {
    HOST,               // Compile for the host machine (default JIT)
    AARCH64_LINUX,      // ARM64 Linux (generic)
    AARCH64_ANDROID,    // ARM64 Android (NDK target)
    X86_64_LINUX,       // x86_64 Linux
};

/// GPU compute target for Vulkan/SPIR-V pipeline
enum class MobileGPUTarget {
    NONE,               // No mobile GPU
    VULKAN_GENERIC,     // Generic Vulkan compute
    MALI,               // ARM Mali GPUs
    ADRENO,             // Qualcomm Adreno GPUs
};

/// Configuration for MLIR compilation
struct MLIRCompileOptions {
    bool enableOptimizations = true;
    bool enableVectorization = true;
    bool enableAffineOptimizations = true;
    bool enableX86Vector = true;
    bool enableAMX = true;
    bool enableArmNeon = true;
    bool enableArmSVE = true;
    bool enableArmSME = true;
    bool enableParallelization = false;
    bool enableGPU = false;
    bool enableVulkan = false;
    int optLevel = 2;  // 0-3
    int tileSize = 32;
    bool debugDumps = false;

    // AOT compilation settings
    AOTTarget aotTarget = AOTTarget::HOST;
    bool aotMode = false;  // true = AOT (emit object file), false = JIT

    // ARM-specific tuning
    int armTileSize = 16;       // Smaller tiles for ARM L1 cache (typically 32-64KB)
    int armVectorWidth = 128;   // NEON is 128-bit; SVE is scalable
    bool armDotProduct = false;  // ARMv8.2+ dot product instructions

    // Mobile GPU settings
    MobileGPUTarget mobileGPU = MobileGPUTarget::NONE;
    int vulkanWorkgroupSize = 64;  // Default Vulkan compute workgroup size
};

/// Main MLIR JIT compilation engine
/// Singleton pattern for global access
class MLIREngine {
public:
    /// Get the singleton instance
    static MLIREngine& getInstance();

    // Delete copy/move constructors
    MLIREngine(const MLIREngine&) = delete;
    MLIREngine& operator=(const MLIREngine&) = delete;
    MLIREngine(MLIREngine&&) = delete;
    MLIREngine& operator=(MLIREngine&&) = delete;

    /// Initialize the MLIR engine
    /// Must be called before any compilation
    bool initialize();

    /// Check if engine is initialized
    bool isInitialized() const { return _initialized; }

    /// Compile an operation for the given context
    /// @param opName Name of the operation (e.g., "matmul", "conv2d")
    /// @param inputShapes Vector of input shapes
    /// @param inputTypes Vector of input data types
    /// @param options Compilation options
    /// @return Compiled kernel or nullptr on failure
    std::shared_ptr<CompiledKernel> compile(
        const std::string& opName,
        const std::vector<std::vector<int64_t>>& inputShapes,
        const std::vector<int>& inputTypes,
        const MLIRCompileOptions& options = MLIRCompileOptions());

    /// Compile with extended op parameters (output shapes, iArgs, tArgs, etc.)
    std::shared_ptr<CompiledKernel> compile(
        const std::string& opName,
        const std::vector<std::vector<int64_t>>& inputShapes,
        const std::vector<int>& inputTypes,
        const MlirOpParams& params,
        const MLIRCompileOptions& options = MLIRCompileOptions());

    /// Get or compile a kernel with caching
    /// Uses cache key based on op name, shapes, and types
    std::shared_ptr<CompiledKernel> getOrCompile(
        const std::string& opName,
        const std::vector<std::vector<int64_t>>& inputShapes,
        const std::vector<int>& inputTypes,
        const MLIRCompileOptions& options = MLIRCompileOptions());

    /// Get or compile with extended op parameters
    std::shared_ptr<CompiledKernel> getOrCompile(
        const std::string& opName,
        const std::vector<std::vector<int64_t>>& inputShapes,
        const std::vector<int>& inputTypes,
        const MlirOpParams& params,
        const MLIRCompileOptions& options = MLIRCompileOptions());

    /// Clear all cached kernels
    void clearCache();

    /// Get cache statistics
    size_t getCacheSize() const;
    size_t getCacheHits() const { return _cacheHits; }
    size_t getCacheMisses() const { return _cacheMisses; }

    /// Set default compilation options
    void setDefaultOptions(const MLIRCompileOptions& options);
    const MLIRCompileOptions& getDefaultOptions() const { return _defaultOptions; }

    /// Enable/disable JIT caching
    void setCachingEnabled(bool enabled) { _cachingEnabled = enabled; }
    bool isCachingEnabled() const { return _cachingEnabled; }

    /// Get the MLIR context (for building modules externally)
    mlir::MLIRContext* getContext() { return _context.get(); }

    /// Compile a pre-built MLIR module through the CPU lowering pipeline.
    /// This bypasses createModuleForOp() — the caller builds the module directly.
    /// @param module Pre-built MLIR module (takes ownership)
    /// @param entryPoint Name of the entry function in the module
    /// @param options Compilation options
    /// @return Compiled kernel or nullptr on failure
    std::shared_ptr<CompiledKernel> compileModule(
        mlir::OwningOpRef<mlir::ModuleOp> module,
        const std::string& entryPoint,
        const MLIRCompileOptions& options = MLIRCompileOptions());

    /// AOT compile a pre-built MLIR module to an object file for cross-compilation.
    /// @param module Pre-built MLIR module (takes ownership)
    /// @param entryPoint Name of the entry function in the module
    /// @param outputPath Path to write the .o object file
    /// @param options Compilation options (must have aotMode=true, aotTarget set)
    /// @return true if compilation succeeded
    bool compileToObjectFile(
        mlir::OwningOpRef<mlir::ModuleOp> module,
        const std::string& entryPoint,
        const std::string& outputPath,
        const MLIRCompileOptions& options);

    /// AOT compile a pre-built MLIR module to Vulkan SPIR-V binary.
    /// @param module Pre-built MLIR module (takes ownership)
    /// @param entryPoint Name of the entry function
    /// @param outputPath Path to write the .spv SPIR-V binary
    /// @param options Compilation options (must have enableVulkan=true)
    /// @return true if compilation succeeded
    bool compileToSPIRV(
        mlir::OwningOpRef<mlir::ModuleOp> module,
        const std::string& entryPoint,
        const std::string& outputPath,
        const MLIRCompileOptions& options);

    /// Get the LLVM target triple string for a given AOT target.
    static std::string getTargetTriple(AOTTarget target);

    /// Check if the current host is an ARM target.
    static bool isArmHost();

    /// Get recommended compile options for an ARM Android target.
    static MLIRCompileOptions getArmAndroidDefaults();

private:
    MLIREngine();
    ~MLIREngine();

    /// Build the CPU lowering pipeline
    void buildCPUPipeline(mlir::PassManager& pm, const MLIRCompileOptions& options);

    /// Build the CPU lowering pipeline with ARM-specific optimizations
    void buildARMCPUPipeline(mlir::PassManager& pm, const MLIRCompileOptions& options);

    /// Build the GPU lowering pipeline (NVVM)
    void buildGPUPipeline(mlir::PassManager& pm, const MLIRCompileOptions& options);

    /// Build the Vulkan/SPIR-V pipeline for mobile GPUs
    void buildVulkanPipeline(mlir::PassManager& pm, const MLIRCompileOptions& options);

    /// Generate cache key from operation parameters
    std::string generateCacheKey(
        const std::string& opName,
        const std::vector<std::vector<int64_t>>& inputShapes,
        const std::vector<int>& inputTypes,
        const MLIRCompileOptions& options);

    /// Create MLIR module for operation
    mlir::OwningOpRef<mlir::ModuleOp> createModuleForOp(
        const std::string& opName,
        const std::vector<std::vector<int64_t>>& inputShapes,
        const std::vector<int>& inputTypes);

    /// Create MLIR module with extended op parameters
    mlir::OwningOpRef<mlir::ModuleOp> createModuleForOp(
        const std::string& opName,
        const std::vector<std::vector<int64_t>>& inputShapes,
        const std::vector<int>& inputTypes,
        const MlirOpParams& params);

private:
    std::unique_ptr<mlir::MLIRContext> _context;
    bool _initialized = false;
    bool _cachingEnabled = true;

    // Kernel cache
    mutable std::mutex _cacheMutex;
    std::unordered_map<std::string, std::shared_ptr<CompiledKernel>> _cache;
    mutable size_t _cacheHits = 0;
    mutable size_t _cacheMisses = 0;

    MLIRCompileOptions _defaultOptions;
};

} // namespace mlir_runtime
} // namespace sd

#endif // SD_MLIR_RUNTIME_MLIRENGINE_H
