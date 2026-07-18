/* ******************************************************************************
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

#ifndef SD_KERNEL_MANAGER_H
#define SD_KERNEL_MANAGER_H

#include <execution/Engine.h>
#include <system/common.h>

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN

/**
 * Information about a kernel implementation
 */
struct SD_LIB_EXPORT KernelInfo {
  std::string opName;           // Operation name (e.g., "conv2d")
  LongType opHash;              // Operation hash
  samediff::Engine engine;      // Backend engine (CUDA, oneDNN, etc.)
  std::string engineName;       // Human-readable engine name
  bool isEnabled;               // Whether this kernel is currently enabled
  bool isUsable;                // Whether this kernel can run on current hardware
  double avgTimeNanos;          // Average execution time (0 if not benchmarked)
  int64_t executionCount;       // Number of times executed
  std::string description;      // Optional description

  KernelInfo()
      : opHash(0), engine(samediff::ENGINE_CPU), isEnabled(true),
        isUsable(false), avgTimeNanos(0), executionCount(0) {}
};

/**
 * Operation info with all available kernels
 */
struct SD_LIB_EXPORT OpKernelInfo {
  std::string opName;
  LongType opHash;
  std::vector<KernelInfo> kernels;
  samediff::Engine preferredEngine;  // Currently preferred engine
  bool hasNativeImplementation;      // Has fallback native CPU implementation

  OpKernelInfo() : opHash(0), preferredEngine(samediff::ENGINE_CPU),
                   hasNativeImplementation(true) {}
};

/**
 * KernelManager provides a unified interface for discovering, enabling,
 * disabling, and configuring kernel implementations for operations.
 *
 * Usage:
 *   auto& km = KernelManager::getInstance();
 *
 *   // List all available kernels for an operation
 *   auto info = km.getOpKernelInfo("conv2d");
 *   for (auto& kernel : info.kernels) {
 *     printf("%s: %s (enabled=%d)\n", kernel.opName.c_str(),
 *            kernel.engineName.c_str(), kernel.isEnabled);
 *   }
 *
 *   // Disable a specific kernel
 *   km.disableKernel("conv2d", samediff::ENGINE_CUDA);
 *
 *   // Set preferred engine for an operation
 *   km.setPreferredEngine("conv2d", samediff::ENGINE_ONEDNN);
 */
class SD_LIB_EXPORT KernelManager {
 private:
  // Per-operation disabled engines
  std::unordered_map<LongType, std::set<samediff::Engine>> _disabledKernels;

  // Per-operation preferred engine
  std::unordered_map<LongType, samediff::Engine> _preferredEngines;

  // Global disabled engines (across all ops)
  std::set<samediff::Engine> _globallyDisabledEngines;

  // Thread safety
  mutable std::mutex _mutex;

  // Singleton
  static KernelManager* _instance;
  KernelManager() = default;

  // Get op hash from name
  LongType getOpHash(const std::string& opName) const;

 public:
  static KernelManager& getInstance();

  // Prevent copying
  KernelManager(const KernelManager&) = delete;
  KernelManager& operator=(const KernelManager&) = delete;

  // ========================
  // Discovery
  // ========================

  /**
   * Get all registered operations with their kernel info
   */
  std::vector<OpKernelInfo> getAllOperations() const;

  /**
   * Get kernel info for a specific operation by name
   */
  OpKernelInfo getOpKernelInfo(const std::string& opName) const;

  /**
   * Get kernel info for a specific operation by hash
   */
  OpKernelInfo getOpKernelInfo(LongType opHash) const;

  /**
   * Get list of all available engines across all operations
   */
  std::vector<samediff::Engine> getAllAvailableEngines() const;

  /**
   * Get list of available engines for a specific operation
   */
  std::vector<samediff::Engine> getAvailableEngines(const std::string& opName) const;
  std::vector<samediff::Engine> getAvailableEngines(LongType opHash) const;

  /**
   * Search for operations by name pattern (supports wildcards)
   */
  std::vector<OpKernelInfo> searchOperations(const std::string& pattern) const;

  // ========================
  // Enable/Disable Kernels
  // ========================

  /**
   * Enable a specific kernel for an operation
   */
  void enableKernel(const std::string& opName, samediff::Engine engine);
  void enableKernel(LongType opHash, samediff::Engine engine);

  /**
   * Disable a specific kernel for an operation
   */
  void disableKernel(const std::string& opName, samediff::Engine engine);
  void disableKernel(LongType opHash, samediff::Engine engine);

  /**
   * Check if a kernel is enabled
   */
  bool isKernelEnabled(const std::string& opName, samediff::Engine engine) const;
  bool isKernelEnabled(LongType opHash, samediff::Engine engine) const;

  /**
   * Enable all kernels for an operation
   */
  void enableAllKernels(const std::string& opName);
  void enableAllKernels(LongType opHash);

  /**
   * Disable all kernels for an operation (will use native implementation)
   */
  void disableAllKernels(const std::string& opName);
  void disableAllKernels(LongType opHash);

  /**
   * Globally enable an engine (across all operations)
   */
  void enableEngineGlobally(samediff::Engine engine);

  /**
   * Globally disable an engine (across all operations)
   */
  void disableEngineGlobally(samediff::Engine engine);

  /**
   * Check if an engine is globally disabled
   */
  bool isEngineGloballyDisabled(samediff::Engine engine) const;

  /**
   * Get list of globally disabled engines
   */
  std::vector<samediff::Engine> getGloballyDisabledEngines() const;

  // ========================
  // Preferred Engine
  // ========================

  /**
   * Set preferred engine for an operation
   * This engine will be used first if available and enabled
   */
  void setPreferredEngine(const std::string& opName, samediff::Engine engine);
  void setPreferredEngine(LongType opHash, samediff::Engine engine);

  /**
   * Get preferred engine for an operation
   */
  samediff::Engine getPreferredEngine(const std::string& opName) const;
  samediff::Engine getPreferredEngine(LongType opHash) const;

  /**
   * Clear preferred engine for an operation (use auto-selection)
   */
  void clearPreferredEngine(const std::string& opName);
  void clearPreferredEngine(LongType opHash);

  /**
   * Set global preferred engine (for all operations without specific preference)
   */
  void setGlobalPreferredEngine(samediff::Engine engine);

  /**
   * Get global preferred engine
   */
  samediff::Engine getGlobalPreferredEngine() const;

  // ========================
  // Configuration
  // ========================

  /**
   * Reset all kernel configurations to defaults
   */
  void resetToDefaults();

  /**
   * Get a summary of kernel configuration
   */
  std::string getConfigurationSummary() const;

  /**
   * Export configuration to JSON string
   */
  std::string exportConfiguration() const;

  /**
   * Import configuration from JSON string
   */
  bool importConfiguration(const std::string& json);

  // ========================
  // Performance Info
  // ========================

  /**
   * Get performance statistics for a kernel
   */
  KernelInfo getKernelPerformance(const std::string& opName, samediff::Engine engine) const;

  /**
   * Clear performance statistics
   */
  void clearPerformanceStats();

  /**
   * Get the best performing kernel for an operation (based on benchmarks)
   */
  samediff::Engine getBestPerformingEngine(const std::string& opName) const;
  samediff::Engine getBestPerformingEngine(LongType opHash) const;
};

// ========================
// C-style API for JNI
// ========================

extern "C" {

/**
 * Get number of available engines for an operation
 */
SD_LIB_EXPORT int kmGetEngineCount(const char* opName);

/**
 * Get engine at index for an operation
 * Returns engine ID or -1 if invalid
 */
SD_LIB_EXPORT int kmGetEngineAt(const char* opName, int index);

/**
 * Get engine name
 */
SD_LIB_EXPORT const char* kmGetEngineName(int engineId);

/**
 * Check if kernel is enabled
 */
SD_LIB_EXPORT bool kmIsKernelEnabled(const char* opName, int engineId);

/**
 * Enable kernel
 */
SD_LIB_EXPORT void kmEnableKernel(const char* opName, int engineId);

/**
 * Disable kernel
 */
SD_LIB_EXPORT void kmDisableKernel(const char* opName, int engineId);

/**
 * Enable all kernels for an operation
 */
SD_LIB_EXPORT void kmEnableAllKernels(const char* opName);

/**
 * Disable all kernels for an operation
 */
SD_LIB_EXPORT void kmDisableAllKernels(const char* opName);

/**
 * Globally enable an engine
 */
SD_LIB_EXPORT void kmEnableEngineGlobally(int engineId);

/**
 * Globally disable an engine
 */
SD_LIB_EXPORT void kmDisableEngineGlobally(int engineId);

/**
 * Check if engine is globally disabled
 */
SD_LIB_EXPORT bool kmIsEngineGloballyDisabled(int engineId);

/**
 * Set preferred engine for an operation
 */
SD_LIB_EXPORT void kmSetPreferredEngine(const char* opName, int engineId);

/**
 * Get preferred engine for an operation
 */
SD_LIB_EXPORT int kmGetPreferredEngine(const char* opName);

/**
 * Clear preferred engine for an operation
 */
SD_LIB_EXPORT void kmClearPreferredEngine(const char* opName);

/**
 * Set global preferred engine
 */
SD_LIB_EXPORT void kmSetGlobalPreferredEngine(int engineId);

/**
 * Get global preferred engine
 */
SD_LIB_EXPORT int kmGetGlobalPreferredEngine();

/**
 * Reset all configurations to defaults
 */
SD_LIB_EXPORT void kmResetToDefaults();

/**
 * Get configuration summary (caller must free the returned string)
 */
SD_LIB_EXPORT const char* kmGetConfigurationSummary();

/**
 * Get number of registered operations
 */
SD_LIB_EXPORT int kmGetOperationCount();

/**
 * Get operation name at index
 */
SD_LIB_EXPORT const char* kmGetOperationNameAt(int index);

/**
 * Get best performing engine for an operation
 */
SD_LIB_EXPORT int kmGetBestPerformingEngine(const char* opName);

/**
 * Get average execution time for a kernel in nanoseconds
 */
SD_LIB_EXPORT double kmGetKernelAvgTime(const char* opName, int engineId);

/**
 * Get execution count for a kernel
 */
SD_LIB_EXPORT int64_t kmGetKernelExecutionCount(const char* opName, int engineId);

/**
 * Free a string returned by kmGet* functions
 */
SD_LIB_EXPORT void kmFreeString(const char* str);

}  // extern "C"

SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd

#endif  // SD_KERNEL_MANAGER_H
