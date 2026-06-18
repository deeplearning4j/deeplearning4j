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

#ifndef SD_MULTI_PLATFORM_DISPATCHER_H
#define SD_MULTI_PLATFORM_DISPATCHER_H

#include <execution/Engine.h>
#include <graph/Context.h>
#include <helpers/HelperVersionRegistry.h>
#include <helpers/KernelAutoTuner.h>
#include <helpers/KernelPerformanceRegistry.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/common.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sd {
namespace ops {

// Forward declaration
class DeclarableOp;

namespace platforms {

/**
 * Dispatch mode for selecting which kernel to use
 */
enum class DispatchMode {
  AUTO,           // Use auto-tuning to find fastest (default)
  FIXED,          // Use a fixed engine priority order
  ROUND_ROBIN,    // Distribute across engines
  BENCHMARK       // Always run all and compare (debugging)
};

/**
 * Statistics for dispatcher usage
 */
struct DispatcherStats {
  std::unordered_map<samediff::Engine, int64_t> engineUseCounts;
  std::unordered_map<samediff::Engine, double> totalTimeByEngine;
  int64_t autoTuneRuns = 0;
  int64_t cacheHits = 0;
  int64_t cacheMisses = 0;
  int64_t fallbackToNative = 0;

  void recordExecution(samediff::Engine engine, double timeNanos) {
    engineUseCounts[engine]++;
    totalTimeByEngine[engine] += timeNanos;
  }

  std::string toString() const;
};

/**
 * MultiPlatformDispatcher manages multiple PlatformHelper implementations
 * for the same operation and intelligently dispatches to the fastest one
 * based on runtime performance characteristics.
 *
 * Key features:
 * - Manages multiple backend implementations (oneDNN, cuDNN, native, etc.)
 * - Auto-tunes at runtime to find fastest kernel for each shape
 * - Caches performance data to avoid repeated benchmarking
 * - Falls back gracefully if preferred backend fails
 * - Thread-safe operation
 *
 * Usage:
 *   auto dispatcher = MultiPlatformDispatcher::getOrCreate(op);
 *   dispatcher->addHelper(onednnHelper);
 *   dispatcher->addHelper(cudnnHelper);
 *   Status result = dispatcher->dispatch(context);
 */
class SD_LIB_EXPORT MultiPlatformDispatcher {
 private:
  // The operation this dispatcher is for
  std::string _opName;
  LongType _opHash;

  // All available helpers for this operation
  std::vector<PlatformHelper*> _helpers;

  // The native (non-helper) implementation
  DeclarableOp* _nativeOp;

  // Dispatch configuration
  DispatchMode _mode;
  std::vector<samediff::Engine> _enginePriority;  // For FIXED mode

  // Statistics
  DispatcherStats _stats;

  // Thread safety
  mutable std::mutex _mutex;

  // Round-robin counter
  std::atomic<int> _roundRobinIndex{0};

  /**
   * Select the best helper based on current mode and context
   */
  PlatformHelper* selectHelper(graph::Context& context);

  /**
   * Get all usable helpers for the given context
   */
  std::vector<PlatformHelper*> getUsableHelpers(graph::Context& context);

  /**
   * Get all version-compatible helpers for the given context
   * This filters out helpers that don't meet version requirements
   */
  std::vector<PlatformHelper*> getVersionCompatibleHelpers(graph::Context& context);

  /**
   * Check if version enforcement is enabled
   */
  bool isVersionEnforcementEnabled() const;

  /**
   * Execute with timing and record performance
   */
  Status executeWithTiming(PlatformHelper* helper, graph::Context& context);

  /**
   * Execute native op with timing
   */
  Status executeNativeWithTiming(graph::Context& context);

 public:
  MultiPlatformDispatcher(const std::string& opName, LongType opHash, DeclarableOp* nativeOp = nullptr);

  virtual ~MultiPlatformDispatcher() = default;

  /**
   * Add a platform helper to this dispatcher
   */
  void addHelper(PlatformHelper* helper);

  /**
   * Remove a platform helper
   */
  void removeHelper(PlatformHelper* helper);

  /**
   * Check if we have any helpers available
   */
  bool hasHelpers() const { return !_helpers.empty(); }

  /**
   * Get the number of registered helpers
   */
  int getHelperCount() const { return static_cast<int>(_helpers.size()); }

  /**
   * Get available engines for this operation
   */
  std::vector<samediff::Engine> getAvailableEngines() const;

  /**
   * Check if a specific engine is available
   */
  bool hasEngine(samediff::Engine engine) const;

  /**
   * Dispatch to the best available kernel
   *
   * @param context The execution context
   * @return Status of execution
   */
  Status dispatch(graph::Context& context);

  /**
   * Force dispatch to a specific engine
   * Falls back to best available if the engine isn't usable
   */
  Status dispatchTo(samediff::Engine engine, graph::Context& context);

  /**
   * Check if any helper is usable for this context
   */
  bool isUsable(graph::Context& context);

  /**
   * Get/set dispatch mode
   */
  DispatchMode getMode() const { return _mode; }
  void setMode(DispatchMode mode) { _mode = mode; }

  /**
   * Set engine priority for FIXED mode
   */
  void setEnginePriority(const std::vector<samediff::Engine>& priority) {
    _enginePriority = priority;
  }

  /**
   * Set native op for fallback
   */
  void setNativeOp(DeclarableOp* op) { _nativeOp = op; }

  /**
   * Get statistics
   */
  const DispatcherStats& getStats() const { return _stats; }
  void resetStats() { _stats = DispatcherStats(); }

  /**
   * Get operation info
   */
  const std::string& getOpName() const { return _opName; }
  LongType getOpHash() const { return _opHash; }

  // ============================================================================
  // Version Validation Methods
  // ============================================================================

  /**
   * Dispatch with version validation
   * Only uses helpers that pass version and capability checks
   * Falls back to native if no version-compatible helper is available
   *
   * @param context The execution context
   * @return Status of execution
   */
  Status dispatchWithVersionValidation(graph::Context& context);

  /**
   * Get version-compatible engines for this operation
   * Returns only engines whose helpers pass version requirements
   */
  std::vector<samediff::Engine> getVersionCompatibleEngines() const;

  /**
   * Get version status for all helpers
   * Returns a map of helper library names to their version status
   */
  std::unordered_map<std::string, std::string> getHelperVersionStatuses() const;

  /**
   * Check if a specific engine is version-compatible
   */
  bool isEngineVersionCompatible(samediff::Engine engine) const;

  /**
   * Log version compatibility warnings for all helpers
   * Useful for debugging version issues
   */
  void logVersionCompatibility() const;

  // Static management
  /**
   * Get or create a dispatcher for an operation
   */
  static MultiPlatformDispatcher* getOrCreate(const std::string& opName, LongType opHash);

  /**
   * Get existing dispatcher (returns nullptr if not found)
   */
  static MultiPlatformDispatcher* get(LongType opHash);

  /**
   * Check if a dispatcher exists for an operation
   */
  static bool exists(LongType opHash);

  /**
   * Global registry of dispatchers
   */
  static std::unordered_map<LongType, std::unique_ptr<MultiPlatformDispatcher>>& getRegistry();
};

/**
 * Helper macro for registering multiple platform implementations
 * with automatic dispatcher setup
 */
#define REGISTER_MULTI_PLATFORM_HELPER(OP_NAME, ENGINE, HELPER_CLASS) \
  static struct __multiPlatformRegistrar_##OP_NAME##_##ENGINE { \
    __multiPlatformRegistrar_##OP_NAME##_##ENGINE() { \
      auto helper = new HELPER_CLASS(); \
      auto dispatcher = MultiPlatformDispatcher::getOrCreate(#OP_NAME, helper->hash()); \
      dispatcher->addHelper(helper); \
    } \
  } __multiPlatformRegistrar_##OP_NAME##_##ENGINE##_instance;

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_MULTI_PLATFORM_DISPATCHER_H
