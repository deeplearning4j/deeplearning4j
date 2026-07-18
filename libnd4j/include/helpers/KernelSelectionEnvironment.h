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

#ifndef SD_KERNEL_SELECTION_ENVIRONMENT_H
#define SD_KERNEL_SELECTION_ENVIRONMENT_H

#include <system/BackendNamespace.h>

#include <helpers/KernelPerformanceRegistry.h>
#include <execution/Engine.h>
#include <graph/Context.h>
#include <system/common.h>

#include <string>
#include <utility>
#include <vector>

namespace sd {
namespace ops {

// Forward declarations
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
class DeclarableOp;
SD_BACKEND_OPS_INLINE_NAMESPACE_END

namespace platforms {
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN

/**
 * Environment configuration for kernel selection.
 *
 * Environment Variables:
 *   SD_KERNEL_STRATEGY         - Selection strategy (FASTEST, FIRST_AVAILABLE, ROUND_ROBIN)
 *   SD_KERNEL_AUTOTUNE        - Enable auto-tuning (1/true to enable)
 *   SD_KERNEL_WARMUP_RUNS     - Number of warmup runs (default: 2)
 *   SD_KERNEL_BENCHMARK_RUNS  - Number of benchmark runs (default: 5)
 *   SD_KERNEL_CACHE_PATH      - Path to cache file
 *   SD_KERNEL_FORCE_ENGINE    - Force an Engine name; default/native selects the artifact engine
 *   SD_KERNEL_DISABLE_ENGINES - Comma-separated list of engines to disable
 *   SD_KERNEL_VERBOSE         - Enable verbose logging (1/true to enable)
 *   SD_KERNEL_PLUGIN_PATH     - Colon-separated plugin search paths
 *   SD_KERNEL_PLUGIN_AUTO     - Auto-load plugins from search paths (1/true)
 */
class SD_LIB_EXPORT KernelSelectionEnvironment {
 private:
  // State is held in a function-local static in the .cpp file
  // to avoid static initialization order fiasco
  static void initialize();

 public:
  /**
   * Check if auto-tuning is enabled
   */
  static bool isAutoTuneEnabled();

  /**
   * Check if verbose logging is enabled
   */
  static bool isVerbose();

  /**
   * Get forced engine (if set)
   */
  static samediff::Engine getForcedEngine();

  /**
   * Check if a specific engine is forced
   */
  static bool hasForcedEngine();

  /**
   * Check if an engine is disabled
   */
  static bool isEngineDisabled(samediff::Engine engine);

  /**
   * Get the cache path
   */
  static const std::string& getCachePath();

  /**
   * Get plugin search paths
   */
  static const std::vector<std::string>& getPluginPaths();

  /**
   * Check if plugin auto-loading is enabled
   */
  static bool isPluginAutoLoadEnabled();

  /**
   * Get engine from string name
   */
  static samediff::Engine engineFromString(const std::string& name);

  /**
   * Get string name from engine
   */
  static std::string engineToString(samediff::Engine engine);

  /**
   * Force re-initialization from environment
   */
  static void reinitialize();
};

/**
 * Helper class for dispatching operations with kernel selection.
 * Integrates with DeclarableOp execution flow.
 */
class SD_LIB_EXPORT KernelDispatchHelper {
 public:
  /**
   * Check if the operation should use a platform helper
   */
  static bool shouldUseHelper(DeclarableOp* op, sd::graph::Context& context);

  /**
   * Dispatch the operation using the best available kernel.
   * Returns pair<bool, Status> where:
   *   - first: true if a helper was used, false if native execution should proceed
   *   - second: the execution status (only valid if first is true)
   */
  static std::pair<bool, Status> dispatchWithAutoTune(DeclarableOp* op, sd::graph::Context& context);
};

SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_KERNEL_SELECTION_ENVIRONMENT_H
