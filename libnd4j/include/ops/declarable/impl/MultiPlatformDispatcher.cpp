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

#include <ops/declarable/MultiPlatformDispatcher.h>
#include <ops/declarable/DeclarableOp.h>

#include <algorithm>
#include <chrono>
#include <sstream>

namespace sd {
namespace ops {
namespace platforms {

// Static registry
static std::mutex _registryMutex;

std::unordered_map<LongType, std::unique_ptr<MultiPlatformDispatcher>>& MultiPlatformDispatcher::getRegistry() {
  static std::unordered_map<LongType, std::unique_ptr<MultiPlatformDispatcher>> registry;
  return registry;
}

MultiPlatformDispatcher* MultiPlatformDispatcher::getOrCreate(const std::string& opName, LongType opHash) {
  std::lock_guard<std::mutex> lock(_registryMutex);

  auto& registry = getRegistry();
  auto it = registry.find(opHash);
  if (it != registry.end()) {
    return it->second.get();
  }

  auto dispatcher = std::make_unique<MultiPlatformDispatcher>(opName, opHash);
  auto* ptr = dispatcher.get();
  registry[opHash] = std::move(dispatcher);
  return ptr;
}

MultiPlatformDispatcher* MultiPlatformDispatcher::get(LongType opHash) {
  std::lock_guard<std::mutex> lock(_registryMutex);

  auto& registry = getRegistry();
  auto it = registry.find(opHash);
  if (it != registry.end()) {
    return it->second.get();
  }
  return nullptr;
}

bool MultiPlatformDispatcher::exists(LongType opHash) {
  std::lock_guard<std::mutex> lock(_registryMutex);
  return getRegistry().find(opHash) != getRegistry().end();
}

MultiPlatformDispatcher::MultiPlatformDispatcher(const std::string& opName, LongType opHash,
                                                   DeclarableOp* nativeOp)
    : _opName(opName), _opHash(opHash), _nativeOp(nativeOp), _mode(DispatchMode::AUTO) {
  // Default engine priority
  _enginePriority = {samediff::ENGINE_CUDA, samediff::ENGINE_ONEDNN,
                     samediff::ENGINE_CPU};
}

void MultiPlatformDispatcher::addHelper(PlatformHelper* helper) {
  std::lock_guard<std::mutex> lock(_mutex);

  // Check if helper already exists
  for (auto* h : _helpers) {
    if (h == helper || h->engine() == helper->engine()) {
      return;  // Already have a helper for this engine
    }
  }

  _helpers.push_back(helper);
}

void MultiPlatformDispatcher::removeHelper(PlatformHelper* helper) {
  std::lock_guard<std::mutex> lock(_mutex);
  _helpers.erase(std::remove(_helpers.begin(), _helpers.end(), helper), _helpers.end());
}

std::vector<samediff::Engine> MultiPlatformDispatcher::getAvailableEngines() const {
  std::lock_guard<std::mutex> lock(_mutex);

  std::vector<samediff::Engine> engines;
  for (auto* helper : _helpers) {
    engines.push_back(helper->engine());
  }

  // Always include CPU as fallback
  if (std::find(engines.begin(), engines.end(), samediff::ENGINE_CPU) == engines.end()) {
    engines.push_back(samediff::ENGINE_CPU);
  }

  return engines;
}

bool MultiPlatformDispatcher::hasEngine(samediff::Engine engine) const {
  std::lock_guard<std::mutex> lock(_mutex);

  for (auto* helper : _helpers) {
    if (helper->engine() == engine) {
      return true;
    }
  }

  return engine == samediff::ENGINE_CPU;  // CPU is always available
}

std::vector<PlatformHelper*> MultiPlatformDispatcher::getUsableHelpers(graph::Context& context) {
  std::vector<PlatformHelper*> usable;

  for (auto* helper : _helpers) {
    if (helper->isUsable(context)) {
      usable.push_back(helper);
    }
  }

  return usable;
}

PlatformHelper* MultiPlatformDispatcher::selectHelper(graph::Context& context) {
  auto usable = getUsableHelpers(context);

  if (usable.empty()) {
    return nullptr;
  }

  if (usable.size() == 1) {
    return usable[0];
  }

  switch (_mode) {
    case DispatchMode::AUTO: {
      // Use auto-tuner to select best
      std::vector<KernelExecutor*> executors;
      std::vector<std::unique_ptr<PlatformHelperExecutor>> ownedExecutors;

      for (auto* helper : usable) {
        auto executor = std::make_unique<PlatformHelperExecutor>(helper);
        executors.push_back(executor.get());
        ownedExecutors.push_back(std::move(executor));
      }

      auto* best = KernelAutoTuner::getInstance().selectBest(context, executors, _opHash);
      if (best != nullptr) {
        // Find the matching helper
        for (auto* helper : usable) {
          if (helper->engine() == best->getEngine()) {
            return helper;
          }
        }
      }
      return usable[0];
    }

    case DispatchMode::FIXED: {
      // Use priority order
      for (auto engine : _enginePriority) {
        for (auto* helper : usable) {
          if (helper->engine() == engine) {
            return helper;
          }
        }
      }
      return usable[0];
    }

    case DispatchMode::ROUND_ROBIN: {
      int index = _roundRobinIndex.fetch_add(1) % static_cast<int>(usable.size());
      return usable[index];
    }

    case DispatchMode::BENCHMARK: {
      // Always benchmark, don't cache
      std::vector<KernelExecutor*> executors;
      std::vector<std::unique_ptr<PlatformHelperExecutor>> ownedExecutors;

      for (auto* helper : usable) {
        auto executor = std::make_unique<PlatformHelperExecutor>(helper);
        executors.push_back(executor.get());
        ownedExecutors.push_back(std::move(executor));
      }

      auto results = KernelAutoTuner::getInstance().benchmarkAll(context, executors);
      if (!results.empty() && results[0].success) {
        for (auto* helper : usable) {
          if (helper->engine() == results[0].engine) {
            return helper;
          }
        }
      }
      return usable[0];
    }
  }

  return usable[0];
}

Status MultiPlatformDispatcher::executeWithTiming(PlatformHelper* helper, graph::Context& context) {
  auto start = std::chrono::high_resolution_clock::now();

  Status status = helper->invokeHelper(context);

  auto end = std::chrono::high_resolution_clock::now();
  double nanos = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

  _stats.recordExecution(helper->engine(), nanos);

  return status;
}

Status MultiPlatformDispatcher::executeNativeWithTiming(graph::Context& context) {
  if (_nativeOp == nullptr) {
    return Status::BAD_INPUT;
  }

  auto start = std::chrono::high_resolution_clock::now();

  Status status = _nativeOp->execute(&context);

  auto end = std::chrono::high_resolution_clock::now();
  double nanos = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

  _stats.recordExecution(samediff::ENGINE_CPU, nanos);

  return status;
}

Status MultiPlatformDispatcher::dispatch(graph::Context& context) {
  std::lock_guard<std::mutex> lock(_mutex);

  // Try to find and execute best helper
  PlatformHelper* helper = selectHelper(context);

  if (helper != nullptr) {
    Status status = executeWithTiming(helper, context);
    if (status == Status::OK) {
      return status;
    }
    // Helper failed, try fallback
  }

  // Fallback to native implementation
  if (_nativeOp != nullptr && KernelSelectionConfig::global().allowFallbackToNative) {
    _stats.fallbackToNative++;
    return executeNativeWithTiming(context);
  }

  return Status::BAD_INPUT;
}

Status MultiPlatformDispatcher::dispatchTo(samediff::Engine engine, graph::Context& context) {
  std::lock_guard<std::mutex> lock(_mutex);

  // Find helper for specified engine
  for (auto* helper : _helpers) {
    if (helper->engine() == engine && helper->isUsable(context)) {
      return executeWithTiming(helper, context);
    }
  }

  // Engine not available, fall back to best available
  return dispatch(context);
}

bool MultiPlatformDispatcher::isUsable(graph::Context& context) {
  std::lock_guard<std::mutex> lock(_mutex);

  for (auto* helper : _helpers) {
    if (helper->isUsable(context)) {
      return true;
    }
  }

  return _nativeOp != nullptr;
}

std::string DispatcherStats::toString() const {
  std::stringstream ss;
  ss << "MultiPlatformDispatcher Statistics:\n";
  ss << "  Auto-tune runs: " << autoTuneRuns << "\n";
  ss << "  Cache hits: " << cacheHits << "\n";
  ss << "  Cache misses: " << cacheMisses << "\n";
  ss << "  Fallback to native: " << fallbackToNative << "\n";
  ss << "  Engine usage:\n";

  for (const auto& pair : engineUseCounts) {
    ss << "    Engine " << static_cast<int>(pair.first) << ": "
       << pair.second << " executions";

    auto timeIt = totalTimeByEngine.find(pair.first);
    if (timeIt != totalTimeByEngine.end() && pair.second > 0) {
      ss << ", avg " << (timeIt->second / pair.second / 1000000.0) << " ms";
    }
    ss << "\n";
  }

  return ss.str();
}

// ============================================================================
// Version Validation Methods
// ============================================================================

bool MultiPlatformDispatcher::isVersionEnforcementEnabled() const {
  return HelperVersionRegistry::getInstance().isStrictVersionEnforcement();
}

std::vector<PlatformHelper*> MultiPlatformDispatcher::getVersionCompatibleHelpers(graph::Context& context) {
  std::vector<PlatformHelper*> compatible;

  for (auto* helper : _helpers) {
    // First check if usable at all
    if (!helper->isUsable(context)) {
      continue;
    }

    // If version enforcement is enabled, check version compatibility
    if (isVersionEnforcementEnabled()) {
      if (!helper->validateRuntimeVersion()) {
        continue;
      }
      if (!helper->hasRequiredCapabilities()) {
        continue;
      }
    }

    compatible.push_back(helper);
  }

  return compatible;
}

Status MultiPlatformDispatcher::dispatchWithVersionValidation(graph::Context& context) {
  std::lock_guard<std::mutex> lock(_mutex);

  // Get version-compatible helpers
  std::vector<PlatformHelper*> compatible = getVersionCompatibleHelpers(context);

  if (compatible.empty()) {
    // Log warning about no compatible helpers
    if (isVersionEnforcementEnabled() && !_helpers.empty()) {
      logVersionCompatibility();
    }

    // Fallback to native implementation
    if (_nativeOp != nullptr && KernelSelectionConfig::global().allowFallbackToNative) {
      _stats.fallbackToNative++;
      return executeNativeWithTiming(context);
    }

    return Status::BAD_INPUT;
  }

  // Select best from compatible helpers
  PlatformHelper* selected = nullptr;

  if (compatible.size() == 1) {
    selected = compatible[0];
  } else {
    // Use the mode-based selection, but only from compatible helpers
    // For now, use priority order from _enginePriority
    for (auto engine : _enginePriority) {
      for (auto* helper : compatible) {
        if (helper->engine() == engine) {
          selected = helper;
          break;
        }
      }
      if (selected != nullptr) break;
    }

    if (selected == nullptr) {
      selected = compatible[0];
    }
  }

  Status status = executeWithTiming(selected, context);

  if (status == Status::OK) {
    return status;
  }

  // Selected helper failed, try next compatible one
  for (auto* helper : compatible) {
    if (helper != selected) {
      status = executeWithTiming(helper, context);
      if (status == Status::OK) {
        return status;
      }
    }
  }

  // All helpers failed, try native
  if (_nativeOp != nullptr && KernelSelectionConfig::global().allowFallbackToNative) {
    _stats.fallbackToNative++;
    return executeNativeWithTiming(context);
  }

  return Status::BAD_INPUT;
}

std::vector<samediff::Engine> MultiPlatformDispatcher::getVersionCompatibleEngines() const {
  std::lock_guard<std::mutex> lock(_mutex);

  std::vector<samediff::Engine> engines;

  for (auto* helper : _helpers) {
    if (helper->validateRuntimeVersion() && helper->hasRequiredCapabilities()) {
      engines.push_back(helper->engine());
    }
  }

  // Always include CPU as fallback
  if (std::find(engines.begin(), engines.end(), samediff::ENGINE_CPU) == engines.end()) {
    engines.push_back(samediff::ENGINE_CPU);
  }

  return engines;
}

std::unordered_map<std::string, std::string> MultiPlatformDispatcher::getHelperVersionStatuses() const {
  std::lock_guard<std::mutex> lock(_mutex);

  std::unordered_map<std::string, std::string> statuses;

  for (auto* helper : _helpers) {
    std::string libraryName = helper->helperLibraryName();
    if (!libraryName.empty()) {
      statuses[libraryName] = helper->getVersionStatusMessage();
    }
  }

  return statuses;
}

bool MultiPlatformDispatcher::isEngineVersionCompatible(samediff::Engine engine) const {
  std::lock_guard<std::mutex> lock(_mutex);

  for (auto* helper : _helpers) {
    if (helper->engine() == engine) {
      return helper->validateRuntimeVersion() && helper->hasRequiredCapabilities();
    }
  }

  // CPU is always compatible
  return engine == samediff::ENGINE_CPU;
}

void MultiPlatformDispatcher::logVersionCompatibility() const {
  // Note: caller should hold the mutex
  sd::Logger::info("Version compatibility for operation: %s\n", _opName.c_str());

  for (auto* helper : _helpers) {
    std::string status = helper->getVersionStatusMessage();
    bool compatible = helper->validateRuntimeVersion() && helper->hasRequiredCapabilities();

    if (compatible) {
      sd::Logger::info("  [OK] Engine %d: %s\n", static_cast<int>(helper->engine()), status.c_str());
    } else {
      sd::Logger::info("  [INCOMPATIBLE] Engine %d: %s\n", static_cast<int>(helper->engine()), status.c_str());
    }
  }
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
