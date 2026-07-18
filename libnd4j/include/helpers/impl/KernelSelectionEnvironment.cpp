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

#include <helpers/KernelSelectionEnvironment.h>
#include <helpers/KernelAutoTuner.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/OpRegistrator.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace sd {
namespace ops {
namespace platforms {
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN

// Use lazy initialization to avoid static initialization order fiasco.
// All state is wrapped in a function-local static struct that's only
// initialized on first access.
struct KernelSelectionState {
  bool initialized = false;
  bool autoTuneEnabled = false;
  bool verbose = false;
  samediff::Engine forcedEngine = DEFAULT_ENGINE;
  bool hasForcedEngine = false;
  std::vector<samediff::Engine> disabledEngines;
  std::string cachePath;
  std::vector<std::string> pluginPaths;
  bool pluginAutoLoad = false;
};

static KernelSelectionState& getState() {
  static KernelSelectionState state;
  return state;
}

void KernelSelectionEnvironment::initialize() {
  auto& state = getState();
  if (state.initialized) return;

  // SD_KERNEL_AUTOTUNE
  const char* autoTuneEnv = std::getenv("SD_KERNEL_AUTOTUNE");
  if (autoTuneEnv != nullptr) {
    std::string val(autoTuneEnv);
    state.autoTuneEnabled = (val == "1" || val == "true" || val == "TRUE");
  }

  // SD_KERNEL_VERBOSE
  const char* verboseEnv = std::getenv("SD_KERNEL_VERBOSE");
  if (verboseEnv != nullptr) {
    std::string val(verboseEnv);
    state.verbose = (val == "1" || val == "true" || val == "TRUE");
  }

  // SD_KERNEL_FORCE_ENGINE
  const char* forceEnv = std::getenv("SD_KERNEL_FORCE_ENGINE");
  if (forceEnv != nullptr) {
    state.forcedEngine = engineFromString(forceEnv);
    state.hasForcedEngine = state.forcedEngine != samediff::ENGINE_ANY;
  }

  // SD_KERNEL_DISABLE_ENGINES
  const char* disableEnv = std::getenv("SD_KERNEL_DISABLE_ENGINES");
  if (disableEnv != nullptr) {
    std::string val(disableEnv);
    std::stringstream ss(val);
    std::string engine;
    while (std::getline(ss, engine, ',')) {
      // Trim whitespace
      engine.erase(0, engine.find_first_not_of(" \t"));
      engine.erase(engine.find_last_not_of(" \t") + 1);
      if (!engine.empty()) {
        state.disabledEngines.push_back(engineFromString(engine));
      }
    }
  }

  // SD_KERNEL_CACHE_PATH
  const char* cacheEnv = std::getenv("SD_KERNEL_CACHE_PATH");
  if (cacheEnv != nullptr) {
    state.cachePath = cacheEnv;
    KernelPerformanceRegistry::getInstance().setCachePath(state.cachePath);
    KernelPerformanceRegistry::getInstance().loadFromFile(state.cachePath);
  }

  // SD_KERNEL_PLUGIN_PATH
  const char* pluginPathEnv = std::getenv("SD_KERNEL_PLUGIN_PATH");
  if (pluginPathEnv != nullptr) {
    std::string val(pluginPathEnv);
#ifdef _WIN32
    char delimiter = ';';
#else
    char delimiter = ':';
#endif
    std::stringstream ss(val);
    std::string path;
    while (std::getline(ss, path, delimiter)) {
      if (!path.empty()) {
        state.pluginPaths.push_back(path);
      }
    }
  }

  // SD_KERNEL_PLUGIN_AUTO
  const char* pluginAutoEnv = std::getenv("SD_KERNEL_PLUGIN_AUTO");
  if (pluginAutoEnv != nullptr) {
    std::string val(pluginAutoEnv);
    state.pluginAutoLoad = (val == "1" || val == "true" || val == "TRUE");
  }

  // Configure the global KernelSelectionConfig
  const char* warmupEnv = std::getenv("SD_KERNEL_WARMUP_RUNS");
  if (warmupEnv != nullptr) {
    try {
      KernelSelectionConfig::global().warmupRuns = std::stoi(warmupEnv);
    } catch (...) {
      // Ignore invalid value
    }
  }

  const char* benchmarkEnv = std::getenv("SD_KERNEL_BENCHMARK_RUNS");
  if (benchmarkEnv != nullptr) {
    try {
      KernelSelectionConfig::global().benchmarkRuns = std::stoi(benchmarkEnv);
    } catch (...) {
      // Ignore invalid value
    }
  }

  KernelSelectionConfig::global().verbose = state.verbose;

  state.initialized = true;
}

bool KernelSelectionEnvironment::isAutoTuneEnabled() {
  initialize();
  return getState().autoTuneEnabled;
}

bool KernelSelectionEnvironment::isVerbose() {
  initialize();
  return getState().verbose;
}

samediff::Engine KernelSelectionEnvironment::getForcedEngine() {
  initialize();
  return getState().forcedEngine;
}

bool KernelSelectionEnvironment::hasForcedEngine() {
  initialize();
  return getState().hasForcedEngine;
}

bool KernelSelectionEnvironment::isEngineDisabled(samediff::Engine engine) {
  initialize();
  auto& state = getState();
  return std::find(state.disabledEngines.begin(), state.disabledEngines.end(), engine) !=
         state.disabledEngines.end();
}

const std::string& KernelSelectionEnvironment::getCachePath() {
  initialize();
  return getState().cachePath;
}

const std::vector<std::string>& KernelSelectionEnvironment::getPluginPaths() {
  initialize();
  return getState().pluginPaths;
}

bool KernelSelectionEnvironment::isPluginAutoLoadEnabled() {
  initialize();
  return getState().pluginAutoLoad;
}

samediff::Engine KernelSelectionEnvironment::engineFromString(
    const std::string& name) {
  std::string lower = name;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });

  if (lower == "default" || lower == "native") return DEFAULT_ENGINE;
  if (lower == "cpu") return samediff::ENGINE_CPU;
  if (lower == "cuda") return samediff::ENGINE_CUDA;
  if (lower == "tpu" || lower == "pjrt") return samediff::ENGINE_TPU;
  if (lower == "zluda_amd" || lower == "zluda-amd")
    return samediff::ENGINE_ZLUDA_AMD;
  if (lower == "zluda_intel" || lower == "zluda-intel")
    return samediff::ENGINE_ZLUDA_INTEL;
  if (lower == "metal") return samediff::ENGINE_METAL;
  if (lower == "vulkan") return samediff::ENGINE_VULKAN;
  if (lower == "opencl") return samediff::ENGINE_OPENCL;
  if (lower == "vlm") return samediff::ENGINE_VLM;
  if (lower == "mps") return samediff::ENGINE_MPS;
  if (lower == "accelerate") return samediff::ENGINE_ACCELERATE;
  if (lower == "onednn" || lower == "mkldnn" || lower == "mkl")
    return samediff::ENGINE_ONEDNN;
  if (lower == "arm" || lower == "armcompute")
    return samediff::ENGINE_ARM;
  if (lower == "any" || lower == "auto") return samediff::ENGINE_ANY;

  const std::string message =
      "Unknown kernel engine '" + name + "'";
  THROW_EXCEPTION(message.c_str());
  return samediff::ENGINE_ANY;
}

std::string KernelSelectionEnvironment::engineToString(
    samediff::Engine engine) {
  return samediff::getEngineName(engine);
}

void KernelSelectionEnvironment::reinitialize() {
  getState() = KernelSelectionState{};
  initialize();
}

// KernelDispatchHelper implementation

bool KernelDispatchHelper::shouldUseHelper(DeclarableOp* op, sd::graph::Context& context) {
  if (op == nullptr) return false;

  // Check if we have any helpers for this operation
  auto& registrator = OpRegistrator::getInstance();
  LongType hash = op->getOpHash();

  if (!registrator.hasAnyHelper(hash)) {
    return false;
  }

  // Check forced engine
  if (KernelSelectionEnvironment::hasForcedEngine()) {
    samediff::Engine forced = KernelSelectionEnvironment::getForcedEngine();
    if (KernelSelectionEnvironment::isEngineDisabled(forced)) {
      return false;
    }
    if (registrator.hasHelper(hash, forced)) {
      auto* helper = registrator.getPlatformHelper(hash, forced);
      return helper != nullptr && helper->isUsable(context);
    }
    return false;
  }

  // Check the configured artifact engine
  auto helpers = registrator.getAllHelpersForOp(hash);
  for (auto* helper : helpers) {
    if (helper->engine() == DEFAULT_ENGINE &&
        !KernelSelectionEnvironment::isEngineDisabled(helper->engine()) &&
        helper->isUsable(context)) {
      return true;
    }
  }

  return false;
}

std::pair<bool, Status> KernelDispatchHelper::dispatchWithAutoTune(DeclarableOp* op, sd::graph::Context& context) {
  if (op == nullptr) {
    return {false, Status::BAD_INPUT};
  }

  auto& registrator = OpRegistrator::getInstance();
  LongType hash = op->getOpHash();

  // Debug: Log helper dispatch attempts
  sd_debug("KernelDispatch: op=%s hash=%lld\n", op->getOpName()->c_str(), hash);

  // If forced engine, use that
  if (KernelSelectionEnvironment::hasForcedEngine()) {
    samediff::Engine forced = KernelSelectionEnvironment::getForcedEngine();
    sd_debug("KernelDispatch: forced engine=%d\n", (int)forced);
    if (KernelSelectionEnvironment::isEngineDisabled(forced)) {
      return {true, Status::BAD_INPUT};
    }
    if (registrator.hasHelper(hash, forced)) {
      auto* helper = registrator.getPlatformHelper(hash, forced);
      if (helper != nullptr && helper->isUsable(context)) {
        sd_debug("KernelDispatch: using forced helper %s\n", helper->name().c_str());
        Status status = helper->invokeHelper(context);
        return {true, status};
      }
    }
    if (forced == DEFAULT_ENGINE) {
      return {false, Status::OK};
    }
    return {true, Status::BAD_INPUT};
  }

  if (KernelSelectionEnvironment::isEngineDisabled(DEFAULT_ENGINE)) {
    return {true, Status::BAD_INPUT};
  }

  // Get the usable helper for this artifact's configured engine
  auto allHelpers = registrator.getAllHelpersForOp(hash);
  sd_debug("KernelDispatch: found %zu helpers for op\n", allHelpers.size());

  std::vector<platforms::PlatformHelper*> usableHelpers;

  for (auto* helper : allHelpers) {
    bool engineDisabled = KernelSelectionEnvironment::isEngineDisabled(helper->engine());
    bool isUsable = helper->isUsable(context);
    sd_debug("KernelDispatch: helper=%s engine=%d disabled=%d usable=%d\n",
             helper->name().c_str(), (int)helper->engine(), engineDisabled, isUsable);
    if (helper->engine() == DEFAULT_ENGINE && !engineDisabled && isUsable) {
      usableHelpers.push_back(helper);
    }
  }

  sd_debug("KernelDispatch: %zu usable helpers\n", usableHelpers.size());

  if (usableHelpers.empty()) {
    return {false, Status::OK};  // Continue on this artifact's native engine
  }

  // If only one helper, use it
  if (usableHelpers.size() == 1) {
    Status status = usableHelpers[0]->invokeHelper(context);
    return {true, status};
  }

  // If auto-tuning is not enabled, use first available
  if (!KernelSelectionEnvironment::isAutoTuneEnabled()) {
    Status status = usableHelpers[0]->invokeHelper(context);
    return {true, status};
  }

  // Use auto-tuning to select best kernel
  std::vector<KernelExecutor*> executors;
  std::vector<std::unique_ptr<PlatformHelperExecutor>> ownedExecutors;

  for (auto* helper : usableHelpers) {
    auto executor = std::make_unique<PlatformHelperExecutor>(helper);
    executors.push_back(executor.get());
    ownedExecutors.push_back(std::move(executor));
  }

  auto* bestExecutor = KernelAutoTuner::getInstance().selectBest(context, executors, hash);
  if (bestExecutor != nullptr) {
    Status status = bestExecutor->execute(context);
    return {true, status};
  }

  // No benchmark winner: execute the eligible artifact-engine helper.
  Status status = usableHelpers[0]->invokeHelper(context);
  return {true, status};
}

SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END
}  // namespace platforms
}  // namespace ops
}  // namespace sd
