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

#include <ops/declarable/KernelManager.h>
#include <ops/declarable/OpRegistrator.h>
#include <helpers/KernelPerformanceRegistry.h>

#include <algorithm>
#include <mutex>
#include <regex>
#include <sstream>

namespace {
// Helper function for engine names (internal linkage)
const char* getEngineNameInternal(samediff::Engine engine) {
  switch (engine) {
    case samediff::ENGINE_CPU:
      return "CPU";
    case samediff::ENGINE_CUDA:
      return "CUDA";
    case samediff::ENGINE_ONEDNN:
      return "oneDNN";
    default:
      return "Unknown";
  }
}
}  // anonymous namespace

namespace sd {
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN

// Singleton instance
KernelManager* KernelManager::_instance = nullptr;

// Global preferred engine
static samediff::Engine _globalPreferredEngine = samediff::ENGINE_CPU;

KernelManager& KernelManager::getInstance() {
  if (_instance == nullptr) {
    _instance = new KernelManager();
  }
  return *_instance;
}

LongType KernelManager::getOpHash(const std::string& opName) const {
  // Use OpRegistrator to get hash
  auto* op = OpRegistrator::getInstance().getOperation(opName.c_str());
  if (op != nullptr) {
    return op->getOpHash();
  }
  // Simple hash if op not found
  LongType hash = 0;
  for (char c : opName) {
    hash = hash * 31 + c;
  }
  return hash;
}

std::vector<OpKernelInfo> KernelManager::getAllOperations() const {
  std::lock_guard<std::mutex> lock(_mutex);
  std::vector<OpKernelInfo> result;

  // Get all operations from OpRegistrator
  auto& registrator = OpRegistrator::getInstance();
  auto opNames = registrator.getAllRegisteredOpNames();

  for (const auto& name : opNames) {
    result.push_back(getOpKernelInfo(name));
  }

  return result;
}

OpKernelInfo KernelManager::getOpKernelInfo(const std::string& opName) const {
  return getOpKernelInfo(getOpHash(opName));
}

OpKernelInfo KernelManager::getOpKernelInfo(LongType opHash) const {
  std::lock_guard<std::mutex> lock(_mutex);

  OpKernelInfo info;
  info.opHash = opHash;
  info.hasNativeImplementation = true;  // Assume all ops have native impl

  // Get operation name
  auto* op = OpRegistrator::getInstance().getOperation(opHash);
  if (op != nullptr) {
    info.opName = op->getOpName()->c_str();
  }

  // Get available helpers
  auto& registrator = OpRegistrator::getInstance();
  auto engines = registrator.getAvailableEnginesForOp(opHash);

  for (auto engine : engines) {
    KernelInfo kernelInfo;
    kernelInfo.opName = info.opName;
    kernelInfo.opHash = opHash;
    kernelInfo.engine = engine;
    kernelInfo.engineName = getEngineNameInternal(engine);

    // Check if enabled
    auto disabledIt = _disabledKernels.find(opHash);
    if (disabledIt != _disabledKernels.end()) {
      kernelInfo.isEnabled = disabledIt->second.find(engine) == disabledIt->second.end();
    } else {
      kernelInfo.isEnabled = !isEngineGloballyDisabled(engine);
    }

    kernelInfo.isUsable = true;  // Would need context to determine
    info.kernels.push_back(kernelInfo);
  }

  // Check preferred engine
  auto prefIt = _preferredEngines.find(opHash);
  if (prefIt != _preferredEngines.end()) {
    info.preferredEngine = prefIt->second;
  } else {
    info.preferredEngine = _globalPreferredEngine;
  }

  return info;
}

std::vector<samediff::Engine> KernelManager::getAllAvailableEngines() const {
  std::set<samediff::Engine> engines;

  // Collect all engines from all operations
  auto& registrator = OpRegistrator::getInstance();
  auto opNames = registrator.getAllRegisteredOpNames();

  for (const auto& name : opNames) {
    auto opEngines = getAvailableEngines(name);
    engines.insert(opEngines.begin(), opEngines.end());
  }

  return std::vector<samediff::Engine>(engines.begin(), engines.end());
}

std::vector<samediff::Engine> KernelManager::getAvailableEngines(const std::string& opName) const {
  return getAvailableEngines(getOpHash(opName));
}

std::vector<samediff::Engine> KernelManager::getAvailableEngines(LongType opHash) const {
  return OpRegistrator::getInstance().getAvailableEnginesForOp(opHash);
}

std::vector<OpKernelInfo> KernelManager::searchOperations(const std::string& pattern) const {
  std::vector<OpKernelInfo> result;

  // Convert wildcard to regex
  std::string regexPattern = pattern;
  size_t pos = 0;
  while ((pos = regexPattern.find('*', pos)) != std::string::npos) {
    regexPattern.replace(pos, 1, ".*");
    pos += 2;
  }

  try {
    std::regex regex(regexPattern, std::regex::icase);

    auto allOps = getAllOperations();
    for (const auto& op : allOps) {
      if (std::regex_match(op.opName, regex)) {
        result.push_back(op);
      }
    }
  } catch (...) {
    // Invalid regex, try simple substring match
    auto allOps = getAllOperations();
    for (const auto& op : allOps) {
      if (op.opName.find(pattern) != std::string::npos) {
        result.push_back(op);
      }
    }
  }

  return result;
}

void KernelManager::enableKernel(const std::string& opName, samediff::Engine engine) {
  enableKernel(getOpHash(opName), engine);
}

void KernelManager::enableKernel(LongType opHash, samediff::Engine engine) {
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _disabledKernels.find(opHash);
  if (it != _disabledKernels.end()) {
    it->second.erase(engine);
  }
}

void KernelManager::disableKernel(const std::string& opName, samediff::Engine engine) {
  disableKernel(getOpHash(opName), engine);
}

void KernelManager::disableKernel(LongType opHash, samediff::Engine engine) {
  std::lock_guard<std::mutex> lock(_mutex);
  _disabledKernels[opHash].insert(engine);
}

bool KernelManager::isKernelEnabled(const std::string& opName, samediff::Engine engine) const {
  return isKernelEnabled(getOpHash(opName), engine);
}

bool KernelManager::isKernelEnabled(LongType opHash, samediff::Engine engine) const {
  std::lock_guard<std::mutex> lock(_mutex);

  // Check global disabled first
  if (_globallyDisabledEngines.find(engine) != _globallyDisabledEngines.end()) {
    return false;
  }

  // Check per-op disabled
  auto it = _disabledKernels.find(opHash);
  if (it != _disabledKernels.end()) {
    return it->second.find(engine) == it->second.end();
  }

  return true;
}

void KernelManager::enableAllKernels(const std::string& opName) {
  enableAllKernels(getOpHash(opName));
}

void KernelManager::enableAllKernels(LongType opHash) {
  std::lock_guard<std::mutex> lock(_mutex);
  _disabledKernels.erase(opHash);
}

void KernelManager::disableAllKernels(const std::string& opName) {
  disableAllKernels(getOpHash(opName));
}

void KernelManager::disableAllKernels(LongType opHash) {
  std::lock_guard<std::mutex> lock(_mutex);

  auto engines = getAvailableEngines(opHash);
  for (auto engine : engines) {
    _disabledKernels[opHash].insert(engine);
  }
}

void KernelManager::enableEngineGlobally(samediff::Engine engine) {
  std::lock_guard<std::mutex> lock(_mutex);
  _globallyDisabledEngines.erase(engine);
}

void KernelManager::disableEngineGlobally(samediff::Engine engine) {
  std::lock_guard<std::mutex> lock(_mutex);
  _globallyDisabledEngines.insert(engine);
}

bool KernelManager::isEngineGloballyDisabled(samediff::Engine engine) const {
  std::lock_guard<std::mutex> lock(_mutex);
  return _globallyDisabledEngines.find(engine) != _globallyDisabledEngines.end();
}

std::vector<samediff::Engine> KernelManager::getGloballyDisabledEngines() const {
  std::lock_guard<std::mutex> lock(_mutex);
  return std::vector<samediff::Engine>(_globallyDisabledEngines.begin(),
                                        _globallyDisabledEngines.end());
}

void KernelManager::setPreferredEngine(const std::string& opName, samediff::Engine engine) {
  setPreferredEngine(getOpHash(opName), engine);
}

void KernelManager::setPreferredEngine(LongType opHash, samediff::Engine engine) {
  std::lock_guard<std::mutex> lock(_mutex);
  _preferredEngines[opHash] = engine;
}

samediff::Engine KernelManager::getPreferredEngine(const std::string& opName) const {
  return getPreferredEngine(getOpHash(opName));
}

samediff::Engine KernelManager::getPreferredEngine(LongType opHash) const {
  std::lock_guard<std::mutex> lock(_mutex);

  auto it = _preferredEngines.find(opHash);
  if (it != _preferredEngines.end()) {
    return it->second;
  }

  return _globalPreferredEngine;
}

void KernelManager::clearPreferredEngine(const std::string& opName) {
  clearPreferredEngine(getOpHash(opName));
}

void KernelManager::clearPreferredEngine(LongType opHash) {
  std::lock_guard<std::mutex> lock(_mutex);
  _preferredEngines.erase(opHash);
}

void KernelManager::setGlobalPreferredEngine(samediff::Engine engine) {
  _globalPreferredEngine = engine;
}

samediff::Engine KernelManager::getGlobalPreferredEngine() const {
  return _globalPreferredEngine;
}

void KernelManager::resetToDefaults() {
  std::lock_guard<std::mutex> lock(_mutex);
  _disabledKernels.clear();
  _preferredEngines.clear();
  _globallyDisabledEngines.clear();
  _globalPreferredEngine = samediff::ENGINE_CPU;
}

std::string KernelManager::getConfigurationSummary() const {
  std::stringstream ss;

  ss << "Kernel Configuration Summary\n";
  ss << "============================\n\n";

  ss << "Global Preferred Engine: " << getEngineNameInternal(_globalPreferredEngine) << "\n\n";

  auto disabled = getGloballyDisabledEngines();
  if (!disabled.empty()) {
    ss << "Globally Disabled Engines:\n";
    for (auto engine : disabled) {
      ss << "  - " << getEngineNameInternal(engine) << "\n";
    }
    ss << "\n";
  }

  {
    std::lock_guard<std::mutex> lock(_mutex);

    if (!_preferredEngines.empty()) {
      ss << "Per-Operation Preferred Engines:\n";
      for (const auto& pair : _preferredEngines) {
        auto* op = OpRegistrator::getInstance().getOperation(pair.first);
        std::string opName = op ? op->getOpName()->c_str() : "unknown";
        ss << "  " << opName << ": " << getEngineNameInternal(pair.second) << "\n";
      }
      ss << "\n";
    }

    if (!_disabledKernels.empty()) {
      ss << "Per-Operation Disabled Kernels:\n";
      for (const auto& pair : _disabledKernels) {
        auto* op = OpRegistrator::getInstance().getOperation(pair.first);
        std::string opName = op ? op->getOpName()->c_str() : "unknown";
        ss << "  " << opName << ":";
        for (auto engine : pair.second) {
          ss << " " << getEngineNameInternal(engine);
        }
        ss << "\n";
      }
    }
  }

  return ss.str();
}

std::string KernelManager::exportConfiguration() const {
  // Simple JSON-like export
  std::stringstream ss;
  ss << "{\n";

  ss << "  \"globalPreferredEngine\": \"" << getEngineNameInternal(_globalPreferredEngine) << "\",\n";

  ss << "  \"globallyDisabledEngines\": [";
  auto disabled = getGloballyDisabledEngines();
  for (size_t i = 0; i < disabled.size(); i++) {
    if (i > 0) ss << ", ";
    ss << "\"" << getEngineNameInternal(disabled[i]) << "\"";
  }
  ss << "],\n";

  ss << "  \"preferredEngines\": {";
  {
    std::lock_guard<std::mutex> lock(_mutex);
    bool first = true;
    for (const auto& pair : _preferredEngines) {
      if (!first) ss << ", ";
      ss << "\"" << pair.first << "\": \"" << getEngineNameInternal(pair.second) << "\"";
      first = false;
    }
  }
  ss << "}\n";

  ss << "}\n";

  return ss.str();
}

bool KernelManager::importConfiguration(const std::string& json) {
  // Parse the simple JSON format produced by exportConfiguration.
  // Supported keys: "globalPreferredEngine", "globallyDisabledEngines", "preferredEngines".
  // Returns false if any required key is missing or unparseable.

  auto parseEngine = [](const std::string& name) -> samediff::Engine {
    if (name == "CPU")    return samediff::ENGINE_CPU;
    if (name == "CUDA")   return samediff::ENGINE_CUDA;
    if (name == "oneDNN") return samediff::ENGINE_ONEDNN;
    return samediff::ENGINE_CPU;
  };

  // Extract a quoted string value following a key, e.g. "key": "value"
  auto extractStringValue = [&](const std::string& key) -> std::string {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    auto colon = json.find(':', pos);
    if (colon == std::string::npos) return "";
    auto q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    auto q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return json.substr(q1 + 1, q2 - q1 - 1);
  };

  std::string globalEngine = extractStringValue("globalPreferredEngine");
  if (globalEngine.empty()) return false;
  setGlobalPreferredEngine(parseEngine(globalEngine));

  // Parse "globallyDisabledEngines": ["ENGINE1", "ENGINE2"]
  {
    auto pos = json.find("\"globallyDisabledEngines\"");
    if (pos != std::string::npos) {
      auto open = json.find('[', pos);
      auto close = json.find(']', pos);
      if (open != std::string::npos && close != std::string::npos) {
        std::string block = json.substr(open + 1, close - open - 1);
        size_t p = 0;
        while (p < block.size()) {
          auto q1 = block.find('"', p);
          if (q1 == std::string::npos) break;
          auto q2 = block.find('"', q1 + 1);
          if (q2 == std::string::npos) break;
          std::string ename = block.substr(q1 + 1, q2 - q1 - 1);
          disableEngineGlobally(parseEngine(ename));
          p = q2 + 1;
        }
      }
    }
  }

  // Parse "preferredEngines": {"opName": "ENGINE", ...}
  {
    auto pos = json.find("\"preferredEngines\"");
    if (pos != std::string::npos) {
      auto open = json.find('{', pos);
      auto close = json.find('}', open + 1);
      if (open != std::string::npos && close != std::string::npos) {
        std::string block = json.substr(open + 1, close - open - 1);
        size_t p = 0;
        while (p < block.size()) {
          auto q1 = block.find('"', p);
          if (q1 == std::string::npos) break;
          auto q2 = block.find('"', q1 + 1);
          if (q2 == std::string::npos) break;
          std::string opName = block.substr(q1 + 1, q2 - q1 - 1);
          auto colon = block.find(':', q2);
          if (colon == std::string::npos) break;
          auto q3 = block.find('"', colon + 1);
          if (q3 == std::string::npos) break;
          auto q4 = block.find('"', q3 + 1);
          if (q4 == std::string::npos) break;
          std::string ename = block.substr(q3 + 1, q4 - q3 - 1);
          setPreferredEngine(opName, parseEngine(ename));
          p = q4 + 1;
        }
      }
    }
  }

  return true;
}

KernelInfo KernelManager::getKernelPerformance(const std::string& opName,
                                                samediff::Engine engine) const {
  KernelInfo info;
  info.opName = opName;
  info.opHash = getOpHash(opName);
  info.engine = engine;
  info.engineName = getEngineNameInternal(engine);
  info.isEnabled = isKernelEnabled(opName, engine);
  info.isUsable = true;

  // Query registry for performance data using a generic (shape-less) signature
  KernelSignature sig = KernelPerformanceRegistry::createSignature(
      info.opHash, {}, {}, sd::DataType::FLOAT32);
  const KernelPerformanceEntry* entry = KernelPerformanceRegistry::getInstance().getPerformance(sig, engine);
  if (entry != nullptr) {
    info.avgTimeNanos = entry->meanTimeNanos;
    info.executionCount = entry->sampleCount;
  }

  return info;
}

void KernelManager::clearPerformanceStats() {
  KernelPerformanceRegistry::getInstance().clear();
}

samediff::Engine KernelManager::getBestPerformingEngine(const std::string& opName) const {
  return getBestPerformingEngine(getOpHash(opName));
}

samediff::Engine KernelManager::getBestPerformingEngine(LongType opHash) const {
  // Query registry for best engine using a generic (shape-less) signature
  KernelSignature sig = KernelPerformanceRegistry::createSignature(
      opHash, {}, {}, sd::DataType::FLOAT32);
  if (KernelPerformanceRegistry::getInstance().hasReliableData(sig)) {
    return KernelPerformanceRegistry::getInstance().getBestEngine(sig);
  }
  return getPreferredEngine(opHash);
}

// ========================
// C-style API Implementation
// ========================

extern "C" {

int kmGetEngineCount(const char* opName) {
  auto engines = KernelManager::getInstance().getAvailableEngines(opName);
  return static_cast<int>(engines.size());
}

int kmGetEngineAt(const char* opName, int index) {
  auto engines = KernelManager::getInstance().getAvailableEngines(opName);
  if (index < 0 || index >= static_cast<int>(engines.size())) {
    return -1;
  }
  return static_cast<int>(engines[index]);
}

const char* kmGetEngineName(int engineId) {
  return getEngineNameInternal(static_cast<samediff::Engine>(engineId));
}

bool kmIsKernelEnabled(const char* opName, int engineId) {
  return KernelManager::getInstance().isKernelEnabled(opName,
                                                       static_cast<samediff::Engine>(engineId));
}

void kmEnableKernel(const char* opName, int engineId) {
  KernelManager::getInstance().enableKernel(opName, static_cast<samediff::Engine>(engineId));
}

void kmDisableKernel(const char* opName, int engineId) {
  KernelManager::getInstance().disableKernel(opName, static_cast<samediff::Engine>(engineId));
}

void kmEnableAllKernels(const char* opName) {
  KernelManager::getInstance().enableAllKernels(opName);
}

void kmDisableAllKernels(const char* opName) {
  KernelManager::getInstance().disableAllKernels(opName);
}

void kmEnableEngineGlobally(int engineId) {
  KernelManager::getInstance().enableEngineGlobally(static_cast<samediff::Engine>(engineId));
}

void kmDisableEngineGlobally(int engineId) {
  KernelManager::getInstance().disableEngineGlobally(static_cast<samediff::Engine>(engineId));
}

bool kmIsEngineGloballyDisabled(int engineId) {
  return KernelManager::getInstance().isEngineGloballyDisabled(
      static_cast<samediff::Engine>(engineId));
}

void kmSetPreferredEngine(const char* opName, int engineId) {
  KernelManager::getInstance().setPreferredEngine(opName, static_cast<samediff::Engine>(engineId));
}

int kmGetPreferredEngine(const char* opName) {
  return static_cast<int>(KernelManager::getInstance().getPreferredEngine(opName));
}

void kmClearPreferredEngine(const char* opName) {
  KernelManager::getInstance().clearPreferredEngine(opName);
}

void kmSetGlobalPreferredEngine(int engineId) {
  KernelManager::getInstance().setGlobalPreferredEngine(static_cast<samediff::Engine>(engineId));
}

int kmGetGlobalPreferredEngine() {
  return static_cast<int>(KernelManager::getInstance().getGlobalPreferredEngine());
}

void kmResetToDefaults() {
  KernelManager::getInstance().resetToDefaults();
}

static std::string _configSummary;

const char* kmGetConfigurationSummary() {
  _configSummary = KernelManager::getInstance().getConfigurationSummary();
  return _configSummary.c_str();
}

int kmGetOperationCount() {
  return static_cast<int>(KernelManager::getInstance().getAllOperations().size());
}

static std::vector<std::string> _opNames;
static std::once_flag _opNamesFlag;

const char* kmGetOperationNameAt(int index) {
  std::call_once(_opNamesFlag, []() {
    auto ops = KernelManager::getInstance().getAllOperations();
    for (const auto& op : ops) {
      _opNames.push_back(op.opName);
    }
  });
  if (index < 0 || index >= static_cast<int>(_opNames.size())) {
    return nullptr;
  }
  return _opNames[index].c_str();
}

int kmGetBestPerformingEngine(const char* opName) {
  return static_cast<int>(KernelManager::getInstance().getBestPerformingEngine(opName));
}

double kmGetKernelAvgTime(const char* opName, int engineId) {
  auto info = KernelManager::getInstance().getKernelPerformance(
      opName, static_cast<samediff::Engine>(engineId));
  return info.avgTimeNanos;
}

int64_t kmGetKernelExecutionCount(const char* opName, int engineId) {
  auto info = KernelManager::getInstance().getKernelPerformance(
      opName, static_cast<samediff::Engine>(engineId));
  return info.executionCount;
}

void kmFreeString(const char* str) {
  // Strings are managed internally, no action needed
}

}  // extern "C"

SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd
