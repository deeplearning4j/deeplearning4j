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

//
// @author Adam Gibson
//

#include <helpers/HelperVersionRegistry.h>
#include <helpers/logger.h>

#include <cstdlib>
#include <mutex>
#include <sstream>

namespace sd {
namespace ops {
namespace platforms {
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN

HelperVersionRegistry::HelperVersionRegistry() { initializeFromEnvironment(); }

HelperVersionRegistry& HelperVersionRegistry::getInstance() {
  static HelperVersionRegistry* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new HelperVersionRegistry();
  });
  return *instance;
}

void HelperVersionRegistry::initializeFromEnvironment() {
  // Check for strict enforcement setting
  const char* enforceEnv = std::getenv("SD_KERNEL_ENFORCE_VERSIONS");
  if (enforceEnv != nullptr) {
    std::string val(enforceEnv);
    _strictEnforcement = (val == "1" || val == "true" || val == "TRUE" || val == "on" || val == "ON");
  } else {
    // Default to strict enforcement per user preference
    _strictEnforcement = true;
  }

  // Check for verbose logging
  const char* verboseEnv = std::getenv("SD_KERNEL_VERSION_VERBOSE");
  if (verboseEnv != nullptr) {
    std::string val(verboseEnv);
    _verboseLogging = (val == "1" || val == "true" || val == "TRUE" || val == "on" || val == "ON");
  }

  // Check for log version mismatches
  const char* logMismatchEnv = std::getenv("SD_KERNEL_LOG_VERSION_MISMATCHES");
  if (logMismatchEnv != nullptr) {
    std::string val(logMismatchEnv);
    if (val == "1" || val == "true" || val == "TRUE" || val == "on" || val == "ON") {
      _verboseLogging = true;
    }
  }
}

void HelperVersionRegistry::registerProvider(const std::string& name, VersionProviderCallback provider) {
  std::lock_guard<std::mutex> lock(_mutex);
  _providers[name] = provider;
  _cacheValid = false;  // Invalidate cache when new provider is added

  if (_verboseLogging) {
    sd_debug("HelperVersionRegistry: Registered provider for %s\n", name.c_str());
  }
}

HelperInfo HelperVersionRegistry::getHelperInfo(const std::string& name) {
  std::lock_guard<std::mutex> lock(_mutex);

  // Check cache first
  if (_cacheValid) {
    auto it = _cache.find(name);
    if (it != _cache.end()) {
      return it->second;
    }
  }

  // Query provider
  auto providerIt = _providers.find(name);
  if (providerIt == _providers.end()) {
    // No provider registered for this helper
    HelperInfo info;
    info.name = name;
    info.available = false;
    info.statusMessage = "No version provider registered";
    return info;
  }

  // Call the provider to get current info
  HelperInfo info = providerIt->second();

  // Update compatibility status
  info.versionCompatible = info.available && info.runtime.inRange(info.minSupported, info.maxSupported);

  // Cache the result
  _cache[name] = info;

  // Log if verbose or if there's a mismatch
  if (_verboseLogging || (!info.versionCompatible && info.available)) {
    logVersionCheck(name, info, info.versionCompatible);
  }

  return info;
}

std::map<std::string, HelperInfo> HelperVersionRegistry::getAllHelperInfo() {
  std::lock_guard<std::mutex> lock(_mutex);

  if (!_cacheValid) {
    _cache.clear();
    for (const auto& pair : _providers) {
      HelperInfo info = pair.second();
      info.versionCompatible = info.available && info.runtime.inRange(info.minSupported, info.maxSupported);
      _cache[pair.first] = info;
    }
    _cacheValid = true;
  }

  return _cache;
}

bool HelperVersionRegistry::checkVersion(const std::string& name, const HelperVersion& minVersion) {
  HelperInfo info = getHelperInfo(name);

  if (!info.available) {
    return false;
  }

  bool result = info.runtime.meetsMinimum(minVersion);

  if (_verboseLogging) {
    sd_debug("HelperVersionRegistry: Version check for %s - runtime %s %s minimum %s\n", name.c_str(),
              info.runtime.toString().c_str(), result ? ">=" : "<", minVersion.toString().c_str());
  }

  return result;
}

bool HelperVersionRegistry::hasCapability(const std::string& name, HelperCapability capability) {
  HelperInfo info = getHelperInfo(name);

  if (!info.available) {
    return false;
  }

  // If strict enforcement is on and version is incompatible, don't report capabilities
  if (_strictEnforcement && !info.versionCompatible) {
    return false;
  }

  return platforms::hasCapability(info.capabilities, capability);
}

bool HelperVersionRegistry::isHelperUsable(const std::string& name) {
  HelperInfo info = getHelperInfo(name);

  if (!info.available) {
    return false;
  }

  if (_strictEnforcement) {
    return info.versionCompatible;
  }

  // If not strict, allow any available helper
  return true;
}

std::vector<std::string> HelperVersionRegistry::getAvailableHelpers() {
  std::vector<std::string> result;
  auto allInfo = getAllHelperInfo();

  for (const auto& pair : allInfo) {
    if (pair.second.available) {
      if (!_strictEnforcement || pair.second.versionCompatible) {
        result.push_back(pair.first);
      }
    }
  }

  return result;
}

std::string HelperVersionRegistry::getStatusSummary() {
  std::ostringstream ss;
  ss << "Helper Library Version Status:\n";
  ss << "==============================\n";
  ss << "Strict enforcement: " << (_strictEnforcement ? "ON" : "OFF") << "\n\n";

  auto allInfo = getAllHelperInfo();

  for (const auto& pair : allInfo) {
    const HelperInfo& info = pair.second;
    ss << info.getDetailedStatus() << "\n";
  }

  return ss.str();
}

void HelperVersionRegistry::refresh() {
  std::lock_guard<std::mutex> lock(_mutex);
  _cacheValid = false;
  _cache.clear();
}

void HelperVersionRegistry::logVersionCheck(const std::string& name, const HelperInfo& info, bool success) {
  if (success) {
    sd_debug("HelperVersionRegistry: %s version check PASSED - runtime %s in range [%s, %s]\n", name.c_str(),
              info.runtime.toString().c_str(), info.minSupported.toString().c_str(),
              info.maxSupported.toString().c_str());
  } else {
    sd_debug(
        "HelperVersionRegistry: %s version check FAILED - runtime %s NOT in range [%s, %s]. %s\n",
        name.c_str(), info.runtime.toString().c_str(), info.minSupported.toString().c_str(),
        info.maxSupported.toString().c_str(),
        _strictEnforcement ? "Helper will be disabled." : "Helper may still be used (strict enforcement OFF).");
  }
}

SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END
}  // namespace platforms
}  // namespace ops
}  // namespace sd
