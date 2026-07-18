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

#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <graph/vulkan/VulkanSpirvDiskCache.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspHashUtils.h>
#include <system/Environment.h>
#include <system/config/VulkanConfig.h>
#include <build_stamp.h>

#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <process.h>
#include <direct.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

namespace sd {
namespace graph {

namespace {

constexpr const char* VULKAN_SPIRV_DISK_CACHE_ABI = "vulkan-spirv-disk-cache-v2";
constexpr uint32_t SPIRV_MAGIC = 0x07230203u;
// Vulkan version encoding is VK_MAKE_API_VERSION(0, major, minor, 0). Keep
// these local so this portable cache helper has no Vulkan-header dependency.
constexpr uint32_t VULKAN_API_1_0 = (1u << 22);
constexpr uint32_t VULKAN_API_1_1 = (1u << 22) | (1u << 12);
constexpr uint32_t VULKAN_API_1_2 = (1u << 22) | (2u << 12);
// A valid SPIR-V module is at least the 5-word header.
constexpr size_t SPIRV_MIN_WORDS = 5;

int capabilityCount(uint32_t mask) {
  int count = 0;
  while (mask != 0) {
    count += static_cast<int>(mask & 1u);
    mask >>= 1u;
  }
  return count;
}

std::string hashToHex(uint64_t hash) {
  std::ostringstream oss;
  oss << std::hex << std::setw(16) << std::setfill('0') << hash;
  return oss.str();
}

std::string tmpSuffix() {
  const auto tidHash = std::hash<std::thread::id>()(std::this_thread::get_id());
  std::ostringstream suffix;
  suffix << ".tmp." << static_cast<long long>(::getpid()) << "." << tidHash;
  return suffix.str();
}

bool parseUIntList(const std::string& text, std::vector<uint32_t>& out) {
  out.clear();
  size_t start = 0;
  while (start <= text.size()) {
    size_t sep = text.find(';', start);
    std::string token = (sep == std::string::npos) ? text.substr(start)
                                                   : text.substr(start, sep - start);
    start = (sep == std::string::npos) ? (text.size() + 1) : (sep + 1);
    if (token.empty()) continue;
    char* endPtr = nullptr;
    unsigned long parsed = std::strtoul(token.c_str(), &endPtr, 10);
    if (endPtr == token.c_str()) return false;
    out.push_back(static_cast<uint32_t>(parsed));
  }
  return true;
}

}  // namespace

// ── Directory helpers (shared with the Tier-2 blob code) ────────────────────

std::string VulkanSpirvDiskCache::configuredOrDefaultDir(
    const std::string& configured, const char* defaultLeaf) {
  if (!configured.empty()) return configured;
  std::string home = sd::Environment::getInstance().homeDirectory();
  if (!home.empty()) return home + "/.kompile/cache/vulkan/" + defaultLeaf;
  return std::string(".kompile/cache/vulkan/") + defaultLeaf;
}

bool VulkanSpirvDiskCache::ensureDir(const std::string& dir) {
  if (dir.empty()) return false;

  std::string currentPath;
  size_t start = 0;
  if (dir[0] == '/') {
    currentPath = "/";
    start = 1;
  }

  while (start <= dir.size()) {
    size_t slashPos = dir.find('/', start);
    std::string part = (slashPos == std::string::npos)
                           ? dir.substr(start)
                           : dir.substr(start, slashPos - start);
    start = (slashPos == std::string::npos) ? (dir.size() + 1) : (slashPos + 1);
    if (part.empty()) continue;

    if (!currentPath.empty() && currentPath.back() != '/') currentPath += "/";
    currentPath += part;

    struct stat st;
    if (stat(currentPath.c_str(), &st) == 0) {
#ifdef _WIN32
      if (!(st.st_mode & _S_IFDIR)) {
#else
      if (!S_ISDIR(st.st_mode)) {
#endif
        DSP_DIAG(JIT, "VulkanSpirvDiskCache: cache path exists but is not a directory: %s",
                 currentPath.c_str());
        return false;
      }
      continue;
    }

    if (errno != ENOENT) {
      DSP_DIAG(JIT, "VulkanSpirvDiskCache: stat failed for cache path %s (errno=%d)",
               currentPath.c_str(), errno);
      return false;
    }

#ifdef _WIN32
    if (_mkdir(currentPath.c_str()) != 0 && errno != EEXIST) {
#else
    if (mkdir(currentPath.c_str(), 0755) != 0 && errno != EEXIST) {
#endif
      DSP_DIAG(JIT, "VulkanSpirvDiskCache: mkdir failed for cache path %s (errno=%d)",
               currentPath.c_str(), errno);
      return false;
    }
  }

  return true;
}

bool VulkanSpirvDiskCache::atomicWrite(const std::string& finalPath,
                                       const void* data, size_t bytes) {
  const std::string tmpPath = finalPath + tmpSuffix();
  {
    std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
      DSP_DIAG(JIT, "VulkanSpirvDiskCache: failed to open temp file %s", tmpPath.c_str());
      return false;
    }
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    out.flush();
    if (!out.good()) {
      DSP_DIAG(JIT, "VulkanSpirvDiskCache: failed to write temp file %s", tmpPath.c_str());
      out.close();
      std::remove(tmpPath.c_str());
      return false;
    }
  }
  if (std::rename(tmpPath.c_str(), finalPath.c_str()) != 0) {
    DSP_DIAG(JIT, "VulkanSpirvDiskCache: failed to finalize cache file %s (errno=%d)",
             finalPath.c_str(), errno);
    std::remove(tmpPath.c_str());
    return false;
  }
  return true;
}

std::string VulkanSpirvDiskCache::cacheDir() {
  return configuredOrDefaultDir(
      sd::config::VulkanConfig::getInstance().spirvCacheDir(), "spirv_cache");
}

std::string VulkanSpirvDiskCache::overrideDir() {
  return configuredOrDefaultDir(
      sd::config::VulkanConfig::getInstance().spirvOverrideDir(), "spirv_override");
}

// ── Key computation ──────────────────────────────────────────────────────────

bool VulkanSpirvDiskCache::active() {
  auto& cfg = sd::config::VulkanConfig::getInstance();
  return cfg.spirvCacheEnabled() && !cfg.alwaysCompile();
}

uint32_t VulkanSpirvDiskCache::normalizeApiVersion(uint32_t apiVersion) {
  const uint32_t major = (apiVersion >> 22u) & 0x7fu;
  const uint32_t minor = (apiVersion >> 12u) & 0x3ffu;
  if (major > 1u || (major == 1u && minor >= 2u)) return VULKAN_API_1_2;
  if (major == 1u && minor >= 1u) return VULKAN_API_1_1;
  return VULKAN_API_1_0;
}

std::string VulkanSpirvDiskCache::computeKey(const std::string& mlirModuleStr,
                                             uint32_t pushConstantBytes,
                                             const DeviceCapsKey& caps) {
  uint64_t hash = dsp::FNV1A64_OFFSET_BASIS;
  dsp::fnv1aMix(hash, VULKAN_SPIRV_DISK_CACHE_ABI,
                std::strlen(VULKAN_SPIRV_DISK_CACHE_ABI));
  // The key is a deployment artifact identity, not a local object-cache key.
  // Never mix LIBND4J_BUILD_STAMP here: it is a per-build timestamp and would
  // make a CI-produced artifact impossible to consume from an Android build.
  const uint32_t apiVersion = normalizeApiVersion(caps.apiVersion);
  const int fp16 = caps.fp16 ? 1 : 0;
  const int storage16 = caps.storage16 ? 1 : 0;
  const int fp64 = caps.fp64 ? 1 : 0;
  const int int64Flag = caps.int64 ? 1 : 0;
  const int int8Flag = caps.int8 ? 1 : 0;
  dsp::fnv1aMix(hash, &apiVersion, sizeof(apiVersion));
  dsp::fnv1aMix(hash, &fp16, sizeof(fp16));
  dsp::fnv1aMix(hash, &storage16, sizeof(storage16));
  dsp::fnv1aMix(hash, &fp64, sizeof(fp64));
  dsp::fnv1aMix(hash, &int64Flag, sizeof(int64Flag));
  dsp::fnv1aMix(hash, &int8Flag, sizeof(int8Flag));
  dsp::fnv1aMix(hash, &pushConstantBytes, sizeof(pushConstantBytes));
  // The MLIR text bakes in shapes, dtypes, and op arguments, so they are
  // implicitly part of the key (the analogue of Triton's TTIR text).
  dsp::fnv1aMix(hash, mlirModuleStr.data(), mlirModuleStr.size());
  return hashToHex(hash);
}

// ── Read path ────────────────────────────────────────────────────────────────

bool VulkanSpirvDiskCache::loadFromDir(const std::string& dir,
                                       const std::string& key,
                                       const std::string& opName,
                                       std::vector<uint32_t>& bytecode,
                                       std::vector<uint32_t>& descriptorBindings) {
  if (dir.empty()) return false;
  const std::string basePath = dir + "/spv_" + key;
  const std::string spvPath = basePath + ".spv";
  const std::string metaPath = basePath + ".meta";

  std::ifstream spvFile(spvPath, std::ios::binary);
  if (!spvFile.good()) return false;
  std::ifstream metaFile(metaPath);
  if (!metaFile.good()) return false;

  std::string spvBytes((std::istreambuf_iterator<char>(spvFile)),
                       std::istreambuf_iterator<char>());
  if (spvBytes.empty() || (spvBytes.size() % sizeof(uint32_t)) != 0) return false;
  const size_t wordCount = spvBytes.size() / sizeof(uint32_t);
  if (wordCount < SPIRV_MIN_WORDS) return false;

  uint32_t magic = 0;
  std::memcpy(&magic, spvBytes.data(), sizeof(uint32_t));
  if (magic != SPIRV_MAGIC) {
    DSP_DIAG(JIT, "VulkanSpirvDiskCache: entry %s rejected (bad SPIR-V magic)", key.c_str());
    return false;
  }

  std::string metaAbi;
  std::string metaOpName;
  std::string metaBindings;
  bool bindingsPresent = false;
  long long metaWords = -1;
  std::string line;
  while (std::getline(metaFile, line)) {
    size_t eqPos = line.find('=');
    if (eqPos == std::string::npos) continue;
    const std::string k = line.substr(0, eqPos);
    const std::string v = line.substr(eqPos + 1);
    if (k == "cacheAbi") {
      metaAbi = v;
    } else if (k == "opName") {
      metaOpName = v;
    } else if (k == "descriptorBindings") {
      metaBindings = v;
      bindingsPresent = true;
    } else if (k == "spirvWords") {
      char* endPtr = nullptr;
      metaWords = std::strtoll(v.c_str(), &endPtr, 10);
      if (endPtr == v.c_str()) metaWords = -1;
    }
  }

  if (metaAbi != VULKAN_SPIRV_DISK_CACHE_ABI) return false;
  if (!metaOpName.empty() && !opName.empty() && metaOpName != opName) {
    DSP_DIAG(JIT, "VulkanSpirvDiskCache: entry %s op name '%s' does not match expected '%s'",
             key.c_str(), metaOpName.c_str(), opName.c_str());
    return false;
  }
  if (metaWords >= 0 && static_cast<size_t>(metaWords) != wordCount) return false;

  // The descriptor ABI is load-bearing: on a hit the MLIR pipeline (where
  // bindings are normally extracted) is skipped entirely.
  std::vector<uint32_t> bindings;
  if (!bindingsPresent || !parseUIntList(metaBindings, bindings)) return false;
  if (bindings.empty()) return false;
  // Mirror the mlirToSpirv() interface validation: set-zero bindings must be
  // sorted and duplicate-free.
  if (!std::is_sorted(bindings.begin(), bindings.end()) ||
      std::adjacent_find(bindings.begin(), bindings.end()) != bindings.end()) {
    return false;
  }

  bytecode.resize(wordCount);
  std::memcpy(bytecode.data(), spvBytes.data(), spvBytes.size());
  descriptorBindings = std::move(bindings);
  return true;
}

bool VulkanSpirvDiskCache::loadFromDirectory(
    const std::string& directory, const std::string& key,
    const std::string& opName, std::vector<uint32_t>& bytecode,
    std::vector<uint32_t>& descriptorBindings) {
  if (key.empty() || directory.empty()) return false;
  return loadFromDir(directory, key, opName, bytecode, descriptorBindings);
}

bool VulkanSpirvDiskCache::loadCompatibleFromDirectory(
    const std::string& directory, const std::string& mlirModuleStr,
    uint32_t pushConstantBytes, const DeviceCapsKey& runtimeCaps,
    const std::string& opName, std::vector<uint32_t>& bytecode,
    std::vector<uint32_t>& descriptorBindings, std::string* matchedKey,
    DeviceCapsKey* matchedTargetCaps) {
  if (directory.empty()) return false;
  bytecode.clear();
  descriptorBindings.clear();
  if (matchedKey != nullptr) matchedKey->clear();

  const uint32_t runtimeApi = normalizeApiVersion(runtimeCaps.apiVersion);
  std::vector<uint32_t> apiCandidates{runtimeApi};
  if (runtimeApi > VULKAN_API_1_1) apiCandidates.push_back(VULKAN_API_1_1);
  if (runtimeApi > VULKAN_API_1_0) apiCandidates.push_back(VULKAN_API_1_0);

  uint32_t availableMask = 0;
  if (runtimeCaps.fp16) availableMask |= 1u << 0u;
  if (runtimeCaps.storage16) availableMask |= 1u << 1u;
  if (runtimeCaps.fp64) availableMask |= 1u << 2u;
  if (runtimeCaps.int64) availableMask |= 1u << 3u;
  if (runtimeCaps.int8) availableMask |= 1u << 4u;

  std::vector<uint32_t> capabilityMasks;
  for (uint32_t mask = 0; mask < (1u << 5u); ++mask) {
    if ((mask & ~availableMask) == 0) capabilityMasks.push_back(mask);
  }
  std::sort(capabilityMasks.begin(), capabilityMasks.end(),
            [](uint32_t lhs, uint32_t rhs) {
              const int lhsCount = capabilityCount(lhs);
              const int rhsCount = capabilityCount(rhs);
              return lhsCount != rhsCount ? lhsCount > rhsCount : lhs > rhs;
            });

  for (uint32_t apiVersion : apiCandidates) {
    for (uint32_t mask : capabilityMasks) {
      const DeviceCapsKey candidate{
          apiVersion,
          (mask & (1u << 0u)) != 0,
          (mask & (1u << 1u)) != 0,
          (mask & (1u << 2u)) != 0,
          (mask & (1u << 3u)) != 0,
          (mask & (1u << 4u)) != 0};
      const std::string key =
          computeKey(mlirModuleStr, pushConstantBytes, candidate);
      std::vector<uint32_t> candidateBytecode;
      std::vector<uint32_t> candidateBindings;
      if (!loadFromDir(directory, key, opName, candidateBytecode,
                       candidateBindings)) {
        continue;
      }
      bytecode = std::move(candidateBytecode);
      descriptorBindings = std::move(candidateBindings);
      if (matchedKey != nullptr) *matchedKey = key;
      if (matchedTargetCaps != nullptr) *matchedTargetCaps = candidate;
      return true;
    }
  }
  return false;
}

bool VulkanSpirvDiskCache::load(const std::string& key, const std::string& opName,
                                std::vector<uint32_t>& bytecode,
                                std::vector<uint32_t>& descriptorBindings) {
  auto& cfg = sd::config::VulkanConfig::getInstance();
  if (key.empty()) return false;

  // Override dir first: read-only pre-seed (APK/deployment path), never written.
  if (loadFromDir(overrideDir(), key, opName, bytecode, descriptorBindings)) {
    cfg.incrementSpirvDiskHits();
    DSP_DIAG(JIT, "VulkanSpirvDiskCache: override HIT for op '%s' (%zu words)",
             opName.c_str(), bytecode.size());
    return true;
  }
  if (loadFromDir(cacheDir(), key, opName, bytecode, descriptorBindings)) {
    cfg.incrementSpirvDiskHits();
    DSP_DIAG(JIT, "VulkanSpirvDiskCache: disk HIT for op '%s' (%zu words)",
             opName.c_str(), bytecode.size());
    return true;
  }
  cfg.incrementSpirvDiskMisses();
  return false;
}

// ── Write path ───────────────────────────────────────────────────────────────

void VulkanSpirvDiskCache::store(const std::string& key, const std::string& opName,
                                 const std::vector<uint32_t>& bytecode,
                                 const std::vector<uint32_t>& descriptorBindings,
                                 uint32_t pushConstantBytes, const DeviceCapsKey& caps,
                                 const std::string& mlirModuleStr) {
  auto& cfg = sd::config::VulkanConfig::getInstance();
  if (key.empty() || bytecode.empty() || descriptorBindings.empty()) return;

  const std::string dir = cacheDir();
  if (!ensureDir(dir)) return;

  const std::string basePath = dir + "/spv_" + key;
  const std::string spvPath = basePath + ".spv";
  const std::string metaPath = basePath + ".meta";

  // .spv first; .meta only if the blob finalized, so a .meta never refers to
  // a missing blob (Triton write ordering).
  if (!atomicWrite(spvPath, bytecode.data(), bytecode.size() * sizeof(uint32_t))) return;

  uint64_t nativeBuildHash = dsp::FNV1A64_OFFSET_BASIS;
  dsp::fnv1aMix(nativeBuildHash, LIBND4J_BUILD_STAMP,
                std::strlen(LIBND4J_BUILD_STAMP));

  std::ostringstream meta;
  meta << "cacheAbi=" << VULKAN_SPIRV_DISK_CACHE_ABI << "\n";
  meta << "nativeBuildInfoHash=" << hashToHex(nativeBuildHash) << "\n";
  meta << "opName=" << opName << "\n";
  meta << "entryPoint=main\n";
  meta << "pushConstantBytes=" << pushConstantBytes << "\n";
  meta << "apiVersion=" << normalizeApiVersion(caps.apiVersion) << "\n";
  meta << "capsFp16=" << (caps.fp16 ? 1 : 0) << "\n";
  meta << "capsStorage16=" << (caps.storage16 ? 1 : 0) << "\n";
  meta << "capsFp64=" << (caps.fp64 ? 1 : 0) << "\n";
  meta << "capsInt64=" << (caps.int64 ? 1 : 0) << "\n";
  meta << "capsInt8=" << (caps.int8 ? 1 : 0) << "\n";
  meta << "descriptorBindings=";
  for (size_t i = 0; i < descriptorBindings.size(); i++) {
    if (i > 0) meta << ";";
    meta << descriptorBindings[i];
  }
  meta << "\n";
  meta << "spirvWords=" << bytecode.size() << "\n";
  meta << "createdAtEpochSec=" << static_cast<long long>(std::time(nullptr)) << "\n";

  const std::string metaStr = meta.str();
  if (!atomicWrite(metaPath, metaStr.data(), metaStr.size())) {
    // Leave the .spv in place — a blob without .meta is simply ignored on read.
    return;
  }

  if (cfg.kernelDump() && !mlirModuleStr.empty()) {
    atomicWrite(basePath + ".mlir", mlirModuleStr.data(), mlirModuleStr.size());
  }

  cfg.incrementSpirvDiskStores();
  DSP_DIAG(JIT, "VulkanSpirvDiskCache: disk STORED for op '%s' (%zu words)",
           opName.c_str(), bytecode.size());
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
