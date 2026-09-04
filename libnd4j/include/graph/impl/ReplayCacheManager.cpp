#include <graph/ReplayCacheManager.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspConstants.h>
#include <system/env_functions.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN
#include <graph/vulkan/VulkanDeviceManager.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>

// std::filesystem requires: C++17, __has_include(<filesystem>), and on macOS
// the deployment target must be >= 10.15. GCC < 9 has <filesystem> but requires
// -lstdc++fs at link time, so exclude it.
#if defined(SD_FILESYSTEM_AVAILABLE)
#define HAS_FILESYSTEM 1
#elif defined(__has_include)
#  if __has_include(<filesystem>) && __cplusplus >= 201703L
#    if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 9
#      define HAS_FILESYSTEM 0
#    elif defined(__APPLE__)
#      if defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500
#        define HAS_FILESYSTEM 1
#      else
#        define HAS_FILESYSTEM 0
#      endif
#    else
#      define HAS_FILESYSTEM 1
#    endif
#  else
#    define HAS_FILESYSTEM 0
#  endif
#else
#define HAS_FILESYSTEM 0
#endif

#if HAS_FILESYSTEM
#include <filesystem>
namespace fs = std::filesystem;
#endif

namespace sd {
namespace graph {

namespace {

constexpr int kReplayCacheMetadataSchema = 1;
constexpr size_t kMaxReplayCacheMetadataBytes = 64 * 1024;

bool isLowerHex64(const std::string& value) {
  if (value.size() != 64) return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

bool isSafeMetadataText(const std::string& value, size_t maxLength = 256) {
  if (value.empty() || value.size() > maxLength) return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return std::isalnum(c) || c == ' ' || c == '_' || c == '-' || c == '.' || c == ':';
  });
}

bool extractJsonString(const std::string& content, const std::string& field,
                       std::string& value) {
  const std::string marker = "\"" + field + "\":\"";
  auto begin = content.find(marker);
  if (begin == std::string::npos) return false;
  begin += marker.size();
  auto end = content.find('"', begin);
  if (end == std::string::npos) return false;
  value = content.substr(begin, end - begin);
  return value.find('\\') == std::string::npos;
}

bool extractJsonLong(const std::string& content, const std::string& field,
                     LongType& value) {
  const std::string marker = "\"" + field + "\":";
  auto begin = content.find(marker);
  if (begin == std::string::npos) return false;
  begin += marker.size();
  char* end = nullptr;
  errno = 0;
  const long long parsed = std::strtoll(content.c_str() + begin, &end, 10);
  if (errno != 0 || end == content.c_str() + begin) return false;
  value = static_cast<LongType>(parsed);
  return true;
}

bool extractJsonInt(const std::string& content, const std::string& field,
                    int& value) {
  LongType parsed = 0;
  if (!extractJsonLong(content, field, parsed) ||
      parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

bool extractJsonBool(const std::string& content, const std::string& field,
                     bool& value) {
  const std::string marker = "\"" + field + "\":";
  auto begin = content.find(marker);
  if (begin == std::string::npos) return false;
  begin += marker.size();
  if (content.compare(begin, 4, "true") == 0) {
    value = true;
    return true;
  }
  if (content.compare(begin, 5, "false") == 0) {
    value = false;
    return true;
  }
  return false;
}

bool validateEntry(const ReplayCacheEntry& entry) {
  return entry.schemaVersion == kReplayCacheMetadataSchema &&
         entry.cacheKey != 0 && isLowerHex64(entry.artifactIdentity) &&
         isLowerHex64(entry.modelKey) && entry.startSlot >= 0 &&
         entry.endSlot >= entry.startSlot && entry.shapeKey != 0 &&
         isSafeMetadataText(entry.backendName) &&
         isSafeMetadataText(entry.backendCacheAbi) &&
         isLowerHex64(entry.deviceFingerprint) && entry.timestamp > 0 &&
         entry.numCaptureBuffers >= 0;
}

bool parseEntry(const std::string& content, ReplayCacheEntry& entry) {
  LongType workspaceHint = 0;
  LongType timestamp = 0;
  if (!extractJsonInt(content, "schemaVersion", entry.schemaVersion) ||
      !extractJsonLong(content, "cacheKey", entry.cacheKey) ||
      !extractJsonString(content, "artifactIdentity", entry.artifactIdentity) ||
      !extractJsonString(content, "modelKey", entry.modelKey) ||
      !extractJsonInt(content, "startSlot", entry.startSlot) ||
      !extractJsonInt(content, "endSlot", entry.endSlot) ||
      !extractJsonLong(content, "shapeKey", entry.shapeKey) ||
      !extractJsonString(content, "backendName", entry.backendName) ||
      !extractJsonString(content, "backendCacheAbi", entry.backendCacheAbi) ||
      !extractJsonString(content, "deviceFingerprint", entry.deviceFingerprint) ||
      !extractJsonLong(content, "workspaceHint", workspaceHint) ||
      !extractJsonInt(content, "numCaptureBuffers", entry.numCaptureBuffers) ||
      !extractJsonLong(content, "timestamp", timestamp) ||
      !extractJsonBool(content, "deviceCachingConfigured",
                       entry.deviceCachingConfigured) ||
      workspaceHint < 0 || timestamp <= 0) {
    return false;
  }
  entry.workspaceHint = static_cast<size_t>(workspaceHint);
  entry.timestamp = static_cast<int64_t>(timestamp);
  return validateEntry(entry);
}

std::string serializeEntry(const ReplayCacheEntry& entry) {
  std::ostringstream output;
  output << "{\n"
         << "\"schemaVersion\":" << entry.schemaVersion << ",\n"
         << "\"cacheKey\":" << entry.cacheKey << ",\n"
         << "\"artifactIdentity\":\"" << entry.artifactIdentity << "\",\n"
         << "\"modelKey\":\"" << entry.modelKey << "\",\n"
         << "\"startSlot\":" << entry.startSlot << ",\n"
         << "\"endSlot\":" << entry.endSlot << ",\n"
         << "\"shapeKey\":" << entry.shapeKey << ",\n"
         << "\"backendName\":\"" << entry.backendName << "\",\n"
         << "\"backendCacheAbi\":\"" << entry.backendCacheAbi << "\",\n"
         << "\"deviceFingerprint\":\"" << entry.deviceFingerprint << "\",\n"
         << "\"workspaceHint\":" << entry.workspaceHint << ",\n"
         << "\"numCaptureBuffers\":" << entry.numCaptureBuffers << ",\n"
         << "\"timestamp\":" << entry.timestamp << ",\n"
         << "\"deviceCachingConfigured\":"
         << (entry.deviceCachingConfigured ? "true" : "false") << "\n}\n";
  return output.str();
}

bool isSafeNamespace(const std::string& value) {
  return !value.empty() && value.size() <= 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return std::isalnum(c) || c == '_' || c == '-' || c == '.';
         });
}

}  // namespace

// ── ReplayCacheDeviceKey ──

ReplayCacheDeviceKey ReplayCacheDeviceKey::fromDeviceManager(DeviceType type, int localIndex) {
  ReplayCacheDeviceKey key;
  key.type = type;
  key.localIndex = localIndex;

#if defined(SD_VULKAN)
#if defined(HAVE_VULKAN) && HAVE_VULKAN
  if (type == DeviceType::VULKAN_GPU) {
    auto& manager = VulkanDeviceManager::getInstance();
    if (manager.initialize()) {
      const auto* info = manager.getDeviceInfo(localIndex);
      if (info != nullptr) {
        std::ostringstream arch;
        arch << "vendor_" << std::hex << info->vendorId << std::dec
             << "_api_" << info->vkMajor << "_" << info->vkMinor
             << "_" << info->name;
        key.archId = arch.str();
        return key;
      }
    }
  }
#endif
  key.archId = "unknown";
  return key;
#else
  auto& dm = DeviceManager::getInstance();
  if (dm.isDeviceAvailable(type, localIndex)) {
    auto info = dm.getDeviceInfo(type, localIndex);
    // Derive archId from device info
    switch (type) {
      case DeviceType::CUDA_GPU:
        key.archId = "sm_" + std::to_string(info.computeCapabilityMajor) +
                     std::to_string(info.computeCapabilityMinor);
        break;
      case DeviceType::CPU:
        // Use device name as arch identifier
        key.archId = info.name.empty() ? "generic" : info.name;
        break;
      default:
        key.archId = info.name.empty() ? "unknown" : info.name;
        break;
    }
  } else {
    key.archId = "unknown";
  }
  return key;
#endif
}

std::string ReplayCacheDeviceKey::toString() const {
  std::string typeStr;
  switch (type) {
    case DeviceType::CPU: typeStr = "cpu"; break;
    case DeviceType::CUDA_GPU: typeStr = "cuda"; break;
    case DeviceType::METAL_GPU: typeStr = "metal"; break;
    case DeviceType::VULKAN_GPU: typeStr = "vulkan"; break;
    case DeviceType::OPENCL_GPU: typeStr = "opencl"; break;
    case DeviceType::TPU: typeStr = "tpu"; break;
    case DeviceType::ACCELERATOR: typeStr = "accel"; break;
    default: typeStr = "other"; break;
  }

  // Sanitize archId for filesystem
  std::string safeArch = archId;
  for (auto& c : safeArch) {
    if (c == ' ' || c == '/' || c == '\\') c = '_';
  }

  return typeStr + "_" + std::to_string(localIndex) + "_" + safeArch;
}

bool ReplayCacheDeviceKey::isCompatibleWith(const ReplayCacheDeviceKey& other) const {
  return type == other.type && archId == other.archId;
}

// ── ReplayCacheManager ──

ReplayCacheManager::ReplayCacheManager() {
  // Read cache dir from Environment (centralized config — no direct getenv)
  std::string cfgDir = sd::env_dspReplayCacheDir();
  if (!cfgDir.empty()) {
    cacheDir_ = cfgDir;
  } else {
    // Default to ~/.ndarray/replay_cache/
    const char* home = std::getenv(sd::graph::dsp::ENV_HOME);
    if (!home) home = std::getenv(sd::graph::dsp::ENV_USERPROFILE);
    if (home) {
      cacheDir_ = std::string(home) + "/.ndarray/replay_cache";
    } else {
      cacheDir_ = "/tmp/ndarray_replay_cache";
    }
  }

  // Read enabled flag from Environment
  enabled_ = sd::env_dspReplayCacheEnabled();
}

ReplayCacheManager& ReplayCacheManager::getInstance() {
  static ReplayCacheManager* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new ReplayCacheManager();
  });
  return *instance;
}

bool ReplayCacheManager::isEnabled() const {
  return enabled_;
}

std::string ReplayCacheManager::getCacheDir() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cacheDir_;
}

std::string ReplayCacheManager::getDeviceCacheDir(const ReplayCacheDeviceKey& device) const {
  return cacheDir_ + "/" + device.toString();
}

bool ReplayCacheManager::configureCacheRoot(const std::string& cacheRoot) {
  if (cacheRoot.empty()) return false;
  std::lock_guard<std::mutex> lock(mutex_);
#if HAS_FILESYSTEM
  const std::string normalized = fs::path(cacheRoot).lexically_normal().string();
#else
  const std::string normalized = cacheRoot;
#endif
  if (normalized == cacheDir_) {
    cacheRootSealed_ = true;
    return true;
  }
  // A process may host multiple plans, but every plan must share one cache root.
  // Once any device namespace has been observed, redirecting later publications
  // would separate metadata from the opaque driver artifact it describes.
  if (cacheRootSealed_) return false;
  cacheDir_ = normalized;
  cacheRootSealed_ = true;
  cacheHits_ = 0;
  cacheMisses_ = 0;
  return true;
}

std::string ReplayCacheManager::getOrCreateBackendCacheDir(
    const ReplayCacheDeviceKey& device, const std::string& backendNamespace) {
  if (!enabled_ || !isSafeNamespace(backendNamespace)) return "";
#if HAS_FILESYSTEM
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    cacheRootSealed_ = true;
    const fs::path directory =
        fs::path(getDeviceCacheDir(device)) / "artifacts" / backendNamespace;
    fs::create_directories(directory);
    if (fs::is_symlink(directory) || !fs::is_directory(directory)) return "";
    return directory.string();
  } catch (...) {
    return "";
  }
#else
  return "";
#endif
}

int ReplayCacheManager::loadAllForDevice(const ReplayCacheDeviceKey& device) {
  if (!enabled_) return 0;

#if HAS_FILESYSTEM
  try {
    int count = 0;
    std::lock_guard<std::mutex> lock(mutex_);
    cacheRootSealed_ = true;
    std::string dir = getDeviceCacheDir(device);
    if (!fs::exists(dir)) return 0;
    std::string deviceKey = device.toString();
    if (loadedDeviceKeys_.find(deviceKey) != loadedDeviceKeys_.end()) return 0;
    auto& entries = deviceCaches_[deviceKey];

    for (const auto& diskEntry : fs::directory_iterator(dir)) {
      try {
        if (diskEntry.path().extension() != ".meta" ||
            fs::is_symlink(diskEntry.path()) ||
            !fs::is_regular_file(diskEntry.path()) ||
            fs::file_size(diskEntry.path()) == 0 ||
            fs::file_size(diskEntry.path()) > kMaxReplayCacheMetadataBytes) {
          continue;
        }
        std::ifstream file(diskEntry.path(), std::ios::binary);
        if (!file.is_open()) continue;
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        ReplayCacheEntry cacheEntry;
        const std::string metadataName = diskEntry.path().filename().string();
        if (!parseEntry(content, cacheEntry) ||
            (metadataName != cacheEntry.artifactIdentity + ".meta" &&
             metadataName.rfind(cacheEntry.artifactIdentity + ".", 0) != 0)) {
          continue;
        }
        auto found = std::find_if(
            entries.begin(), entries.end(), [&](const ReplayCacheEntry& existing) {
              return existing.artifactIdentity == cacheEntry.artifactIdentity;
            });
        if (found == entries.end()) {
          entries.push_back(std::move(cacheEntry));
          count++;
        }
      } catch (...) {
        // One malformed or concurrently replaced entry must not hide valid peers.
      }
    }
    loadedDeviceKeys_.insert(deviceKey);
    return count;
  } catch (...) {
    return 0;
  }
#else
  return 0;
#endif
}

bool ReplayCacheManager::findEntry(const ReplayCacheDeviceKey& device,
                                   const std::string& artifactIdentity,
                                   ReplayCacheEntry* entry) {
  if (!enabled_ || !isLowerHex64(artifactIdentity)) return false;
  loadAllForDevice(device);
  std::lock_guard<std::mutex> lock(mutex_);
  auto cache = deviceCaches_.find(device.toString());
  if (cache != deviceCaches_.end()) {
    auto found = std::find_if(
        cache->second.begin(), cache->second.end(),
        [&](const ReplayCacheEntry& candidate) {
          return candidate.artifactIdentity == artifactIdentity;
        });
    if (found != cache->second.end()) {
      if (entry != nullptr) *entry = *found;
      cacheHits_++;
      return true;
    }
  }
  cacheMisses_++;
  return false;
}

bool ReplayCacheManager::saveEntry(const ReplayCacheDeviceKey& device,
                                   const ReplayCacheEntry& entry) {
  if (!enabled_ || !validateEntry(entry)) return false;
#if HAS_FILESYSTEM
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    cacheRootSealed_ = true;
    const fs::path directory(getDeviceCacheDir(device));
    fs::create_directories(directory);
    if (fs::is_symlink(directory) || !fs::is_directory(directory)) return false;
    auto& entries = deviceCaches_[device.toString()];
    auto found = std::find_if(
        entries.begin(), entries.end(), [&](const ReplayCacheEntry& existing) {
          return existing.artifactIdentity == entry.artifactIdentity;
        });
    if (found != entries.end()) {
      // Metadata identities are immutable. A candidate already loaded from disk
      // describes the same opaque artifact and must not be replaced in place.
      return true;
    }
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path reservation;
    std::string immutableSuffix;
    for (int attempt = 0; attempt < 1024; ++attempt) {
      immutableSuffix = std::to_string(static_cast<long long>(nonce)) + "." +
                        std::to_string(attempt);
      reservation = directory / (".immutable." + entry.artifactIdentity + "." +
                                   immutableSuffix);
      std::error_code reservationError;
      if (fs::create_directory(reservation, reservationError)) break;
      reservation.clear();
    }
    if (reservation.empty()) return false;
    const fs::path destination = directory /
        (entry.artifactIdentity + "." + immutableSuffix + ".meta");
    const fs::path temporary = reservation / "metadata.tmp";
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      if (!output.is_open()) {
        fs::remove(reservation);
        return false;
      }
      output << serializeEntry(entry);
      output.flush();
      if (!output.good()) {
        output.close();
        fs::remove(temporary);
        fs::remove(reservation);
        return false;
      }
    }
    std::error_code renameError;
    fs::rename(temporary, destination, renameError);
    if (renameError) {
      std::error_code cleanupError;
      fs::remove(temporary, cleanupError);
      fs::remove(reservation, cleanupError);
      return false;
    }
    // The atomic rename publishes the immutable metadata outside the reservation.
    // Remove the now-empty reservation directory on success as well; otherwise
    // every cache insertion leaks one .immutable.* directory indefinitely.
    std::error_code reservationCleanupError;
    fs::remove(reservation, reservationCleanupError);

    entries.push_back(entry);
    loadedDeviceKeys_.insert(device.toString());
    return true;
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

std::vector<ReplayCacheDeviceKey> ReplayCacheManager::getCachedDevices() const {
  std::vector<ReplayCacheDeviceKey> result;
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& pair : deviceCaches_) {
    const std::string& deviceKeyStr = pair.first;
    const std::vector<ReplayCacheEntry>& entries = pair.second;
    if (!entries.empty()) {
      // Parse device key string back into ReplayCacheDeviceKey
      // Format: "type_index_arch"
      auto pos1 = deviceKeyStr.find('_');
      auto pos2 = deviceKeyStr.find('_', pos1 + 1);
      if (pos1 != std::string::npos && pos2 != std::string::npos) {
        ReplayCacheDeviceKey key;
        std::string typeStr = deviceKeyStr.substr(0, pos1);
        if (typeStr == "cpu") key.type = DeviceType::CPU;
        else if (typeStr == "cuda") key.type = DeviceType::CUDA_GPU;
        else if (typeStr == "metal") key.type = DeviceType::METAL_GPU;
        else if (typeStr == "vulkan") key.type = DeviceType::VULKAN_GPU;
        else if (typeStr == "opencl") key.type = DeviceType::OPENCL_GPU;
        else if (typeStr == "tpu") key.type = DeviceType::TPU;
        else if (typeStr == "accel") key.type = DeviceType::ACCELERATOR;
        else continue;
        key.localIndex = std::stoi(deviceKeyStr.substr(pos1 + 1, pos2 - pos1 - 1));
        key.archId = deviceKeyStr.substr(pos2 + 1);
        result.push_back(key);
      }
    }
  }
  return result;
}

int ReplayCacheManager::getDeviceCacheEntryCount(const ReplayCacheDeviceKey& device) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = deviceCaches_.find(device.toString());
  if (it != deviceCaches_.end()) {
    return static_cast<int>(it->second.size());
  }
  return 0;
}

void ReplayCacheManager::clearDevice(const ReplayCacheDeviceKey& device) {
  std::lock_guard<std::mutex> lock(mutex_);
  deviceCaches_.erase(device.toString());
  loadedDeviceKeys_.erase(device.toString());

#if HAS_FILESYSTEM
  try {
    fs::remove_all(getDeviceCacheDir(device));
  } catch (...) {}
#endif
}

bool ReplayCacheManager::migrateDeviceCache(const ReplayCacheDeviceKey& from,
                                             const ReplayCacheDeviceKey& to) {
  if (!from.isCompatibleWith(to)) return false;

  std::lock_guard<std::mutex> lock(mutex_);
  auto fromIt = deviceCaches_.find(from.toString());
  if (fromIt == deviceCaches_.end() || fromIt->second.empty()) return false;

  auto& toEntries = deviceCaches_[to.toString()];
  for (const auto& entry : fromIt->second) {
    toEntries.push_back(entry);
  }
  return true;
}

int ReplayCacheManager::pruneStaleDevices() {
  auto currentDevices = discoverCurrentDevices();
  int pruned = 0;

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = deviceCaches_.begin();
  while (it != deviceCaches_.end()) {
    bool found = false;
    for (const auto& dev : currentDevices) {
      if (dev.toString() == it->first) { found = true; break; }
    }
    if (!found) {
      it = deviceCaches_.erase(it);
      pruned++;
    } else {
      ++it;
    }
  }
  return pruned;
}

std::vector<ReplayCacheDeviceKey> ReplayCacheManager::discoverCurrentDevices() const {
  std::vector<ReplayCacheDeviceKey> result;

#if defined(SD_VULKAN)
#if defined(HAVE_VULKAN) && HAVE_VULKAN
  auto& vulkanManager = VulkanDeviceManager::getInstance();
  if (vulkanManager.initialize()) {
    for (int i = 0; i < vulkanManager.deviceCount(); ++i) {
      result.push_back(
          ReplayCacheDeviceKey::fromDeviceManager(DeviceType::VULKAN_GPU, i));
    }
  }
#endif
#else
  // Always add CPU
  result.push_back(ReplayCacheDeviceKey::fromDeviceManager(DeviceType::CPU, 0));

  // Add CUDA GPUs if available
  auto& dm = DeviceManager::getInstance();
  int cudaCount = dm.getCudaGpuCount();
  for (int i = 0; i < cudaCount; ++i) {
    result.push_back(ReplayCacheDeviceKey::fromDeviceManager(DeviceType::CUDA_GPU, i));
  }
#endif

  return result;
}

void ReplayCacheManager::clearAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  deviceCaches_.clear();
  loadedDeviceKeys_.clear();
  cacheHits_ = 0;
  cacheMisses_ = 0;

#if HAS_FILESYSTEM
  try {
    fs::remove_all(cacheDir_);
  } catch (...) {}
#endif
}

std::string ReplayCacheManager::getDeviceCacheStatsJson() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ostringstream ss;
  ss << "{\"hits\":" << cacheHits_.load()
     << ",\"misses\":" << cacheMisses_.load()
     << ",\"devices\":[";

  bool first = true;
  for (const auto& pair : deviceCaches_) {
    const std::string& deviceKey = pair.first;
    const std::vector<ReplayCacheEntry>& entries = pair.second;
    if (!first) ss << ",";
    first = false;
    ss << "{\"key\":\"" << deviceKey << "\",\"entries\":" << entries.size() << "}";
  }
  ss << "]}";
  return ss.str();
}

std::string ReplayCacheManager::getCachedDevicesJson() const {
  auto devices = getCachedDevices();
  std::ostringstream ss;
  ss << "[";
  bool first = true;
  for (const auto& dev : devices) {
    if (!first) ss << ",";
    first = false;
    ss << "{\"type\":" << static_cast<int>(dev.type)
       << ",\"index\":" << dev.localIndex
       << ",\"archId\":\"" << dev.archId << "\""
       << ",\"key\":\"" << dev.toString() << "\"}";
  }
  ss << "]";
  return ss.str();
}

}  // namespace graph
}  // namespace sd
