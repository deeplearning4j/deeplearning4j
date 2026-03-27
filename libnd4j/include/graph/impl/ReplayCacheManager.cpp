#include <graph/ReplayCacheManager.h>
#include <graph/NativeDynamicShapePlan.h>

#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <algorithm>

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

// FNV-1a 64-bit hash
static LongType fnv1a64(const void* data, size_t len) {
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  LongType hash = 14695981039346656037ULL;
  for (size_t i = 0; i < len; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

static LongType fnv1a64String(const std::string& s) {
  return fnv1a64(s.data(), s.size());
}

// ── ReplayCacheDeviceKey ──

ReplayCacheDeviceKey ReplayCacheDeviceKey::fromDeviceManager(DeviceType type, int localIndex) {
  ReplayCacheDeviceKey key;
  key.type = type;
  key.localIndex = localIndex;

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
}

ReplayCacheDeviceKey ReplayCacheDeviceKey::fromCurrentDevice() {
#ifdef SD_CUDA
  int devId = sd::AffinityManager::currentDeviceId();
  return fromDeviceManager(DeviceType::CUDA_GPU, devId);
#else
  return fromDeviceManager(DeviceType::CPU, 0);
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
  // Check env var for cache dir
  const char* envDir = std::getenv("ND4J_REPLAY_CACHE_DIR");
  if (envDir && envDir[0] != '\0') {
    cacheDir_ = envDir;
  } else {
    // Default to ~/.ndarray/replay_cache/
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    if (home) {
      cacheDir_ = std::string(home) + "/.ndarray/replay_cache";
    } else {
      cacheDir_ = "/tmp/ndarray_replay_cache";
    }
  }

  // Check env var for enable/disable
  const char* envEnabled = std::getenv("ND4J_REPLAY_CACHE_ENABLED");
  enabled_ = (envEnabled == nullptr || std::string(envEnabled) != "false");
}

ReplayCacheManager& ReplayCacheManager::getInstance() {
  static ReplayCacheManager instance;
  return instance;
}

bool ReplayCacheManager::isEnabled() const {
  return enabled_;
}

std::string ReplayCacheManager::getCacheDir() const {
  return cacheDir_;
}

LongType ReplayCacheManager::computeCacheKey(const GraphSegment& seg, LongType shapeKey,
                                              const ReplayCacheDeviceKey& device) const {
  // Combine slot range + shape key + device key into a single hash
  std::string keyStr = std::to_string(seg.startSlot) + ":" +
                       std::to_string(seg.endSlot) + ":" +
                       std::to_string(shapeKey) + ":" +
                       device.toString();
  return fnv1a64String(keyStr);
}

std::string ReplayCacheManager::getDeviceCacheDir(const ReplayCacheDeviceKey& device) const {
  return cacheDir_ + "/" + device.toString();
}

std::string ReplayCacheManager::getEntryFilePath(const ReplayCacheDeviceKey& device,
                                                   LongType cacheKey) const {
  return getDeviceCacheDir(device) + "/replay_" + std::to_string(cacheKey) + ".meta";
}

bool ReplayCacheManager::saveSegmentMetadata(const GraphSegment& seg, LongType shapeKey,
                                              const ReplayCacheDeviceKey& device) {
  if (!enabled_) return false;

  LongType key = computeCacheKey(seg, shapeKey, device);

  // Build entry
  ReplayCacheEntry entry;
  entry.cacheKey = key;
  entry.startSlot = seg.startSlot;
  entry.endSlot = seg.endSlot;
  entry.shapeKey = shapeKey;
  entry.backendName = seg.compiledByBackend;
  entry.workspaceHint = 0;
  entry.numCaptureBuffers = 0;
  entry.timestamp = static_cast<int64_t>(std::time(nullptr));

  if (seg.replayHandle) {
    auto& bufs = seg.replayHandle->getCaptureBuffers();
    entry.numCaptureBuffers = static_cast<int>(bufs.size());
  }

  // Store in memory
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string deviceKey = device.toString();
    auto& entries = deviceCaches_[deviceKey];

    // Replace or add
    bool replaced = false;
    for (auto& e : entries) {
      if (e.cacheKey == key) {
        e = entry;
        replaced = true;
        break;
      }
    }
    if (!replaced) {
      entries.push_back(entry);
    }
  }

#if HAS_FILESYSTEM
  try {
    // Ensure directory exists
    fs::create_directories(getDeviceCacheDir(device));

    // Write entry to file
    std::string path = getEntryFilePath(device, key);
    std::ofstream file(path);
    if (!file.is_open()) return false;

    file << "{"
         << "\"cacheKey\":" << key
         << ",\"startSlot\":" << entry.startSlot
         << ",\"endSlot\":" << entry.endSlot
         << ",\"shapeKey\":" << entry.shapeKey
         << ",\"backendName\":\"" << entry.backendName << "\""
         << ",\"workspaceHint\":" << entry.workspaceHint
         << ",\"numCaptureBuffers\":" << entry.numCaptureBuffers
         << ",\"timestamp\":" << entry.timestamp
         << "}";
    return true;
  } catch (...) {
    return false;
  }
#else
  return true;  // Memory-only cache when no filesystem support
#endif
}

bool ReplayCacheManager::loadSegmentMetadata(const GraphSegment& seg, LongType shapeKey,
                                              const ReplayCacheDeviceKey& device,
                                              size_t& outWorkspaceHint, int& outNumCaptureBuffers,
                                              std::string& outBackendName) {
  if (!enabled_) {
    cacheMisses_++;
    return false;
  }

  LongType key = computeCacheKey(seg, shapeKey, device);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string deviceKey = device.toString();
    auto it = deviceCaches_.find(deviceKey);
    if (it != deviceCaches_.end()) {
      for (const auto& e : it->second) {
        if (e.cacheKey == key) {
          outWorkspaceHint = e.workspaceHint;
          outNumCaptureBuffers = e.numCaptureBuffers;
          outBackendName = e.backendName;
          cacheHits_++;
          return true;
        }
      }
    }
  }

  cacheMisses_++;
  return false;
}

int ReplayCacheManager::loadAllForDevice(const ReplayCacheDeviceKey& device) {
  if (!enabled_) return 0;

#if HAS_FILESYSTEM
  try {
    std::string dir = getDeviceCacheDir(device);
    if (!fs::exists(dir)) return 0;

    int count = 0;
    std::lock_guard<std::mutex> lock(mutex_);
    std::string deviceKey = device.toString();
    auto& entries = deviceCaches_[deviceKey];

    for (const auto& entry : fs::directory_iterator(dir)) {
      if (entry.path().extension() == ".meta") {
        // Simple JSON parsing - read file and extract fields
        std::ifstream file(entry.path());
        if (!file.is_open()) continue;

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        ReplayCacheEntry cacheEntry;
        // Basic JSON field extraction
        auto extractLong = [&](const std::string& field) -> LongType {
          auto pos = content.find("\"" + field + "\":");
          if (pos == std::string::npos) return 0;
          pos = content.find(':', pos) + 1;
          return std::stoll(content.substr(pos));
        };
        auto extractInt = [&](const std::string& field) -> int {
          auto pos = content.find("\"" + field + "\":");
          if (pos == std::string::npos) return 0;
          pos = content.find(':', pos) + 1;
          return std::stoi(content.substr(pos));
        };
        auto extractString = [&](const std::string& field) -> std::string {
          auto pos = content.find("\"" + field + "\":\"");
          if (pos == std::string::npos) return "";
          pos = content.find(":\"", pos) + 2;
          auto end = content.find("\"", pos);
          return content.substr(pos, end - pos);
        };

        cacheEntry.cacheKey = extractLong("cacheKey");
        cacheEntry.startSlot = extractInt("startSlot");
        cacheEntry.endSlot = extractInt("endSlot");
        cacheEntry.shapeKey = extractLong("shapeKey");
        cacheEntry.backendName = extractString("backendName");
        cacheEntry.workspaceHint = static_cast<size_t>(extractLong("workspaceHint"));
        cacheEntry.numCaptureBuffers = extractInt("numCaptureBuffers");
        cacheEntry.timestamp = extractLong("timestamp");

        // Check for duplicates
        bool found = false;
        for (const auto& e : entries) {
          if (e.cacheKey == cacheEntry.cacheKey) { found = true; break; }
        }
        if (!found) {
          entries.push_back(cacheEntry);
          count++;
        }
      }
    }
    return count;
  } catch (...) {
    return 0;
  }
#else
  return 0;
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
        else key.type = DeviceType::CPU;
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

  // Always add CPU
  result.push_back(ReplayCacheDeviceKey::fromDeviceManager(DeviceType::CPU, 0));

  // Add CUDA GPUs if available
  auto& dm = DeviceManager::getInstance();
  int cudaCount = dm.getCudaGpuCount();
  for (int i = 0; i < cudaCount; ++i) {
    result.push_back(ReplayCacheDeviceKey::fromDeviceManager(DeviceType::CUDA_GPU, i));
  }

  return result;
}

void ReplayCacheManager::clearAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  deviceCaches_.clear();
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
