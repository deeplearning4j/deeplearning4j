#ifndef REPLAYCACHEMANAGER_H
#define REPLAYCACHEMANAGER_H

#include <system/common.h>
#include <types/types.h>
#include <execution/ModelParallelConfig.h>
#include <execution/DeviceManager.h>

#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

namespace sd {
namespace graph {

// Forward declare
struct GraphSegment;

// Pull in the device types from modelparallel namespace
using modelparallel::DeviceType;
using modelparallel::DeviceManager;
using modelparallel::DeviceInfo;

/**
 * Device descriptor for replay cache keying.
 * Identifies a specific compute device (GPU, CPU, etc.) with architecture info.
 */
struct SD_LIB_EXPORT ReplayCacheDeviceKey {
  DeviceType type;       // CPU, CUDA_GPU, METAL_GPU, VULKAN_GPU, etc.
  int localIndex;        // Device index within type (GPU 0, GPU 1, CPU 0, ...)
  std::string archId;    // Architecture string: "sm_89", "avx512", "apple_m2", etc.

  ReplayCacheDeviceKey() : type(DeviceType::CPU), localIndex(0) {}
  ReplayCacheDeviceKey(DeviceType t, int idx, const std::string& arch)
      : type(t), localIndex(idx), archId(arch) {}

  /** Create from DeviceManager info. */
  static ReplayCacheDeviceKey fromDeviceManager(DeviceType type, int localIndex);

  /** Create from current thread's affinity device. */
  static ReplayCacheDeviceKey fromCurrentDevice();

  /** String key for filesystem: "cuda_0_sm89", "cpu_0_avx512", etc. */
  std::string toString() const;

  /** Two devices are migration-compatible if same type + same archId. */
  bool isCompatibleWith(const ReplayCacheDeviceKey& other) const;

  bool operator==(const ReplayCacheDeviceKey& other) const {
    return type == other.type && localIndex == other.localIndex && archId == other.archId;
  }
};

/**
 * Cached metadata for a single segment on a specific device.
 */
struct ReplayCacheEntry {
  LongType cacheKey;           // FNV1a hash of {slot_range, op_names, shape_key, device_key}
  int startSlot;
  int endSlot;
  LongType shapeKey;
  std::string backendName;     // "Triton", "oneDNN", "CUDA", etc.
  size_t workspaceHint;        // Workspace size that worked
  int numCaptureBuffers;       // Number of staging buffers needed
  int64_t timestamp;           // When this entry was created
};

/**
 * Multi-platform per-device replay cache manager.
 *
 * Caches compilation metadata (workspace sizes, capture buffer layout, backend hints)
 * to enable faster warm re-capture on process restart. Does NOT cache compiled GPU
 * artifacts (those are device-bound), only the metadata needed to speed up re-capture.
 */
class SD_LIB_EXPORT ReplayCacheManager {
 public:
  static ReplayCacheManager& getInstance();

  // ── Per-device save/load ──

  bool saveSegmentMetadata(const GraphSegment& seg, LongType shapeKey,
                           const ReplayCacheDeviceKey& device);

  bool loadSegmentMetadata(const GraphSegment& seg, LongType shapeKey,
                           const ReplayCacheDeviceKey& device,
                           size_t& outWorkspaceHint, int& outNumCaptureBuffers,
                           std::string& outBackendName);

  /** Load ALL cached segments for a device (bulk warm-start). Returns count loaded. */
  int loadAllForDevice(const ReplayCacheDeviceKey& device);

  // ── Per-device management ──

  std::vector<ReplayCacheDeviceKey> getCachedDevices() const;
  int getDeviceCacheEntryCount(const ReplayCacheDeviceKey& device) const;
  void clearDevice(const ReplayCacheDeviceKey& device);

  /** Migrate cache between compatible devices (same type + arch). */
  bool migrateDeviceCache(const ReplayCacheDeviceKey& from,
                          const ReplayCacheDeviceKey& to);

  /** Remove entries for absent/changed devices. */
  int pruneStaleDevices();

  /** Auto-detect all available devices and their architecture strings. */
  std::vector<ReplayCacheDeviceKey> discoverCurrentDevices() const;

  // ── General management ──

  void clearAll();
  bool isEnabled() const;
  std::string getCacheDir() const;

  // Stats
  int getCacheHits() const { return cacheHits_.load(); }
  int getCacheMisses() const { return cacheMisses_.load(); }
  std::string getDeviceCacheStatsJson() const;
  std::string getCachedDevicesJson() const;

 private:
  ReplayCacheManager();

  std::string getDeviceCacheDir(const ReplayCacheDeviceKey& device) const;
  std::string getEntryFilePath(const ReplayCacheDeviceKey& device, LongType cacheKey) const;
  LongType computeCacheKey(const GraphSegment& seg, LongType shapeKey,
                           const ReplayCacheDeviceKey& device) const;

  // In-memory cache (loaded from disk)
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::vector<ReplayCacheEntry>> deviceCaches_;
  std::string cacheDir_;
  bool enabled_;

  std::atomic<int> cacheHits_{0};
  std::atomic<int> cacheMisses_{0};
};

}  // namespace graph
}  // namespace sd

#endif  // REPLAYCACHEMANAGER_H
