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

#ifndef SD_KERNEL_PERFORMANCE_REGISTRY_H
#define SD_KERNEL_PERFORMANCE_REGISTRY_H

#include <system/BackendNamespace.h>

#include <execution/Engine.h>
#include <system/common.h>
#include <array/DataType.h>
#include <math/templatemath.h>

#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN

/**
 * Signature for identifying a kernel execution context.
 * Combines operation, shape characteristics, and data type.
 */
struct SD_LIB_EXPORT KernelSignature {
  LongType opHash;
  std::vector<int> shapeBucket;  // Bucketed shape for caching
  DataType dataType;

  bool operator==(const KernelSignature& other) const {
    return opHash == other.opHash && shapeBucket == other.shapeBucket && dataType == other.dataType;
  }
};

/**
 * Hash function for KernelSignature
 */
struct KernelSignatureHash {
  size_t operator()(const KernelSignature& sig) const {
    size_t h = std::hash<LongType>()(sig.opHash);
    for (int dim : sig.shapeBucket) {
      h ^= std::hash<int>()(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    h ^= std::hash<int>()(static_cast<int>(sig.dataType)) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

/**
 * Performance data for a single kernel implementation
 */
struct SD_LIB_EXPORT KernelPerformanceEntry {
  double meanTimeNanos = 0.0;
  double varianceNanos = 0.0;
  int sampleCount = 0;
  int64_t lastUpdated = 0;

  void update(double newTimeNanos) {
    sampleCount++;
    double delta = newTimeNanos - meanTimeNanos;
    meanTimeNanos += delta / sampleCount;
    double delta2 = newTimeNanos - meanTimeNanos;
    varianceNanos += delta * delta2;
    lastUpdated = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
  }

  double getStdDev() const {
    if (sampleCount < 2) return 0.0;
    return sd::math::sd_sqrt<double, double>(varianceNanos / (sampleCount - 1));
  }

  bool isReliable(int minSamples = 3, double maxCoefficientOfVariation = 0.5) const {
    if (sampleCount < minSamples) return false;
    if (meanTimeNanos <= 0) return false;
    double cv = getStdDev() / meanTimeNanos;
    return cv < maxCoefficientOfVariation;
  }
};

/**
 * Configuration for kernel performance benchmarking
 */
struct SD_LIB_EXPORT KernelSelectionConfig {
  int warmupRuns = 2;
  int benchmarkRuns = 5;
  int64_t maxBenchmarkTimeMs = 1000;
  double performanceThreshold = 1.2;  // 20% faster required to switch
  bool allowFallbackToNative = true;
  bool verbose = false;

  static KernelSelectionConfig& global() {
    static KernelSelectionConfig config;
    return config;
  }
};

/**
 * Thread-safe registry for kernel performance data.
 * Caches benchmark results to avoid repeated performance measurement.
 */
class SD_LIB_EXPORT KernelPerformanceRegistry {
 private:
  // Map: signature -> (engine -> performance)
  std::unordered_map<KernelSignature,
                     std::unordered_map<samediff::Engine, KernelPerformanceEntry>,
                     KernelSignatureHash>
      _cache;

  mutable std::mutex _mutex;
  std::string _cachePath;
  bool _dirty = false;

  // Shape bucketing for cache efficiency
  static int bucketDimension(LongType dim);

 public:
  static KernelPerformanceRegistry& getInstance();

  // Prevent copying
  KernelPerformanceRegistry(const KernelPerformanceRegistry&) = delete;
  KernelPerformanceRegistry& operator=(const KernelPerformanceRegistry&) = delete;

  /**
   * Create a signature from operation and input shapes
   */
  static KernelSignature createSignature(LongType opHash, const std::vector<const LongType*>& shapes,
                                          const std::vector<int>& ranks, DataType dtype);

  /**
   * Record a performance measurement
   */
  void recordPerformance(const KernelSignature& sig, samediff::Engine engine, double timeNanos);

  /**
   * Get best engine for a signature based on cached performance
   * Returns ENGINE_CPU if no data available
   */
  samediff::Engine getBestEngine(const KernelSignature& sig) const;

  /**
   * Get performance entry for a specific signature and engine
   */
  const KernelPerformanceEntry* getPerformance(const KernelSignature& sig, samediff::Engine engine) const;

  /**
   * Check if we have reliable performance data for a signature
   */
  bool hasReliableData(const KernelSignature& sig) const;

  /**
   * Get all performance data for a signature
   */
  std::unordered_map<samediff::Engine, KernelPerformanceEntry> getAllPerformance(
      const KernelSignature& sig) const;

  /**
   * Clear all cached data
   */
  void clear();

  /**
   * Clear data for a specific operation
   */
  void clearForOp(LongType opHash);

  /**
   * Save cache to file
   */
  bool saveToFile(const std::string& path);

  /**
   * Load cache from file
   */
  bool loadFromFile(const std::string& path);

  /**
   * Set auto-save path (will save when dirty)
   */
  void setCachePath(const std::string& path) { _cachePath = path; }

  /**
   * Get cache statistics
   */
  size_t getCacheSize() const;
  size_t getOperationCount() const;

 private:
  KernelPerformanceRegistry() = default;
  ~KernelPerformanceRegistry();
};

SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd

#endif  // SD_KERNEL_PERFORMANCE_REGISTRY_H
