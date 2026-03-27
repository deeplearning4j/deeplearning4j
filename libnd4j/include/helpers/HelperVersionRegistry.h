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
// Unified helper version registry for tracking library versions and capabilities
// across all platform helpers (cuDNN, OneDNN, ARM Compute, MPS, Accelerate, llama.cpp)
//

#ifndef SD_HELPER_VERSION_REGISTRY_H
#define SD_HELPER_VERSION_REGISTRY_H

#include <system/common.h>

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace sd {
namespace ops {
namespace platforms {

/**
 * Version structure representing major.minor.patch versioning
 */
struct SD_LIB_EXPORT HelperVersion {
  // Field names avoid 'major'/'minor' which collide with glibc macros
  // from <sys/sysmacros.h> on older Linux toolchains (devtoolset-7).
  int majorVersion = 0;
  int minorVersion = 0;
  int patchVersion = 0;
  std::string buildInfo;  // Additional build info (e.g., "CUDA 12.0", "AVX512")

  HelperVersion() = default;
  HelperVersion(int maj, int min, int pat, const std::string& info = "")
      : majorVersion(maj), minorVersion(min), patchVersion(pat), buildInfo(info) {}

  /**
   * Check if this version meets the minimum required version
   * @param min Minimum required version
   * @return true if this version >= min
   */
  bool meetsMinimum(const HelperVersion& min) const {
    if (majorVersion > min.majorVersion) return true;
    if (majorVersion < min.majorVersion) return false;
    if (minorVersion > min.minorVersion) return true;
    if (minorVersion < min.minorVersion) return false;
    return patchVersion >= min.patchVersion;
  }

  /**
   * Check if this version is within a range [min, max]
   * @param min Minimum version (inclusive)
   * @param max Maximum version (inclusive)
   * @return true if min <= this <= max
   */
  bool inRange(const HelperVersion& min, const HelperVersion& max) const {
    return meetsMinimum(min) && max.meetsMinimum(*this);
  }

  /**
   * Convert version to string representation
   */
  std::string toString() const {
    std::string result =
        std::to_string(majorVersion) + "." + std::to_string(minorVersion) + "." + std::to_string(patchVersion);
    if (!buildInfo.empty()) {
      result += " (" + buildInfo + ")";
    }
    return result;
  }

  /**
   * Convert version to integer for comparison (major*10000 + minor*100 + patch)
   */
  int toInt() const { return majorVersion * 10000 + minorVersion * 100 + patchVersion; }

  /**
   * Create version from integer representation
   */
  static HelperVersion fromInt(int version) {
    return HelperVersion(version / 10000, (version / 100) % 100, version % 100);
  }

  bool operator==(const HelperVersion& other) const {
    return majorVersion == other.majorVersion && minorVersion == other.minorVersion && patchVersion == other.patchVersion;
  }

  bool operator!=(const HelperVersion& other) const { return !(*this == other); }

  bool operator<(const HelperVersion& other) const { return toInt() < other.toInt(); }

  bool operator<=(const HelperVersion& other) const { return toInt() <= other.toInt(); }

  bool operator>(const HelperVersion& other) const { return toInt() > other.toInt(); }

  bool operator>=(const HelperVersion& other) const { return toInt() >= other.toInt(); }
};

/**
 * Capability flags for helper libraries
 * Used for feature detection independent of version numbers
 */
enum class HelperCapability : uint64_t {
  NONE = 0,

  // Data type support
  FLOAT16_COMPUTE = 1ULL << 0,
  BFLOAT16_COMPUTE = 1ULL << 1,
  INT8_COMPUTE = 1ULL << 2,
  INT4_COMPUTE = 1ULL << 3,

  // Attention/Transformer features
  FLASH_ATTENTION = 1ULL << 4,
  MULTI_HEAD_ATTENTION = 1ULL << 5,
  GROUPED_QUERY_ATTENTION = 1ULL << 6,
  KV_CACHE = 1ULL << 7,

  // Hardware acceleration
  TENSOR_CORES = 1ULL << 8,
  MATRIX_CORES = 1ULL << 9,
  AMX = 1ULL << 10,  // Apple Matrix Extension / Intel AMX
  SVE = 1ULL << 11,  // ARM Scalable Vector Extension
  SVE2 = 1ULL << 12,
  NEON = 1ULL << 13,
  AVX2 = 1ULL << 14,
  AVX512 = 1ULL << 15,

  // Memory features
  UNIFIED_MEMORY = 1ULL << 16,
  P2P_TRANSFER = 1ULL << 17,
  ASYNC_COPY = 1ULL << 18,

  // Execution features
  GRAPH_EXECUTION = 1ULL << 19,
  DYNAMIC_SHAPES = 1ULL << 20,
  MODEL_PARALLEL = 1ULL << 21,

  // Quantization
  QUANTIZATION_Q4 = 1ULL << 22,
  QUANTIZATION_Q8 = 1ULL << 23,
  QUANTIZATION_DYNAMIC = 1ULL << 24,

  // Normalization variants
  LAYER_NORM = 1ULL << 25,
  RMS_NORM = 1ULL << 26,
  GROUP_NORM = 1ULL << 27,
  INSTANCE_NORM = 1ULL << 28,

  // Activation variants
  GELU = 1ULL << 29,
  SWISH = 1ULL << 30,
  MISH = 1ULL << 31,
  HARDSWISH = 1ULL << 32,
  SOFTPLUS = 1ULL << 33,

  // RNN features
  NEW_RNN_API = 1ULL << 34,
  LSTM_PEEPHOLE = 1ULL << 35,
  GRU = 1ULL << 36,

  // Convolution features
  WINOGRAD = 1ULL << 37,
  FFT_CONV = 1ULL << 38,
  DEPTHWISE_CONV = 1ULL << 39,

  // Backend-specific
  CUDA_GRAPHS = 1ULL << 40,
  METAL_SHADERS = 1ULL << 41,
  VULKAN_COMPUTE = 1ULL << 42,
  SYCL_BACKEND = 1ULL << 43,
};

/**
 * Bitwise OR operator for combining capabilities
 */
inline HelperCapability operator|(HelperCapability a, HelperCapability b) {
  return static_cast<HelperCapability>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}

/**
 * Bitwise AND operator for checking capabilities
 */
inline HelperCapability operator&(HelperCapability a, HelperCapability b) {
  return static_cast<HelperCapability>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}

/**
 * Check if a capability set contains a specific capability
 */
inline bool hasCapability(HelperCapability capabilities, HelperCapability check) {
  return (static_cast<uint64_t>(capabilities) & static_cast<uint64_t>(check)) != 0;
}

/**
 * Known helper library names
 */
struct HelperNames {
  static constexpr const char* CUDNN = "cuDNN";
  static constexpr const char* ONEDNN = "oneDNN";
  static constexpr const char* ARM_COMPUTE = "ARM Compute";
  static constexpr const char* MPS = "MPS";
  static constexpr const char* ACCELERATE = "Accelerate";
  static constexpr const char* GGML = "GGML";
  static constexpr const char* LLAMA_CPP = "llama.cpp";
  static constexpr const char* ZLUDA = "ZLUDA";
  static constexpr const char* MIOPEN = "MIOpen";
};

/**
 * Information about a helper library
 */
struct SD_LIB_EXPORT HelperInfo {
  std::string name;                    // Library name (e.g., "cuDNN", "oneDNN")
  HelperVersion compileTime;           // Version at compile time
  HelperVersion runtime;               // Version detected at runtime
  HelperVersion minSupported;          // Minimum supported version
  HelperVersion maxSupported;          // Maximum supported version
  HelperCapability capabilities;       // Available capabilities
  bool available = false;              // Whether the helper is available
  bool versionCompatible = false;      // Whether runtime version is compatible
  std::string statusMessage;           // Human-readable status
  std::string libraryPath;             // Path to loaded library (if applicable)

  HelperInfo() : capabilities(HelperCapability::NONE) {}

  /**
   * Check if the runtime version is compatible with supported range
   */
  bool isCompatible() const {
    return available && runtime.inRange(minSupported, maxSupported);
  }

  /**
   * Get a detailed status string
   */
  std::string getDetailedStatus() const {
    std::string status = name + ": ";
    if (!available) {
      status += "NOT AVAILABLE";
      if (!statusMessage.empty()) {
        status += " - " + statusMessage;
      }
    } else if (!versionCompatible) {
      status += "VERSION MISMATCH - runtime " + runtime.toString() + " not in supported range [" +
                minSupported.toString() + ", " + maxSupported.toString() + "]";
    } else {
      status += "OK (runtime " + runtime.toString() + ")";
    }
    return status;
  }
};

/**
 * Callback type for version providers
 * Each library registers a function that returns its HelperInfo
 */
using VersionProviderCallback = std::function<HelperInfo()>;

/**
 * Central registry for helper library versions and capabilities
 * Thread-safe singleton that manages all helper version information
 */
class SD_LIB_EXPORT HelperVersionRegistry {
 public:
  /**
   * Get the singleton instance
   */
  static HelperVersionRegistry& getInstance();

  /**
   * Register a version provider for a helper library
   * @param name Library name
   * @param provider Callback that returns HelperInfo
   */
  void registerProvider(const std::string& name, VersionProviderCallback provider);

  /**
   * Get information about a helper library
   * @param name Library name
   * @return HelperInfo for the library, or empty info if not found
   */
  HelperInfo getHelperInfo(const std::string& name);

  /**
   * Get information about all registered helpers
   * @return Map of library name to HelperInfo
   */
  std::map<std::string, HelperInfo> getAllHelperInfo();

  /**
   * Check if a helper meets minimum version requirements
   * @param name Library name
   * @param minVersion Minimum required version
   * @return true if available and version >= minVersion
   */
  bool checkVersion(const std::string& name, const HelperVersion& minVersion);

  /**
   * Check if a helper has a specific capability
   * @param name Library name
   * @param capability Capability to check
   * @return true if helper has the capability
   */
  bool hasCapability(const std::string& name, HelperCapability capability);

  /**
   * Check if a helper is available and version-compatible
   * @param name Library name
   * @return true if helper is usable
   */
  bool isHelperUsable(const std::string& name);

  /**
   * Get a list of all available helper names
   */
  std::vector<std::string> getAvailableHelpers();

  /**
   * Get a summary of all helper statuses
   */
  std::string getStatusSummary();

  /**
   * Refresh cached helper information (re-queries all providers)
   */
  void refresh();

  /**
   * Enable or disable strict version enforcement
   * When enabled, helpers with version mismatches will not be used
   */
  void setStrictVersionEnforcement(bool strict) { _strictEnforcement = strict; }
  bool isStrictVersionEnforcement() const { return _strictEnforcement; }

  /**
   * Enable or disable verbose logging of version checks
   */
  void setVerboseLogging(bool verbose) { _verboseLogging = verbose; }
  bool isVerboseLogging() const { return _verboseLogging; }

 private:
  HelperVersionRegistry();
  ~HelperVersionRegistry() = default;

  // Disable copy and move
  HelperVersionRegistry(const HelperVersionRegistry&) = delete;
  HelperVersionRegistry& operator=(const HelperVersionRegistry&) = delete;

  void initializeFromEnvironment();
  void logVersionCheck(const std::string& name, const HelperInfo& info, bool success);

  std::mutex _mutex;
  std::map<std::string, VersionProviderCallback> _providers;
  std::map<std::string, HelperInfo> _cache;
  bool _cacheValid = false;
  bool _strictEnforcement = true;  // Default to strict per user preference
  bool _verboseLogging = false;
};

/**
 * RAII helper for registering version providers at static initialization time
 */
class SD_LIB_EXPORT VersionProviderRegistrar {
 public:
  VersionProviderRegistrar(const std::string& name, VersionProviderCallback provider) {
    HelperVersionRegistry::getInstance().registerProvider(name, provider);
  }
};

/**
 * Macro for registering a version provider
 * Usage: REGISTER_VERSION_PROVIDER("cuDNN", getCudnnVersionInfo)
 */
#define REGISTER_VERSION_PROVIDER(name, provider) \
  static sd::ops::platforms::VersionProviderRegistrar _versionProvider_##__LINE__(name, provider)

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_HELPER_VERSION_REGISTRY_H
