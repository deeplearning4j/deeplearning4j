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
// ARM Compute Library version provider with version range support
//

#ifndef SD_ARM_COMPUTE_VERSION_PROVIDER_H
#define SD_ARM_COMPUTE_VERSION_PROVIDER_H

#include <helpers/HelperVersionRegistry.h>

#if HAVE_ARMCOMPUTE
#include <arm_compute/core/Version.h>
#endif

#include <system/BackendNamespace.h>

namespace sd {
namespace ops {
namespace platforms {
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN
namespace armcompute {

// ============================================================================
// ARM Compute Library Version Constants
// Based on cmake/Dependencies.cmake line 526: v25.04
// Now with version range support for v24.x - v25.x
// ============================================================================

// Minimum supported version
constexpr int ACL_MIN_VERSION_MAJOR = 24;
constexpr int ACL_MIN_VERSION_MINOR = 0;
constexpr int ACL_MIN_VERSION_PATCH = 0;

// Maximum supported version
constexpr int ACL_MAX_VERSION_MAJOR = 25;
constexpr int ACL_MAX_VERSION_MINOR = 99;
constexpr int ACL_MAX_VERSION_PATCH = 99;

// Feature availability versions
constexpr int ACL_DYNAMIC_FUSION_VERSION_MAJOR = 23;
constexpr int ACL_DYNAMIC_FUSION_VERSION_MINOR = 8;

constexpr int ACL_SVE2_VERSION_MAJOR = 21;
constexpr int ACL_SVE2_VERSION_MINOR = 8;

constexpr int ACL_I8MM_VERSION_MAJOR = 21;
constexpr int ACL_I8MM_VERSION_MINOR = 11;

constexpr int ACL_BF16_VERSION_MAJOR = 21;
constexpr int ACL_BF16_VERSION_MINOR = 5;

// ============================================================================
// ArmComputeVersionProvider class
// ============================================================================

class SD_LIB_EXPORT ArmComputeVersionProvider {
 public:
  /**
   * Parse ARM Compute version string (e.g., "v25.04" -> {25, 4, 0})
   */
  static HelperVersion parseVersionString(const std::string& versionStr) {
    // ARM Compute uses format like "v25.04" or "25.04"
    std::string ver = versionStr;

    // Remove 'v' prefix if present
    if (!ver.empty() && ver[0] == 'v') {
      ver = ver.substr(1);
    }

    int major = 0, minor = 0, patch = 0;

    size_t dotPos = ver.find('.');
    if (dotPos != std::string::npos) {
      major = std::stoi(ver.substr(0, dotPos));
      std::string rest = ver.substr(dotPos + 1);

      size_t dotPos2 = rest.find('.');
      if (dotPos2 != std::string::npos) {
        minor = std::stoi(rest.substr(0, dotPos2));
        patch = std::stoi(rest.substr(dotPos2 + 1));
      } else {
        minor = std::stoi(rest);
      }
    }

    return HelperVersion(major, minor, patch);
  }

  /**
   * Get compile-time ARM Compute version
   */
  static HelperVersion getCompileTimeVersion() {
#if HAVE_ARMCOMPUTE
    // ARM_COMPUTE_VERSION_STR is typically "vYY.MM"
    return parseVersionString(arm_compute::build_information());
#else
    return HelperVersion(0, 0, 0);
#endif
  }

  /**
   * Get runtime ARM Compute version
   * Note: ARM Compute doesn't distinguish between compile and runtime versions
   * since it's statically linked
   */
  static HelperVersion getRuntimeVersion() {
    return getCompileTimeVersion();
  }

  /**
   * Check if ARM Compute is available
   */
  static bool isAvailable() {
#if HAVE_ARMCOMPUTE
    return true;
#else
    return false;
#endif
  }

  /**
   * Get full HelperInfo for the registry
   */
  static HelperInfo getInfo() {
    HelperInfo info;
    info.name = HelperNames::ARM_COMPUTE;

#if HAVE_ARMCOMPUTE
    info.available = true;
    info.compileTime = getCompileTimeVersion();
    info.runtime = getRuntimeVersion();
    info.minSupported =
        HelperVersion(ACL_MIN_VERSION_MAJOR, ACL_MIN_VERSION_MINOR, ACL_MIN_VERSION_PATCH);
    info.maxSupported =
        HelperVersion(ACL_MAX_VERSION_MAJOR, ACL_MAX_VERSION_MINOR, ACL_MAX_VERSION_PATCH);
    info.versionCompatible = info.runtime.inRange(info.minSupported, info.maxSupported);

    // Set capabilities
    info.capabilities = HelperCapability::NONE;

    // NEON is always available on ARM64
    info.capabilities = info.capabilities | HelperCapability::NEON;

    // SVE/SVE2 detection
    if (hasSve()) {
      info.capabilities = info.capabilities | HelperCapability::SVE;
    }
    if (hasSve2()) {
      info.capabilities = info.capabilities | HelperCapability::SVE2;
    }

    // BFloat16 support
    HelperVersion bf16Ver(ACL_BF16_VERSION_MAJOR, ACL_BF16_VERSION_MINOR, 0);
    if (info.runtime.meetsMinimum(bf16Ver) && hasBf16Hardware()) {
      info.capabilities = info.capabilities | HelperCapability::BFLOAT16_COMPUTE;
    }

    // Int8 matrix multiply
    HelperVersion i8mmVer(ACL_I8MM_VERSION_MAJOR, ACL_I8MM_VERSION_MINOR, 0);
    if (info.runtime.meetsMinimum(i8mmVer) && hasI8mm()) {
      info.capabilities = info.capabilities | HelperCapability::INT8_COMPUTE;
    }

    // Dynamic fusion (graph optimization)
    HelperVersion dynFusionVer(ACL_DYNAMIC_FUSION_VERSION_MAJOR, ACL_DYNAMIC_FUSION_VERSION_MINOR, 0);
    if (info.runtime.meetsMinimum(dynFusionVer)) {
      info.capabilities = info.capabilities | HelperCapability::GRAPH_EXECUTION;
    }

    // Float16 is always available on ARM64 with NEON
    info.capabilities = info.capabilities | HelperCapability::FLOAT16_COMPUTE;

    // Depthwise convolution always available
    info.capabilities = info.capabilities | HelperCapability::DEPTHWISE_CONV;

    // Winograd convolution
    info.capabilities = info.capabilities | HelperCapability::WINOGRAD;

    info.statusMessage = "ARM Compute v" + info.runtime.toString();
#else
    info.available = false;
    info.statusMessage = "ARM Compute not available (HAVE_ARMCOMPUTE not defined)";
#endif

    return info;
  }

  // ============================================================================
  // CPU Feature Detection
  // ============================================================================

  static bool hasNeon() {
#if HAVE_ARMCOMPUTE
    // NEON is mandatory on ARM64
    return true;
#else
    return false;
#endif
  }

  static bool hasSve() {
#if HAVE_ARMCOMPUTE
    // Check for SVE support via ARM Compute's CPU info
    // This is a simplified check - actual detection uses CPUID
#if defined(__ARM_FEATURE_SVE)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
  }

  static bool hasSve2() {
#if HAVE_ARMCOMPUTE
#if defined(__ARM_FEATURE_SVE2)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
  }

  static bool hasI8mm() {
#if HAVE_ARMCOMPUTE
#if defined(__ARM_FEATURE_MATMUL_INT8)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
  }

  static bool hasBf16Hardware() {
#if HAVE_ARMCOMPUTE
#if defined(__ARM_FEATURE_BF16)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
  }

  /**
   * Check if version is in supported range
   */
  static bool isVersionCompatible() {
    if (!isAvailable()) return false;

    HelperVersion runtime = getRuntimeVersion();
    HelperVersion minVer(ACL_MIN_VERSION_MAJOR, ACL_MIN_VERSION_MINOR, ACL_MIN_VERSION_PATCH);
    HelperVersion maxVer(ACL_MAX_VERSION_MAJOR, ACL_MAX_VERSION_MINOR, ACL_MAX_VERSION_PATCH);

    return runtime.inRange(minVer, maxVer);
  }

  /**
   * Get version requirement message for version mismatch errors
   */
  static std::string getVersionRequirementMessage() {
    return "ARM Compute Library v" + std::to_string(ACL_MIN_VERSION_MAJOR) + "." +
           std::to_string(ACL_MIN_VERSION_MINOR) + " - v" + std::to_string(ACL_MAX_VERSION_MAJOR) +
           "." + std::to_string(ACL_MAX_VERSION_MINOR);
  }
};

}  // namespace armcompute
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_ARM_COMPUTE_VERSION_PROVIDER_H
