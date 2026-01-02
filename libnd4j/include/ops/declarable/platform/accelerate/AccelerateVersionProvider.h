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
// Apple Accelerate framework version provider
// Covers vecLib (BLAS), vDSP, BNNS, vForce, and LAPACK
//

#ifndef SD_ACCELERATE_VERSION_PROVIDER_H
#define SD_ACCELERATE_VERSION_PROVIDER_H

#include <helpers/HelperVersionRegistry.h>

#ifdef HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace sd {
namespace ops {
namespace platforms {
namespace accelerate {

// ============================================================================
// Accelerate Version Constants (based on macOS versions)
// ============================================================================

// Accelerate availability by macOS version:
// - macOS 10.3+: Basic Accelerate (vecLib, vDSP)
// - macOS 10.11+: BNNS (Basic Neural Network Subroutines)
// - macOS 11+: BNNS enhancements, Apple Silicon optimizations
// - macOS 12+: More BNNS operations, improved performance
// - macOS 13+: Extended BNNS, better Apple Silicon support
// - macOS 14+: AMX optimizations, new operations

// Minimum supported macOS version
constexpr int ACCELERATE_MIN_MACOS_MAJOR = 11;  // Big Sur
constexpr int ACCELERATE_MIN_MACOS_MINOR = 0;

// Maximum supported (tested) macOS version
constexpr int ACCELERATE_MAX_MACOS_MAJOR = 15;
constexpr int ACCELERATE_MAX_MACOS_MINOR = 99;

// Feature availability by macOS version
constexpr int ACCELERATE_BNNS_GRAPH_MACOS_MAJOR = 12;  // BNNSGraph API
constexpr int ACCELERATE_AMX_MACOS_MAJOR = 14;         // AMX optimizations

// ============================================================================
// AccelerateVersionProvider class
// ============================================================================

class SD_LIB_EXPORT AccelerateVersionProvider {
 public:
  /**
   * Get macOS version (Accelerate version is tied to OS)
   */
  static HelperVersion getMacOSVersion() {
#ifdef HAVE_ACCELERATE
#if defined(__APPLE__)
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 150000
    return HelperVersion(15, 0, 0);
#elif __MAC_OS_X_VERSION_MIN_REQUIRED >= 140000
    return HelperVersion(14, 0, 0);
#elif __MAC_OS_X_VERSION_MIN_REQUIRED >= 130000
    return HelperVersion(13, 0, 0);
#elif __MAC_OS_X_VERSION_MIN_REQUIRED >= 120000
    return HelperVersion(12, 0, 0);
#elif __MAC_OS_X_VERSION_MIN_REQUIRED >= 110000
    return HelperVersion(11, 0, 0);
#else
    return HelperVersion(10, 15, 0);
#endif
#else
    return HelperVersion(0, 0, 0);
#endif
#else
    return HelperVersion(0, 0, 0);
#endif
  }

  /**
   * Check if Accelerate is available
   */
  static bool isAvailable() {
#ifdef HAVE_ACCELERATE
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
    info.name = HelperNames::ACCELERATE;

#ifdef HAVE_ACCELERATE
    info.available = true;
    info.compileTime = getMacOSVersion();
    info.runtime = getMacOSVersion();  // Same for system framework
    info.minSupported = HelperVersion(ACCELERATE_MIN_MACOS_MAJOR, ACCELERATE_MIN_MACOS_MINOR, 0);
    info.maxSupported = HelperVersion(ACCELERATE_MAX_MACOS_MAJOR, ACCELERATE_MAX_MACOS_MINOR, 0);
    info.versionCompatible = info.runtime.inRange(info.minSupported, info.maxSupported);

    // Set capabilities
    info.capabilities = HelperCapability::NONE;

    // Basic capabilities (always available in Accelerate)
    // These are provided by vecLib/vDSP/vForce

#if defined(__arm64__)
    // Apple Silicon specific capabilities
    info.capabilities = info.capabilities | HelperCapability::NEON;

    // AMX (Apple Matrix Extension) - available on M1 and later
    if (info.runtime.major >= ACCELERATE_AMX_MACOS_MAJOR || hasAmx()) {
      info.capabilities = info.capabilities | HelperCapability::AMX;
    }
#else
    // Intel Mac - check for AVX
    if (hasAvx2()) {
      info.capabilities = info.capabilities | HelperCapability::AVX2;
    }
    if (hasAvx512()) {
      info.capabilities = info.capabilities | HelperCapability::AVX512;
    }
#endif

    // BNNS (Basic Neural Network Subroutines)
    // Available on all supported macOS versions
    info.capabilities = info.capabilities | HelperCapability::DEPTHWISE_CONV;
    info.capabilities = info.capabilities | HelperCapability::WINOGRAD;

    // BNNS Graph API (macOS 12+)
    if (info.runtime.major >= ACCELERATE_BNNS_GRAPH_MACOS_MAJOR) {
      info.capabilities = info.capabilities | HelperCapability::GRAPH_EXECUTION;
    }

    // Float16 support on Apple Silicon
#if defined(__arm64__)
    info.capabilities = info.capabilities | HelperCapability::FLOAT16_COMPUTE;
#endif

    // BFloat16 on recent Apple Silicon
#if defined(__arm64__) && defined(__ARM_FEATURE_BF16)
    info.capabilities = info.capabilities | HelperCapability::BFLOAT16_COMPUTE;
#endif

    // Activation functions via BNNS
    info.capabilities = info.capabilities | HelperCapability::GELU;
    info.capabilities = info.capabilities | HelperCapability::SWISH;
    info.capabilities = info.capabilities | HelperCapability::SOFTPLUS;

    // LAPACK operations
    // Always available through vecLib

    info.statusMessage = "Accelerate (macOS " + info.runtime.toString() + ")";
#else
    info.available = false;
    info.statusMessage = "Accelerate not available (HAVE_ACCELERATE not defined)";
#endif

    return info;
  }

  // ============================================================================
  // CPU Feature Detection
  // ============================================================================

  static bool hasAmx() {
#if defined(__arm64__) && defined(__APPLE__)
    // AMX is available on M1 and later
    // This is a compile-time check; runtime would require sysctlbyname
    return true;
#else
    return false;
#endif
  }

  static bool hasAvx2() {
#if defined(__x86_64__) && defined(__AVX2__)
    return true;
#else
    return false;
#endif
  }

  static bool hasAvx512() {
#if defined(__x86_64__) && defined(__AVX512F__)
    return true;
#else
    return false;
#endif
  }

  static bool hasBnns() {
#ifdef HAVE_ACCELERATE
    // BNNS is available in all supported Accelerate versions
    return true;
#else
    return false;
#endif
  }

  static bool hasBnnsGraph() {
#ifdef HAVE_ACCELERATE
    return getMacOSVersion().major >= ACCELERATE_BNNS_GRAPH_MACOS_MAJOR;
#else
    return false;
#endif
  }

  /**
   * Check if version is compatible
   */
  static bool isVersionCompatible() {
    if (!isAvailable()) return false;

    HelperVersion runtime = getMacOSVersion();
    HelperVersion minVer(ACCELERATE_MIN_MACOS_MAJOR, ACCELERATE_MIN_MACOS_MINOR, 0);
    HelperVersion maxVer(ACCELERATE_MAX_MACOS_MAJOR, ACCELERATE_MAX_MACOS_MINOR, 0);

    return runtime.inRange(minVer, maxVer);
  }

  /**
   * Get description of available components
   */
  static std::string getComponentsDescription() {
    std::string components = "vecLib, vDSP, vForce, LAPACK";
    if (hasBnns()) {
      components += ", BNNS";
    }
    if (hasBnnsGraph()) {
      components += ", BNNSGraph";
    }
    return components;
  }
};

}  // namespace accelerate
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_ACCELERATE_VERSION_PROVIDER_H
