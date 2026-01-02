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
// Metal Performance Shaders (MPS) version provider for Apple GPU acceleration
//

#ifndef SD_MPS_VERSION_PROVIDER_H
#define SD_MPS_VERSION_PROVIDER_H

#include <helpers/HelperVersionRegistry.h>

// MPS version is tied to macOS/iOS version
// We detect features based on the deployment target

namespace sd {
namespace ops {
namespace platforms {
namespace mps {

// ============================================================================
// MPS Version Constants (based on macOS versions)
// ============================================================================

// MPS availability by macOS version:
// - macOS 10.13 (High Sierra): Basic MPS
// - macOS 10.14 (Mojave): MPSNNGraph
// - macOS 10.15 (Catalina): Improved convolutions
// - macOS 11 (Big Sur): Apple Silicon support, MLCompute
// - macOS 12 (Monterey): MPSGraph enhancements
// - macOS 13 (Ventura): Enhanced MPS for M1/M2
// - macOS 14 (Sonoma): Further M-series optimizations

// Minimum supported macOS version
constexpr int MPS_MIN_MACOS_MAJOR = 11;  // Big Sur (for Apple Silicon)
constexpr int MPS_MIN_MACOS_MINOR = 0;

// Maximum supported (tested) macOS version
constexpr int MPS_MAX_MACOS_MAJOR = 15;
constexpr int MPS_MAX_MACOS_MINOR = 99;

// Feature availability by macOS version
constexpr int MPS_GRAPH_MACOS_MAJOR = 12;      // MPSGraph API
constexpr int MPS_TRANSFORMER_MACOS_MAJOR = 14; // Transformer optimizations
constexpr int MPS_FLASH_ATTN_MACOS_MAJOR = 14;  // Flash attention-like ops

// ============================================================================
// MpsVersionProvider class
// ============================================================================

class SD_LIB_EXPORT MpsVersionProvider {
 public:
  /**
   * Get macOS version (MPS version is tied to OS)
   */
  static HelperVersion getMacOSVersion() {
#ifdef HAVE_MPS
#if defined(__APPLE__)
    // Get macOS version from deployment target
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
   * Check if MPS is available
   */
  static bool isAvailable() {
#ifdef HAVE_MPS
    return true;
#else
    return false;
#endif
  }

  /**
   * Get GPU family (Apple7, Apple8, etc.)
   */
  static std::string getGpuFamily() {
#ifdef HAVE_MPS
    // This would require runtime detection via Metal API
    // For now, return based on compile target
#if defined(__arm64__)
    return "Apple Silicon";
#else
    return "Intel/AMD";
#endif
#else
    return "N/A";
#endif
  }

  /**
   * Get full HelperInfo for the registry
   */
  static HelperInfo getInfo() {
    HelperInfo info;
    info.name = HelperNames::MPS;

#ifdef HAVE_MPS
    info.available = true;
    info.compileTime = getMacOSVersion();
    info.runtime = getMacOSVersion();  // Same for statically linked
    info.minSupported = HelperVersion(MPS_MIN_MACOS_MAJOR, MPS_MIN_MACOS_MINOR, 0);
    info.maxSupported = HelperVersion(MPS_MAX_MACOS_MAJOR, MPS_MAX_MACOS_MINOR, 0);
    info.versionCompatible = info.runtime.inRange(info.minSupported, info.maxSupported);

    // Set capabilities
    info.capabilities = HelperCapability::NONE;

    // Basic capabilities (always available)
    info.capabilities = info.capabilities | HelperCapability::FLOAT16_COMPUTE;
    info.capabilities = info.capabilities | HelperCapability::METAL_SHADERS;

#if defined(__arm64__)
    // Apple Silicon specific capabilities
    info.capabilities = info.capabilities | HelperCapability::MATRIX_CORES;  // Neural Engine
    info.capabilities = info.capabilities | HelperCapability::UNIFIED_MEMORY;
#endif

    // MPSGraph (macOS 12+)
    if (info.runtime.major >= MPS_GRAPH_MACOS_MAJOR) {
      info.capabilities = info.capabilities | HelperCapability::GRAPH_EXECUTION;
    }

    // Transformer optimizations (macOS 14+)
    if (info.runtime.major >= MPS_TRANSFORMER_MACOS_MAJOR) {
      info.capabilities = info.capabilities | HelperCapability::MULTI_HEAD_ATTENTION;
      info.capabilities = info.capabilities | HelperCapability::FLASH_ATTENTION;
    }

    // Standard neural network operations
    info.capabilities = info.capabilities | HelperCapability::DEPTHWISE_CONV;
    info.capabilities = info.capabilities | HelperCapability::LAYER_NORM;
    info.capabilities = info.capabilities | HelperCapability::INSTANCE_NORM;
    info.capabilities = info.capabilities | HelperCapability::GROUP_NORM;

    // Activation functions
    info.capabilities = info.capabilities | HelperCapability::GELU;
    info.capabilities = info.capabilities | HelperCapability::SWISH;
    info.capabilities = info.capabilities | HelperCapability::MISH;
    info.capabilities = info.capabilities | HelperCapability::SOFTPLUS;

    info.statusMessage = "MPS (macOS " + info.runtime.toString() + ", " + getGpuFamily() + ")";
#else
    info.available = false;
    info.statusMessage = "MPS not available (HAVE_MPS not defined or not on Apple platform)";
#endif

    return info;
  }

  // ============================================================================
  // Feature Detection
  // ============================================================================

  static bool hasMPSGraph() {
#ifdef HAVE_MPS
    return getMacOSVersion().major >= MPS_GRAPH_MACOS_MAJOR;
#else
    return false;
#endif
  }

  static bool hasTransformerOptimizations() {
#ifdef HAVE_MPS
    return getMacOSVersion().major >= MPS_TRANSFORMER_MACOS_MAJOR;
#else
    return false;
#endif
  }

  static bool hasAppleSilicon() {
#if defined(__arm64__) && defined(HAVE_MPS)
    return true;
#else
    return false;
#endif
  }

  static bool hasNeuralEngine() {
    // Neural Engine is available on Apple Silicon
    return hasAppleSilicon();
  }

  /**
   * Check if version is compatible
   */
  static bool isVersionCompatible() {
    if (!isAvailable()) return false;

    HelperVersion runtime = getMacOSVersion();
    HelperVersion minVer(MPS_MIN_MACOS_MAJOR, MPS_MIN_MACOS_MINOR, 0);
    HelperVersion maxVer(MPS_MAX_MACOS_MAJOR, MPS_MAX_MACOS_MINOR, 0);

    return runtime.inRange(minVer, maxVer);
  }
};

}  // namespace mps
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_MPS_VERSION_PROVIDER_H
