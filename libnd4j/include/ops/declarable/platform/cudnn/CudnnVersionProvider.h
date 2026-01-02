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
// Centralized cuDNN version provider
// Replaces scattered #if CUDNN_VERSION checks with unified runtime validation
//

#ifndef SD_CUDNN_VERSION_PROVIDER_H
#define SD_CUDNN_VERSION_PROVIDER_H

#include <helpers/HelperVersionRegistry.h>

#ifdef HAVE_CUDNN
#include <cudnn.h>
#endif

namespace sd {
namespace ops {
namespace platforms {
namespace cudnn {

// ============================================================================
// Centralized cuDNN Version Constants
// Previously scattered in cudnnUtils.h lines 37-38, 136-159
// ============================================================================

// Minimum supported cuDNN version
constexpr int CUDNN_MIN_SUPPORTED_VERSION = 7000;  // cuDNN 7.0.0

// Maximum supported cuDNN version (update as we test new versions)
constexpr int CUDNN_MAX_SUPPORTED_VERSION = 99999;  // No upper limit currently

// API change versions
constexpr int CUDNN_NEW_RNN_API_VER = 8001;      // cuDNN 8.0.1 - New RNN descriptor API
constexpr int CUDNN_CLIPPING_API_VER = 7201;    // cuDNN 7.2.1 - RNN clipping support

// Feature availability versions
constexpr int CUDNN_SWISH_VERSION = 8000;       // cuDNN 8.0.0 - Swish activation
constexpr int CUDNN_MISH_VERSION = 8000;        // cuDNN 8.0.0 - Mish activation
constexpr int CUDNN_HARDSIGMOID_VERSION = 8000; // cuDNN 8.0.0 - Hard sigmoid
constexpr int CUDNN_HARDSWISH_VERSION = 8200;   // cuDNN 8.2.0 - Hard swish
constexpr int CUDNN_GELU_VERSION = 8300;        // cuDNN 8.3.0 - GELU activation
constexpr int CUDNN_SOFTPLUS_VERSION = 8500;    // cuDNN 8.5.0 - Softplus activation
constexpr int CUDNN_LAYER_NORM_VERSION = 8100;  // cuDNN 8.1.0 - Layer normalization
constexpr int CUDNN_INSTANCE_NORM_VERSION = 8000; // cuDNN 8.0.0 - Instance normalization
constexpr int CUDNN_FLASH_ATTENTION_VERSION = 8900; // cuDNN 8.9.0 - Flash attention

// ============================================================================
// CudnnVersionProvider class
// ============================================================================

class SD_LIB_EXPORT CudnnVersionProvider {
 public:
  /**
   * Get compile-time cuDNN version
   */
  static HelperVersion getCompileTimeVersion() {
#ifdef HAVE_CUDNN
    return HelperVersion(CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
#else
    return HelperVersion(0, 0, 0);
#endif
  }

  /**
   * Get runtime cuDNN version
   */
  static HelperVersion getRuntimeVersion() {
#ifdef HAVE_CUDNN
    size_t version = cudnnGetVersion();
    int major = static_cast<int>(version / 1000);
    int minor = static_cast<int>((version / 100) % 10);
    int patch = static_cast<int>(version % 100);
    return HelperVersion(major, minor, patch);
#else
    return HelperVersion(0, 0, 0);
#endif
  }

  /**
   * Get runtime version as integer (for direct comparison with constants)
   */
  static int getRuntimeVersionInt() {
#ifdef HAVE_CUDNN
    return static_cast<int>(cudnnGetVersion());
#else
    return 0;
#endif
  }

  /**
   * Check if cuDNN is available
   */
  static bool isAvailable() {
#ifdef HAVE_CUDNN
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
    info.name = HelperNames::CUDNN;

#ifdef HAVE_CUDNN
    info.available = true;
    info.compileTime = getCompileTimeVersion();
    info.runtime = getRuntimeVersion();
    info.minSupported = HelperVersion::fromInt(CUDNN_MIN_SUPPORTED_VERSION);
    info.maxSupported = HelperVersion::fromInt(CUDNN_MAX_SUPPORTED_VERSION);
    info.versionCompatible = info.runtime.inRange(info.minSupported, info.maxSupported);

    // Set capabilities based on runtime version
    info.capabilities = HelperCapability::NONE;
    int ver = getRuntimeVersionInt();

    // Data type support (available in all supported versions)
    info.capabilities = info.capabilities | HelperCapability::FLOAT16_COMPUTE;

    // Activation capabilities
    if (ver >= CUDNN_SWISH_VERSION) {
      info.capabilities = info.capabilities | HelperCapability::SWISH;
    }
    if (ver >= CUDNN_MISH_VERSION) {
      info.capabilities = info.capabilities | HelperCapability::MISH;
    }
    if (ver >= CUDNN_GELU_VERSION) {
      info.capabilities = info.capabilities | HelperCapability::GELU;
    }
    if (ver >= CUDNN_HARDSWISH_VERSION) {
      info.capabilities = info.capabilities | HelperCapability::HARDSWISH;
    }
    if (ver >= CUDNN_SOFTPLUS_VERSION) {
      info.capabilities = info.capabilities | HelperCapability::SOFTPLUS;
    }

    // Normalization capabilities
    if (ver >= CUDNN_LAYER_NORM_VERSION) {
      info.capabilities = info.capabilities | HelperCapability::LAYER_NORM;
    }
    if (ver >= CUDNN_INSTANCE_NORM_VERSION) {
      info.capabilities = info.capabilities | HelperCapability::INSTANCE_NORM;
    }

    // RNN capabilities
    if (ver >= CUDNN_NEW_RNN_API_VER) {
      info.capabilities = info.capabilities | HelperCapability::NEW_RNN_API;
    }

    // Advanced features
    if (ver >= CUDNN_FLASH_ATTENTION_VERSION) {
      info.capabilities = info.capabilities | HelperCapability::FLASH_ATTENTION;
    }

    // Tensor cores (available with appropriate hardware, indicated by version)
    if (ver >= 7000) {
      info.capabilities = info.capabilities | HelperCapability::TENSOR_CORES;
    }

    // Graph execution (cuDNN 8.0+)
    if (ver >= 8000) {
      info.capabilities = info.capabilities | HelperCapability::GRAPH_EXECUTION;
    }

    info.statusMessage = "cuDNN " + info.runtime.toString();
#else
    info.available = false;
    info.statusMessage = "cuDNN not available (HAVE_CUDNN not defined)";
#endif

    return info;
  }

  // ============================================================================
  // Feature availability checks (replace scattered #if CUDNN_VERSION checks)
  // ============================================================================

  static bool hasSwish() {
    return isAvailable() && getRuntimeVersionInt() >= CUDNN_SWISH_VERSION;
  }

  static bool hasMish() {
    return isAvailable() && getRuntimeVersionInt() >= CUDNN_MISH_VERSION;
  }

  static bool hasHardsigmoid() {
    return isAvailable() && getRuntimeVersionInt() >= CUDNN_HARDSIGMOID_VERSION;
  }

  static bool hasHardswish() {
    return isAvailable() && getRuntimeVersionInt() >= CUDNN_HARDSWISH_VERSION;
  }

  static bool hasGelu() {
    return isAvailable() && getRuntimeVersionInt() >= CUDNN_GELU_VERSION;
  }

  static bool hasSoftplus() {
    return isAvailable() && getRuntimeVersionInt() >= CUDNN_SOFTPLUS_VERSION;
  }

  static bool hasLayerNorm() {
    return isAvailable() && getRuntimeVersionInt() >= CUDNN_LAYER_NORM_VERSION;
  }

  static bool hasInstanceNorm() {
    return isAvailable() && getRuntimeVersionInt() >= CUDNN_INSTANCE_NORM_VERSION;
  }

  static bool hasNewRnnApi() {
    return isAvailable() && getRuntimeVersionInt() >= CUDNN_NEW_RNN_API_VER;
  }

  static bool hasClippingApi() {
    return isAvailable() && getRuntimeVersionInt() >= CUDNN_CLIPPING_API_VER;
  }

  static bool hasFlashAttention() {
    return isAvailable() && getRuntimeVersionInt() >= CUDNN_FLASH_ATTENTION_VERSION;
  }

  /**
   * Check if a specific operation is supported at runtime
   * @param opName Operation name (e.g., "swish", "gelu", "softplus")
   * @return true if operation is supported
   */
  static bool supportsOperation(const std::string& opName) {
    if (!isAvailable()) return false;

    if (opName == "swish") return hasSwish();
    if (opName == "mish") return hasMish();
    if (opName == "hardsigmoid") return hasHardsigmoid();
    if (opName == "hardswish") return hasHardswish();
    if (opName == "gelu") return hasGelu();
    if (opName == "softplus") return hasSoftplus();
    if (opName == "layer_norm") return hasLayerNorm();
    if (opName == "instance_norm") return hasInstanceNorm();
    if (opName == "flash_attention") return hasFlashAttention();

    // Default: assume supported for unknown operations
    return true;
  }

  /**
   * Get version requirement message for an operation
   */
  static std::string getVersionRequirement(const std::string& opName) {
    if (opName == "swish") return "cuDNN 8.0.0+";
    if (opName == "mish") return "cuDNN 8.0.0+";
    if (opName == "hardsigmoid") return "cuDNN 8.0.0+";
    if (opName == "hardswish") return "cuDNN 8.2.0+";
    if (opName == "gelu") return "cuDNN 8.3.0+";
    if (opName == "softplus") return "cuDNN 8.5.0+";
    if (opName == "layer_norm") return "cuDNN 8.1.0+";
    if (opName == "instance_norm") return "cuDNN 8.0.0+";
    if (opName == "flash_attention") return "cuDNN 8.9.0+";
    return "cuDNN 7.0.0+";
  }
};

}  // namespace cudnn
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_CUDNN_VERSION_PROVIDER_H
