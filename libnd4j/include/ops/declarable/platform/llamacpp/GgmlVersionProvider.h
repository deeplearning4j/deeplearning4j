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
// GGML/llama.cpp version provider with backend capability detection
//

#ifndef SD_GGML_VERSION_PROVIDER_H
#define SD_GGML_VERSION_PROVIDER_H

#include <helpers/HelperVersionRegistry.h>

#if HAVE_LLAMACPP
#include <ggml.h>
#endif

namespace sd {
namespace ops {
namespace platforms {
namespace llamacpp {

// ============================================================================
// GGML Version Constants
// ============================================================================

// Minimum supported GGML version (build number based)
constexpr int GGML_MIN_BUILD = 1000;

// Feature availability (approximate build numbers)
constexpr int GGML_FLASH_ATTENTION_BUILD = 1500;
constexpr int GGML_QUANTIZATION_Q4_K_BUILD = 1200;
constexpr int GGML_QUANTIZATION_Q6_K_BUILD = 1300;
constexpr int GGML_METAL_BACKEND_BUILD = 1100;
constexpr int GGML_VULKAN_BACKEND_BUILD = 1400;
constexpr int GGML_SYCL_BACKEND_BUILD = 1600;

// ============================================================================
// GgmlVersionProvider class
// ============================================================================

class SD_LIB_EXPORT GgmlVersionProvider {
 public:
  /**
   * Get GGML version
   * GGML uses a build number system rather than semantic versioning
   */
  static HelperVersion getGgmlVersion() {
#if HAVE_LLAMACPP
    // GGML_FILE_VERSION or similar could be used
    // For now, use compile-time detection
#ifdef GGML_FILE_VERSION
    int build = GGML_FILE_VERSION;
    return HelperVersion(build / 1000, (build / 100) % 10, build % 100);
#else
    // Fallback to a reasonable default
    return HelperVersion(1, 0, 0);
#endif
#else
    return HelperVersion(0, 0, 0);
#endif
  }

  /**
   * Get llama.cpp version if available
   */
  static HelperVersion getLlamaVersion() {
#if HAVE_LLAMACPP
    // llama.cpp build number
    return getGgmlVersion();  // Usually same as GGML
#else
    return HelperVersion(0, 0, 0);
#endif
  }

  /**
   * Check if llama.cpp/GGML is available
   */
  static bool isAvailable() {
#if HAVE_LLAMACPP
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
    info.name = HelperNames::GGML;

#if HAVE_LLAMACPP
    info.available = true;
    info.compileTime = getGgmlVersion();
    info.runtime = getGgmlVersion();  // No runtime distinction for statically linked
    info.minSupported = HelperVersion(1, 0, 0);
    info.maxSupported = HelperVersion(99, 99, 99);
    // Actually validate version against supported range
    info.versionCompatible = info.runtime.inRange(info.minSupported, info.maxSupported);

    // Set capabilities
    info.capabilities = HelperCapability::NONE;

    // Quantization support
    info.capabilities = info.capabilities | HelperCapability::QUANTIZATION_Q4;
    info.capabilities = info.capabilities | HelperCapability::QUANTIZATION_Q8;

    // Backend detection
    if (hasCudaBackend()) {
      info.capabilities = info.capabilities | HelperCapability::CUDA_GRAPHS;
    }
    if (hasMetalBackend()) {
      info.capabilities = info.capabilities | HelperCapability::METAL_SHADERS;
    }
    if (hasVulkanBackend()) {
      info.capabilities = info.capabilities | HelperCapability::VULKAN_COMPUTE;
    }
    if (hasSyclBackend()) {
      info.capabilities = info.capabilities | HelperCapability::SYCL_BACKEND;
    }

    // Feature detection
    if (supportsFlashAttention()) {
      info.capabilities = info.capabilities | HelperCapability::FLASH_ATTENTION;
    }

    // Model parallelism
    if (supportsModelParallel()) {
      info.capabilities = info.capabilities | HelperCapability::MODEL_PARALLEL;
    }

    // KV Cache
    info.capabilities = info.capabilities | HelperCapability::KV_CACHE;

    // RMS Norm (used in LLaMA models)
    info.capabilities = info.capabilities | HelperCapability::RMS_NORM;

    // GQA (Grouped Query Attention)
    info.capabilities = info.capabilities | HelperCapability::GROUPED_QUERY_ATTENTION;

    info.statusMessage = "GGML/llama.cpp " + info.runtime.toString() + " (" + getActiveBackends() + ")";
#else
    info.available = false;
    info.statusMessage = "GGML/llama.cpp not available (HAVE_LLAMACPP not defined)";
#endif

    return info;
  }

  // ============================================================================
  // Backend Detection
  // ============================================================================

  static bool hasCudaBackend() {
#if HAVE_LLAMACPP
#if defined(GGML_USE_CUDA) || defined(GGML_USE_CUBLAS)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
  }

  static bool hasMetalBackend() {
#if HAVE_LLAMACPP
#if defined(GGML_USE_METAL)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
  }

  static bool hasVulkanBackend() {
#if HAVE_LLAMACPP
#if defined(GGML_USE_VULKAN)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
  }

  static bool hasSyclBackend() {
#if HAVE_LLAMACPP
#if defined(GGML_USE_SYCL)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
  }

  static bool hasOpenClBackend() {
#if HAVE_LLAMACPP
#if defined(GGML_USE_OPENCL)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
  }

  static std::string getActiveBackends() {
    std::string backends;
    if (hasCudaBackend()) backends += "CUDA,";
    if (hasMetalBackend()) backends += "Metal,";
    if (hasVulkanBackend()) backends += "Vulkan,";
    if (hasSyclBackend()) backends += "SYCL,";
    if (hasOpenClBackend()) backends += "OpenCL,";

    if (backends.empty()) {
      return "CPU";
    }

    // Remove trailing comma
    backends.pop_back();
    return backends;
  }

  // ============================================================================
  // Feature Detection
  // ============================================================================

  static bool supportsFlashAttention() {
#if HAVE_LLAMACPP
    // Flash attention is typically available in recent GGML builds
#if defined(GGML_USE_FLASH_ATTN) || defined(GGML_FLASH_ATTN)
    return true;
#else
    // May still be available through CUDA backend
    return hasCudaBackend();
#endif
#else
    return false;
#endif
  }

  static bool supportsModelParallel() {
#if HAVE_LLAMACPP
    // Model parallelism requires CUDA
    return hasCudaBackend();
#else
    return false;
#endif
  }

  /**
   * Check if a specific quantization type is supported
   */
  static bool supportsQuantType(int ggmlType) {
#if HAVE_LLAMACPP
    switch (ggmlType) {
      case 2:   // GGML_TYPE_Q4_0
      case 3:   // GGML_TYPE_Q4_1
      case 6:   // GGML_TYPE_Q5_0
      case 7:   // GGML_TYPE_Q5_1
      case 8:   // GGML_TYPE_Q8_0
      case 9:   // GGML_TYPE_Q8_1
        return true;

      case 10:  // GGML_TYPE_Q2_K
      case 11:  // GGML_TYPE_Q3_K
      case 12:  // GGML_TYPE_Q4_K
      case 13:  // GGML_TYPE_Q5_K
      case 14:  // GGML_TYPE_Q6_K
      case 15:  // GGML_TYPE_Q8_K
        // K-quants require newer GGML
        return true;

      default:
        return false;
    }
#else
    return false;
#endif
  }

  /**
   * Check if version is compatible
   */
  static bool isVersionCompatible() {
    if (!isAvailable()) return false;

    HelperVersion runtime = getGgmlVersion();
    HelperVersion minVer(1, 0, 0);
    HelperVersion maxVer(99, 99, 99);

    return runtime.inRange(minVer, maxVer);
  }
};

}  // namespace llamacpp
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_GGML_VERSION_PROVIDER_H
