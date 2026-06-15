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
// OneDNN (formerly MKL-DNN) version provider
//

#ifndef SD_ONEDNN_VERSION_PROVIDER_H
#define SD_ONEDNN_VERSION_PROVIDER_H

#include <helpers/HelperVersionRegistry.h>

#if HAVE_ONEDNN
#include <dnnl.h>
#include <dnnl_version.h>
#endif

namespace sd {
namespace ops {
namespace platforms {
namespace onednn {

// ============================================================================
// OneDNN Version Constants
// Based on cmake/Dependencies.cmake line 442: version 3.8.1
// ============================================================================

// Minimum supported version
constexpr int ONEDNN_MIN_VERSION_MAJOR = 3;
constexpr int ONEDNN_MIN_VERSION_MINOR = 0;
constexpr int ONEDNN_MIN_VERSION_PATCH = 0;

// Maximum supported version (tested up to)
constexpr int ONEDNN_MAX_VERSION_MAJOR = 3;
constexpr int ONEDNN_MAX_VERSION_MINOR = 99;
constexpr int ONEDNN_MAX_VERSION_PATCH = 99;

// Feature availability versions
constexpr int ONEDNN_BF16_VERSION_MAJOR = 1;
constexpr int ONEDNN_BF16_VERSION_MINOR = 5;

constexpr int ONEDNN_INT8_VERSION_MAJOR = 1;
constexpr int ONEDNN_INT8_VERSION_MINOR = 0;

constexpr int ONEDNN_GRAPH_API_VERSION_MAJOR = 2;
constexpr int ONEDNN_GRAPH_API_VERSION_MINOR = 0;

// ============================================================================
// OnednnVersionProvider class
// ============================================================================

class SD_LIB_EXPORT OnednnVersionProvider {
 public:
  /**
   * Get compile-time OneDNN version
   */
  static HelperVersion getCompileTimeVersion() {
#if HAVE_ONEDNN
    return HelperVersion(DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR, DNNL_VERSION_PATCH);
#else
    return HelperVersion(0, 0, 0);
#endif
  }

  /**
   * Get runtime OneDNN version
   */
  static HelperVersion getRuntimeVersion() {
#if HAVE_ONEDNN
    const dnnl_version_t* ver = dnnl_version();
    if (ver) {
      return HelperVersion(ver->major, ver->minor, ver->patch);
    }
    return HelperVersion(0, 0, 0);
#else
    return HelperVersion(0, 0, 0);
#endif
  }

  /**
   * Check if OneDNN is available
   */
  static bool isAvailable() {
#if HAVE_ONEDNN
    return true;
#else
    return false;
#endif
  }

  /**
   * Get CPU ISA (Instruction Set Architecture) name
   */
  static std::string getCpuIsa() {
#if HAVE_ONEDNN
    // OneDNN 3.x provides ISA detection
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    switch (isa) {
      case dnnl_cpu_isa_avx512_core_amx:
        return "AVX512_AMX";
      case dnnl_cpu_isa_avx512_core_amx_fp16:
        return "AVX512_AMX_FP16";
      case dnnl_cpu_isa_avx512_core_bf16:
        return "AVX512_BF16";
      case dnnl_cpu_isa_avx512_core:
        return "AVX512_CORE";
      case dnnl_cpu_isa_avx2:
        return "AVX2";
      case dnnl_cpu_isa_avx:
        return "AVX";
      case dnnl_cpu_isa_sse41:
        return "SSE4.1";
      default:
        return "UNKNOWN";
    }
#else
    return "N/A";
#endif
  }

  /**
   * Get full HelperInfo for the registry
   */
  static HelperInfo getInfo() {
    HelperInfo info;
    info.name = HelperNames::ONEDNN;

#if HAVE_ONEDNN
    info.available = true;
    info.compileTime = getCompileTimeVersion();
    info.runtime = getRuntimeVersion();
    info.minSupported =
        HelperVersion(ONEDNN_MIN_VERSION_MAJOR, ONEDNN_MIN_VERSION_MINOR, ONEDNN_MIN_VERSION_PATCH);
    info.maxSupported =
        HelperVersion(ONEDNN_MAX_VERSION_MAJOR, ONEDNN_MAX_VERSION_MINOR, ONEDNN_MAX_VERSION_PATCH);
    info.versionCompatible = info.runtime.inRange(info.minSupported, info.maxSupported);

    // Set capabilities
    info.capabilities = HelperCapability::NONE;

    // Check CPU ISA for capabilities
    std::string isa = getCpuIsa();
    if (isa.find("AVX2") != std::string::npos || isa.find("AVX512") != std::string::npos) {
      info.capabilities = info.capabilities | HelperCapability::AVX2;
    }
    if (isa.find("AVX512") != std::string::npos) {
      info.capabilities = info.capabilities | HelperCapability::AVX512;
    }
    if (isa.find("AMX") != std::string::npos) {
      info.capabilities = info.capabilities | HelperCapability::AMX;
    }

    // BFloat16 support (v1.5+)
    HelperVersion bf16Ver(ONEDNN_BF16_VERSION_MAJOR, ONEDNN_BF16_VERSION_MINOR, 0);
    if (info.runtime.meetsMinimum(bf16Ver)) {
      info.capabilities = info.capabilities | HelperCapability::BFLOAT16_COMPUTE;
    }

    // Int8 support (v1.0+)
    HelperVersion int8Ver(ONEDNN_INT8_VERSION_MAJOR, ONEDNN_INT8_VERSION_MINOR, 0);
    if (info.runtime.meetsMinimum(int8Ver)) {
      info.capabilities = info.capabilities | HelperCapability::INT8_COMPUTE;
    }

    // Graph API (v2.0+)
    HelperVersion graphVer(ONEDNN_GRAPH_API_VERSION_MAJOR, ONEDNN_GRAPH_API_VERSION_MINOR, 0);
    if (info.runtime.meetsMinimum(graphVer)) {
      info.capabilities = info.capabilities | HelperCapability::GRAPH_EXECUTION;
    }

    // Layer norm (available in all 3.x versions)
    if (info.runtime.majorVersion >= 3) {
      info.capabilities = info.capabilities | HelperCapability::LAYER_NORM;
      info.capabilities = info.capabilities | HelperCapability::GROUP_NORM;
    }

    info.statusMessage = "oneDNN " + info.runtime.toString() + " (" + isa + ")";
#else
    info.available = false;
    info.statusMessage = "oneDNN not available (HAVE_ONEDNN not defined)";
#endif

    return info;
  }

  // ============================================================================
  // Feature availability checks
  // ============================================================================

  static bool hasBfloat16Support() {
#if HAVE_ONEDNN
    HelperVersion runtime = getRuntimeVersion();
    HelperVersion required(ONEDNN_BF16_VERSION_MAJOR, ONEDNN_BF16_VERSION_MINOR, 0);
    return runtime.meetsMinimum(required);
#else
    return false;
#endif
  }

  static bool hasInt8Support() {
#if HAVE_ONEDNN
    HelperVersion runtime = getRuntimeVersion();
    HelperVersion required(ONEDNN_INT8_VERSION_MAJOR, ONEDNN_INT8_VERSION_MINOR, 0);
    return runtime.meetsMinimum(required);
#else
    return false;
#endif
  }

  static bool hasGraphApi() {
#if HAVE_ONEDNN
    HelperVersion runtime = getRuntimeVersion();
    HelperVersion required(ONEDNN_GRAPH_API_VERSION_MAJOR, ONEDNN_GRAPH_API_VERSION_MINOR, 0);
    return runtime.meetsMinimum(required);
#else
    return false;
#endif
  }

  static bool hasAvx512() {
#if HAVE_ONEDNN
    std::string isa = getCpuIsa();
    return isa.find("AVX512") != std::string::npos;
#else
    return false;
#endif
  }

  static bool hasAmx() {
#if HAVE_ONEDNN
    std::string isa = getCpuIsa();
    return isa.find("AMX") != std::string::npos;
#else
    return false;
#endif
  }

  /**
   * Check version compatibility with strict enforcement
   */
  static bool isVersionCompatible() {
    if (!isAvailable()) return false;

    HelperVersion runtime = getRuntimeVersion();
    HelperVersion minVer(ONEDNN_MIN_VERSION_MAJOR, ONEDNN_MIN_VERSION_MINOR, ONEDNN_MIN_VERSION_PATCH);
    HelperVersion maxVer(ONEDNN_MAX_VERSION_MAJOR, ONEDNN_MAX_VERSION_MINOR, ONEDNN_MAX_VERSION_PATCH);

    return runtime.inRange(minVer, maxVer);
  }
};

}  // namespace onednn
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_ONEDNN_VERSION_PROVIDER_H
