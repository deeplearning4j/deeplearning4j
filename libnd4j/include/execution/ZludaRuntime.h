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
// ZLUDA Runtime Utilities
// Provides runtime detection and initialization for ZLUDA transpiler support
//

#ifndef SD_ZLUDA_RUNTIME_H
#define SD_ZLUDA_RUNTIME_H

#include <execution/Engine.h>
#include <cstdlib>
#include <cstring>

namespace sd {
namespace zluda {

/**
 * Check if ZLUDA runtime is available.
 * This checks for the ZLUDA_PATH environment variable which indicates
 * ZLUDA libraries are installed and configured.
 *
 * @return true if ZLUDA runtime appears to be available
 */
inline bool isZludaRuntimeAvailable() {
#if defined(HAVE_ZLUDA)
    const char* zludaPath = std::getenv("ZLUDA_PATH");
    return zludaPath != nullptr && std::strlen(zludaPath) > 0;
#else
    return false;
#endif
}

/**
 * Detect the ZLUDA target backend based on environment and available GPUs.
 * This is a runtime detection that complements the compile-time configuration.
 *
 * @return The detected engine type, or ENGINE_CPU if ZLUDA is not available
 */
inline samediff::Engine detectZludaBackend() {
#if defined(HAVE_ZLUDA)
    if (!isZludaRuntimeAvailable()) {
        return samediff::ENGINE_CPU;
    }

    // Check for explicit target setting via environment variable
    const char* zludaTarget = std::getenv("ZLUDA_TARGET");
    if (zludaTarget != nullptr) {
        if (std::strcmp(zludaTarget, "AMD") == 0 || std::strcmp(zludaTarget, "amd") == 0) {
            return samediff::ENGINE_ZLUDA_AMD;
        }
        if (std::strcmp(zludaTarget, "INTEL") == 0 || std::strcmp(zludaTarget, "intel") == 0) {
            return samediff::ENGINE_ZLUDA_INTEL;
        }
    }

    // Use compile-time detected target
#if defined(ZLUDA_TARGET_AMD)
    return samediff::ENGINE_ZLUDA_AMD;
#elif defined(ZLUDA_TARGET_INTEL)
    return samediff::ENGINE_ZLUDA_INTEL;
#else
    // Default to AMD if no specific target was configured
    return samediff::ENGINE_ZLUDA_AMD;
#endif

#else
    return samediff::ENGINE_CPU;
#endif
}

/**
 * Check if a specific ZLUDA feature is supported.
 * This can be used to check for runtime feature availability.
 *
 * @param featureName The name of the feature to check
 * @return true if the feature is supported
 */
inline bool isFeatureSupported(const char* featureName) {
#if defined(HAVE_ZLUDA)
    if (featureName == nullptr) {
        return false;
    }

    // Check for MIOpen support (AMD DNN library)
    if (std::strcmp(featureName, "miopen") == 0 || std::strcmp(featureName, "MIOPEN") == 0) {
#if defined(HAVE_MIOPEN)
        return true;
#else
        return false;
#endif
    }

    // Check for oneDNN support (Intel DNN library)
    if (std::strcmp(featureName, "onednn") == 0 || std::strcmp(featureName, "ONEDNN") == 0) {
#if defined(HAVE_ONEDNN) && defined(ZLUDA_USE_ONEDNN)
        return true;
#else
        return false;
#endif
    }

    // Check for ROCm/HIP support
    if (std::strcmp(featureName, "rocm") == 0 || std::strcmp(featureName, "hip") == 0) {
#if defined(ZLUDA_TARGET_AMD)
        return true;
#else
        return false;
#endif
    }

    // Check for Level Zero support
    if (std::strcmp(featureName, "level_zero") == 0 || std::strcmp(featureName, "levelzero") == 0) {
#if defined(ZLUDA_TARGET_INTEL)
        return true;
#else
        return false;
#endif
    }

    return false;
#else
    return false;
#endif
}

/**
 * Get the ZLUDA target backend name as a string.
 *
 * @return "AMD", "INTEL", or "NONE" if ZLUDA is not configured
 */
inline const char* getZludaTargetName() {
#if defined(HAVE_ZLUDA)
#if defined(ZLUDA_TARGET_AMD)
    return "AMD";
#elif defined(ZLUDA_TARGET_INTEL)
    return "INTEL";
#else
    return "UNKNOWN";
#endif
#else
    return "NONE";
#endif
}

/**
 * Initialize ZLUDA runtime.
 * This should be called early in application startup when using ZLUDA.
 * Currently this is a no-op but may be extended for future initialization needs.
 */
inline void initializeZludaRuntime() {
#if defined(HAVE_ZLUDA)
    // Future: Add any ZLUDA-specific initialization here
    // For now, ZLUDA initialization is handled by the library loader
#endif
}

/**
 * Get ZLUDA version information if available.
 *
 * @return Version string or "unknown" if not available
 */
inline const char* getZludaVersion() {
#if defined(HAVE_ZLUDA)
    // ZLUDA doesn't currently expose a version API
    // This could be extended to query the ZLUDA library
    return "unknown";
#else
    return "not_installed";
#endif
}

}  // namespace zluda
}  // namespace sd

#endif  // SD_ZLUDA_RUNTIME_H
