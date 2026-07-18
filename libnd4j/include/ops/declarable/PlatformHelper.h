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
// @author raver119@gmail.com
//

#ifndef SD_PLATFORMHELPER_H
#define SD_PLATFORMHELPER_H

#include <system/BackendNamespace.h>
#include <execution/Engine.h>
#include <graph/Context.h>
#include <helpers/HelperVersionRegistry.h>
#include <helpers/ShapeUtils.h>
#include <system/RequirementsHelper.h>

#include <string>

namespace sd {
namespace ops {
namespace platforms {
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN
/**
 * This abstract class defines methods used by platform-specific helpers implementations
 */
class SD_LIB_EXPORT PlatformHelper {
 protected:
  // target engine for this impl
  samediff::Engine _engine;

  // name of the operation this helper is built for
  std::string _name;

  // hash of the operation this helper is built for
  LongType _hash;

  // ============================================================================
  // Version Metadata Fields
  // ============================================================================

  // Name of the helper library (e.g., "cuDNN", "OneDNN", "ARM_COMPUTE")
  std::string _helperLibraryName;

  // Minimum required version for this helper
  HelperVersion _minVersion{0, 0, 0};

  // Maximum supported version for this helper
  HelperVersion _maxVersion{99, 99, 99};

  // Required capabilities for this helper
  HelperCapability _requiredCapabilities = HelperCapability::NONE;

 public:
  PlatformHelper(const char *name, samediff::Engine engine);

  /**
   * Constructor with version requirements
   */
  PlatformHelper(const char *name, samediff::Engine engine, const std::string &helperLibrary,
                 const HelperVersion &minVersion = {0, 0, 0},
                 const HelperVersion &maxVersion = {99, 99, 99},
                 HelperCapability requiredCapabilities = HelperCapability::NONE);

  virtual ~PlatformHelper() = default;

  std::string name();

  samediff::Engine engine();

  LongType hash();

  /**
   * This method checks, if given helper can be used with given input/output/configuration options
   *
   * @param context
   * @return
   */
  virtual bool isUsable(graph::Context &context) = 0;

  /**
   * This method invokes helper. Typically this method replaces actual op execution
   *
   * @param context
   * @return
   */
  virtual Status invokeHelper(graph::Context &context) = 0;

  /**
   * Helper method, needed for compatibility with DeclarableOp macros
   * @param ctx
   * @param inputId
   * @return
   */
  NDArray *getZ(graph::Context &ctx, int inputId);

  /**
   * Helper method, needed for compatibility with DeclarableOp macros
   * @param ctx
   * @param inputId
   * @return
   */
  NDArray *getNullifiedZ(graph::Context &ctx, int inputId);

  // ============================================================================
  // Version Validation Methods
  // ============================================================================

  /**
   * Get the helper library name (e.g., "cuDNN", "OneDNN")
   */
  std::string helperLibraryName() const { return _helperLibraryName; }

  /**
   * Get minimum required version
   */
  HelperVersion minVersion() const { return _minVersion; }

  /**
   * Get maximum supported version
   */
  HelperVersion maxVersion() const { return _maxVersion; }

  /**
   * Get required capabilities
   */
  HelperCapability requiredCapabilities() const { return _requiredCapabilities; }

  /**
   * Set version requirements (for helpers that need to configure this dynamically)
   */
  void setVersionRequirements(const HelperVersion &minVersion, const HelperVersion &maxVersion) {
    _minVersion = minVersion;
    _maxVersion = maxVersion;
  }

  /**
   * Set required capabilities
   */
  void setRequiredCapabilities(HelperCapability capabilities) { _requiredCapabilities = capabilities; }

  /**
   * Set helper library name
   */
  void setHelperLibraryName(const std::string &name) { _helperLibraryName = name; }

  /**
   * Validate that the runtime version of the helper library meets requirements
   * @return true if version is compatible, false otherwise
   */
  virtual bool validateRuntimeVersion() const {
    if (_helperLibraryName.empty()) {
      // No helper library specified, assume compatible
      return true;
    }

    auto &registry = HelperVersionRegistry::getInstance();
    auto info = registry.getHelperInfo(_helperLibraryName);

    if (!info.available) {
      return false;
    }

    return info.runtime.inRange(_minVersion, _maxVersion);
  }

  /**
   * Check if the helper has all required capabilities
   * @return true if all required capabilities are present
   */
  virtual bool hasRequiredCapabilities() const {
    if (_helperLibraryName.empty() || _requiredCapabilities == HelperCapability::NONE) {
      return true;
    }

    auto &registry = HelperVersionRegistry::getInstance();
    auto info = registry.getHelperInfo(_helperLibraryName);

    if (!info.available) {
      return false;
    }

    return hasCapability(info.capabilities, _requiredCapabilities);
  }

  /**
   * Combined check: version compatible AND has required capabilities
   * Use this in isUsable() implementations for complete validation
   * @param context The execution context
   * @return true if the helper is usable with version checks
   */
  virtual bool isUsableWithVersionCheck(graph::Context &context) {
    // First check version and capabilities
    if (!validateRuntimeVersion()) {
      return false;
    }

    if (!hasRequiredCapabilities()) {
      return false;
    }

    // Then delegate to the actual isUsable check
    return isUsable(context);
  }

  /**
   * Get a detailed status message about version compatibility
   */
  std::string getVersionStatusMessage() const {
    if (_helperLibraryName.empty()) {
      return "No version requirements specified";
    }

    auto &registry = HelperVersionRegistry::getInstance();
    auto info = registry.getHelperInfo(_helperLibraryName);

    std::stringstream ss;
    ss << _helperLibraryName << ": ";

    if (!info.available) {
      ss << "NOT AVAILABLE";
    } else {
      ss << "v" << info.runtime.toString();
      if (validateRuntimeVersion()) {
        ss << " (compatible)";
      } else {
        ss << " (INCOMPATIBLE - requires v" << _minVersion.toString() << " - v" << _maxVersion.toString() << ")";
      }
    }

    return ss.str();
  }
};
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_PLATFORMHELPER_H
