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

#if !defined(SD_VULKAN)
#error "system/config/vulkan/VulkanConfig.cpp is only valid for SD_VULKAN"
#endif

#include <system/config/VulkanConfig.h>
#include <system/config/EnvHelper.h>

namespace sd {
namespace config {

VulkanConfig& VulkanConfig::getInstance() {
  static VulkanConfig config;
  return config;
}

VulkanConfig::VulkanConfig() {
  initFromEnvironment();
}

void VulkanConfig::initFromEnvironment() {
  // Tier 1 — SPIR-V module disk cache
  {
    int v = readBoolEnvTriState("ND4J_VULKAN_SPIRV_CACHE_ENABLE");
    if (v >= 0) setSpirvCacheEnabled(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_VULKAN_ALWAYS_COMPILE");
    if (v >= 0) setAlwaysCompile(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_VULKAN_KERNEL_DUMP");
    if (v >= 0) setKernelDump(v == 1);
  }

  // Tier 2 — VkPipelineCache driver-blob persistence
  {
    int v = readBoolEnvTriState("ND4J_VULKAN_PIPELINE_CACHE_ENABLE");
    if (v >= 0) setPipelineCacheEnabled(v == 1);
  }
  {
    int64_t v = readInt64Env("ND4J_VULKAN_PIPELINE_CACHE_MAX_BYTES", -1);
    if (v >= 0) setPipelineCacheMaxBytes(v);
  }

  // Directories
  {
    std::string v = readStringEnv("ND4J_VULKAN_SPIRV_CACHE_DIR");
    if (!v.empty()) setSpirvCacheDir(v);
  }
  {
    std::string v = readStringEnv("ND4J_VULKAN_SPIRV_OVERRIDE_DIR");
    if (!v.empty()) setSpirvOverrideDir(v);
  }
  {
    std::string v = readStringEnv("ND4J_VULKAN_PIPELINE_CACHE_DIR");
    if (!v.empty()) setPipelineCacheDir(v);
  }
}

}  // namespace config
}  // namespace sd
