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

#ifndef LIBND4J_VULKAN_CONFIG_H
#define LIBND4J_VULKAN_CONFIG_H

#include <system/common.h>

#include <atomic>
#include <string>

namespace sd {
namespace config {

/**
 * Vulkan backend disk-cache configuration (ADR 0115).
 *
 * Tier 1 — SPIR-V module disk cache: persists MLIR→SPIR-V lowering results
 * (spv_<16hex>.spv + .meta) so warm process starts skip the MLIR pass
 * pipeline entirely. Mirrors the Triton kernel disk cache.
 *
 * Tier 2 — VkPipelineCache driver-blob persistence: persists the driver's
 * pipeline cache (vkpc_<16hex>.bin, keyed by pipelineCacheUUID) so warm
 * starts also skip the driver's SPIR-V→ISA compile.
 *
 * This class stays free of Vulkan API dependencies, but its implementation and
 * singleton lifetime belong exclusively to the Vulkan native artifact.
 *
 * Directory resolution for each dir option: explicit value here (set from a
 * Java system property via the Environment setter, or from the matching
 * ND4J_VULKAN_* env var at initFromEnvironment) → ~/.kompile/cache/vulkan/<leaf>
 * → .kompile/cache/vulkan/<leaf> when $HOME is empty.
 */
class SD_LIB_EXPORT VulkanConfig {
 private:
  // Tier 1 — SPIR-V module disk cache
  std::atomic<bool> _spirvCacheEnabled{true};
  // Bypass Tier-1 reads AND skip writes (Triton ND4J_TRITON_ALWAYS_COMPILE semantics).
  std::atomic<bool> _alwaysCompile{false};
  // Dump the input MLIR module next to stored cache entries (<key>.mlir).
  std::atomic<bool> _kernelDump{false};

  // Tier 2 — VkPipelineCache driver-blob persistence
  std::atomic<bool> _pipelineCacheEnabled{true};
  // Blob growth control: VkPipelineCache has no eviction API. Blobs larger
  // than this are not loaded (fresh cache regenerates) and not saved.
  std::atomic<int64_t> _pipelineCacheMaxBytes{67108864LL};  // 64 MB

  // Test/diagnostic counters. sd_printf writes to native fd 1 which Java's
  // System.setOut cannot intercept, so tests observe cache behavior through
  // these counters instead (same pattern as TritonConfig's
  // _moduleResidencyWarnFireCount).
  std::atomic<int64_t> _spirvDiskHits{0};
  std::atomic<int64_t> _spirvDiskMisses{0};
  std::atomic<int64_t> _spirvDiskStores{0};
  std::atomic<int64_t> _pipelineBlobLoads{0};
  std::atomic<int64_t> _pipelineBlobSaves{0};

  // Directories (empty = use the ~/.kompile/cache/vulkan/<leaf> default)
  std::string _spirvCacheDir;
  std::string _spirvOverrideDir;
  std::string _pipelineCacheDir;

 public:
  static VulkanConfig& getInstance();

  VulkanConfig();

  // --- Tier 1: SPIR-V module disk cache ---
  bool spirvCacheEnabled() { return _spirvCacheEnabled.load(); }
  void setSpirvCacheEnabled(bool v) { _spirvCacheEnabled.store(v); }
  bool alwaysCompile() { return _alwaysCompile.load(); }
  void setAlwaysCompile(bool v) { _alwaysCompile.store(v); }
  bool kernelDump() { return _kernelDump.load(); }
  void setKernelDump(bool v) { _kernelDump.store(v); }

  // --- Tier 2: VkPipelineCache driver-blob persistence ---
  bool pipelineCacheEnabled() { return _pipelineCacheEnabled.load(); }
  void setPipelineCacheEnabled(bool v) { _pipelineCacheEnabled.store(v); }
  int64_t pipelineCacheMaxBytes() { return _pipelineCacheMaxBytes.load(); }
  void setPipelineCacheMaxBytes(int64_t bytes) {
    _pipelineCacheMaxBytes.store(bytes < 0 ? 0 : bytes);
  }

  // --- Directories ---
  std::string spirvCacheDir() const { return _spirvCacheDir; }
  void setSpirvCacheDir(const std::string& v) { _spirvCacheDir = v; }
  std::string spirvOverrideDir() const { return _spirvOverrideDir; }
  void setSpirvOverrideDir(const std::string& v) { _spirvOverrideDir = v; }
  std::string pipelineCacheDir() const { return _pipelineCacheDir; }
  void setPipelineCacheDir(const std::string& v) { _pipelineCacheDir = v; }

  // --- Counters (test/diagnostic observability) ---
  int64_t spirvDiskHits() { return _spirvDiskHits.load(); }
  int64_t spirvDiskMisses() { return _spirvDiskMisses.load(); }
  int64_t spirvDiskStores() { return _spirvDiskStores.load(); }
  int64_t pipelineBlobLoads() { return _pipelineBlobLoads.load(); }
  int64_t pipelineBlobSaves() { return _pipelineBlobSaves.load(); }
  void incrementSpirvDiskHits() { _spirvDiskHits.fetch_add(1, std::memory_order_relaxed); }
  void incrementSpirvDiskMisses() { _spirvDiskMisses.fetch_add(1, std::memory_order_relaxed); }
  void incrementSpirvDiskStores() { _spirvDiskStores.fetch_add(1, std::memory_order_relaxed); }
  void incrementPipelineBlobLoads() { _pipelineBlobLoads.fetch_add(1, std::memory_order_relaxed); }
  void incrementPipelineBlobSaves() { _pipelineBlobSaves.fetch_add(1, std::memory_order_relaxed); }
  void clearCacheCounters() {
    _spirvDiskHits.store(0);
    _spirvDiskMisses.store(0);
    _spirvDiskStores.store(0);
    _pipelineBlobLoads.store(0);
    _pipelineBlobSaves.store(0);
  }

  /**
   * Initialize from ND4J_VULKAN_* environment variables. Values already set
   * programmatically are only overwritten when the env var is present, so
   * Java-property application (which runs later, at backend init) wins.
   */
  void initFromEnvironment();
};

}  // namespace config
}  // namespace sd

#endif  // LIBND4J_VULKAN_CONFIG_H
