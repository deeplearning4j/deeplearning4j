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

#ifndef LIBND4J_VULKAN_SPIRV_DISK_CACHE_H
#define LIBND4J_VULKAN_SPIRV_DISK_CACHE_H

#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <cstdint>
#include <string>
#include <vector>

namespace sd {
namespace graph {

/**
 * VulkanSpirvDiskCache — Tier-1 disk cache for MLIR→SPIR-V lowering results
 * (ADR 0115), modeled directly on the Triton kernel disk cache
 * (TritonGraphBackend_cache.cpp).
 *
 * On-disk layout: spv_<16hex>.spv (raw SPIR-V words) + spv_<16hex>.meta
 * (key=value text sidecar carrying the descriptor-binding ABI that
 * VulkanPipelineCache normally extracts during mlirToSpirv()).
 *
 * Portable artifact key: FNV-1a 64 over a stable compiler/cache ABI literal,
 * the normalized SPIR-V target environment, pushConstantBytes, and the full
 * MLIR module text. Per-build timestamps and device identity are deliberately
 * excluded: a bundle produced by CI must remain usable by a separately built
 * Android runtime. Driver-specific state lives in the Tier-2 VkPipelineCache
 * blob.
 *
 * Directories (each resolved: configured value → ~/.kompile/cache/vulkan/<leaf>
 * → relative fallback when $HOME is empty):
 *   cache:    read/write,  leaf "spirv_cache"
 *   override: read-only pre-seed checked first, leaf "spirv_override"
 *
 * Writes are atomic (tmp.<pid>.<tid-hash> + rename); .spv is finalized
 * before .meta so a .meta never refers to a missing blob. Concurrent
 * writers produce identical bytes, so last-writer-wins is idempotent.
 *
 * This class has no Vulkan or MLIR API dependencies (pure hashing + file
 * I/O); it is gated on HAVE_VULKAN only so VulkanDeviceContext can reuse
 * the directory helpers for the Tier-2 blob without requiring MLIR.
 */
class SD_LIB_EXPORT VulkanSpirvDiskCache {
 public:
  /** Device capabilities that alter SPIR-V codegen — the Vulkan analogue of
   *  Triton's targetArch. Mirrors the fields VulkanPipelineCache captures. */
  struct DeviceCapsKey {
    uint32_t apiVersion = 0;
    bool fp16 = false;
    bool storage16 = false;
    bool fp64 = false;
    bool int64 = false;
    bool int8 = false;
  };

  /** Tier-1 enabled AND not bypassed by alwaysCompile. */
  static bool active();

  /**
   * Canonical Vulkan target band used by keying and compatibility checks.
   * Vulkan 1.2+ maps to the SPIR-V 1.5 target, 1.1 to SPIR-V 1.3, and older
   * or unspecified versions to Vulkan 1.0 / SPIR-V 1.0.
   */
  static uint32_t normalizeApiVersion(uint32_t apiVersion);

  /** 16-lowercase-hex stable portable artifact key (see class doc). */
  static std::string computeKey(const std::string& mlirModuleStr,
                                uint32_t pushConstantBytes,
                                const DeviceCapsKey& caps);

  /**
   * Load and validate one entry from an explicit read-only artifact directory.
   * This is the bundle-scoped path used by precompiled-only mobile runtimes.
   */
  static bool loadFromDirectory(
      const std::string& directory, const std::string& key,
      const std::string& opName, std::vector<uint32_t>& bytecode,
      std::vector<uint32_t>& descriptorBindings);

  /**
   * Load the richest bundle artifact whose target is supported by the runtime
   * device. This first tries the exact normalized target, then lower Vulkan API
   * bands and capability subsets. It lets an artifact generated for a mobile
   * baseline run on a stronger phone without allowing the inverse.
   */
  static bool loadCompatibleFromDirectory(
      const std::string& directory, const std::string& mlirModuleStr,
      uint32_t pushConstantBytes, const DeviceCapsKey& runtimeCaps,
      const std::string& opName, std::vector<uint32_t>& bytecode,
      std::vector<uint32_t>& descriptorBindings,
      std::string* matchedKey = nullptr,
      DeviceCapsKey* matchedTargetCaps = nullptr);

  /**
   * Try override dir then cache dir. On hit fills bytecode + bindings and
   * returns true. Any validation failure (bad magic, ABI mismatch, missing
   * bindings, op-name mismatch) is treated as a miss so the caller falls
   * back to the MLIR pipeline and overwrites the entry.
   */
  static bool load(const std::string& key, const std::string& opName,
                   std::vector<uint32_t>& bytecode,
                   std::vector<uint32_t>& descriptorBindings);

  /** Atomic store of .spv + .meta into the cache dir (never the override dir). */
  static void store(const std::string& key, const std::string& opName,
                    const std::vector<uint32_t>& bytecode,
                    const std::vector<uint32_t>& descriptorBindings,
                    uint32_t pushConstantBytes, const DeviceCapsKey& caps,
                    const std::string& mlirModuleStr);

  /** Resolved Tier-1 directories. */
  static std::string cacheDir();
  static std::string overrideDir();

  // ── Shared helpers (also used by the Tier-2 blob code) ──────────────────

  /** configured → ~/.kompile/cache/vulkan/<leaf> → .kompile/cache/vulkan/<leaf>. */
  static std::string configuredOrDefaultDir(const std::string& configured,
                                            const char* defaultLeaf);

  /** Recursive mkdir (Triton ensureDiskCacheDir semantics). */
  static bool ensureDir(const std::string& dir);

  /** Atomic write: tmp.<pid>.<tid-hash> + rename. Returns false on any IO failure. */
  static bool atomicWrite(const std::string& finalPath, const void* data,
                          size_t bytes);

 private:
  static bool loadFromDir(const std::string& dir, const std::string& key,
                          const std::string& opName,
                          std::vector<uint32_t>& bytecode,
                          std::vector<uint32_t>& descriptorBindings);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
#endif  // LIBND4J_VULKAN_SPIRV_DISK_CACHE_H
