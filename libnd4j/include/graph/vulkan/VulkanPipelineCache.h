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

#ifndef LIBND4J_VULKAN_PIPELINE_CACHE_H
#define LIBND4J_VULKAN_PIPELINE_CACHE_H

#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <vulkan/vulkan.h>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

struct VulkanDeviceCaps;

/**
 * VulkanPipelineCache — compiles MLIR modules to SPIR-V and caches the
 * resulting VkPipeline objects for reuse.
 *
 * Lifecycle:
 *   1. Construct with a VkDevice and VkPhysicalDevice (typically from
 *      VulkanReplayHandle after initVulkan()).
 *   2. Call getOrCompile(opName, mlirModuleStr, device) to obtain a
 *      VkPipeline.  On the first call for a given (opName, mlirModuleStr)
 *      pair the MLIR module is lowered to SPIR-V, a VkShaderModule is
 *      created, and a compute VkPipeline is built and cached.  Subsequent
 *      calls for the same key return the cached pipeline instantly.
 *   3. Call clear() to invalidate cache lookup entries. Vulkan objects are
 *      retired until cache destruction so recorded or in-flight command buffers
 *      can continue to reference them safely.
 *
 * Cache key:
 *   std::string key = opName + "|push=" + pushConstantBytes + "|" +
 *                     mlirModuleStr
 *   Shapes, types, and immutable op metadata are part of the emitted MLIR and
 *   therefore part of the cache identity.
 *
 * Error handling:
 *   All compile / Vulkan-API failures are logged via sd_printf and the
 *   method returns VK_NULL_HANDLE.  The caller must check for VK_NULL_HANDLE
 *   before recording a dispatch.
 *
 * Thread safety:
 *   Cache lookup, compilation, invalidation, and introspection are serialized.
 *   Distinct device-bound cache instances may compile concurrently.
 */
class SD_LIB_EXPORT VulkanPipelineCache {
 public:
  /**
   * Construct a pipeline cache bound to the given Vulkan device.
   *
   * @param device      Logical device used for VkPipeline / VkShaderModule creation.
   * @param physDevice  Physical device that owns the logical device.
   * @param caps        Features actually enabled on that logical device; these
   *                    define the MLIR SPIR-V target contract.
   * @param driverPipelineCache The device-lifetime VkPipelineCache from
   *                    VulkanDeviceContext (ADR 0115 Tier 2). Passed to
   *                    vkCreateComputePipelines so the driver reuses and
   *                    accumulates compiled shader state across all pipeline
   *                    creations on this device. VK_NULL_HANDLE disables it.
   */
  VulkanPipelineCache(VkDevice device, VkPhysicalDevice physDevice,
                      const VulkanDeviceCaps& caps,
                      VkPipelineCache driverPipelineCache = VK_NULL_HANDLE);

  /**
   * Destroy all cached VkPipeline, VkPipelineLayout, and VkDescriptorSetLayout
   * objects.  Must be called before the owning VkDevice is destroyed.
   */
  ~VulkanPipelineCache();

  // Non-copyable, non-movable (owns Vulkan handles)
  VulkanPipelineCache(const VulkanPipelineCache&) = delete;
  VulkanPipelineCache& operator=(const VulkanPipelineCache&) = delete;

  /**
   * Thread-scoped policy used while one DSP segment executes. It applies to
   * recorder and eager Vulkan paths, so warmup cannot compile behind the
   * portable runtime's back.
   */
  class SD_LIB_EXPORT ScopedCompilationPolicy {
   public:
    ScopedCompilationPolicy(bool allowRuntimeCompilation,
                            const std::string& artifactDirectory);
    ~ScopedCompilationPolicy();

    ScopedCompilationPolicy(const ScopedCompilationPolicy&) = delete;
    ScopedCompilationPolicy& operator=(const ScopedCompilationPolicy&) = delete;

   private:
    bool previousAllowRuntimeCompilation_;
    std::string previousArtifactDirectory_;
  };

  /**
   * Return a compute VkPipeline for the given MLIR module, compiling on the
   * first call and returning the cached pipeline on subsequent calls.
   *
   * @param opName        Human-readable op name used as part of the cache key
   *                      and in diagnostic messages.
   * @param mlirModuleStr Textual MLIR module containing exactly one GPU compute
   *                      kernel to lower to SPIR-V.
   * @param device        Logical Vulkan device (must match the one passed at
   *                      construction).
   * @param pushConstantBytes Compute-stage push-constant interface size. Zero
   *                      means the shader has no push-constant interface.
   * @return              A valid VkPipeline on success, VK_NULL_HANDLE on any
   *                      compile or Vulkan-API failure.
   */
  VkPipeline getOrCompile(const std::string& opName,
                          const std::string& mlirModuleStr,
                          VkDevice device,
                          uint32_t pushConstantBytes = 0);

  /**
   * Invalidate all lookup entries. Their Vulkan objects remain retired until
   * this cache is destroyed, because already-recorded command buffers may still
   * reference them. Safe to call with an empty cache.
   */
  void clear();

  /** Return the number of pipelines currently held in the cache. */
  size_t size() const;

  /** Process-wide counters for the Vulkan MLIR/SPIR-V compiler path. */
  static uint64_t totalKernelLaunches();
  static uint64_t totalCacheHits();
  static void recordKernelLaunches(uint64_t launches);
  static void resetCounters();

  /** Invalidate every live device-bound Vulkan pipeline cache. */
  static void invalidateAll();

  /**
   * Return the VkPipelineLayout for a previously compiled pipeline.
   *
   * This accessor is required by VulkanReplayHandle::recordDispatch(MLIR) to
   * pass the correct layout to vkCmdBindDescriptorSets.  The pipeline must
   * already be in the cache (i.e. getOrCompile() must have been called first
   * with the same opName and mlirModuleStr).
   *
   * @param opName        Same opName passed to getOrCompile().
   * @param mlirModuleStr Same mlirModuleStr passed to getOrCompile().
   * @return              The VkPipelineLayout, or VK_NULL_HANDLE if the key
   *                      is not present in the cache.
   */
  VkPipelineLayout getPipelineLayout(
      const std::string& opName, const std::string& mlirModuleStr,
      uint32_t pushConstantBytes = 0) const;

  /**
   * Return the VkDescriptorSetLayout for a previously compiled pipeline.
   *
   * @param opName        Same opName passed to getOrCompile().
   * @param mlirModuleStr Same mlirModuleStr passed to getOrCompile().
   * @return              The VkDescriptorSetLayout, or VK_NULL_HANDLE if the
   *                      key is not present in the cache.
   */
  VkDescriptorSetLayout getDescriptorSetLayout(
      const std::string& opName, const std::string& mlirModuleStr,
      uint32_t pushConstantBytes = 0) const;

  /**
   * Return the ordered descriptor binding numbers declared by the compiled
   * SPIR-V interface for descriptor set zero.
   */
  std::vector<uint32_t> getDescriptorBindings(
      const std::string& opName, const std::string& mlirModuleStr,
      uint32_t pushConstantBytes = 0) const;

 private:
  // ── Per-pipeline Vulkan objects ────────────────────────────────────────────

  struct CachedPipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
    std::vector<uint32_t> descriptorBindings;
    uint32_t pushConstantBytes = 0;
  };

  struct SpirvModule {
    std::vector<uint32_t> bytecode;
    std::vector<uint32_t> descriptorBindings;

    bool empty() const { return bytecode.empty(); }
  };

  // ── Members ────────────────────────────────────────────────────────────────

  VkDevice device_;
  VkPhysicalDevice physDevice_;
  // Device-lifetime driver pipeline cache (Tier 2, ADR 0115). Owned by
  // VulkanDeviceContext; internally synchronized by the driver, so both the
  // device-context and replay-handle cache instances may share it.
  VkPipelineCache driverPipelineCache_ = VK_NULL_HANDLE;
  uint32_t apiVersion_;
  bool fp16_;
  bool storage16_;
  bool fp64_;
  bool int64_;
  bool int8_;

  /** key → compiled pipeline */
  mutable std::mutex mutex_;
  std::map<std::string, CachedPipeline> pipelineCache_;

  /**
   * Invalidated objects retained until cache destruction. Vulkan command
   * buffers capture raw pipeline/layout handles, so lookup invalidation cannot
   * destroy those handles while a recorded replay may still reference them.
   */
  std::vector<CachedPipeline> retiredPipelines_;

  static std::mutex registryMutex_;
  static std::unordered_set<VulkanPipelineCache*> instances_;
  static std::atomic<uint64_t> totalKernelLaunches_;
  static std::atomic<uint64_t> totalCacheHits_;

  // ── Private helpers ────────────────────────────────────────────────────────

  /**
   * Lower the given textual MLIR module to SPIR-V bytecode.
   *
   * The pipeline applied is:
   *   GPU dialect → SPIR-V dialect → serialised SPIR-V binary
   *
   * The pass sequence uses MLIR public APIs only:
   *   - mlir::PassManager with convertGpuOpsToSPIRVOps and
   *     mlir::spirv::serialize.
   *
   * @param mlirModuleStr Textual MLIR module.
   * @return              Serialized SPIR-V and its exact set-zero descriptor
   *                      bindings, or an empty result on any failure.
   */
  SpirvModule mlirToSpirv(const std::string& mlirModuleStr);

  /**
   * Create a Vulkan compute pipeline from the given SPIR-V bytecode.
   *
   * Steps:
   *   1. vkCreateShaderModule from spirvBytecode.
   *   2. Create a VkDescriptorSetLayout from the exact storage-buffer bindings
   *      declared by the lowered SPIR-V interface.
   *   3. vkCreatePipelineLayout from the descriptor set layout.
   *   4. vkCreateComputePipelines with the shader module at entry "main".
   *   5. vkDestroyShaderModule (the pipeline retains the compiled bytecode).
   *
   * @param spirvBytecode      SPIR-V binary as a uint32_t vector.
   * @param descriptorBindings Ordered descriptor-set-zero storage bindings.
   * @param device             Logical Vulkan device.
   * @param pushConstantBytes Compute-stage push-constant range size.
   * @param outLayout          Receives the VkPipelineLayout on success.
   * @param outDescLayout      Receives the VkDescriptorSetLayout on success.
   * @return                   A valid VkPipeline on success, VK_NULL_HANDLE on
   *                           any Vulkan-API failure.
   */
  VkPipeline createComputePipeline(const std::vector<uint32_t>& spirvBytecode,
                                   const std::vector<uint32_t>& descriptorBindings,
                                   VkDevice device,
                                   uint32_t pushConstantBytes,
                                   VkPipelineLayout& outLayout,
                                   VkDescriptorSetLayout& outDescLayout);

  /**
   * Destroy one CachedPipeline's Vulkan handles in reverse creation order.
   * Safe to call on a zero-initialised struct.
   */
  void destroyCachedPipeline(CachedPipeline& cp);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
#endif  // LIBND4J_VULKAN_PIPELINE_CACHE_H
