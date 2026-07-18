/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.factory.config;

/**
 * Vulkan backend disk-cache configuration (ADR 0115): the Tier-1 SPIR-V
 * module disk cache (skips the MLIR→SPIR-V pass pipeline on warm starts) and
 * the Tier-2 VkPipelineCache driver-blob persistence (skips the driver's
 * SPIR-V→ISA compile).
 *
 * <p>All methods have default implementations that return safe defaults, so
 * non-Vulkan backends need not implement them. The Vulkan backend delegates
 * to the native {@code sd::config::VulkanConfig} via generated bindings, and
 * applies the matching {@code nd4j.environment.vulkan*} system properties at
 * backend init so properties win over {@code ND4J_VULKAN_*} env vars —
 * matching the DSP plan disk cache resolution contract.</p>
 */
public interface VulkanEnvironmentConfig {

    // --- Tier 1: SPIR-V module disk cache ---

    /** Whether the SPIR-V module disk cache is enabled. Default: true. */
    default boolean vulkanSpirvCacheEnabled() { return true; }
    default void setVulkanSpirvCacheEnabled(boolean enabled) {}

    /** SPIR-V cache directory override, or empty for ~/.kompile/cache/vulkan/spirv_cache/. */
    default String vulkanSpirvCacheDir() { return ""; }
    default void setVulkanSpirvCacheDir(String dir) {}

    /** Read-only pre-seed directory override, or empty for ~/.kompile/cache/vulkan/spirv_override/. */
    default String vulkanSpirvOverrideDir() { return ""; }
    default void setVulkanSpirvOverrideDir(String dir) {}

    /** Bypass Tier-1 reads and skip writes (Triton alwaysCompile semantics). Default: false. */
    default boolean vulkanAlwaysCompile() { return false; }
    default void setVulkanAlwaysCompile(boolean alwaysCompile) {}

    /** Dump the input MLIR module next to stored cache entries. Default: false. */
    default boolean vulkanKernelDump() { return false; }
    default void setVulkanKernelDump(boolean kernelDump) {}

    // --- Tier 2: VkPipelineCache driver-blob persistence ---

    /** Whether driver pipeline-cache blob persistence is enabled. Default: true. */
    default boolean vulkanPipelineCacheEnabled() { return true; }
    default void setVulkanPipelineCacheEnabled(boolean enabled) {}

    /** Blob directory override, or empty for ~/.kompile/cache/vulkan/pipeline_cache/. */
    default String vulkanPipelineCacheDir() { return ""; }
    default void setVulkanPipelineCacheDir(String dir) {}

    /** Blob size budget in bytes; larger blobs are dropped and regenerate. Default: 64 MB. */
    default long vulkanPipelineCacheMaxBytes() { return 67108864L; }
    default void setVulkanPipelineCacheMaxBytes(long maxBytes) {}

    // --- Cache observability (test/diagnostic counters) ---

    /** Tier-1 disk hits since process start (or last counter reset). */
    default long vulkanSpirvDiskHits() { return 0L; }
    /** Tier-1 disk misses since process start (or last counter reset). */
    default long vulkanSpirvDiskMisses() { return 0L; }
    /** Tier-1 disk stores since process start (or last counter reset). */
    default long vulkanSpirvDiskStores() { return 0L; }
    /** Tier-2 blob loads since process start (or last counter reset). */
    default long vulkanPipelineBlobLoads() { return 0L; }
    /** Tier-2 blob saves since process start (or last counter reset). */
    default long vulkanPipelineBlobSaves() { return 0L; }
    /** Reset all Vulkan cache counters to zero. */
    default void clearVulkanCacheCounters() {}
}
