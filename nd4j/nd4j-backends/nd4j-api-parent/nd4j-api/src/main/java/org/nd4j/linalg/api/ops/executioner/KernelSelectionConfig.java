/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.linalg.api.ops.executioner;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Configuration for the kernel selection and auto-tuning system.
 * <p>
 * The kernel selection system allows ND4J to automatically choose the fastest
 * implementation for each operation based on runtime benchmarking. Different
 * backends (oneDNN, cuDNN, native, custom plugins) can provide implementations
 * for the same operations, and this system selects the best one.
 * </p>
 * <p>
 * Example usage:
 * <pre>{@code
 * KernelSelectionConfig config = KernelSelectionConfig.builder()
 *     .strategy(KernelSelectionStrategy.FASTEST)
 *     .autoTuneEnabled(true)
 *     .warmupRuns(2)
 *     .benchmarkRuns(5)
 *     .build();
 *
 * Nd4j.getKernelSelector().configure(config);
 * }</pre>
 * </p>
 *
 * @author Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class KernelSelectionConfig implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * Kernel selection strategy
     */
    public enum Strategy {
        /**
         * Always use the fastest kernel based on benchmarks (default)
         */
        FASTEST,

        /**
         * Use first available helper (legacy behavior)
         */
        FIRST_AVAILABLE,

        /**
         * Distribute load across backends in round-robin fashion
         */
        ROUND_ROBIN,

        /**
         * Prefer kernels with lower memory usage
         */
        MEMORY_OPTIMIZED,

        /**
         * Prefer power-efficient kernels (useful for mobile/edge)
         */
        POWER_OPTIMIZED
    }

    /**
     * Compute engine types
     */
    public enum Engine {
        CPU,
        CUDA,
        ONEDNN,
        MPS,
        ACCELERATE,
        ARM,
        VULKAN,
        OPENCL,
        TPU,
        LLAMACPP,
        VLM;

        public static Engine fromString(String name) {
            if (name == null) return CPU;
            switch (name.toLowerCase()) {
                case "cpu":
                case "native":
                    return CPU;
                case "cuda":
                case "gpu":
                    return CUDA;
                case "onednn":
                case "mkldnn":
                case "mkl":
                    return ONEDNN;
                case "mps":
                case "metal":
                    return MPS;
                case "accelerate":
                case "apple":
                    return ACCELERATE;
                case "arm":
                case "armcl":
                    return ARM;
                case "vulkan":
                    return VULKAN;
                case "opencl":
                    return OPENCL;
                case "tpu":
                    return TPU;
                case "llamacpp":
                case "llama":
                    return LLAMACPP;
                case "vlm":
                    return VLM;
                default:
                    return CPU;
            }
        }
    }

    /**
     * The kernel selection strategy
     */
    @Builder.Default
    private Strategy strategy = Strategy.FASTEST;

    /**
     * Enable/disable auto-tuning
     */
    @Builder.Default
    private boolean autoTuneEnabled = true;

    /**
     * Number of warmup runs before timing
     */
    @Builder.Default
    private int warmupRuns = 2;

    /**
     * Number of benchmark runs for timing
     */
    @Builder.Default
    private int benchmarkRuns = 5;

    /**
     * Maximum time to spend benchmarking one kernel (ms)
     */
    @Builder.Default
    private long maxBenchmarkTimeMs = 1000;

    /**
     * Path to persist the kernel performance cache
     */
    private String cachePath;

    /**
     * Cache entry expiry time in seconds
     */
    @Builder.Default
    private long cacheExpirySeconds = 86400; // 24 hours

    /**
     * Performance threshold for switching kernels (1.2 = 20% faster required)
     */
    @Builder.Default
    private double performanceThreshold = 1.2;

    /**
     * Allow fallback to native implementation
     */
    @Builder.Default
    private boolean allowFallbackToNative = true;

    /**
     * Preferred engine for tie-breaking
     */
    @Builder.Default
    private Engine preferredEngine = Engine.CPU;

    /**
     * Force a specific engine (ignores auto-tuning)
     */
    private Engine forcedEngine;

    /**
     * Engines to disable
     */
    @Builder.Default
    private Set<Engine> disabledEngines = new HashSet<>();

    /**
     * Enable verbose logging of kernel selection
     */
    @Builder.Default
    private boolean verbose = false;

    /**
     * Engine priority preferences (higher = more preferred)
     */
    @Builder.Default
    private Map<Engine, Integer> enginePreferences = new HashMap<>();

    /**
     * Create default configuration
     */
    public static KernelSelectionConfig defaultConfig() {
        Map<Engine, Integer> defaultPrefs = new HashMap<>();
        defaultPrefs.put(Engine.CUDA, 100);
        defaultPrefs.put(Engine.ONEDNN, 90);
        defaultPrefs.put(Engine.MPS, 85);
        defaultPrefs.put(Engine.ACCELERATE, 80);
        defaultPrefs.put(Engine.ARM, 75);
        defaultPrefs.put(Engine.CPU, 50);

        return KernelSelectionConfig.builder()
                .strategy(Strategy.FASTEST)
                .autoTuneEnabled(true)
                .warmupRuns(2)
                .benchmarkRuns(5)
                .maxBenchmarkTimeMs(1000)
                .cacheExpirySeconds(86400)
                .performanceThreshold(1.2)
                .allowFallbackToNative(true)
                .preferredEngine(Engine.CPU)
                .enginePreferences(defaultPrefs)
                .verbose(false)
                .build();
    }

    /**
     * Create configuration from environment variables
     */
    public static KernelSelectionConfig fromEnvironment() {
        KernelSelectionConfigBuilder builder = KernelSelectionConfig.builder();

        String strategyEnv = System.getenv("SD_KERNEL_STRATEGY");
        if (strategyEnv != null) {
            try {
                builder.strategy(Strategy.valueOf(strategyEnv.toUpperCase()));
            } catch (IllegalArgumentException e) {
                // Use default
            }
        }

        String autoTuneEnv = System.getenv("SD_KERNEL_AUTOTUNE");
        if (autoTuneEnv != null) {
            builder.autoTuneEnabled("1".equals(autoTuneEnv) || "true".equalsIgnoreCase(autoTuneEnv));
        }

        String warmupEnv = System.getenv("SD_KERNEL_WARMUP_RUNS");
        if (warmupEnv != null) {
            try {
                builder.warmupRuns(Integer.parseInt(warmupEnv));
            } catch (NumberFormatException e) {
                // Use default
            }
        }

        String benchmarkEnv = System.getenv("SD_KERNEL_BENCHMARK_RUNS");
        if (benchmarkEnv != null) {
            try {
                builder.benchmarkRuns(Integer.parseInt(benchmarkEnv));
            } catch (NumberFormatException e) {
                // Use default
            }
        }

        String cachePathEnv = System.getenv("SD_KERNEL_CACHE_PATH");
        if (cachePathEnv != null) {
            builder.cachePath(cachePathEnv);
        }

        String preferredEnv = System.getenv("SD_KERNEL_PREFERRED_ENGINE");
        if (preferredEnv != null) {
            builder.preferredEngine(Engine.fromString(preferredEnv));
        }

        String forceEnv = System.getenv("SD_KERNEL_FORCE_ENGINE");
        if (forceEnv != null) {
            builder.forcedEngine(Engine.fromString(forceEnv));
        }

        String verboseEnv = System.getenv("SD_KERNEL_VERBOSE");
        if (verboseEnv != null) {
            builder.verbose("1".equals(verboseEnv) || "true".equalsIgnoreCase(verboseEnv));
        }

        String disableEnv = System.getenv("SD_KERNEL_DISABLE_ENGINES");
        if (disableEnv != null) {
            Set<Engine> disabled = new HashSet<>();
            for (String eng : disableEnv.split(",")) {
                disabled.add(Engine.fromString(eng.trim()));
            }
            builder.disabledEngines(disabled);
        }

        return builder.build();
    }

    /**
     * Disable a specific engine
     */
    public KernelSelectionConfig disableEngine(Engine engine) {
        if (disabledEngines == null) {
            disabledEngines = new HashSet<>();
        }
        disabledEngines.add(engine);
        return this;
    }

    /**
     * Enable a previously disabled engine
     */
    public KernelSelectionConfig enableEngine(Engine engine) {
        if (disabledEngines != null) {
            disabledEngines.remove(engine);
        }
        return this;
    }

    /**
     * Set engine preference
     */
    public KernelSelectionConfig setEnginePreference(Engine engine, int priority) {
        if (enginePreferences == null) {
            enginePreferences = new HashMap<>();
        }
        enginePreferences.put(engine, priority);
        return this;
    }

    /**
     * Check if an engine is disabled
     */
    public boolean isEngineDisabled(Engine engine) {
        return disabledEngines != null && disabledEngines.contains(engine);
    }

    /**
     * Get engine preference (default 0)
     */
    public int getEnginePreference(Engine engine) {
        if (enginePreferences == null) {
            return 0;
        }
        return enginePreferences.getOrDefault(engine, 0);
    }
}
