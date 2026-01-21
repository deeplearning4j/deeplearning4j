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

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

/**
 * Interface for the kernel selection and auto-tuning system.
 * <p>
 * The kernel selector manages the selection of optimal kernel implementations
 * for operations based on runtime performance benchmarking. It supports:
 * <ul>
 *   <li>Auto-tuning to find the fastest kernel</li>
 *   <li>Caching of benchmark results</li>
 *   <li>Multiple selection strategies</li>
 *   <li>Engine preference configuration</li>
 * </ul>
 * </p>
 * <p>
 * Example usage:
 * <pre>{@code
 * KernelSelector selector = Nd4j.getKernelSelector();
 *
 * // Configure auto-tuning
 * selector.configure(KernelSelectionConfig.builder()
 *     .autoTuneEnabled(true)
 *     .strategy(KernelSelectionConfig.Strategy.FASTEST)
 *     .build());
 *
 * // Force retune for an operation
 * selector.retune("conv2d");
 *
 * // Get performance statistics
 * System.out.println(selector.getPerformanceSummary());
 * }</pre>
 * </p>
 *
 * @author Eclipse Deeplearning4j Development Team
 */
public interface KernelSelector {

    /**
     * Performance statistics for a kernel
     */
    @Data
    class KernelPerformanceStats {
        private String operationName;
        private KernelSelectionConfig.Engine engine;
        private double avgTimeNanos;
        private double minTimeNanos;
        private double maxTimeNanos;
        private double stdDevNanos;
        private int sampleCount;
        private boolean reliable;
        private long lastUpdateTime;
    }

    /**
     * Result of kernel selection
     */
    @Data
    class SelectionResult {
        private String operationName;
        private KernelSelectionConfig.Engine selectedEngine;
        private List<KernelSelectionConfig.Engine> availableEngines;
        private double selectionTimeNanos;
        private boolean fromCache;
    }

    /**
     * Configure the kernel selector.
     *
     * @param config Configuration to apply
     */
    void configure(KernelSelectionConfig config);

    /**
     * Get current configuration.
     *
     * @return Current configuration
     */
    KernelSelectionConfig getConfig();

    /**
     * Enable or disable auto-tuning.
     *
     * @param enabled true to enable
     */
    void setAutoTuneEnabled(boolean enabled);

    /**
     * Check if auto-tuning is enabled.
     *
     * @return true if enabled
     */
    boolean isAutoTuneEnabled();

    /**
     * Set the kernel selection strategy.
     *
     * @param strategy Strategy to use
     */
    void setStrategy(KernelSelectionConfig.Strategy strategy);

    /**
     * Get current selection strategy.
     *
     * @return Current strategy
     */
    KernelSelectionConfig.Strategy getStrategy();

    /**
     * Force a specific engine for all operations.
     * Pass null to clear forced engine.
     *
     * @param engine Engine to force, or null
     */
    void forceEngine(KernelSelectionConfig.Engine engine);

    /**
     * Get forced engine, if any.
     *
     * @return Forced engine, or null if none
     */
    KernelSelectionConfig.Engine getForcedEngine();

    /**
     * Disable a specific engine.
     *
     * @param engine Engine to disable
     */
    void disableEngine(KernelSelectionConfig.Engine engine);

    /**
     * Enable a previously disabled engine.
     *
     * @param engine Engine to enable
     */
    void enableEngine(KernelSelectionConfig.Engine engine);

    /**
     * Check if an engine is disabled.
     *
     * @param engine Engine to check
     * @return true if disabled
     */
    boolean isEngineDisabled(KernelSelectionConfig.Engine engine);

    /**
     * Get available engines for an operation.
     *
     * @param operationName Name of the operation
     * @return List of available engines
     */
    List<KernelSelectionConfig.Engine> getAvailableEngines(String operationName);

    /**
     * Force retune for a specific operation.
     * Clears cached performance data and triggers new benchmarks.
     *
     * @param operationName Name of the operation to retune
     */
    void retune(String operationName);

    /**
     * Clear all cached performance data.
     */
    void clearCache();

    /**
     * Get performance statistics for an operation.
     *
     * @param operationName Name of the operation
     * @return Map of engine to performance stats
     */
    Map<KernelSelectionConfig.Engine, KernelPerformanceStats> getPerformanceStats(String operationName);

    /**
     * Get performance statistics for an operation with specific input shapes.
     *
     * @param operationName Name of the operation
     * @param inputShapes Shapes of input arrays
     * @return Map of engine to performance stats
     */
    Map<KernelSelectionConfig.Engine, KernelPerformanceStats> getPerformanceStats(
            String operationName, long[]... inputShapes);

    /**
     * Run benchmarks for an operation.
     *
     * @param operationName Name of the operation
     * @param inputs Sample input arrays
     * @return Map of engine to performance stats
     */
    Map<KernelSelectionConfig.Engine, KernelPerformanceStats> benchmark(
            String operationName, INDArray... inputs);

    /**
     * Get cache hit rate.
     *
     * @return Hit rate (0.0 to 1.0)
     */
    double getCacheHitRate();

    /**
     * Get total cache hits.
     *
     * @return Number of cache hits
     */
    long getCacheHits();

    /**
     * Get total cache misses.
     *
     * @return Number of cache misses
     */
    long getCacheMisses();

    /**
     * Get total benchmark runs.
     *
     * @return Number of benchmarks run
     */
    long getBenchmarkCount();

    /**
     * Save performance cache to a file.
     *
     * @param path File path
     * @return true if saved successfully
     */
    boolean saveCache(String path);

    /**
     * Load performance cache from a file.
     *
     * @param path File path
     * @return true if loaded successfully
     */
    boolean loadCache(String path);

    /**
     * Enable verbose logging of kernel selection decisions.
     *
     * @param verbose true to enable verbose logging
     */
    void setVerbose(boolean verbose);

    /**
     * Check if verbose logging is enabled.
     *
     * @return true if verbose
     */
    boolean isVerbose();

    /**
     * Get a human-readable summary of kernel selection statistics.
     *
     * @return Summary string
     */
    String getPerformanceSummary();

    /**
     * Get statistics summary for a specific operation.
     *
     * @param operationName Name of the operation
     * @return Summary string
     */
    String getOperationSummary(String operationName);

    /**
     * Get the kernel plugin manager.
     *
     * @return Plugin manager instance
     */
    KernelPluginManager getPluginManager();
}
