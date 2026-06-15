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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Per-layer KV cache policies that allow different cache configurations per transformer layer.
 *
 * <p>Research has shown that different layers in a transformer model attend to different
 * context windows. Early layers often focus on local context, middle layers capture
 * global patterns, and late layers again focus locally. Per-layer policies exploit this
 * by allocating different amounts of KV cache memory per layer.</p>
 *
 * <h3>Policy presets</h3>
 * <ul>
 *   <li><b>Pyramid:</b> Early layers get small windows, deep layers get large windows.
 *       Suitable for models where deep layers need the most context.</li>
 *   <li><b>Hourglass:</b> Early and late layers get small windows, middle layers get
 *       large windows. Based on the observation that middle layers handle
 *       long-range dependencies.</li>
 *   <li><b>Uniform:</b> Same window size for all layers (baseline).</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see PerLayerPagedKVCache
 */
@Slf4j
public class PerLayerKVPolicy {

    /**
     * Policy configuration for a single layer.
     */
    public static class LayerPolicy {
        /** Maximum context window size in tokens for this layer. */
        @Getter
        private final int windowSize;

        /** Maximum number of KV cache blocks for this layer. */
        @Getter
        private final int maxBlocks;

        /** Data type for this layer's KV cache. */
        @Getter
        private final DataType kvDataType;

        /** Eviction priority (lower = evicted first). */
        @Getter
        private final int evictionPriority;

        /** Whether this layer uses a sliding window (evicts oldest blocks when full). */
        @Getter
        private final boolean slidingWindow;

        /**
         * Create a layer policy.
         *
         * @param windowSize       maximum context window in tokens
         * @param maxBlocks        maximum KV cache blocks
         * @param kvDataType       data type for KV tensors
         * @param evictionPriority eviction priority (lower = evicted first)
         * @param slidingWindow    whether to use sliding window eviction
         */
        public LayerPolicy(int windowSize, int maxBlocks, DataType kvDataType,
                           int evictionPriority, boolean slidingWindow) {
            this.windowSize = windowSize;
            this.maxBlocks = maxBlocks;
            this.kvDataType = kvDataType;
            this.evictionPriority = evictionPriority;
            this.slidingWindow = slidingWindow;
        }

        @Override
        public String toString() {
            return String.format("LayerPolicy[window=%d, blocks=%d, dtype=%s, priority=%d, sliding=%s]",
                    windowSize, maxBlocks, kvDataType, evictionPriority, slidingWindow);
        }
    }

    /** Per-layer policies, indexed by layer number. */
    private final List<LayerPolicy> layerPolicies;

    /** Number of layers. */
    @Getter
    private final int numLayers;

    private PerLayerKVPolicy(List<LayerPolicy> layerPolicies) {
        this.layerPolicies = Collections.unmodifiableList(layerPolicies);
        this.numLayers = layerPolicies.size();
    }

    /**
     * Create a pyramid policy: early layers small, deep layers large.
     *
     * <p>Layers 0 to numLayers/3 use baseWindow. Layers numLayers/3 to 2*numLayers/3
     * use 4x baseWindow. Layers 2*numLayers/3 to end use 16x baseWindow.</p>
     *
     * @param numLayers  total number of transformer layers
     * @param baseWindow smallest window size (for early layers)
     * @return a pyramid per-layer policy
     */
    public static PerLayerKVPolicy pyramid(int numLayers, int baseWindow) {
        return pyramid(numLayers, baseWindow, PagedKVCache.DEFAULT_BLOCK_SIZE, DataType.FLOAT);
    }

    /**
     * Create a pyramid policy with configurable block size and data type.
     *
     * @param numLayers  total number of transformer layers
     * @param baseWindow smallest window size
     * @param blockSize  tokens per block
     * @param dataType   KV cache data type
     * @return a pyramid per-layer policy
     */
    public static PerLayerKVPolicy pyramid(int numLayers, int baseWindow, int blockSize, DataType dataType) {
        List<LayerPolicy> policies = new ArrayList<>(numLayers);
        int tier1End = numLayers / 3;
        int tier2End = 2 * numLayers / 3;

        for (int layer = 0; layer < numLayers; layer++) {
            int window;
            int priority;
            if (layer < tier1End) {
                window = baseWindow;
                priority = 1; // Low priority (evicted first)
            } else if (layer < tier2End) {
                window = baseWindow * 4;
                priority = 5; // Medium priority
            } else {
                window = baseWindow * 16;
                priority = 10; // High priority (evicted last)
            }

            int maxBlocks = (window + blockSize - 1) / blockSize;
            policies.add(new LayerPolicy(window, maxBlocks, dataType, priority, layer < tier1End));
        }

        log.info("Created pyramid policy: {} layers, baseWindow={}, tiers at [{}, {}, {}]",
                numLayers, baseWindow, tier1End, tier2End, numLayers);

        return new PerLayerKVPolicy(policies);
    }

    /**
     * Create an hourglass policy: early/late layers small, middle layers large.
     *
     * @param numLayers  total number of transformer layers
     * @param baseWindow smallest window size (for early/late layers)
     * @return an hourglass per-layer policy
     */
    public static PerLayerKVPolicy hourglass(int numLayers, int baseWindow) {
        return hourglass(numLayers, baseWindow, PagedKVCache.DEFAULT_BLOCK_SIZE, DataType.FLOAT);
    }

    /**
     * Create an hourglass policy with configurable block size and data type.
     *
     * @param numLayers  total number of transformer layers
     * @param baseWindow smallest window size
     * @param blockSize  tokens per block
     * @param dataType   KV cache data type
     * @return an hourglass per-layer policy
     */
    public static PerLayerKVPolicy hourglass(int numLayers, int baseWindow, int blockSize, DataType dataType) {
        List<LayerPolicy> policies = new ArrayList<>(numLayers);
        int quarter = numLayers / 4;
        int threeQuarters = 3 * numLayers / 4;

        for (int layer = 0; layer < numLayers; layer++) {
            int window;
            int priority;
            boolean sliding;

            if (layer < quarter) {
                // Early layers: small window, sliding
                window = baseWindow;
                priority = 2;
                sliding = true;
            } else if (layer >= threeQuarters) {
                // Late layers: small window, sliding
                window = baseWindow;
                priority = 2;
                sliding = true;
            } else {
                // Middle layers: large window
                // Scale based on distance from edges
                double centerDist = 1.0 - Math.abs(layer - numLayers / 2.0) / (numLayers / 2.0);
                int multiplier = Math.max(1, (int) (16 * centerDist));
                window = baseWindow * multiplier;
                priority = 8;
                sliding = false;
            }

            int maxBlocks = (window + blockSize - 1) / blockSize;
            policies.add(new LayerPolicy(window, maxBlocks, dataType, priority, sliding));
        }

        log.info("Created hourglass policy: {} layers, baseWindow={}", numLayers, baseWindow);

        return new PerLayerKVPolicy(policies);
    }

    /**
     * Create a uniform policy: same window size for all layers.
     *
     * @param numLayers  total number of transformer layers
     * @param windowSize window size for all layers
     * @return a uniform per-layer policy
     */
    public static PerLayerKVPolicy uniform(int numLayers, int windowSize) {
        return uniform(numLayers, windowSize, PagedKVCache.DEFAULT_BLOCK_SIZE, DataType.FLOAT);
    }

    /**
     * Create a uniform policy with configurable block size and data type.
     *
     * @param numLayers  total number of transformer layers
     * @param windowSize window size for all layers
     * @param blockSize  tokens per block
     * @param dataType   KV cache data type
     * @return a uniform per-layer policy
     */
    public static PerLayerKVPolicy uniform(int numLayers, int windowSize, int blockSize, DataType dataType) {
        List<LayerPolicy> policies = new ArrayList<>(numLayers);
        int maxBlocks = (windowSize + blockSize - 1) / blockSize;

        for (int layer = 0; layer < numLayers; layer++) {
            policies.add(new LayerPolicy(windowSize, maxBlocks, dataType, 5, false));
        }

        return new PerLayerKVPolicy(policies);
    }

    /**
     * Get the policy for a specific layer.
     *
     * @param layerIdx layer index (0-based)
     * @return the layer's KV cache policy
     */
    public LayerPolicy getPolicy(int layerIdx) {
        if (layerIdx < 0 || layerIdx >= numLayers) {
            throw new IndexOutOfBoundsException("Layer index " + layerIdx + " out of range [0, " + numLayers + ")");
        }
        return layerPolicies.get(layerIdx);
    }

    /**
     * Get the total maximum blocks across all layers.
     *
     * @return sum of maxBlocks for all layers
     */
    public int getTotalMaxBlocks() {
        int total = 0;
        for (LayerPolicy policy : layerPolicies) {
            total += policy.maxBlocks;
        }
        return total;
    }

    /**
     * Estimate total memory usage across all layers.
     *
     * @param numKvHeads number of KV attention heads
     * @param headDim    dimension per head
     * @return estimated memory in bytes
     */
    public long estimateMemoryBytes(int numKvHeads, int headDim) {
        long total = 0;
        for (LayerPolicy policy : layerPolicies) {
            // Each block: blockSize * numKvHeads * headDim * dataType.width() for both key and value
            int blockSize = policy.windowSize; // Approximate: actual blocks from maxBlocks
            long bytesPerBlock = (long) policy.maxBlocks * numKvHeads * headDim * policy.kvDataType.width();
            total += 2L * bytesPerBlock; // 2x for key + value
        }
        return total;
    }

    @Override
    public String toString() {
        int totalBlocks = getTotalMaxBlocks();
        return String.format("PerLayerKVPolicy[layers=%d, totalBlocks=%d]", numLayers, totalBlocks);
    }
}
