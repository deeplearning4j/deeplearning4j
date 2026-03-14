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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class PerLayerKVPolicyTest extends BaseNd4jTestWithBackends {

    // ==================== Pyramid Policy ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPyramidPolicyTiers(Nd4jBackend backend) {
        int numLayers = 12;
        int baseWindow = 64;
        PerLayerKVPolicy policy = PerLayerKVPolicy.pyramid(numLayers, baseWindow);

        assertEquals(numLayers, policy.getNumLayers());

        // Tier boundaries: tier1End = 4, tier2End = 8
        // Layers 0-3: baseWindow (64)
        for (int i = 0; i < 4; i++) {
            PerLayerKVPolicy.LayerPolicy lp = policy.getPolicy(i);
            assertEquals(baseWindow, lp.getWindowSize(), "Layer " + i + " should have baseWindow");
            assertEquals(1, lp.getEvictionPriority(), "Early layers should have low priority");
            assertTrue(lp.isSlidingWindow(), "Early layers should use sliding window");
        }

        // Layers 4-7: 4x baseWindow (256)
        for (int i = 4; i < 8; i++) {
            PerLayerKVPolicy.LayerPolicy lp = policy.getPolicy(i);
            assertEquals(baseWindow * 4, lp.getWindowSize(), "Layer " + i + " should have 4x baseWindow");
            assertEquals(5, lp.getEvictionPriority(), "Middle layers should have medium priority");
            assertFalse(lp.isSlidingWindow());
        }

        // Layers 8-11: 16x baseWindow (1024)
        for (int i = 8; i < 12; i++) {
            PerLayerKVPolicy.LayerPolicy lp = policy.getPolicy(i);
            assertEquals(baseWindow * 16, lp.getWindowSize(), "Layer " + i + " should have 16x baseWindow");
            assertEquals(10, lp.getEvictionPriority(), "Deep layers should have high priority");
            assertFalse(lp.isSlidingWindow());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPyramidPolicyMonotonicallyIncreasingWindows(Nd4jBackend backend) {
        PerLayerKVPolicy policy = PerLayerKVPolicy.pyramid(24, 32);

        int prevWindow = 0;
        for (int i = 0; i < policy.getNumLayers(); i++) {
            int window = policy.getPolicy(i).getWindowSize();
            assertTrue(window >= prevWindow, "Window sizes must be non-decreasing in pyramid policy");
            prevWindow = window;
        }
    }

    // ==================== Hourglass Policy ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHourglassPolicySymmetry(Nd4jBackend backend) {
        int numLayers = 16;
        int baseWindow = 64;
        PerLayerKVPolicy policy = PerLayerKVPolicy.hourglass(numLayers, baseWindow);

        assertEquals(numLayers, policy.getNumLayers());

        // quarter = 4, threeQuarters = 12
        // Early layers (0-3): baseWindow, sliding
        for (int i = 0; i < 4; i++) {
            PerLayerKVPolicy.LayerPolicy lp = policy.getPolicy(i);
            assertEquals(baseWindow, lp.getWindowSize(), "Early layer " + i);
            assertTrue(lp.isSlidingWindow(), "Early layers should use sliding window");
        }

        // Late layers (12-15): baseWindow, sliding
        for (int i = 12; i < 16; i++) {
            PerLayerKVPolicy.LayerPolicy lp = policy.getPolicy(i);
            assertEquals(baseWindow, lp.getWindowSize(), "Late layer " + i);
            assertTrue(lp.isSlidingWindow(), "Late layers should use sliding window");
        }

        // Middle layers (4-11): larger windows, not sliding
        for (int i = 4; i < 12; i++) {
            PerLayerKVPolicy.LayerPolicy lp = policy.getPolicy(i);
            assertTrue(lp.getWindowSize() >= baseWindow,
                    "Middle layer " + i + " should have window >= baseWindow");
            assertFalse(lp.isSlidingWindow(), "Middle layers should NOT use sliding window");
        }

        // Symmetric property: early and late should have same window sizes
        for (int i = 0; i < 4; i++) {
            int earlyWindow = policy.getPolicy(i).getWindowSize();
            int lateWindow = policy.getPolicy(numLayers - 1 - i).getWindowSize();
            assertEquals(earlyWindow, lateWindow,
                    "Early layer " + i + " and late layer " + (numLayers - 1 - i) +
                            " should have same window in hourglass");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHourglassMiddleLargerThanEdges(Nd4jBackend backend) {
        PerLayerKVPolicy policy = PerLayerKVPolicy.hourglass(12, 32);
        int midLayer = 6; // middle of the hourglass
        int edgeLayer = 0;

        assertTrue(policy.getPolicy(midLayer).getWindowSize() > policy.getPolicy(edgeLayer).getWindowSize(),
                "Middle layers should have larger windows than edge layers in hourglass");
    }

    // ==================== Uniform Policy ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testUniformPolicyConsistency(Nd4jBackend backend) {
        int numLayers = 8;
        int windowSize = 256;
        PerLayerKVPolicy policy = PerLayerKVPolicy.uniform(numLayers, windowSize);

        assertEquals(numLayers, policy.getNumLayers());

        for (int i = 0; i < numLayers; i++) {
            PerLayerKVPolicy.LayerPolicy lp = policy.getPolicy(i);
            assertEquals(windowSize, lp.getWindowSize(), "All layers should have the same window");
            assertEquals(DataType.FLOAT, lp.getKvDataType());
            assertFalse(lp.isSlidingWindow());
            assertEquals(5, lp.getEvictionPriority(), "Uniform should have equal priority");
        }

        // All maxBlocks should be the same
        int expectedMaxBlocks = policy.getPolicy(0).getMaxBlocks();
        for (int i = 1; i < numLayers; i++) {
            assertEquals(expectedMaxBlocks, policy.getPolicy(i).getMaxBlocks());
        }
    }

    // ==================== getTotalMaxBlocks / estimateMemoryBytes ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTotalMaxBlocks(Nd4jBackend backend) {
        PerLayerKVPolicy policy = PerLayerKVPolicy.uniform(4, 128);

        int singleLayerBlocks = policy.getPolicy(0).getMaxBlocks();
        assertEquals(singleLayerBlocks * 4, policy.getTotalMaxBlocks());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEstimateMemoryBytes(Nd4jBackend backend) {
        PerLayerKVPolicy policy = PerLayerKVPolicy.uniform(4, 128);

        long memBytes = policy.estimateMemoryBytes(8, 64);
        assertTrue(memBytes > 0, "Memory estimate should be positive");

        // Pyramid should use less memory than uniform with equivalent maxWindow
        PerLayerKVPolicy pyramidPolicy = PerLayerKVPolicy.pyramid(12, 32);
        PerLayerKVPolicy uniformPolicy = PerLayerKVPolicy.uniform(12, 32 * 16);

        // Pyramid total blocks should be less than uniform at max window
        assertTrue(pyramidPolicy.getTotalMaxBlocks() < uniformPolicy.getTotalMaxBlocks(),
                "Pyramid should allocate fewer total blocks than uniform at max window");
    }

    // ==================== Boundary / Error Cases ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetPolicyOutOfBoundsThrows(Nd4jBackend backend) {
        PerLayerKVPolicy policy = PerLayerKVPolicy.uniform(4, 128);

        assertThrows(IndexOutOfBoundsException.class, () -> policy.getPolicy(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> policy.getPolicy(4));
    }

    // ==================== PerLayerPagedKVCache ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPerLayerCacheCreation(Nd4jBackend backend) {
        int numLayers = 6;
        int blockSize = 16;
        PerLayerKVPolicy policy = PerLayerKVPolicy.uniform(numLayers, 128, blockSize, DataType.FLOAT);

        try (PerLayerPagedKVCache cache = new PerLayerPagedKVCache(policy, 2, 4, 8, blockSize)) {
            assertEquals(numLayers, cache.getNumLayers());
            assertEquals(2, cache.getMaxBatchSize());
            assertEquals(4, cache.getNumKvHeads());
            assertEquals(8, cache.getHeadDim());
            assertEquals(blockSize, cache.getBlockSize());

            // Each layer should have a PagedKVCache
            for (int layer = 0; layer < numLayers; layer++) {
                PagedKVCache layerCache = cache.getLayerCache(layer);
                assertNotNull(layerCache, "Layer " + layer + " cache should not be null");
            }

            assertTrue(cache.getTotalBlocks() > 0);
            assertEquals(cache.getTotalBlocks(), cache.getTotalFreeBlocks(),
                    "All blocks should be free initially");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPerLayerCacheAppend(Nd4jBackend backend) {
        int numLayers = 3;
        int blockSize = 8;
        int numKvHeads = 2;
        int headDim = 4;
        PerLayerKVPolicy policy = PerLayerKVPolicy.uniform(numLayers, 64, blockSize, DataType.FLOAT);

        try (PerLayerPagedKVCache cache = new PerLayerPagedKVCache(policy, 1, numKvHeads, headDim, blockSize)) {
            int numTokens = 5;
            INDArray keys = Nd4j.rand(DataType.FLOAT, numTokens, numKvHeads, headDim);
            INDArray vals = Nd4j.rand(DataType.FLOAT, numTokens, numKvHeads, headDim);

            // Append to each layer independently
            for (int layer = 0; layer < numLayers; layer++) {
                cache.append(layer, 0, keys, vals);
                assertEquals(numTokens, cache.getLayerCache(layer).getSequenceLength(0),
                        "Layer " + layer + " should have " + numTokens + " tokens");
            }

            // Verify free blocks decreased
            assertTrue(cache.getTotalFreeBlocks() < cache.getTotalBlocks(),
                    "Free blocks should decrease after appending");

            keys.close();
            vals.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPerLayerFreeSequenceAcrossLayers(Nd4jBackend backend) {
        int numLayers = 4;
        int blockSize = 8;
        int numKvHeads = 2;
        int headDim = 4;
        PerLayerKVPolicy policy = PerLayerKVPolicy.uniform(numLayers, 64, blockSize, DataType.FLOAT);

        try (PerLayerPagedKVCache cache = new PerLayerPagedKVCache(policy, 2, numKvHeads, headDim, blockSize)) {
            int totalBlocksBefore = cache.getTotalFreeBlocks();

            INDArray keys = Nd4j.rand(DataType.FLOAT, 4, numKvHeads, headDim);
            INDArray vals = Nd4j.rand(DataType.FLOAT, 4, numKvHeads, headDim);

            // Append to all layers for seq 0
            for (int layer = 0; layer < numLayers; layer++) {
                cache.append(layer, 0, keys, vals);
            }

            int freeAfterAppend = cache.getTotalFreeBlocks();
            assertTrue(freeAfterAppend < totalBlocksBefore, "Should have fewer free blocks after append");

            // Free sequence 0 across all layers
            cache.freeSequence(0);

            int freeAfterRelease = cache.getTotalFreeBlocks();
            assertEquals(totalBlocksBefore, freeAfterRelease,
                    "All blocks should be returned after freeSequence");

            // Verify all layers show zero length for seq 0
            for (int layer = 0; layer < numLayers; layer++) {
                assertEquals(0, cache.getLayerCache(layer).getSequenceLength(0));
            }

            keys.close();
            vals.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBudgetProportionalAllocation(Nd4jBackend backend) {
        // Pyramid gives different block budgets per tier
        int numLayers = 12;
        int baseWindow = 32;
        int blockSize = 16;
        PerLayerKVPolicy policy = PerLayerKVPolicy.pyramid(numLayers, baseWindow, blockSize, DataType.FLOAT);

        // Early layers (0-3): baseWindow=32, maxBlocks = ceil(32/16) = 2
        // Mid layers (4-7): 4*32=128, maxBlocks = ceil(128/16) = 8
        // Deep layers (8-11): 16*32=512, maxBlocks = ceil(512/16) = 32

        for (int i = 0; i < 4; i++) {
            assertEquals(2, policy.getPolicy(i).getMaxBlocks(), "Early layer " + i);
        }
        for (int i = 4; i < 8; i++) {
            assertEquals(8, policy.getPolicy(i).getMaxBlocks(), "Mid layer " + i);
        }
        for (int i = 8; i < 12; i++) {
            assertEquals(32, policy.getPolicy(i).getMaxBlocks(), "Deep layer " + i);
        }

        // Total = 4*2 + 4*8 + 4*32 = 8 + 32 + 128 = 168
        assertEquals(168, policy.getTotalMaxBlocks());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPerLayerCacheOutOfBoundsThrows(Nd4jBackend backend) {
        int numLayers = 3;
        PerLayerKVPolicy policy = PerLayerKVPolicy.uniform(numLayers, 64, 8, DataType.FLOAT);

        try (PerLayerPagedKVCache cache = new PerLayerPagedKVCache(policy, 1, 2, 4, 8)) {
            INDArray keys = Nd4j.rand(DataType.FLOAT, 1, 2, 4);
            INDArray vals = Nd4j.rand(DataType.FLOAT, 1, 2, 4);

            assertThrows(IndexOutOfBoundsException.class, () -> cache.append(-1, 0, keys, vals));
            assertThrows(IndexOutOfBoundsException.class, () -> cache.append(numLayers, 0, keys, vals));
            assertThrows(IndexOutOfBoundsException.class, () -> cache.getLayerCache(-1));
            assertThrows(IndexOutOfBoundsException.class, () -> cache.getLayerCache(numLayers));

            keys.close();
            vals.close();
        }
    }
}
