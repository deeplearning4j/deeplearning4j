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

import org.eclipse.deeplearning4j.llm.generation.kvcache.MLAKVCache;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link MLAKVCache} -- Multi-head Latent Attention KV cache.
 * Java-only tests: validates construction, data flow, memory calculations, and lifecycle.
 *
 * Adam Gibson
 */
class TestMLAKVCache {

    // ==================== Construction Tests ====================

    @Test
    void testCreationWithRoPE() {
        int batchSize = 2, maxSeqLen = 128, latentDim = 512, ropeHeadDim = 64, numLayers = 4;
        try (MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT)) {
            assertEquals(batchSize, cache.getBatchSize());
            assertEquals(maxSeqLen, cache.getMaxSeqLen());
            assertEquals(latentDim, cache.getLatentDim());
            assertEquals(ropeHeadDim, cache.getRopeHeadDim());
            assertEquals(numLayers, cache.getNumLayers());
            assertEquals(DataType.FLOAT, cache.getDataType());

            // Latent cache shape
            INDArray latent = cache.getLatentCache();
            assertNotNull(latent);
            assertArrayEquals(new long[]{batchSize, maxSeqLen, latentDim}, latent.shape());

            // RoPE cache should exist
            INDArray rope = cache.getRopeCache();
            assertNotNull(rope);
            assertArrayEquals(new long[]{batchSize, maxSeqLen, ropeHeadDim}, rope.shape());
        }
    }

    @Test
    void testCreationWithoutRoPE() {
        int batchSize = 1, maxSeqLen = 64, latentDim = 256, ropeHeadDim = 0, numLayers = 2;
        try (MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT)) {
            assertEquals(0, cache.getRopeHeadDim());
            assertNull(cache.getRopeCache(), "RoPE cache should be null when ropeHeadDim == 0");

            INDArray latent = cache.getLatentCache();
            assertNotNull(latent);
            assertArrayEquals(new long[]{batchSize, maxSeqLen, latentDim}, latent.shape());
        }
    }

    @Test
    void testCreationHalfPrecision() {
        try (MLAKVCache cache = new MLAKVCache(1, 32, 128, 16, 1, DataType.HALF)) {
            assertEquals(DataType.HALF, cache.getDataType());
            assertEquals(DataType.HALF, cache.getLatentCache().dataType());
            assertEquals(DataType.HALF, cache.getRopeCache().dataType());
        }
    }

    // ==================== Validation Tests ====================

    @Test
    void testInvalidBatchSize() {
        assertThrows(IllegalArgumentException.class,
                () -> new MLAKVCache(0, 64, 256, 32, 2, DataType.FLOAT));
        assertThrows(IllegalArgumentException.class,
                () -> new MLAKVCache(-1, 64, 256, 32, 2, DataType.FLOAT));
    }

    @Test
    void testInvalidMaxSeqLen() {
        assertThrows(IllegalArgumentException.class,
                () -> new MLAKVCache(1, 0, 256, 32, 2, DataType.FLOAT));
    }

    @Test
    void testInvalidLatentDim() {
        assertThrows(IllegalArgumentException.class,
                () -> new MLAKVCache(1, 64, 0, 32, 2, DataType.FLOAT));
    }

    @Test
    void testInvalidRopeHeadDimNegative() {
        assertThrows(IllegalArgumentException.class,
                () -> new MLAKVCache(1, 64, 256, -1, 2, DataType.FLOAT));
    }

    @Test
    void testInvalidNumLayers() {
        assertThrows(IllegalArgumentException.class,
                () -> new MLAKVCache(1, 64, 256, 32, 0, DataType.FLOAT));
    }

    // ==================== Append Tests ====================

    @Test
    void testAppendLatentData() {
        int batchSize = 1, maxSeqLen = 16, latentDim = 8, ropeHeadDim = 0, numLayers = 1;
        try (MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT)) {
            // Create data with known values
            INDArray newLatent = Nd4j.ones(DataType.FLOAT, batchSize, 3, latentDim).mul(2.5);
            cache.append(newLatent, null, 0);

            INDArray latentCache = cache.getLatentCache();
            // Positions 0-2 should have the appended values
            for (int s = 0; s < 3; s++) {
                double val = latentCache.getDouble(0, s, 0);
                assertEquals(2.5, val, 1e-5, "Appended value at position " + s + " should be 2.5");
            }
            // Position 3 should still be zero
            assertEquals(0.0, latentCache.getDouble(0, 3, 0), 1e-5, "Position 3 should still be zero");
        }
    }

    @Test
    void testAppendWithRoPE() {
        int batchSize = 1, maxSeqLen = 16, latentDim = 8, ropeHeadDim = 4, numLayers = 1;
        try (MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT)) {
            INDArray newLatent = Nd4j.ones(DataType.FLOAT, batchSize, 2, latentDim).mul(3.0);
            INDArray newRope = Nd4j.ones(DataType.FLOAT, batchSize, 2, ropeHeadDim).mul(7.0);
            cache.append(newLatent, newRope, 0);

            // Check latent
            assertEquals(3.0, cache.getLatentCache().getDouble(0, 0, 0), 1e-5);
            assertEquals(3.0, cache.getLatentCache().getDouble(0, 1, 0), 1e-5);
            assertEquals(0.0, cache.getLatentCache().getDouble(0, 2, 0), 1e-5);

            // Check RoPE
            INDArray ropeCache = cache.getRopeCache();
            assertNotNull(ropeCache);
            assertEquals(7.0, ropeCache.getDouble(0, 0, 0), 1e-5);
            assertEquals(7.0, ropeCache.getDouble(0, 1, 0), 1e-5);
            assertEquals(0.0, ropeCache.getDouble(0, 2, 0), 1e-5);
        }
    }

    @Test
    void testAppendAtOffset() {
        int batchSize = 1, maxSeqLen = 16, latentDim = 4, ropeHeadDim = 0, numLayers = 1;
        try (MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT)) {
            // First append at position 0
            INDArray first = Nd4j.ones(DataType.FLOAT, batchSize, 2, latentDim).mul(1.0);
            cache.append(first, null, 0);

            // Second append at position 2
            INDArray second = Nd4j.ones(DataType.FLOAT, batchSize, 2, latentDim).mul(5.0);
            cache.append(second, null, 2);

            INDArray latentCache = cache.getLatentCache();
            assertEquals(1.0, latentCache.getDouble(0, 0, 0), 1e-5);
            assertEquals(1.0, latentCache.getDouble(0, 1, 0), 1e-5);
            assertEquals(5.0, latentCache.getDouble(0, 2, 0), 1e-5);
            assertEquals(5.0, latentCache.getDouble(0, 3, 0), 1e-5);
            assertEquals(0.0, latentCache.getDouble(0, 4, 0), 1e-5);
        }
    }

    @Test
    void testAppendMultipleBatches() {
        int batchSize = 2, maxSeqLen = 8, latentDim = 4, ropeHeadDim = 0, numLayers = 1;
        try (MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT)) {
            // Create data where batch 0 has value 1.0 and batch 1 has value 2.0
            INDArray newLatent = Nd4j.zeros(DataType.FLOAT, batchSize, 1, latentDim);
            newLatent.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all()).assign(1.0);
            newLatent.get(org.nd4j.linalg.indexing.NDArrayIndex.point(1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all()).assign(2.0);
            cache.append(newLatent, null, 0);

            assertEquals(1.0, cache.getLatentCache().getDouble(0, 0, 0), 1e-5);
            assertEquals(2.0, cache.getLatentCache().getDouble(1, 0, 0), 1e-5);
        }
    }

    // ==================== Layer Weights Tests ====================

    @Test
    void testSetAndGetLayerWeights() {
        int latentDim = 64, numLayers = 3;
        try (MLAKVCache cache = new MLAKVCache(1, 16, latentDim, 0, numLayers, DataType.FLOAT)) {
            int outDim = 128;
            for (int layer = 0; layer < numLayers; layer++) {
                INDArray weight = Nd4j.randn(DataType.FLOAT, latentDim, outDim).mul(0.01 * (layer + 1));
                cache.setLayerWeights(layer, weight);
            }

            for (int layer = 0; layer < numLayers; layer++) {
                INDArray retrieved = cache.getKvDownProj(layer);
                assertNotNull(retrieved);
                assertArrayEquals(new long[]{latentDim, outDim}, retrieved.shape());
            }
        }
    }

    @Test
    void testGetKvDownProjWithoutSettingThrows() {
        try (MLAKVCache cache = new MLAKVCache(1, 16, 64, 0, 2, DataType.FLOAT)) {
            assertThrows(Exception.class, () -> cache.getKvDownProj(0),
                    "Should throw when weights not set");
        }
    }

    @Test
    void testSetLayerWeightsInvalidLayerIdx() {
        try (MLAKVCache cache = new MLAKVCache(1, 16, 64, 0, 2, DataType.FLOAT)) {
            INDArray weight = Nd4j.zeros(DataType.FLOAT, 64, 128);
            assertThrows(IllegalArgumentException.class, () -> cache.setLayerWeights(-1, weight));
            assertThrows(IllegalArgumentException.class, () -> cache.setLayerWeights(2, weight));
        }
    }

    @Test
    void testSetLayerWeightsWrongRank() {
        try (MLAKVCache cache = new MLAKVCache(1, 16, 64, 0, 2, DataType.FLOAT)) {
            INDArray weight3d = Nd4j.zeros(DataType.FLOAT, 64, 128, 4);
            assertThrows(IllegalArgumentException.class, () -> cache.setLayerWeights(0, weight3d));
        }
    }

    @Test
    void testSetLayerWeightsWrongDim0() {
        try (MLAKVCache cache = new MLAKVCache(1, 16, 64, 0, 2, DataType.FLOAT)) {
            INDArray weight = Nd4j.zeros(DataType.FLOAT, 32, 128); // dim0 should be 64
            assertThrows(IllegalArgumentException.class, () -> cache.setLayerWeights(0, weight));
        }
    }

    // ==================== Memory Calculations Tests ====================

    @Test
    void testMemoryBytes() {
        int batchSize = 1, maxSeqLen = 1024, latentDim = 512, ropeHeadDim = 64, numLayers = 60;
        try (MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT)) {
            long expected = (long) batchSize * maxSeqLen * latentDim * 4
                    + (long) batchSize * maxSeqLen * ropeHeadDim * 4;
            assertEquals(expected, cache.getMemoryBytes());
        }
    }

    @Test
    void testMemoryBytesWithoutRoPE() {
        int batchSize = 1, maxSeqLen = 1024, latentDim = 512, ropeHeadDim = 0, numLayers = 60;
        try (MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT)) {
            long expected = (long) batchSize * maxSeqLen * latentDim * 4;
            assertEquals(expected, cache.getMemoryBytes());
        }
    }

    @Test
    void testMemorySavingsRatio() {
        // DeepSeek-V2-like config: 60 layers, 8 KV heads, headDim=128, latentDim=512
        int batchSize = 1, maxSeqLen = 1024, latentDim = 512, ropeHeadDim = 64, numLayers = 60;
        int numKvHeads = 8, headDim = 128;
        try (MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT)) {
            double ratio = cache.getMemorySavingsRatio(numKvHeads, headDim);
            // Standard: 60 * 2 * 1 * 1024 * 8 * 128 * 4 = very large
            // MLA: 1 * 1024 * (512 + 64) * 4 = much smaller
            assertTrue(ratio > 1.0, "MLA should save memory compared to standard KV cache, got ratio=" + ratio);

            long standardBytes = cache.getEquivalentStandardMemoryBytes(numKvHeads, headDim);
            long mlaBytes = cache.getMemoryBytes();
            assertEquals((double) standardBytes / mlaBytes, ratio, 1e-10);
        }
    }

    @Test
    void testMemorySavingsRatioTypicalConfig() {
        // With typical config, ratio should be significant (4-8x)
        int batchSize = 1, maxSeqLen = 2048, latentDim = 512, ropeHeadDim = 64, numLayers = 60;
        int numKvHeads = 8, headDim = 128;
        try (MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT)) {
            double ratio = cache.getMemorySavingsRatio(numKvHeads, headDim);
            // 60 * 2 * 8 * 128 / (512 + 64) = 122880 / 576 ~ 213x
            // That is very high because we share 1 latent across all layers
            assertTrue(ratio > 4.0, "Expected significant savings, got ratio=" + ratio);
        }
    }

    // ==================== Reset Tests ====================

    @Test
    void testResetZerosCache() {
        int batchSize = 1, maxSeqLen = 8, latentDim = 4, ropeHeadDim = 2, numLayers = 1;
        try (MLAKVCache cache = new MLAKVCache(batchSize, maxSeqLen, latentDim, ropeHeadDim, numLayers, DataType.FLOAT)) {
            // Append some data
            INDArray newLatent = Nd4j.ones(DataType.FLOAT, batchSize, 2, latentDim).mul(5.0);
            INDArray newRope = Nd4j.ones(DataType.FLOAT, batchSize, 2, ropeHeadDim).mul(3.0);
            cache.append(newLatent, newRope, 0);

            // Verify data is present
            assertTrue(cache.getLatentCache().getDouble(0, 0, 0) != 0.0);
            assertTrue(cache.getRopeCache().getDouble(0, 0, 0) != 0.0);

            // Reset
            cache.reset();

            // All values should be zero
            assertEquals(0.0, cache.getLatentCache().sumNumber().doubleValue(), 1e-5,
                    "Latent cache should be all zeros after reset");
            assertEquals(0.0, cache.getRopeCache().sumNumber().doubleValue(), 1e-5,
                    "RoPE cache should be all zeros after reset");
        }
    }

    @Test
    void testResetWithoutRoPE() {
        try (MLAKVCache cache = new MLAKVCache(1, 8, 4, 0, 1, DataType.FLOAT)) {
            INDArray newLatent = Nd4j.ones(DataType.FLOAT, 1, 2, 4).mul(5.0);
            cache.append(newLatent, null, 0);
            cache.reset();
            assertEquals(0.0, cache.getLatentCache().sumNumber().doubleValue(), 1e-5);
        }
    }

    // ==================== Close Tests ====================

    @Test
    void testCloseDoesNotThrow() {
        MLAKVCache cache = new MLAKVCache(1, 8, 4, 2, 1, DataType.FLOAT);
        assertDoesNotThrow(cache::close);
    }

    @Test
    void testAutoCloseableInTryWithResources() {
        assertDoesNotThrow(() -> {
            try (MLAKVCache cache = new MLAKVCache(1, 8, 4, 0, 1, DataType.FLOAT)) {
                cache.append(Nd4j.ones(DataType.FLOAT, 1, 1, 4), null, 0);
            }
        });
    }
}
