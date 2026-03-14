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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link QuantizedPagedKVCache} -- structure and lifecycle tests only.
 * Does not test append/dequantize since those require native quantization ops.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
class TestQuantizedKVCache {

    private static final int MAX_BATCH = 4;
    private static final int MAX_SEQ = 256;
    private static final int NUM_KV_HEADS = 8;
    private static final int HEAD_DIM = 64;

    // ==================== Creation Tests ====================

    @ParameterizedTest
    @EnumSource(QuantizedPagedKVCache.QuantFormat.class)
    void testCreationWithEachQuantFormat(QuantizedPagedKVCache.QuantFormat format) {
        try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                MAX_BATCH, MAX_SEQ, NUM_KV_HEADS, HEAD_DIM, format, DataType.FLOAT)) {
            assertEquals(format, cache.getQuantFormat());
            assertEquals(MAX_BATCH, cache.getMaxBatchSize());
            assertEquals(NUM_KV_HEADS, cache.getNumKvHeads());
            assertEquals(HEAD_DIM, cache.getHeadDim());
            assertEquals(DataType.FLOAT, cache.getOriginalDataType());
            assertEquals(QuantizedPagedKVCache.DEFAULT_BLOCK_SIZE, cache.getBlockSize());
        }
    }

    @Test
    void testCreationWithCustomBlockSizeAndPoolFactor() {
        int blockSize = 32;
        double poolFactor = 1.5;
        try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                MAX_BATCH, MAX_SEQ, NUM_KV_HEADS, HEAD_DIM,
                QuantizedPagedKVCache.QuantFormat.INT8,
                blockSize, DataType.FLOAT, poolFactor)) {
            assertEquals(blockSize, cache.getBlockSize());

            // Verify pool is larger due to pool factor
            int maxBlocksPerSeq = (MAX_SEQ + blockSize - 1) / blockSize;
            int minBlocks = MAX_BATCH * maxBlocksPerSeq;
            int expectedBlocks = (int) Math.ceil(minBlocks * poolFactor);
            assertEquals(expectedBlocks, cache.getNumBlocks());
        }
    }

    @Test
    void testDefaultConstructorPoolSize() {
        try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                MAX_BATCH, MAX_SEQ, NUM_KV_HEADS, HEAD_DIM,
                QuantizedPagedKVCache.QuantFormat.INT8, DataType.FLOAT)) {
            // Default pool factor is 1.2
            int blockSize = QuantizedPagedKVCache.DEFAULT_BLOCK_SIZE;
            int maxBlocksPerSeq = (MAX_SEQ + blockSize - 1) / blockSize;
            int minBlocks = MAX_BATCH * maxBlocksPerSeq;
            int expectedBlocks = (int) Math.ceil(minBlocks * 1.2);
            assertEquals(expectedBlocks, cache.getNumBlocks());

            // All blocks should be free initially
            assertEquals(expectedBlocks, cache.getNumFreeBlocks());
        }
    }

    // ==================== QuantFormat Enum Tests ====================

    @Test
    void testQuantFormatNativeCodes() {
        assertEquals(0, QuantizedPagedKVCache.QuantFormat.INT8.getNativeCode());
        assertEquals(1, QuantizedPagedKVCache.QuantFormat.FP8_E4M3.getNativeCode());
        assertEquals(2, QuantizedPagedKVCache.QuantFormat.FP8_E5M2.getNativeCode());
        assertEquals(3, QuantizedPagedKVCache.QuantFormat.INT4.getNativeCode());
    }

    @Test
    void testQuantFormatValuesCount() {
        assertEquals(4, QuantizedPagedKVCache.QuantFormat.values().length);
    }

    // ==================== Pool Shape Tests ====================

    @Test
    void testPoolShapes() {
        int blockSize = 16;
        try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                2, 64, NUM_KV_HEADS, HEAD_DIM,
                QuantizedPagedKVCache.QuantFormat.INT8,
                blockSize, DataType.FLOAT, 1.0)) {
            int numBlocks = cache.getNumBlocks();

            // Key/Value block pools: [numBlocks, blockSize, numKvHeads, headDim] in INT8
            INDArray keyPool = cache.getKeyBlockPool();
            assertNotNull(keyPool);
            assertArrayEquals(new long[]{numBlocks, blockSize, NUM_KV_HEADS, HEAD_DIM}, keyPool.shape());
            assertEquals(DataType.INT8, keyPool.dataType());

            INDArray valPool = cache.getValueBlockPool();
            assertNotNull(valPool);
            assertArrayEquals(new long[]{numBlocks, blockSize, NUM_KV_HEADS, HEAD_DIM}, valPool.shape());
            assertEquals(DataType.INT8, valPool.dataType());

            // Scale pools: [numBlocks, blockSize, numKvHeads] in FLOAT32
            INDArray keyScales = cache.getKeyScalePool();
            assertNotNull(keyScales);
            assertArrayEquals(new long[]{numBlocks, blockSize, NUM_KV_HEADS}, keyScales.shape());
            assertEquals(DataType.FLOAT, keyScales.dataType());

            INDArray valScales = cache.getValueScalePool();
            assertNotNull(valScales);
            assertArrayEquals(new long[]{numBlocks, blockSize, NUM_KV_HEADS}, valScales.shape());
            assertEquals(DataType.FLOAT, valScales.dataType());
        }
    }

    // ==================== Sequence Management Tests ====================

    @Test
    void testSequenceLengthInitiallyZero() {
        try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                MAX_BATCH, MAX_SEQ, NUM_KV_HEADS, HEAD_DIM,
                QuantizedPagedKVCache.QuantFormat.INT8, DataType.FLOAT)) {
            for (int i = 0; i < MAX_BATCH; i++) {
                assertEquals(0, cache.getSequenceLength(i), "Sequence " + i + " should start at length 0");
            }
        }
    }

    @Test
    void testNumAllocatedBlocksInitiallyZero() {
        try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                MAX_BATCH, MAX_SEQ, NUM_KV_HEADS, HEAD_DIM,
                QuantizedPagedKVCache.QuantFormat.INT8, DataType.FLOAT)) {
            for (int i = 0; i < MAX_BATCH; i++) {
                assertEquals(0, cache.getNumAllocatedBlocks(i));
            }
        }
    }

    @Test
    void testFreeSequenceOnEmptySlot() {
        try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                MAX_BATCH, MAX_SEQ, NUM_KV_HEADS, HEAD_DIM,
                QuantizedPagedKVCache.QuantFormat.INT8, DataType.FLOAT)) {
            int initialFree = cache.getNumFreeBlocks();
            // Freeing an already-empty sequence should be a no-op
            cache.freeSequence(0);
            assertEquals(initialFree, cache.getNumFreeBlocks());
            assertEquals(0, cache.getSequenceLength(0));
        }
    }

    @Test
    void testFreeSequenceOutOfRange() {
        try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                MAX_BATCH, MAX_SEQ, NUM_KV_HEADS, HEAD_DIM,
                QuantizedPagedKVCache.QuantFormat.INT8, DataType.FLOAT)) {
            // Should not throw for out-of-range (implementation returns early)
            assertDoesNotThrow(() -> cache.freeSequence(-1));
            assertDoesNotThrow(() -> cache.freeSequence(MAX_BATCH));
        }
    }

    @Test
    void testReassignSlot() {
        try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                MAX_BATCH, MAX_SEQ, NUM_KV_HEADS, HEAD_DIM,
                QuantizedPagedKVCache.QuantFormat.INT8, DataType.FLOAT)) {
            int initialFree = cache.getNumFreeBlocks();
            // reassignSlot delegates to freeSequence
            cache.reassignSlot(0);
            assertEquals(initialFree, cache.getNumFreeBlocks());
            assertEquals(0, cache.getSequenceLength(0));
        }
    }

    // ==================== Page Table Tests ====================

    @Test
    void testPageTableInitiallyEmpty() {
        try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                MAX_BATCH, MAX_SEQ, NUM_KV_HEADS, HEAD_DIM,
                QuantizedPagedKVCache.QuantFormat.INT8, DataType.FLOAT)) {
            // With 0 allocated blocks, page table should be empty (length 0)
            INDArray pageTable = cache.getPageTableArray(0);
            assertNotNull(pageTable);
            assertEquals(0, pageTable.length(),
                    "Page table should be empty for a sequence with no allocated blocks");
        }
    }

    // ==================== Close Tests ====================

    @Test
    void testCloseDoesNotThrow() {
        QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                2, 64, NUM_KV_HEADS, HEAD_DIM,
                QuantizedPagedKVCache.QuantFormat.FP8_E4M3, DataType.FLOAT);
        assertDoesNotThrow(cache::close);
    }

    @Test
    void testAutoCloseableInTryWithResources() {
        assertDoesNotThrow(() -> {
            try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                    2, 64, NUM_KV_HEADS, HEAD_DIM,
                    QuantizedPagedKVCache.QuantFormat.INT4, DataType.HALF)) {
                // Just verify it constructs and closes cleanly
                assertNotNull(cache.getKeyBlockPool());
            }
        });
    }

    // ==================== toString Tests ====================

    @Test
    void testToStringContainsFormat() {
        try (QuantizedPagedKVCache cache = new QuantizedPagedKVCache(
                2, 64, NUM_KV_HEADS, HEAD_DIM,
                QuantizedPagedKVCache.QuantFormat.FP8_E5M2, DataType.FLOAT)) {
            String str = cache.toString();
            assertNotNull(str);
            assertTrue(str.contains("FP8_E5M2"), "toString should contain the quant format");
            assertTrue(str.contains("QuantizedPagedKVCache"), "toString should contain class name");
        }
    }
}
