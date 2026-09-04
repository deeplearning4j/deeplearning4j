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

import org.eclipse.deeplearning4j.llm.generation.kvcache.PagedKVCache;
import org.eclipse.deeplearning4j.llm.generation.kvcache.PerLayerKVPolicy;
import org.eclipse.deeplearning4j.llm.generation.kvcache.PerLayerPagedKVCache;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the native-append PagedKVCache: correctness of the batched
 * {@code paged_kv_append} routing vs a dense reference, sliding-window eviction
 * safety (page-table shift + no double-free), zero-copy prefix sharing refcounts,
 * and the slack (reserved-block) band.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public class TestNativePagedKvAppend extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helper: append and read back the full sequence from the pool
    // ─────────────────────────────────────────────────────────────────────────

    private static INDArray readBackSequence(PagedKVCache cache, int seqIdx) {
        int len = cache.getSequenceLength(seqIdx);
        INDArray out = Nd4j.create(DataType.FLOAT, 1, len, cache.getNumKvHeads(), cache.getHeadDim());

        INDArray tableArr = cache.getPageTableArray(seqIdx);
        int[] table = tableArr.toIntVector();
        tableArr.close();

        int blockSize = cache.getBlockSize();
        int pos = 0;
        for (int logical = 0; logical < table.length; logical++) {
            int physical = table[logical];
            if (physical < 0) break;
            int tokensInBlock = Math.min(blockSize, len - pos);
            // blockPool[physical, 0:tokensInBlock, :, :] is [tokensInBlock, kvHeads, headDim]
            INDArray blockKeys = cache.getKeyBlockPool().get(
                    NDArrayIndex.point(physical), NDArrayIndex.interval(0, tokensInBlock),
                    NDArrayIndex.all(), NDArrayIndex.all());
            out.get(NDArrayIndex.point(0), NDArrayIndex.interval(pos, pos + tokensInBlock),
                    NDArrayIndex.all(), NDArrayIndex.all()).assign(blockKeys);
            pos += tokensInBlock;
        }
        return out;
    }

    private static INDArray makeKeys(int len, int kvHeads, int headDim, float seed) {
        INDArray keys = Nd4j.create(DataType.FLOAT, 1, len, kvHeads, headDim);
        for (int t = 0; t < len; t++) {
            for (int h = 0; h < kvHeads; h++) {
                for (int d = 0; d < headDim; d++) {
                    keys.putScalar(new int[]{0, t, h, d}, seed + t * 0.01f + h * 0.1f + d * 0.001f);
                }
            }
        }
        return keys;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Native append correctness: values land at the right physical slots
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAppendRoundTripExactValues(Nd4jBackend backend) {
        int blockSize = 8;
        PagedKVCache cache = new PagedKVCache(1, 100, 2, 4, blockSize, DataType.FLOAT, 1.0);
        try {
            // Multi-block append crossing two block boundaries
            INDArray keys = makeKeys(20, 2, 4, 1.0f);

            cache.append(0, keys, keys);

            assertEquals(20, cache.getSequenceLength(0));
            int allocated = cache.getNumAllocatedBlocks(0);
            assertEquals((20 + blockSize - 1) / blockSize, allocated, "should span 3 blocks");

            INDArray readKeys = readBackSequence(cache, 0);
            assertArrayEquals(keys.shape(), readKeys.shape());
            for (int t = 0; t < 20; t++) {
                for (int h = 0; h < 2; h++) {
                    for (int d = 0; d < 4; d++) {
                        assertEquals(keys.getFloat(0, t, h, d), readKeys.getFloat(0, t, h, d), 1e-6f,
                                "key mismatch at [t=" + t + ",h=" + h + ",d=" + d + "]");
                    }
                }
            }
        } finally {
            cache.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIncrementalSingleTokenAppends(Nd4jBackend backend) {
        int blockSize = 4;
        PagedKVCache cache = new PagedKVCache(1, 40, 2, 3, blockSize, DataType.FLOAT, 1.0);
        try {
            for (int step = 0; step < 10; step++) {
                INDArray keys = makeKeys(1, 2, 3, step * 1.0f);
                cache.append(0, keys, keys);
                assertEquals(step + 1, cache.getSequenceLength(0));
            }

            // Verify token 7 landed correctly (block 1, offset 3)
            INDArray readKeys = readBackSequence(cache, 0);
            // Step-7 append was a 1-token tensor with seed 7.0, so token value at
            // [0, 7, 0, 0] = 7.0 + 0*0.01 (t=0 within the append)
            assertEquals(7.0f, readKeys.getFloat(0, 7, 0, 0), 1e-6f);
            assertEquals(7.0f + 0.1f, readKeys.getFloat(0, 7, 1, 0), 1e-6f);
        } finally {
            cache.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAppendRank3Input(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(1, 32, 2, 4, 8, DataType.FLOAT, 1.0);
        try {
            // [newLen, kvHeads, headDim] without the batch dim
            INDArray keys = makeKeys(5, 2, 4, 2.0f).get(NDArrayIndex.point(0));

            cache.append(0, keys, keys);
            assertEquals(5, cache.getSequenceLength(0));

            INDArray readKeys = readBackSequence(cache, 0);
            assertEquals(2.0f, readKeys.getFloat(0, 0, 0, 0), 1e-6f);
            assertEquals(2.0f + 4 * 0.01f, readKeys.getFloat(0, 4, 0, 0), 1e-6f);
        } finally {
            cache.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAppendNonContiguousInput(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(1, 32, 2, 4, 8, DataType.FLOAT, 1.0);
        try {
            // A genuinely non-contiguous view: single permute from a BHSD source to a
            // BSHD logical layout — exactly what UnifiedKvCacheManager feeds append
            // for prefill outputs. Shape [1, 6, 2, 4], strided buffer.
            INDArray bhsd = Nd4j.create(DataType.FLOAT, 1, 2, 6, 4);   // BHSD source
            // Logical BSHD content: value at [0, t, h, d] = seed + t*0.01 + h*0.1 + d*0.001
            float seed = 5.0f;
            for (int t = 0; t < 6; t++) {
                for (int h = 0; h < 2; h++) {
                    for (int d = 0; d < 4; d++) {
                        bhsd.putScalar(new int[]{0, h, t, d}, seed + t * 0.01f + h * 0.1f + d * 0.001f);
                    }
                }
            }
            INDArray bshdView = bhsd.permute(0, 2, 1, 3);   // [1, 6, 2, 4] logical, strided
            assertFalse(bshdView.ordering() == 'c' && Shape.strideDescendingCAscendingF(bshdView),
                    "test precondition: input should be a non-contiguous view");

            cache.append(0, bshdView, bshdView);
            assertEquals(6, cache.getSequenceLength(0));

            INDArray readKeys = readBackSequence(cache, 0);
            for (int t = 0; t < 6; t++) {
                assertEquals(seed + t * 0.01f, readKeys.getFloat(0, t, 0, 0), 1e-6f,
                        "non-contiguous append must preserve logical token content, mismatch at token " + t);
            }
        } finally {
            cache.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAppendExceedingMaxBlocksThrows(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(1, 16, 2, 4, 8, DataType.FLOAT, 1.0);
        try {
            INDArray keys = makeKeys(16, 2, 4, 1.0f);
            cache.append(0, keys, keys);
            assertEquals(16, cache.getSequenceLength(0));

            INDArray more = makeKeys(1, 2, 4, 9.0f);
            assertThrows(IllegalStateException.class, () -> cache.append(0, more, more),
                    "append past maxSeqLen must throw");
            assertEquals(16, cache.getSequenceLength(0));
        } finally {
            cache.close();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Free/reuse with slack band
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFreeSequenceReturnsBlocksAndReuse(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(2, 32, 2, 4, 8, DataType.FLOAT, 1.0);
        try {
            int total = cache.getNumFreeBlocks();
            assertEquals(cache.getNumBlocks(), total);

            INDArray keys = makeKeys(10, 2, 4, 1.0f);
            cache.append(0, keys, keys);
            assertEquals(total - 2, cache.getNumFreeBlocks());

            cache.freeSequence(0);
            assertEquals(total, cache.getNumFreeBlocks(), "all blocks free again");
            assertEquals(0, cache.getSequenceLength(0));

            cache.append(0, keys, keys);
            assertEquals(10, cache.getSequenceLength(0));
        } finally {
            cache.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSlackBandRespectsLimit(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(2, 64, 2, 4, 8, DataType.FLOAT, 1.0);
        try {
            cache.setReservedBlockLimit(3);
            assertEquals(3, cache.getReservedBlockLimit());

            INDArray keys = makeKeys(32, 2, 4, 1.0f);
            cache.append(0, keys, keys);   // 4 blocks
            cache.freeSequence(0);

            assertTrue(cache.getNumReservedBlocks() <= 3,
                    "reserved band must respect its limit, got " + cache.getNumReservedBlocks());
            assertEquals(cache.getNumBlocks(), cache.getNumFreeBlocks(),
                    "free count includes reserved band");
        } finally {
            cache.close();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Zero-copy prefix sharing refcounts
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSharePrefixBlocksZeroCopy(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(3, 64, 2, 4, 8, DataType.FLOAT, 1.0);
        try {
            int total = cache.getNumFreeBlocks();

            INDArray keys = makeKeys(16, 2, 4, 1.0f);
            cache.append(0, keys, keys);   // 2 blocks
            assertEquals(total - 2, cache.getNumFreeBlocks());

            // Share seq0's 2 blocks into empty seq1 — no new allocation
            int shared = cache.sharePrefixBlocks(0, 2, 1);
            assertEquals(2, shared);
            assertEquals(total - 2, cache.getNumFreeBlocks(), "sharing must not allocate");
            assertEquals(16, cache.getSequenceLength(1));

            // Both page tables point at the same physical blocks with refcount 2
            INDArray t0 = cache.getPageTableArray(0);
            INDArray t1 = cache.getPageTableArray(1);
            int[] table0 = t0.toIntVector();
            int[] table1 = t1.toIntVector();
            t0.close();
            t1.close();
            assertArrayEquals(table0, table1, "shared prefix page tables must match");
            assertEquals(2, cache.getBlockRefCount(table0[0]));

            // seq1 continues appending — new blocks, not shared
            INDArray more = makeKeys(4, 2, 4, 50.0f);
            cache.append(1, more, more);
            assertEquals(20, cache.getSequenceLength(1));
            assertEquals(total - 3, cache.getNumFreeBlocks(), "only one new block allocated");

            // Free seq1: shared blocks drop to refcount 1, private block freed
            cache.freeSequence(1);
            assertEquals(1, cache.getBlockRefCount(table0[0]), "shared block still owned by seq0");
            assertEquals(total - 2, cache.getNumFreeBlocks());

            // Free seq0: shared blocks finally return to the pool
            cache.freeSequence(0);
            assertEquals(total, cache.getNumFreeBlocks());
            assertEquals(0, cache.getBlockRefCount(table0[0]));
        } finally {
            cache.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSharePrefixRejectsNonEmptyDestination(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(2, 64, 2, 4, 8, DataType.FLOAT, 1.0);
        try {
            INDArray keys = makeKeys(8, 2, 4, 1.0f);
            cache.append(0, keys, keys);
            cache.append(1, keys, keys);

            assertThrows(IllegalStateException.class, () -> cache.sharePrefixBlocks(0, 1, 1),
                    "destination must be empty");
            assertThrows(IllegalArgumentException.class, () -> cache.sharePrefixBlocks(0, 1, 0),
                    "self-sharing is rejected");
        } finally {
            cache.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSharedBlockDataReadableFromBothSequences(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(2, 64, 2, 4, 8, DataType.FLOAT, 1.0);
        try {
            INDArray keys = makeKeys(10, 2, 4, 7.0f);
            cache.append(0, keys, keys);
            cache.sharePrefixBlocks(0, cache.getNumAllocatedBlocks(0), 1);

            INDArray readKeys1 = readBackSequence(cache, 1);
            for (int t = 0; t < 10; t++) {
                assertEquals(keys.getFloat(0, t, 0, 0), readKeys1.getFloat(0, t, 0, 0), 1e-6f,
                        "shared data must read identically from the forking sequence");
            }
        } finally {
            cache.close();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. evictOldestBlocks: page-table shift + no double-free
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEvictOldestBlocksShiftsPageTable(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(1, 64, 2, 4, 8, DataType.FLOAT, 1.0);
        try {
            INDArray keys = makeKeys(24, 2, 4, 1.0f);
            cache.append(0, keys, keys);   // 3 blocks

            INDArray before = cache.getPageTableArray(0);
            int[] beforeTable = before.toIntVector();
            before.close();

            int evicted = cache.evictOldestBlocks(0, 1);
            assertEquals(1, evicted);
            assertEquals(16, cache.getSequenceLength(0), "length drops by blockSize");

            INDArray after = cache.getPageTableArray(0);
            int[] afterTable = after.toIntVector();
            after.close();
            assertEquals(2, afterTable.length);
            assertEquals(beforeTable[1], afterTable[0], "logical block 0 must now be the old block 1");
            assertEquals(beforeTable[2], afterTable[1], "logical block 1 must now be the old block 2");

            // No double-free: freeSequence must succeed cleanly
            cache.freeSequence(0);
            assertEquals(cache.getNumBlocks(), cache.getNumFreeBlocks());
        } finally {
            cache.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEvictMoreThanAllocatedIsCapped(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(1, 32, 2, 4, 8, DataType.FLOAT, 1.0);
        try {
            INDArray keys = makeKeys(8, 2, 4, 1.0f);
            cache.append(0, keys, keys);   // 1 block

            assertEquals(1, cache.evictOldestBlocks(0, 5), "evict count is capped at allocated");
            assertEquals(0, cache.getSequenceLength(0));
            assertEquals(cache.getNumBlocks(), cache.getNumFreeBlocks());
        } finally {
            cache.close();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 5. PerLayerPagedKVCache sliding window (the double-free regression)
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSlidingWindowNeverExceedsAndNeverDoubleFrees(Nd4jBackend backend) {
        int windowSize = 16;
        int blockSize = 8;
        int numLayers = 2;

        PerLayerKVPolicy policy = PerLayerKVPolicy.uniformSlidingWindow(numLayers, windowSize, blockSize, DataType.FLOAT);
        PerLayerPagedKVCache cache = new PerLayerPagedKVCache(policy, 1, 2, 4, blockSize);
        try {
            // Stream well past the window: 60 tokens through a 16-token window
            for (int step = 0; step < 60; step++) {
                INDArray keys = makeKeys(1, 2, 4, step * 1.0f);
                for (int layer = 0; layer < numLayers; layer++) {
                    cache.append(layer, 0, keys, keys);
                    int len = cache.getLayerCache(layer).getSequenceLength(0);
                    assertTrue(len <= windowSize,
                            "layer " + layer + " length " + len + " exceeds window " + windowSize
                                    + " at step " + step);
                }
            }

            // Everything frees cleanly at the end — the old code double-freed here
            assertDoesNotThrow(() -> cache.freeSequence(0));
            assertEquals(cache.getTotalBlocks(), cache.getTotalFreeBlocks(),
                    "all blocks must be free after freeSequence (no double-free, no leak)");
        } finally {
            cache.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSlidingWindowSurvivorDataPreserved(Nd4jBackend backend) {
        // After eviction, the SURVIVING tokens' data must still be readable through
        // the (shifted) page table — the old code left entries pointing at freed blocks.
        int windowSize = 16;
        int blockSize = 8;
        PerLayerKVPolicy policy = PerLayerKVPolicy.uniformSlidingWindow(1, windowSize, blockSize, DataType.FLOAT);
        PerLayerPagedKVCache cache = new PerLayerPagedKVCache(policy, 1, 2, 4, blockSize);
        try {
            PagedKVCache layer0 = cache.getLayerCache(0);

            // Fill to exactly the window
            for (int step = 0; step < windowSize; step++) {
                INDArray keys = makeKeys(1, 2, 4, step * 1.0f);
                cache.append(0, 0, keys, keys);
            }
            assertEquals(windowSize, layer0.getSequenceLength(0));

            // Append one more -> eviction fires, keeping the append inside the window
            INDArray next = makeKeys(1, 2, 4, 16.0f);
            cache.append(0, 0, next, next);
            int len = layer0.getSequenceLength(0);
            assertTrue(len <= windowSize, "length must stay within window, got " + len);

            // The newest token must be readable at the tail via the shifted page table
            INDArray tableArr = layer0.getPageTableArray(0);
            int[] table = tableArr.toIntVector();
            tableArr.close();
            int lastPos = len - 1;
            int physical = table[lastPos / blockSize];
            int offset = lastPos % blockSize;
            float tail = layer0.getKeyBlockPool().getFloat(physical, offset, 0, 0);
            assertEquals(16.0f, tail, 1e-6f,
                    "newest token must be readable after window eviction");
        } finally {
            cache.close();
        }
    }
}
