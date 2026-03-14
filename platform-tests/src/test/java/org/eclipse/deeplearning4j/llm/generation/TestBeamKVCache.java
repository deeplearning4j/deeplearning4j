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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for BeamKVCacheManager and KVCachePrefixTree.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
class TestBeamKVCache {

    private PagedKVCache kvCache;

    private static final int BLOCK_SIZE = 4;
    private static final int NUM_KV_HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final int MAX_BATCH_SIZE = 8;
    private static final int MAX_SEQ_LEN = 64;

    @BeforeEach
    void setUp() {
        kvCache = new PagedKVCache(MAX_BATCH_SIZE, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM,
                BLOCK_SIZE, DataType.FLOAT, 2.0);
    }

    @AfterEach
    void tearDown() {
        if (kvCache != null) {
            kvCache.close();
        }
    }

    // ==================== KVCachePrefixTree Tests ====================

    @Test
    void testPrefixTree_construction() {
        KVCachePrefixTree tree = new KVCachePrefixTree(4);
        assertEquals(4, tree.getBlockSize());
        assertEquals(0, tree.getNodeCount());
        assertEquals(0, tree.getSharedBlockCount());
    }

    @Test
    void testPrefixTree_invalidBlockSize() {
        assertThrows(IllegalArgumentException.class, () -> new KVCachePrefixTree(0));
        assertThrows(IllegalArgumentException.class, () -> new KVCachePrefixTree(-1));
    }

    @Test
    void testPrefixTree_registerSingleSequence() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        AtomicInteger nextBlock = new AtomicInteger(0);

        int[] tokens = {10, 20, 30, 40, 50, 60, 70, 80};
        List<KVCachePrefixTree.BlockMapping> mappings = tree.registerSequence(tokens, nextBlock::getAndIncrement);

        assertEquals(2, mappings.size(), "8 tokens / block_size 4 = 2 blocks");
        assertEquals(2, tree.getNodeCount());
        assertEquals(0, tree.getSharedBlockCount(), "Single sequence has no shared blocks");

        // Verify mappings
        assertEquals(0, mappings.get(0).getLogicalBlockIndex());
        assertEquals(1, mappings.get(1).getLogicalBlockIndex());
        assertFalse(mappings.get(0).isShared());
        assertFalse(mappings.get(1).isShared());
    }

    @Test
    void testPrefixTree_sharedPrefix() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        AtomicInteger nextBlock = new AtomicInteger(0);

        int[] seqA = {10, 20, 30, 40, 50, 60, 70, 80};
        int[] seqB = {10, 20, 30, 40, 90, 91, 92, 93};

        tree.registerSequence(seqA, nextBlock::getAndIncrement);
        List<KVCachePrefixTree.BlockMapping> mappingsB = tree.registerSequence(seqB, nextBlock::getAndIncrement);

        // First block [10,20,30,40] is shared
        assertTrue(mappingsB.get(0).isShared(), "First block should be shared");
        // Second block [90,91,92,93] is unique
        assertFalse(mappingsB.get(1).isShared(), "Second block should not be shared");

        assertEquals(3, tree.getNodeCount(), "Should have 3 nodes (shared prefix + 2 unique suffixes)");
        assertEquals(1, tree.getSharedBlockCount(), "One block is shared");
    }

    @Test
    void testPrefixTree_unregisterFreesBlocks() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        AtomicInteger nextBlock = new AtomicInteger(0);

        int[] tokens = {10, 20, 30, 40, 50, 60, 70, 80};
        tree.registerSequence(tokens, nextBlock::getAndIncrement);

        List<Integer> freed = tree.unregisterSequence(tokens);
        assertEquals(2, freed.size(), "Unregistering sole reference should free both blocks");
        assertEquals(0, tree.getNodeCount(), "All nodes should be removed");
    }

    @Test
    void testPrefixTree_unregisterSharedDoesNotFreeShared() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        AtomicInteger nextBlock = new AtomicInteger(0);

        int[] seqA = {10, 20, 30, 40, 50, 60, 70, 80};
        int[] seqB = {10, 20, 30, 40, 90, 91, 92, 93};

        tree.registerSequence(seqA, nextBlock::getAndIncrement);
        tree.registerSequence(seqB, nextBlock::getAndIncrement);

        // Unregister seqA -- shared block stays, seqA's unique block is freed
        List<Integer> freed = tree.unregisterSequence(seqA);
        assertEquals(1, freed.size(), "Only the unique suffix block should be freed");
        assertEquals(0, tree.getSharedBlockCount(), "After one unregister, the prefix block is no longer shared");
        assertEquals(2, tree.getNodeCount(), "Shared prefix node + seqB suffix node should remain");
    }

    @Test
    void testPrefixTree_needsCopyOnWrite() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        AtomicInteger nextBlock = new AtomicInteger(0);

        int[] tokens = {10, 20, 30, 40};
        tree.registerSequence(tokens, nextBlock::getAndIncrement);

        // Single reference -- no CoW needed
        assertFalse(tree.needsCopyOnWrite(tokens));

        // Register again to make it shared
        tree.registerSequence(tokens, nextBlock::getAndIncrement);
        assertTrue(tree.needsCopyOnWrite(tokens), "Shared block needs copy-on-write");
    }

    @Test
    void testPrefixTree_lookupPrefix() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        AtomicInteger nextBlock = new AtomicInteger(0);

        int[] tokens = {10, 20, 30, 40, 50, 60, 70, 80};
        tree.registerSequence(tokens, nextBlock::getAndIncrement);

        // Full prefix lookup
        List<Integer> blockIds = tree.lookupPrefix(tokens);
        assertEquals(2, blockIds.size());

        // Partial prefix lookup
        blockIds = tree.lookupPrefix(new int[]{10, 20, 30, 40});
        assertEquals(1, blockIds.size());

        // Non-matching prefix
        blockIds = tree.lookupPrefix(new int[]{99, 99, 99, 99});
        assertEquals(0, blockIds.size());
    }

    @Test
    void testPrefixTree_partialBlock() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        AtomicInteger nextBlock = new AtomicInteger(0);

        // 6 tokens = 1 full block + 1 partial block (2 valid tokens)
        int[] tokens = {10, 20, 30, 40, 50, 60};
        List<KVCachePrefixTree.BlockMapping> mappings = tree.registerSequence(tokens, nextBlock::getAndIncrement);

        assertEquals(2, mappings.size());
        assertEquals(4, mappings.get(0).getValidTokenCount());
        assertEquals(2, mappings.get(1).getValidTokenCount());
    }

    @Test
    void testPrefixTree_toString() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        String str = tree.toString();
        assertTrue(str.contains("KVCachePrefixTree"));
        assertTrue(str.contains("blockSize=" + BLOCK_SIZE));
    }

    @Test
    void testPrefixTree_memorySaved() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        AtomicInteger nextBlock = new AtomicInteger(0);

        int[] tokens = {10, 20, 30, 40};
        tree.registerSequence(tokens, nextBlock::getAndIncrement);
        tree.registerSequence(tokens, nextBlock::getAndIncrement);

        // 1 shared block with refCount=2, saves (2-1)=1 block copy
        long saved = tree.getMemorySavedBytes(1024);
        assertEquals(1024, saved, "Should save 1 block's worth of memory");
    }

    // ==================== BeamKVCacheManager Tests ====================

    @Test
    void testBeamManager_construction() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 4);
        assertEquals(4, manager.getNumBeams());
        assertSame(kvCache, manager.getKvCache());
        assertSame(tree, manager.getPrefixTree());
    }

    @Test
    void testBeamManager_nullKvCacheThrows() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        assertThrows(IllegalArgumentException.class, () -> new BeamKVCacheManager(null, tree, 4));
    }

    @Test
    void testBeamManager_nullPrefixTreeThrows() {
        assertThrows(IllegalArgumentException.class, () -> new BeamKVCacheManager(kvCache, null, 4));
    }

    @Test
    void testBeamManager_invalidNumBeams() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        assertThrows(IllegalArgumentException.class, () -> new BeamKVCacheManager(kvCache, tree, 0));
    }

    @Test
    void testBeamManager_numBeamsExceedsBatchSize() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        assertThrows(IllegalArgumentException.class,
                () -> new BeamKVCacheManager(kvCache, tree, MAX_BATCH_SIZE + 1));
    }

    @Test
    void testBeamManager_initializeBeams() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 3);

        int[] prompt = {1, 2, 3, 4, 5, 6, 7, 8};
        manager.initializeBeams(prompt);

        for (int i = 0; i < 3; i++) {
            assertTrue(manager.isBeamActive(i), "Beam " + i + " should be active after initialization");
            int[] tokens = manager.getBeamTokens(i);
            assertArrayEquals(prompt, tokens, "All beams should start with the prompt tokens");
        }
    }

    @Test
    void testBeamManager_initializeBeams_nullThrows() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 2);

        assertThrows(IllegalArgumentException.class, () -> manager.initializeBeams(null));
    }

    @Test
    void testBeamManager_initializeBeams_emptyThrows() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 2);

        assertThrows(IllegalArgumentException.class, () -> manager.initializeBeams(new int[0]));
    }

    @Test
    void testBeamManager_isBeamActive_initiallyInactive() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 3);

        for (int i = 0; i < 3; i++) {
            assertFalse(manager.isBeamActive(i), "Beams should be inactive before initialization");
        }
    }

    @Test
    void testBeamManager_isBeamActive_invalidIdx() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 3);

        assertThrows(IllegalArgumentException.class, () -> manager.isBeamActive(-1));
        assertThrows(IllegalArgumentException.class, () -> manager.isBeamActive(3));
    }

    @Test
    void testBeamManager_getBeamTokens_returnsCopy() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 2);

        int[] prompt = {1, 2, 3, 4};
        manager.initializeBeams(prompt);

        int[] tokens = manager.getBeamTokens(0);
        assertArrayEquals(prompt, tokens);

        // Modifying the returned array should not affect the internal state
        tokens[0] = 999;
        int[] tokens2 = manager.getBeamTokens(0);
        assertEquals(1, tokens2[0], "getBeamTokens should return a copy");
    }

    @Test
    void testBeamManager_forkBeam() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 4);

        int[] prompt = {1, 2, 3, 4};
        manager.initializeBeams(prompt);

        // Fork beam 0 to beam 3 (beam 3 is already active from init, but forkBeam handles that)
        manager.forkBeam(0, 3);

        assertTrue(manager.isBeamActive(3));
        assertArrayEquals(manager.getBeamTokens(0), manager.getBeamTokens(3),
                "Forked beam should have same tokens as parent");
    }

    @Test
    void testBeamManager_forkBeam_fromInactiveThrows() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        // Use only 2 beams but 4 slots, so beams 2-3 remain inactive
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 4);

        // Beam 0 is inactive (not initialized), forking from it should throw
        assertThrows(IllegalStateException.class, () -> manager.forkBeam(0, 1));
    }

    @Test
    void testBeamManager_forkBeam_invalidIdx() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 3);

        assertThrows(IllegalArgumentException.class, () -> manager.forkBeam(-1, 0));
        assertThrows(IllegalArgumentException.class, () -> manager.forkBeam(0, 5));
    }

    @Test
    void testBeamManager_freeBeam() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 3);

        int[] prompt = {1, 2, 3, 4};
        manager.initializeBeams(prompt);

        assertTrue(manager.isBeamActive(1));
        manager.freeBeam(1);
        assertFalse(manager.isBeamActive(1), "Freed beam should be inactive");
        assertArrayEquals(new int[0], manager.getBeamTokens(1), "Freed beam should have empty tokens");
    }

    @Test
    void testBeamManager_freeBeam_alreadyInactive() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 3);

        // Freeing an already-inactive beam should be a no-op (not throw)
        assertFalse(manager.isBeamActive(0));
        manager.freeBeam(0);
        assertFalse(manager.isBeamActive(0));
    }

    @Test
    void testBeamManager_appendToken() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 2);

        int[] prompt = {1, 2, 3, 4};
        manager.initializeBeams(prompt);

        // Append a token to beam 0
        INDArray newKey = Nd4j.rand(DataType.FLOAT, 1, 1, NUM_KV_HEADS, HEAD_DIM);
        INDArray newVal = Nd4j.rand(DataType.FLOAT, 1, 1, NUM_KV_HEADS, HEAD_DIM);
        manager.appendToken(0, 100, newKey, newVal);

        int[] tokens = manager.getBeamTokens(0);
        assertEquals(5, tokens.length, "After appending, beam should have prompt + 1 token");
        assertEquals(100, tokens[4], "Appended token ID should be at the end");

        // Beam 1 should still have the original prompt
        assertEquals(4, manager.getBeamTokens(1).length);
    }

    @Test
    void testBeamManager_appendToken_toInactiveThrows() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 2);

        INDArray newKey = Nd4j.rand(DataType.FLOAT, 1, 1, NUM_KV_HEADS, HEAD_DIM);
        INDArray newVal = Nd4j.rand(DataType.FLOAT, 1, 1, NUM_KV_HEADS, HEAD_DIM);

        assertThrows(IllegalStateException.class, () -> manager.appendToken(0, 100, newKey, newVal));
    }

    @Test
    void testBeamManager_toString() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 3);

        String str = manager.toString();
        assertTrue(str.contains("BeamKVCacheManager"));
        assertTrue(str.contains("0/3 active"));

        manager.initializeBeams(new int[]{1, 2, 3, 4});
        str = manager.toString();
        assertTrue(str.contains("3/3 active"));
    }

    @Test
    void testBeamManager_reindexBeams_wrongLengthThrows() {
        KVCachePrefixTree tree = new KVCachePrefixTree(BLOCK_SIZE);
        BeamKVCacheManager manager = new BeamKVCacheManager(kvCache, tree, 3);
        manager.initializeBeams(new int[]{1, 2, 3, 4});

        assertThrows(IllegalArgumentException.class, () -> manager.reindexBeams(new int[]{0, 1}));
    }
}
