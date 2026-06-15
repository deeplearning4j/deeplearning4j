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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for H2OEvictionPolicy, StreamingLLMEvictionPolicy, and EvictablePagedKVCache.
 *
 * Adam Gibson
 */
class TestTokenEviction {

    // ==================== H2OEvictionPolicy Tests ====================

    @Test
    void testH2OConstruction_valid() {
        H2OEvictionPolicy policy = new H2OEvictionPolicy(128, 4, 10, 8);
        assertEquals(128, policy.getMaxSeqLen());
        assertEquals(4, policy.getNumLayers());
        assertEquals(10, policy.getTopK());
        assertEquals(8, policy.getRecentWindow());
    }

    @Test
    void testH2OConstruction_invalidMaxSeqLen() {
        assertThrows(IllegalArgumentException.class, () -> new H2OEvictionPolicy(0, 4, 10, 8));
    }

    @Test
    void testH2OConstruction_invalidNumLayers() {
        assertThrows(IllegalArgumentException.class, () -> new H2OEvictionPolicy(128, 0, 10, 8));
    }

    @Test
    void testH2OConstruction_negativeTopK() {
        assertThrows(IllegalArgumentException.class, () -> new H2OEvictionPolicy(128, 4, -1, 8));
    }

    @Test
    void testH2OConstruction_negativeRecentWindow() {
        assertThrows(IllegalArgumentException.class, () -> new H2OEvictionPolicy(128, 4, 10, -1));
    }

    @Test
    void testH2O_noEvictionWhenWithinBudget() {
        H2OEvictionPolicy policy = new H2OEvictionPolicy(64, 2, 5, 4);
        int[] evicted = policy.selectTokensToEvict(0, 8, 10);
        assertEquals(0, evicted.length, "No eviction needed when currentLen <= budget");
    }

    @Test
    void testH2O_noEvictionWhenExactlyAtBudget() {
        H2OEvictionPolicy policy = new H2OEvictionPolicy(64, 2, 5, 4);
        int[] evicted = policy.selectTokensToEvict(0, 10, 10);
        assertEquals(0, evicted.length, "No eviction needed when currentLen == budget");
    }

    @Test
    void testH2O_evictsWhenOverBudget() {
        H2OEvictionPolicy policy = new H2OEvictionPolicy(64, 1, 3, 2);
        // currentLen=10, budget=5, topK=3, recentWindow=2
        // Should keep top 3 by score + last 2, evict the rest
        int[] evicted = policy.selectTokensToEvict(0, 10, 5);
        assertTrue(evicted.length > 0, "Should evict tokens when over budget");
    }

    @Test
    void testH2O_updateScoresAndEviction() {
        H2OEvictionPolicy policy = new H2OEvictionPolicy(16, 1, 2, 2);

        // Create attention weights where positions 0 and 3 are heavy hitters
        // Shape: [batch=1, heads=1, queryLen=1, keyLen=8]
        INDArray attnWeights = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 8);
        attnWeights.putScalar(new int[]{0, 0, 0, 0}, 10.0f); // pos 0: heavy hitter
        attnWeights.putScalar(new int[]{0, 0, 0, 1}, 0.1f);  // pos 1: low
        attnWeights.putScalar(new int[]{0, 0, 0, 2}, 0.1f);  // pos 2: low
        attnWeights.putScalar(new int[]{0, 0, 0, 3}, 8.0f);  // pos 3: heavy hitter
        attnWeights.putScalar(new int[]{0, 0, 0, 4}, 0.2f);  // pos 4: low
        attnWeights.putScalar(new int[]{0, 0, 0, 5}, 0.1f);  // pos 5: low
        attnWeights.putScalar(new int[]{0, 0, 0, 6}, 1.0f);  // pos 6: recent
        attnWeights.putScalar(new int[]{0, 0, 0, 7}, 1.0f);  // pos 7: recent

        policy.updateScores(0, attnWeights);

        // budget=4, topK=2, recentWindow=2
        // Should keep positions 0,3 (top-K) and 6,7 (recent), evict 1,2,4,5
        int[] evicted = policy.selectTokensToEvict(0, 8, 4);
        assertEquals(4, evicted.length, "Should evict 4 tokens to reach budget of 4");

        Set<Integer> evictedSet = new HashSet<>();
        for (int pos : evicted) {
            evictedSet.add(pos);
        }
        // Heavy hitters 0,3 should NOT be evicted
        assertFalse(evictedSet.contains(0), "Heavy hitter at pos 0 should be kept");
        assertFalse(evictedSet.contains(3), "Heavy hitter at pos 3 should be kept");
        // Recent window 6,7 should NOT be evicted
        assertFalse(evictedSet.contains(6), "Recent pos 6 should be kept");
        assertFalse(evictedSet.contains(7), "Recent pos 7 should be kept");
        // Low-score positions should be evicted
        assertTrue(evictedSet.contains(1), "Low-score pos 1 should be evicted");
        assertTrue(evictedSet.contains(2), "Low-score pos 2 should be evicted");
        assertTrue(evictedSet.contains(4), "Low-score pos 4 should be evicted");
        assertTrue(evictedSet.contains(5), "Low-score pos 5 should be evicted");
    }

    @Test
    void testH2O_cumulativeScores() {
        H2OEvictionPolicy policy = new H2OEvictionPolicy(16, 1, 2, 2);

        // First update: position 1 has higher score
        INDArray attn1 = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 8);
        attn1.putScalar(new int[]{0, 0, 0, 1}, 5.0f);
        attn1.putScalar(new int[]{0, 0, 0, 2}, 1.0f);
        policy.updateScores(0, attn1);

        // Second update: position 2 gets a higher score
        INDArray attn2 = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 8);
        attn2.putScalar(new int[]{0, 0, 0, 1}, 1.0f);
        attn2.putScalar(new int[]{0, 0, 0, 2}, 8.0f);
        policy.updateScores(0, attn2);

        // Cumulative: pos1=6.0, pos2=9.0
        // With budget=4, topK=2, recentWindow=2, currentLen=8
        // Recent: 6,7. TopK from candidates [0..5]: pos2(9.0), pos1(6.0)
        int[] evicted = policy.selectTokensToEvict(0, 8, 4);
        Set<Integer> evictedSet = new HashSet<>();
        for (int pos : evicted) {
            evictedSet.add(pos);
        }
        assertFalse(evictedSet.contains(1), "Cumulative heavy hitter at pos 1 should be kept");
        assertFalse(evictedSet.contains(2), "Cumulative heavy hitter at pos 2 should be kept");
    }

    @Test
    void testH2O_reset() {
        H2OEvictionPolicy policy = new H2OEvictionPolicy(16, 1, 2, 2);

        INDArray attn = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 8);
        attn.putScalar(new int[]{0, 0, 0, 0}, 100.0f);
        policy.updateScores(0, attn);

        policy.reset();

        // After reset, all scores should be zero -- no tokens are heavy hitters
        // Any eviction result should not favor position 0 based on pre-reset scores
        int[] evicted = policy.selectTokensToEvict(0, 8, 4);
        // With all scores zero, the top-K selection is arbitrary but deterministic
        // The important thing is that the policy does not crash and produces valid output
        assertTrue(evicted.length > 0, "Should still evict when over budget after reset");
    }

    @Test
    void testH2O_outOfRangeLayerIndex() {
        H2OEvictionPolicy policy = new H2OEvictionPolicy(16, 2, 2, 2);
        // Layer index out of range should return empty
        int[] evicted = policy.selectTokensToEvict(-1, 10, 5);
        assertEquals(0, evicted.length);
        evicted = policy.selectTokensToEvict(5, 10, 5);
        assertEquals(0, evicted.length);
    }

    @Test
    void testH2O_perLayerBudgets() {
        H2OEvictionPolicy policy = new H2OEvictionPolicy(32, 2, 3, 2);
        policy.setPerLayerBudgets(new int[]{5, 8});

        // Layer 0 has budget override of 5
        int[] evictedL0 = policy.selectTokensToEvict(0, 10, 100);
        assertTrue(evictedL0.length > 0, "Layer 0 should evict based on per-layer budget 5");

        // Layer 1 has budget override of 8
        int[] evictedL1 = policy.selectTokensToEvict(1, 10, 100);
        assertTrue(evictedL1.length > 0, "Layer 1 should evict based on per-layer budget 8");
    }

    @Test
    void testH2O_perLayerBudgets_wrongLength() {
        H2OEvictionPolicy policy = new H2OEvictionPolicy(32, 2, 3, 2);
        assertThrows(IllegalArgumentException.class, () -> policy.setPerLayerBudgets(new int[]{5}));
    }

    @Test
    void testH2O_toString() {
        H2OEvictionPolicy policy = new H2OEvictionPolicy(64, 4, 10, 8);
        String str = policy.toString();
        assertTrue(str.contains("H2OEvictionPolicy"));
        assertTrue(str.contains("64"));
        assertTrue(str.contains("10"));
    }

    // ==================== StreamingLLMEvictionPolicy Tests ====================

    @Test
    void testStreamingLLM_construction() {
        StreamingLLMEvictionPolicy policy = new StreamingLLMEvictionPolicy(2, 32);
        assertEquals(2, policy.getNumSinks());
        assertEquals(32, policy.getWindowSize());
    }

    @Test
    void testStreamingLLM_invalidNumSinks() {
        assertThrows(IllegalArgumentException.class, () -> new StreamingLLMEvictionPolicy(-1, 32));
    }

    @Test
    void testStreamingLLM_invalidWindowSize() {
        assertThrows(IllegalArgumentException.class, () -> new StreamingLLMEvictionPolicy(2, 0));
    }

    @Test
    void testStreamingLLM_noEvictionWhenShort() {
        StreamingLLMEvictionPolicy policy = new StreamingLLMEvictionPolicy(2, 4);
        // currentLen=6 = numSinks(2) + windowSize(4), no eviction needed
        int[] evicted = policy.selectTokensToEvict(0, 6, 100);
        assertEquals(0, evicted.length, "No eviction when sequence fits in sinks + window");
    }

    @Test
    void testStreamingLLM_noEvictionWhenShorterThanSinksWindow() {
        StreamingLLMEvictionPolicy policy = new StreamingLLMEvictionPolicy(2, 4);
        int[] evicted = policy.selectTokensToEvict(0, 3, 100);
        assertEquals(0, evicted.length, "No eviction for very short sequence");
    }

    @Test
    void testStreamingLLM_evictsMiddle() {
        StreamingLLMEvictionPolicy policy = new StreamingLLMEvictionPolicy(2, 3);
        // currentLen=10, sinks=2, window=3
        // Keep positions [0,1] (sinks) and [7,8,9] (window)
        // Evict positions [2,3,4,5,6]
        int[] evicted = policy.selectTokensToEvict(0, 10, 100);
        assertEquals(5, evicted.length, "Should evict 5 tokens from the middle");

        for (int i = 0; i < evicted.length; i++) {
            assertEquals(2 + i, evicted[i], "Evicted positions should be contiguous from sinks end");
        }
    }

    @Test
    void testStreamingLLM_keepsSinksAndWindow() {
        StreamingLLMEvictionPolicy policy = new StreamingLLMEvictionPolicy(3, 4);
        // currentLen=20
        // Sinks: [0,1,2], Window: [16,17,18,19], Evict: [3..15] = 13 positions
        int[] evicted = policy.selectTokensToEvict(0, 20, 100);
        assertEquals(13, evicted.length);

        Set<Integer> evictedSet = new HashSet<>();
        for (int pos : evicted) {
            evictedSet.add(pos);
        }
        // Sinks should not be evicted
        for (int i = 0; i < 3; i++) {
            assertFalse(evictedSet.contains(i), "Sink position " + i + " should not be evicted");
        }
        // Window should not be evicted
        for (int i = 16; i < 20; i++) {
            assertFalse(evictedSet.contains(i), "Window position " + i + " should not be evicted");
        }
    }

    @Test
    void testStreamingLLM_updateScoresIsNoop() {
        StreamingLLMEvictionPolicy policy = new StreamingLLMEvictionPolicy(2, 4);
        // Should not throw -- updateScores is a no-op for position-based policy
        INDArray attn = Nd4j.ones(DataType.FLOAT, 1, 1, 1, 8);
        policy.updateScores(0, attn);
    }

    @Test
    void testStreamingLLM_resetIsNoop() {
        StreamingLLMEvictionPolicy policy = new StreamingLLMEvictionPolicy(2, 4);
        // Should not throw
        policy.reset();
        // Behavior should be identical before and after reset
        int[] before = policy.selectTokensToEvict(0, 10, 100);
        policy.reset();
        int[] after = policy.selectTokensToEvict(0, 10, 100);
        assertArrayEquals(before, after, "Reset should not change position-based eviction");
    }

    @Test
    void testStreamingLLM_budgetIgnored() {
        // StreamingLLM ignores the budget parameter -- it always evicts based on sinks + window
        StreamingLLMEvictionPolicy policy = new StreamingLLMEvictionPolicy(2, 3);
        int[] evicted1 = policy.selectTokensToEvict(0, 10, 5);
        int[] evicted2 = policy.selectTokensToEvict(0, 10, 100);
        assertArrayEquals(evicted1, evicted2, "StreamingLLM should not use the budget parameter");
    }

    @Test
    void testStreamingLLM_toString() {
        StreamingLLMEvictionPolicy policy = new StreamingLLMEvictionPolicy(2, 32);
        String str = policy.toString();
        assertTrue(str.contains("StreamingLLM"));
        assertTrue(str.contains("2"));
        assertTrue(str.contains("32"));
    }

    // ==================== EvictablePagedKVCache Tests ====================

    @Test
    void testEvictableCache_constructionWithDefaults() {
        try (EvictablePagedKVCache cache = new EvictablePagedKVCache(4, 64, 2, 16, DataType.FLOAT, 32)) {
            assertEquals(32, cache.getTokenBudget());
            assertNull(cache.getEvictionPolicy());
            assertEquals(4, cache.getMaxBatchSize());
        }
    }

    @Test
    void testEvictableCache_constructionWithBlockSize() {
        try (EvictablePagedKVCache cache = new EvictablePagedKVCache(4, 64, 2, 16, 8, DataType.FLOAT, 1.5, 32)) {
            assertEquals(32, cache.getTokenBudget());
            assertEquals(8, cache.getBlockSize());
        }
    }

    @Test
    void testEvictableCache_setEvictionPolicy() {
        try (EvictablePagedKVCache cache = new EvictablePagedKVCache(4, 64, 2, 16, DataType.FLOAT, 32)) {
            assertNull(cache.getEvictionPolicy());
            StreamingLLMEvictionPolicy policy = new StreamingLLMEvictionPolicy(2, 10);
            cache.setEvictionPolicy(policy);
            assertSame(policy, cache.getEvictionPolicy());
        }
    }

    @Test
    void testEvictableCache_appendWithoutPolicy() {
        try (EvictablePagedKVCache cache = new EvictablePagedKVCache(2, 128, 2, 8, 4, DataType.FLOAT, 1.5, 10)) {
            // Append 5 tokens without a policy -- no eviction should occur
            INDArray keys = Nd4j.rand(DataType.FLOAT, 5, 2, 8);
            INDArray values = Nd4j.rand(DataType.FLOAT, 5, 2, 8);
            cache.append(0, keys, values);
            assertEquals(5, cache.getSequenceLength(0));
        }
    }

    @Test
    void testEvictableCache_evictTokensReducesLength() {
        try (EvictablePagedKVCache cache = new EvictablePagedKVCache(2, 128, 2, 8, 4, DataType.FLOAT, 1.5, 20)) {
            // Append 8 tokens
            INDArray keys = Nd4j.rand(DataType.FLOAT, 8, 2, 8);
            INDArray values = Nd4j.rand(DataType.FLOAT, 8, 2, 8);
            cache.append(0, keys, values);
            assertEquals(8, cache.getSequenceLength(0));

            // Evict 3 tokens
            cache.evictTokens(0, new int[]{1, 3, 5});
            assertEquals(5, cache.getSequenceLength(0), "Sequence length should decrease by evicted count");
        }
    }

    @Test
    void testEvictableCache_evictEmptyArray() {
        try (EvictablePagedKVCache cache = new EvictablePagedKVCache(2, 128, 2, 8, 4, DataType.FLOAT, 1.5, 20)) {
            INDArray keys = Nd4j.rand(DataType.FLOAT, 4, 2, 8);
            INDArray values = Nd4j.rand(DataType.FLOAT, 4, 2, 8);
            cache.append(0, keys, values);

            // Evict nothing -- should not change state
            cache.evictTokens(0, new int[]{});
            assertEquals(4, cache.getSequenceLength(0));
        }
    }

    @Test
    void testEvictableCache_evictTokensInvalidSeqIdx() {
        try (EvictablePagedKVCache cache = new EvictablePagedKVCache(2, 64, 2, 8, 4, DataType.FLOAT, 1.5, 20)) {
            assertThrows(IllegalArgumentException.class, () -> cache.evictTokens(-1, new int[]{0}));
            assertThrows(IllegalArgumentException.class, () -> cache.evictTokens(5, new int[]{0}));
        }
    }

    @Test
    void testEvictableCache_setTokenBudget() {
        try (EvictablePagedKVCache cache = new EvictablePagedKVCache(2, 64, 2, 8, DataType.FLOAT, 20)) {
            assertEquals(20, cache.getTokenBudget());
            cache.setTokenBudget(10);
            assertEquals(10, cache.getTokenBudget());
        }
    }

    @Test
    void testEvictableCache_toString() {
        try (EvictablePagedKVCache cache = new EvictablePagedKVCache(2, 64, 2, 8, DataType.FLOAT, 20)) {
            String str = cache.toString();
            assertTrue(str.contains("EvictablePagedKVCache"));
            assertTrue(str.contains("budget=20"));
            assertTrue(str.contains("none"));

            cache.setEvictionPolicy(new StreamingLLMEvictionPolicy(2, 10));
            str = cache.toString();
            assertTrue(str.contains("StreamingLLMEvictionPolicy"));
        }
    }

    @Test
    void testEvictableCache_compactBlocksFreesBlocks() {
        try (EvictablePagedKVCache cache = new EvictablePagedKVCache(2, 128, 2, 8, 4, DataType.FLOAT, 2.0, 20)) {
            int freeBlocksBefore = cache.getNumFreeBlocks();

            // Append 8 tokens (uses 2 blocks of size 4)
            INDArray keys = Nd4j.rand(DataType.FLOAT, 8, 2, 8);
            INDArray values = Nd4j.rand(DataType.FLOAT, 8, 2, 8);
            cache.append(0, keys, values);

            int freeBlocksAfterAppend = cache.getNumFreeBlocks();
            assertTrue(freeBlocksAfterAppend < freeBlocksBefore, "Appending should use blocks from the pool");

            // Evict half the tokens -- compaction should potentially free a block
            cache.evictTokens(0, new int[]{0, 1, 2, 3});
            int freeBlocksAfterEvict = cache.getNumFreeBlocks();
            assertTrue(freeBlocksAfterEvict > freeBlocksAfterAppend,
                    "Evicting a full block's worth of tokens should free blocks after compaction");
        }
    }
}
