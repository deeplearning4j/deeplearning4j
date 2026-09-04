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

package org.eclipse.deeplearning4j.llm.generation.kvcache;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies the observability counters that feed the serving /api/llm/status
 * {@code prefixCacheStats} snapshot: lookups, hits, hit rate, restored tokens,
 * stored tokens, and the immutable stats map shape.
 */
class KvPrefixBlockPoolStatsTest {

    private static final int BLOCK_SIZE = 4;
    private static final int KV_H = 2;
    private static final int HEAD_DIM = 3;
    private static final int NUM_LAYERS = 2;   // one key layer + one value layer

    @Test
    void freshPoolReportsZeroedCounters() {
        try (KvPrefixBlockPool pool = new KvPrefixBlockPool(BLOCK_SIZE, 0, 64, 16)) {
            assertEquals(0, pool.getTotalLookups());
            assertEquals(0, pool.getTotalHits());
            assertEquals(0.0d, pool.getHitRate());
            assertEquals(0, pool.getTotalHitTokens());
            assertEquals(0, pool.getTotalStoredTokens());
            assertEquals(0, pool.getResidentBlockCount());
        }
    }

    @Test
    void statsMapExposesAllMonitoringFields() {
        try (KvPrefixBlockPool pool = new KvPrefixBlockPool(BLOCK_SIZE, 0, 64, 16)) {
            Map<String, Object> stats = pool.toStatsMap();
            assertNotNull(stats);
            assertEquals(BLOCK_SIZE, stats.get("blockSize"));
            assertEquals(0L, stats.get("maxByteBudget"));
            assertEquals(0, stats.get("residentBlocks"));
            assertEquals(0L, stats.get("currentBytes"));
            assertEquals(0L, stats.get("totalLookups"));
            assertEquals(0L, stats.get("totalHits"));
            assertEquals(0.0d, stats.get("hitRate"));
            assertEquals(0L, stats.get("totalHitTokens"));
            assertEquals(0L, stats.get("totalStoredTokens"));
        }
    }

    @Test
    void storeThenLookupCountsHitAndTokens() {
        try (KvPrefixBlockPool pool = new KvPrefixBlockPool(BLOCK_SIZE, 0, 64, 16)) {
            int[] prompt = {10, 11, 12, 13, 14, 15, 16, 17};   // exactly 2 blocks

            // Store the first 8 tokens via the same path the pipeline uses.
            LinkedHashMap<String, INDArray> kv = buffers(8);
            pool.storeCompletedPrefill(prompt, prompt.length, kv,
                    List.of("k0", "v0"), null);
            assertEquals(8, pool.getTotalStoredTokens());
            assertEquals(2, pool.getResidentBlockCount());
            assertEquals(0, pool.getTotalLookups());

            // First lookup: identical prompt — hit covering all 8 tokens.
            LinkedHashMap<String, INDArray> restore = buffers(8);
            int restored = pool.restoreBlocks(prompt, restore, List.of("k0", "v0"));
            assertEquals(8, restored);
            assertEquals(1, pool.getTotalLookups());
            assertEquals(1, pool.getTotalHits());
            assertEquals(8, pool.getTotalHitTokens());
            assertEquals(1.0d, pool.getHitRate());

            // Second lookup: disjoint prompt — miss.
            int[] other = {90, 91, 92, 93, 94, 95, 96, 97};
            assertEquals(0, pool.restoreBlocks(other, buffers(8), List.of("k0", "v0")));
            assertEquals(2, pool.getTotalLookups());
            assertEquals(1, pool.getTotalHits());
            assertEquals(8, pool.getTotalHitTokens());
            assertEquals(0.5d, pool.getHitRate());
        }
    }

    @Test
    void recordLookupCountsProbesWithoutRestoring() {
        try (KvPrefixBlockPool pool = new KvPrefixBlockPool(BLOCK_SIZE, 0, 64, 16)) {
            int[] prompt = {10, 11, 12, 13, 14, 15, 16, 17};
            LinkedHashMap<String, INDArray> kv = buffers(8);
            pool.storeCompletedPrefill(prompt, prompt.length, kv,
                    List.of("k0", "v0"), null);

            // Hit-probe path used by GenerationPipeline.attemptPrefixCacheHit.
            PrefixLookupResult probe = pool.getRadixCache().lookup(prompt);
            pool.recordLookup(probe);
            assertEquals(1, pool.getTotalLookups());
            assertEquals(1, pool.getTotalHits());
            assertEquals(8, pool.getTotalHitTokens());

            // Null/malformed probes count as lookups but never as hits.
            pool.recordLookup(null);
            assertEquals(2, pool.getTotalLookups());
            assertEquals(1, pool.getTotalHits());
        }
    }

    @Test
    void partialPrefixLookupCountsOnlyMatchedTokens() {
        try (KvPrefixBlockPool pool = new KvPrefixBlockPool(BLOCK_SIZE, 0, 64, 16)) {
            int[] prompt = {10, 11, 12, 13, 14, 15, 16, 17};
            pool.storeCompletedPrefill(prompt, prompt.length, buffers(8),
                    List.of("k0", "v0"), null);

            // Shared first block only (4 tokens), then diverges.
            int[] shared = {10, 11, 12, 13, 99, 98, 97, 96};
            assertEquals(4, pool.restoreBlocks(shared, buffers(8), List.of("k0", "v0")));
            assertEquals(1, pool.getTotalHits());
            assertEquals(4, pool.getTotalHitTokens());
        }
    }

    /** Allocates zeroed KV buffers shaped [1, kvLen, kvH, headDim] for k0/v0. */
    private static LinkedHashMap<String, INDArray> buffers(int kvLen) {
        LinkedHashMap<String, INDArray> map = new LinkedHashMap<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            map.put(i == 0 ? "k0" : "v0",
                    Nd4j.zeros(1, kvLen, KV_H, HEAD_DIM));
        }
        return map;
    }
}
