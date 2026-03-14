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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for RadixPrefixCache and PrefixLookupResult.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
class RadixPrefixCacheTest {

    private static final int BLOCK_SIZE = 4;
    private static final int MAX_ENTRIES = 100;
    private static final long BYTES_PER_BLOCK = 1024;

    private RadixPrefixCache cache;

    @BeforeEach
    void setUp() {
        cache = new RadixPrefixCache(BLOCK_SIZE, MAX_ENTRIES, BYTES_PER_BLOCK);
    }

    // ==================== PrefixLookupResult Tests ====================

    @Test
    void testNoMatchResult() {
        PrefixLookupResult result = PrefixLookupResult.noMatch();
        assertEquals(0, result.getMatchedTokenCount());
        assertEquals(0, result.getSharedBlockIds().length);
        assertEquals(0, result.getMemorySavedBytes());
        assertFalse(result.hasMatch());
    }

    @Test
    void testMatchResult() {
        PrefixLookupResult result = new PrefixLookupResult(8, new int[]{0, 1}, 2048);
        assertEquals(8, result.getMatchedTokenCount());
        assertArrayEquals(new int[]{0, 1}, result.getSharedBlockIds());
        assertEquals(2048, result.getMemorySavedBytes());
        assertTrue(result.hasMatch());
    }

    @Test
    void testResultToString() {
        PrefixLookupResult result = new PrefixLookupResult(4, new int[]{5}, 1024);
        String str = result.toString();
        assertNotNull(str);
        assertTrue(str.contains("4"));
        assertTrue(str.contains("1024"));
    }

    // ==================== RadixPrefixCache Basic Tests ====================

    @Test
    void testConstructorParameters() {
        assertEquals(BLOCK_SIZE, cache.getBlockSize());
        assertEquals(MAX_ENTRIES, cache.getMaxCacheEntries());
        assertEquals(BYTES_PER_BLOCK, cache.getBytesPerBlock());
    }

    @Test
    void testConstructorValidation_blockSize() {
        assertThrows(IllegalArgumentException.class,
                () -> new RadixPrefixCache(0, 100, 1024));
        assertThrows(IllegalArgumentException.class,
                () -> new RadixPrefixCache(-1, 100, 1024));
    }

    @Test
    void testConstructorValidation_maxCacheEntries() {
        assertThrows(IllegalArgumentException.class,
                () -> new RadixPrefixCache(4, 0, 1024));
        assertThrows(IllegalArgumentException.class,
                () -> new RadixPrefixCache(4, -1, 1024));
    }

    @Test
    void testEmptyCacheLookup() {
        PrefixLookupResult result = cache.lookup(new int[]{1, 2, 3, 4});
        assertFalse(result.hasMatch());
        assertEquals(0, result.getMatchedTokenCount());
    }

    @Test
    void testEmptyCacheSize() {
        assertEquals(0, cache.size());
    }

    // ==================== Register and Lookup Tests ====================

    @Test
    void testRegisterAndLookupExactMatch() {
        int[] tokens = {10, 20, 30, 40};  // Exactly one block
        int[] blockIds = {0};

        cache.register(tokens, blockIds);
        assertEquals(1, cache.size());

        PrefixLookupResult result = cache.lookup(tokens);
        assertTrue(result.hasMatch());
        assertEquals(4, result.getMatchedTokenCount());
        assertArrayEquals(new int[]{0}, result.getSharedBlockIds());
        assertEquals(BYTES_PER_BLOCK, result.getMemorySavedBytes());
    }

    @Test
    void testRegisterAndLookupMultipleBlocks() {
        // Two blocks worth of tokens
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8};
        int[] blockIds = {10, 20};

        cache.register(tokens, blockIds);

        PrefixLookupResult result = cache.lookup(tokens);
        assertTrue(result.hasMatch());
        assertEquals(8, result.getMatchedTokenCount());
        assertArrayEquals(new int[]{10, 20}, result.getSharedBlockIds());
        assertEquals(2 * BYTES_PER_BLOCK, result.getMemorySavedBytes());
    }

    @Test
    void testPartialPrefixMatch() {
        // Register a longer prefix
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8};
        int[] blockIds = {10, 20};
        cache.register(tokens, blockIds);

        // Lookup with a longer sequence that shares the prefix
        int[] longerTokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        PrefixLookupResult result = cache.lookup(longerTokens);
        assertTrue(result.hasMatch());
        assertEquals(8, result.getMatchedTokenCount());
        assertArrayEquals(new int[]{10, 20}, result.getSharedBlockIds());
    }

    @Test
    void testNoMatchWhenPrefixDiffers() {
        int[] tokens = {1, 2, 3, 4};
        int[] blockIds = {0};
        cache.register(tokens, blockIds);

        // Different first token -- no prefix match
        int[] different = {99, 2, 3, 4};
        PrefixLookupResult result = cache.lookup(different);
        assertFalse(result.hasMatch());
    }

    @Test
    void testCrossRequestSharing() {
        // Simulate a shared system prompt prefix
        int[] systemPrompt = {100, 200, 300, 400, 500, 600, 700, 800};
        int[] blockIds = {0, 1};
        cache.register(systemPrompt, blockIds);

        // Request 1: system prompt + user query A
        int[] request1 = {100, 200, 300, 400, 500, 600, 700, 800, 901, 902, 903, 904};
        PrefixLookupResult result1 = cache.lookup(request1);
        assertTrue(result1.hasMatch());
        assertEquals(8, result1.getMatchedTokenCount());
        assertArrayEquals(new int[]{0, 1}, result1.getSharedBlockIds());

        // Request 2: same system prompt + different user query B
        int[] request2 = {100, 200, 300, 400, 500, 600, 700, 800, 777, 888, 999, 111};
        PrefixLookupResult result2 = cache.lookup(request2);
        assertTrue(result2.hasMatch());
        assertEquals(8, result2.getMatchedTokenCount());
        assertArrayEquals(new int[]{0, 1}, result2.getSharedBlockIds());
    }

    // ==================== LRU Eviction Tests ====================

    @Test
    void testLruEvictionWhenFull() {
        // Create cache with max 3 entries
        RadixPrefixCache smallCache = new RadixPrefixCache(BLOCK_SIZE, 3, BYTES_PER_BLOCK);

        // Register 3 entries
        smallCache.register(new int[]{1, 2, 3, 4}, new int[]{0});
        smallCache.register(new int[]{5, 6, 7, 8}, new int[]{1});
        smallCache.register(new int[]{9, 10, 11, 12}, new int[]{2});
        assertEquals(3, smallCache.size());

        // Register a 4th -- should evict the LRU (first registered)
        smallCache.register(new int[]{13, 14, 15, 16}, new int[]{3});
        assertEquals(3, smallCache.size());
    }

    @Test
    void testExplicitEvictLru() {
        cache.register(new int[]{1, 2, 3, 4}, new int[]{0});
        cache.register(new int[]{5, 6, 7, 8}, new int[]{1});
        assertEquals(2, cache.size());

        cache.evictLru();
        assertEquals(1, cache.size());

        cache.evictLru();
        assertEquals(0, cache.size());
    }

    @Test
    void testEvictLruOnEmptyCache() {
        // Should not throw
        cache.evictLru();
        assertEquals(0, cache.size());
    }

    // ==================== Clear Tests ====================

    @Test
    void testClear() {
        cache.register(new int[]{1, 2, 3, 4}, new int[]{0});
        cache.register(new int[]{5, 6, 7, 8}, new int[]{1});
        assertEquals(2, cache.size());

        cache.clear();
        assertEquals(0, cache.size());

        // After clear, lookups should return no match
        PrefixLookupResult result = cache.lookup(new int[]{1, 2, 3, 4});
        assertFalse(result.hasMatch());
    }

    // ==================== Disk Persistence Tests ====================

    @Test
    void testSaveAndLoadFromDisk(@TempDir Path tempDir) throws IOException {
        // Register some entries
        cache.register(new int[]{1, 2, 3, 4}, new int[]{10});
        cache.register(new int[]{5, 6, 7, 8, 9, 10, 11, 12}, new int[]{20, 30});
        cache.register(new int[]{100, 200, 300, 400}, new int[]{40});

        Path cacheFile = tempDir.resolve("prefix_cache.bin");
        cache.saveToDisk(cacheFile);

        // Load from disk
        RadixPrefixCache loaded = RadixPrefixCache.loadFromDisk(cacheFile);

        assertEquals(cache.getBlockSize(), loaded.getBlockSize());
        assertEquals(cache.getMaxCacheEntries(), loaded.getMaxCacheEntries());
        assertEquals(cache.getBytesPerBlock(), loaded.getBytesPerBlock());
        assertEquals(cache.size(), loaded.size());

        // Verify lookups work on loaded cache
        PrefixLookupResult result1 = loaded.lookup(new int[]{1, 2, 3, 4});
        assertTrue(result1.hasMatch());
        assertEquals(4, result1.getMatchedTokenCount());
        assertArrayEquals(new int[]{10}, result1.getSharedBlockIds());

        PrefixLookupResult result2 = loaded.lookup(new int[]{5, 6, 7, 8, 9, 10, 11, 12});
        assertTrue(result2.hasMatch());
        assertEquals(8, result2.getMatchedTokenCount());
        assertArrayEquals(new int[]{20, 30}, result2.getSharedBlockIds());
    }

    @Test
    void testSaveEmptyCache(@TempDir Path tempDir) throws IOException {
        Path cacheFile = tempDir.resolve("empty_cache.bin");
        cache.saveToDisk(cacheFile);

        RadixPrefixCache loaded = RadixPrefixCache.loadFromDisk(cacheFile);
        assertEquals(0, loaded.size());
        assertEquals(BLOCK_SIZE, loaded.getBlockSize());
    }

    // ==================== Concurrency Tests ====================

    @Test
    void testConcurrentLookups() throws Exception {
        // Pre-populate cache with a known prefix
        int[] sharedPrefix = {1, 2, 3, 4, 5, 6, 7, 8};
        cache.register(sharedPrefix, new int[]{0, 1});

        int numThreads = 8;
        int lookupsPerThread = 100;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        AtomicInteger matchCount = new AtomicInteger(0);
        AtomicInteger errorCount = new AtomicInteger(0);

        List<Future<?>> futures = new ArrayList<>();
        for (int t = 0; t < numThreads; t++) {
            futures.add(executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < lookupsPerThread; i++) {
                        int[] query = {1, 2, 3, 4, 5, 6, 7, 8, 99, 100, 101, 102};
                        PrefixLookupResult result = cache.lookup(query);
                        if (result.hasMatch() && result.getMatchedTokenCount() == 8) {
                            matchCount.incrementAndGet();
                        }
                    }
                } catch (Exception e) {
                    errorCount.incrementAndGet();
                }
            }));
        }

        // Release all threads at once
        startLatch.countDown();

        for (Future<?> f : futures) {
            f.get(30, TimeUnit.SECONDS);
        }

        executor.shutdown();
        assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));

        assertEquals(0, errorCount.get(), "No errors should occur during concurrent lookups");
        assertEquals(numThreads * lookupsPerThread, matchCount.get(),
                "All concurrent lookups should find the prefix match");
    }

    @Test
    void testConcurrentRegisterAndLookup() throws Exception {
        int numThreads = 4;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        AtomicInteger errorCount = new AtomicInteger(0);

        List<Future<?>> futures = new ArrayList<>();

        // Half threads register, half threads lookup
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            futures.add(executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < 50; i++) {
                        if (threadId % 2 == 0) {
                            // Register thread
                            int base = threadId * 1000 + i * 10;
                            int[] tokens = {base, base + 1, base + 2, base + 3};
                            cache.register(tokens, new int[]{base});
                        } else {
                            // Lookup thread
                            int base = (threadId - 1) * 1000 + i * 10;
                            int[] tokens = {base, base + 1, base + 2, base + 3};
                            // Just verify no exceptions
                            cache.lookup(tokens);
                        }
                    }
                } catch (Exception e) {
                    errorCount.incrementAndGet();
                }
            }));
        }

        startLatch.countDown();

        for (Future<?> f : futures) {
            f.get(30, TimeUnit.SECONDS);
        }

        executor.shutdown();
        assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));

        assertEquals(0, errorCount.get(),
                "No errors should occur during concurrent register + lookup");
    }

    // ==================== Edge Case Tests ====================

    @Test
    void testTokensNotBlockAligned() {
        // 6 tokens with blockSize=4: first block [1,2,3,4], second block [5,6,_,_]
        int[] tokens = {1, 2, 3, 4, 5, 6};
        int[] blockIds = {0, 1};

        cache.register(tokens, blockIds);

        PrefixLookupResult result = cache.lookup(tokens);
        assertTrue(result.hasMatch());
        assertEquals(6, result.getMatchedTokenCount());
    }

    @Test
    void testSingleTokenBlock() {
        RadixPrefixCache singleBlockCache = new RadixPrefixCache(1, 100, 512);

        int[] tokens = {42, 43, 44};
        int[] blockIds = {0, 1, 2};

        singleBlockCache.register(tokens, blockIds);

        PrefixLookupResult result = singleBlockCache.lookup(tokens);
        assertTrue(result.hasMatch());
        assertEquals(3, result.getMatchedTokenCount());
        assertArrayEquals(new int[]{0, 1, 2}, result.getSharedBlockIds());
    }

    @Test
    void testToStringDoesNotThrow() {
        assertNotNull(cache.toString());
        assertTrue(cache.toString().contains("RadixPrefixCache"));

        cache.register(new int[]{1, 2, 3, 4}, new int[]{0});
        assertTrue(cache.toString().contains("1"));
    }

    @Test
    void testOverwriteExistingPrefix() {
        int[] tokens = {1, 2, 3, 4};

        cache.register(tokens, new int[]{10});
        cache.register(tokens, new int[]{20});

        // The second registration may create a new LRU entry
        // but the trie node's blockId should be updated to 20
        PrefixLookupResult result = cache.lookup(tokens);
        assertTrue(result.hasMatch());
        assertEquals(4, result.getMatchedTokenCount());
        // Block ID should be the latest registered value
        assertArrayEquals(new int[]{20}, result.getSharedBlockIds());
    }

    @Test
    void testMemorySavedCalculation() {
        long bytesPerBlock = 4096;
        RadixPrefixCache largeBlockCache = new RadixPrefixCache(BLOCK_SIZE, MAX_ENTRIES, bytesPerBlock);

        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8};  // 2 blocks
        largeBlockCache.register(tokens, new int[]{0, 1});

        PrefixLookupResult result = largeBlockCache.lookup(tokens);
        assertEquals(2 * bytesPerBlock, result.getMemorySavedBytes());
    }
}
