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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Cross-request KV block pool for prefix caching.
 *
 * <p>Stores device-resident KV blocks extracted from completed prefills and makes them
 * available for reuse when a subsequent request shares a token prefix. Paired with
 * {@link RadixPrefixCache} for trie-based longest-prefix lookup.</p>
 *
 * <h3>Block layout</h3>
 * <p>Each block covers {@code blockSize} consecutive token positions and stores the
 * full KV slice {@code [1, blockSize, numKVHeads, headDim]} per layer. Blocks are
 * identified by a monotonically-increasing integer ID allocated by
 * {@link #allocateBlockId()}.</p>
 *
 * <h3>Memory budget</h3>
 * <p>Total device bytes across all stored blocks is tracked and bounded by
 * {@code maxByteBudget}. When a new block would exceed the budget, the LRU block is
 * evicted (freeing its {@link INDArray} data) before the new block is stored. The
 * radix trie is notified of evictions so stale block IDs are not returned by
 * subsequent lookups.</p>
 *
 * <h3>Recurrent state snapshots</h3>
 * <p>For GDN/recurrent-state models, the recurrent state at the end of the full prefill
 * is stored as a per-prefix snapshot (keyed by the number of prefix tokens that were
 * cached). This snapshot is valid ONLY for exact-full-boundary reuse: the cached
 * prefix covers all token positions 0..cachedLen-1 and the new request shares exactly
 * that prefix. Partial-boundary hits do NOT reuse the recurrent state — they fall back
 * to a full re-prefill of the uncached suffix from position 0 with the recurrent state
 * properly derived.</p>
 *
 * <h3>Thread safety</h3>
 * <p>All public methods are guarded by an internal {@link ReentrantReadWriteLock}.</p>
 *
 * @see RadixPrefixCache
 * @see PrefixLookupResult
 */
@Slf4j
public class KvPrefixBlockPool implements AutoCloseable {

    /** Tokens per block (block granularity for the radix trie). */
    @Getter
    private final int blockSize;

    /** Maximum total device bytes across all stored blocks (0 = unlimited). */
    @Getter
    private final long maxByteBudget;

    /** The radix trie that maps token sequences to block IDs. */
    @Getter
    private final RadixPrefixCache radixCache;

    // ── Block storage ───────────────────────────────────────────────────────────────

    /** Per-block KV data: blockId → array[layerIdx] of shape [1,blockSize,kvH,headDim]. */
    private final LinkedHashMap<Integer, INDArray[]> blockData;

    /** Per-block byte cost (sum of all layers). Needed for LRU budget accounting. */
    private final LinkedHashMap<Integer, Long> blockBytes;

    /** Per-block recurrent-state snapshot (may be null when no recurrent state). */
    private final LinkedHashMap<Integer, Map<String, INDArray>> blockRecurrentSnapshots;

    /** Mapping of (full-prefix token count) → blockId of the last block in that prefix. */
    private final LinkedHashMap<Integer, Integer> prefixEndBlockId;

    /** Current total bytes of all stored blocks. */
    private long currentBytes;

    /** Monotonically-increasing block-ID allocator. */
    private final AtomicInteger nextBlockId = new AtomicInteger(0);

    /** Total radix-trie lookups issued through this pool (hit or miss). */
    private final AtomicLong totalLookups = new AtomicLong();

    /** Lookups whose matched token count was positive (usable cache hit). */
    private final AtomicLong totalHits = new AtomicLong();

    /** Sum of tokens restored from cache across all hits. */
    private final AtomicLong totalHitTokens = new AtomicLong();

    /** Total prefill tokens covered by registered prefixes. */
    private final AtomicLong totalStoredTokens = new AtomicLong();

    /** Lock for all pool state. */
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    /**
     * Default block size: 16 tokens. Matches fine-grained hit rates for typical
     * system prompts while keeping per-block overhead manageable.
     */
    public static final int DEFAULT_BLOCK_SIZE = 16;

    /**
     * Default maximum cache entries in the radix trie.
     * This is the number of distinct token-prefix sequences registered, not blocks.
     */
    public static final int DEFAULT_MAX_CACHE_ENTRIES = 256;

    /**
     * Create a block pool.
     *
     * @param blockSize      tokens per block (must be >= 1)
     * @param maxByteBudget  maximum total device bytes across all blocks (0 = unlimited)
     * @param bytesPerBlock  estimated bytes per block for the radix trie's savings estimate
     * @param maxCacheEntries maximum registered prefix entries in the radix trie
     */
    public KvPrefixBlockPool(int blockSize, long maxByteBudget, long bytesPerBlock, int maxCacheEntries) {
        if (blockSize < 1) throw new IllegalArgumentException("blockSize must be >= 1");
        this.blockSize = blockSize;
        this.maxByteBudget = maxByteBudget;
        this.radixCache = new RadixPrefixCache(blockSize, maxCacheEntries, bytesPerBlock);
        this.blockData = new LinkedHashMap<>(16, 0.75f, true);   // access-order LRU
        this.blockBytes = new LinkedHashMap<>(16, 0.75f, true);
        this.blockRecurrentSnapshots = new LinkedHashMap<>(16, 0.75f, true);
        this.prefixEndBlockId = new LinkedHashMap<>();
        this.currentBytes = 0L;
    }

    /**
     * Allocate a fresh block ID (does NOT store anything — call
     * {@link #storeBlock(int, INDArray[], long)} after).
     */
    public int allocateBlockId() {
        return nextBlockId.getAndIncrement();
    }

    /**
     * Store KV slices for a block.
     *
     * <p>If the budget would be exceeded, the least-recently-used block is evicted first.
     * The data arrays are {@code dup()}'d so the caller retains ownership of the originals.</p>
     *
     * @param blockId   the ID allocated by {@link #allocateBlockId()}
     * @param kvSlices  per-layer KV slices, each shape [1, blockSize, kvH, headDim]
     * @param byteCost  total byte cost of this block (sum across all layers)
     */
    public void storeBlock(int blockId, INDArray[] kvSlices, long byteCost) {
        lock.writeLock().lock();
        try {
            // Evict LRU blocks until we are within budget (or budget is 0 = unlimited).
            while (maxByteBudget > 0 && currentBytes + byteCost > maxByteBudget && !blockData.isEmpty()) {
                evictLruBlock();
            }
            INDArray[] stored = new INDArray[kvSlices.length];
            for (int i = 0; i < kvSlices.length; i++) {
                stored[i] = kvSlices[i] != null ? kvSlices[i].dup() : null;
            }
            blockData.put(blockId, stored);
            blockBytes.put(blockId, byteCost);
            currentBytes += byteCost;
            log.trace("[KvPrefixBlockPool] stored blockId={} cost={}B totalBytes={}", blockId, byteCost, currentBytes);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Store a recurrent-state snapshot associated with a given prefix length.
     *
     * <p>The snapshot is a per-layer deep copy of the recurrent state after prefilling
     * exactly {@code prefixTokenCount} tokens. Valid only for exact-boundary reuse.</p>
     *
     * @param prefixTokenCount total tokens covered by the cached prefix (must be block-aligned)
     * @param lastBlockId      the block ID of the last block in this prefix (for LRU linkage)
     * @param stateSnapshot    map of recurrent-state-input-name → INDArray (will be dup'd)
     */
    public void storeRecurrentSnapshot(int prefixTokenCount, int lastBlockId,
                                       Map<String, INDArray> stateSnapshot) {
        if (stateSnapshot == null || stateSnapshot.isEmpty()) return;
        lock.writeLock().lock();
        try {
            Map<String, INDArray> snap = new java.util.LinkedHashMap<>();
            for (Map.Entry<String, INDArray> e : stateSnapshot.entrySet()) {
                snap.put(e.getKey(), e.getValue() != null ? e.getValue().dup() : null);
            }
            blockRecurrentSnapshots.put(lastBlockId, snap);
            prefixEndBlockId.put(prefixTokenCount, lastBlockId);
            log.trace("[KvPrefixBlockPool] stored recurrent snapshot for prefix={} lastBlockId={}",
                    prefixTokenCount, lastBlockId);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Look up and restore cached KV blocks into pre-allocated static KV buffers.
     *
     * <p>Uses the radix trie to find the longest matching prefix. For each matched block,
     * copies the stored KV slice into the corresponding position of each layer's static
     * KV buffer. Returns the number of tokens restored (0 if no match, or the number
     * must be a multiple of {@code blockSize}).</p>
     *
     * <p>The static KV buffers must have shape {@code [1, maxKvLen, kvH, headDim]}.
     * Blocks are written at positions {@code [0..blockSize-1], [blockSize..2*blockSize-1]},
     * etc. Caller is responsible for zeroing the buffers before calling this if they may
     * contain stale data.</p>
     *
     * @param tokenIds         the full prompt token sequence
     * @param staticKvBuffers  layer-keyed static KV buffers (in same key order as layerNames)
     * @param layerNames       ordered list of KV cache layer names (keys followed by values,
     *                         matching the iteration order of staticKvBuffers)
     * @return number of tokens successfully restored from cache (0 if no hit)
     */
    public int restoreBlocks(int[] tokenIds, Map<String, INDArray> staticKvBuffers,
                             List<String> layerNames) {
        PrefixLookupResult result = radixCache.lookup(tokenIds);
        totalLookups.incrementAndGet();
        if (result.hasMatch() && result.getMatchedTokenCount() > 0) {
            totalHits.incrementAndGet();
            totalHitTokens.addAndGet(result.getMatchedTokenCount());
        }
        if (!result.hasMatch()) return 0;

        lock.readLock().lock();
        try {
            int[] blockIds = result.getSharedBlockIds();
            int restored = 0;

            for (int bi = 0; bi < blockIds.length; bi++) {
                int blockId = blockIds[bi];
                INDArray[] slices = blockData.get(blockId);
                if (slices == null) {
                    // Block was evicted — stop at the first hole; prefix is broken.
                    log.debug("[KvPrefixBlockPool] blockId={} not found (evicted); stopping at {} tokens",
                            blockId, restored);
                    break;
                }
                int tokenOffset = bi * blockSize;
                int layerIdx = 0;
                for (String layerName : layerNames) {
                    INDArray buf = staticKvBuffers.get(layerName);
                    if (buf == null || layerIdx >= slices.length || slices[layerIdx] == null) {
                        layerIdx++;
                        continue;
                    }
                    INDArray slice = slices[layerIdx];
                    long sliceLen = slice.shape()[1];  // actual stored block length
                    buf.get(NDArrayIndex.all(),
                            NDArrayIndex.interval(tokenOffset, tokenOffset + sliceLen),
                            NDArrayIndex.all(),
                            NDArrayIndex.all()).assign(slice);
                    layerIdx++;
                }
                restored += (int) Math.min(blockSize,
                        result.getMatchedTokenCount() - tokenOffset);
            }

            log.debug("[KvPrefixBlockPool] restored {} tokens from {} blocks", restored, blockIds.length);
            return restored;

        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Retrieve the recurrent-state snapshot for an exact-boundary prefix hit, or null.
     *
     * <p>Returns a snapshot only if {@code prefixTokenCount} exactly matches a registered
     * boundary and the associated block is still in the pool. Callers must check for
     * {@code null} and fall back to full re-prefill if no snapshot is available.</p>
     *
     * @param prefixTokenCount the number of tokens matched by the cache lookup
     * @return per-state-name snapshot map, or null if not available (partial hit or evicted)
     */
    public Map<String, INDArray> getRecurrentSnapshot(int prefixTokenCount) {
        lock.readLock().lock();
        try {
            Integer lastBlockId = prefixEndBlockId.get(prefixTokenCount);
            if (lastBlockId == null) return null;
            if (!blockData.containsKey(lastBlockId)) {
                // Associated block was evicted
                return null;
            }
            Map<String, INDArray> snap = blockRecurrentSnapshots.get(lastBlockId);
            if (snap == null) return null;
            // Return shallow copies so caller can use them without ownership confusion
            Map<String, INDArray> result = new java.util.LinkedHashMap<>();
            for (Map.Entry<String, INDArray> e : snap.entrySet()) {
                result.put(e.getKey(), e.getValue());  // not dup'd — caller must not close these
            }
            return result;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Register the completed prefix in the radix trie.
     *
     * @param tokenIds the full token sequence whose prefix was stored
     * @param blockIds the block IDs allocated (one per block-aligned chunk)
     */
    public void registerPrefix(int[] tokenIds, int[] blockIds) {
        radixCache.register(tokenIds, blockIds);
    }

    /**
     * Extract and store a block-aligned prefix from completed static KV buffers.
     *
     * <p>Called after a successful prefill. Slices the static KV buffers into blocks of
     * {@code blockSize} tokens, allocates block IDs, stores the slices in the pool, and
     * registers the prefix in the radix trie. If the recurrent state map is non-null and
     * the prefix is exactly block-aligned, a recurrent snapshot is also stored.</p>
     *
     * @param tokenIds            the full prompt token IDs
     * @param prefillLen          actual (non-padded) prefill length
     * @param staticKvBuffers     the filled static KV buffers (shape [1, maxKvLen, kvH, headDim])
     * @param layerNames          ordered layer names matching staticKvBuffers
     * @param recurrentState      the recurrent state after prefill, or null
     */
    public void storeCompletedPrefill(int[] tokenIds, int prefillLen,
                                      Map<String, INDArray> staticKvBuffers,
                                      List<String> layerNames,
                                      Map<String, INDArray> recurrentState) {
        int numFullBlocks = prefillLen / blockSize;
        if (numFullBlocks == 0) {
            log.debug("[KvPrefixBlockPool] prefillLen={} < blockSize={} — nothing to store", prefillLen, blockSize);
            return;
        }
        int numLayers = layerNames.size();
        List<Integer> allocatedBlockIds = new ArrayList<>();

        for (int bi = 0; bi < numFullBlocks; bi++) {
            int tokenOffset = bi * blockSize;
            INDArray[] slices = new INDArray[numLayers];
            long blockByteCost = 0L;
            int li = 0;
            for (String layerName : layerNames) {
                INDArray buf = staticKvBuffers.get(layerName);
                if (buf == null) { li++; continue; }
                // slice shape: [1, blockSize, kvH, headDim]
                INDArray slice = buf.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(tokenOffset, tokenOffset + blockSize),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()).dup();
                slices[li] = slice;
                blockByteCost += slice.length() * slice.dataType().width();
                li++;
            }
            int blockId = allocateBlockId();
            storeBlock(blockId, slices, blockByteCost);
            // Close the temp dup slices (storeBlock dup'd them again internally)
            for (INDArray s : slices) { if (s != null) s.close(); }
            allocatedBlockIds.add(blockId);
        }

        // Register in radix trie (only cover the block-aligned prefix)
        int coveredTokens = numFullBlocks * blockSize;
        int[] prefixTokenIds = new int[coveredTokens];
        System.arraycopy(tokenIds, 0, prefixTokenIds, 0, coveredTokens);
        int[] blockIdsArr = allocatedBlockIds.stream().mapToInt(Integer::intValue).toArray();
        registerPrefix(prefixTokenIds, blockIdsArr);
        totalStoredTokens.addAndGet(coveredTokens);

        // Store recurrent snapshot if we have exactly block-aligned coverage
        if (recurrentState != null && !recurrentState.isEmpty()
                && coveredTokens == prefillLen && !allocatedBlockIds.isEmpty()) {
            int lastBlockId = allocatedBlockIds.get(allocatedBlockIds.size() - 1);
            storeRecurrentSnapshot(coveredTokens, lastBlockId, recurrentState);
        }

        log.info("[KvPrefixBlockPool] stored {} blocks for {} tokens (prefillLen={} coveredTokens={})",
                allocatedBlockIds.size(), prefixTokenIds.length, prefillLen, coveredTokens);
    }

    /**
     * Current total byte usage of all stored blocks.
     */
    public long getCurrentBytes() {
        lock.readLock().lock();
        try { return currentBytes; }
        finally { lock.readLock().unlock(); }
    }

    /** Total radix-trie lookups issued through this pool (hit or miss). */
    public long getTotalLookups() { return totalLookups.get(); }

    /**
     * Record an externally-issued radix lookup (for callers that use
     * {@code getRadixCache().lookup} directly, e.g. hit-probing before buffer
     * allocation). Safe to call alongside {@link #restoreBlocks}, which records
     * its own lookup; double-counting within one request is acceptable because
     * both calls represent real lookups.
     */
    public void recordLookup(PrefixLookupResult result) {
        totalLookups.incrementAndGet();
        if (result != null && result.hasMatch() && result.getMatchedTokenCount() > 0) {
            totalHits.incrementAndGet();
            totalHitTokens.addAndGet(result.getMatchedTokenCount());
        }
    }

    /** Lookups that found a usable cached prefix. */
    public long getTotalHits() { return totalHits.get(); }

    /** Cache hit rate in [0,1]; 0 when no lookups have been issued. */
    public double getHitRate() {
        long lookups = totalLookups.get();
        return lookups == 0 ? 0.0d : (double) totalHits.get() / lookups;
    }

    /** Sum of tokens restored from cache across all hits. */
    public long getTotalHitTokens() { return totalHitTokens.get(); }

    /** Total prefill tokens covered by registered prefixes. */
    public long getTotalStoredTokens() { return totalStoredTokens.get(); }

    /** Current number of distinct stored blocks resident in the pool. */
    public int getResidentBlockCount() {
        lock.readLock().lock();
        try {
            return blockData.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    /** Immutable snapshot of pool observability counters for status endpoints. */
    public Map<String, Object> toStatsMap() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("blockSize", blockSize);
        stats.put("maxByteBudget", maxByteBudget);
        stats.put("residentBlocks", getResidentBlockCount());
        stats.put("currentBytes", getCurrentBytes());
        stats.put("totalLookups", getTotalLookups());
        stats.put("totalHits", getTotalHits());
        stats.put("hitRate", getHitRate());
        stats.put("totalHitTokens", getTotalHitTokens());
        stats.put("totalStoredTokens", getTotalStoredTokens());
        return stats;
    }

    /**
     * Number of blocks currently in the pool.
     */
    public int blockCount() {
        lock.readLock().lock();
        try { return blockData.size(); }
        finally { lock.readLock().unlock(); }
    }

    /**
     * Clear all stored blocks and the radix trie. Frees all device memory.
     */
    public void clear() {
        lock.writeLock().lock();
        try {
            for (INDArray[] slices : blockData.values()) {
                for (INDArray s : slices) {
                    if (s != null && !s.wasClosed()) try { s.close(); } catch (Exception ignore) {}
                }
            }
            for (Map<String, INDArray> snap : blockRecurrentSnapshots.values()) {
                if (snap != null) {
                    for (INDArray a : snap.values()) {
                        if (a != null && !a.wasClosed()) try { a.close(); } catch (Exception ignore) {}
                    }
                }
            }
            blockData.clear();
            blockBytes.clear();
            blockRecurrentSnapshots.clear();
            prefixEndBlockId.clear();
            currentBytes = 0L;
            radixCache.clear();
            log.info("[KvPrefixBlockPool] cleared");
        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public void close() {
        clear();
    }

    // ── Internal helpers ────────────────────────────────────────────────────────────

    /** Evict the least-recently-used block. Caller holds the write lock. */
    private void evictLruBlock() {
        if (blockData.isEmpty()) return;
        Map.Entry<Integer, INDArray[]> lruEntry = blockData.entrySet().iterator().next();
        int evictedId = lruEntry.getKey();
        INDArray[] slices = lruEntry.getValue();
        long cost = blockBytes.getOrDefault(evictedId, 0L);
        blockData.remove(evictedId);
        blockBytes.remove(evictedId);
        currentBytes = Math.max(0, currentBytes - cost);
        // Free device memory
        for (INDArray s : slices) {
            if (s != null && !s.wasClosed()) {
                try { s.close(); } catch (Exception ignore) {}
            }
        }
        // Remove recurrent snapshot if present
        Map<String, INDArray> snap = blockRecurrentSnapshots.remove(evictedId);
        if (snap != null) {
            for (INDArray a : snap.values()) {
                if (a != null && !a.wasClosed()) try { a.close(); } catch (Exception ignore) {}
            }
        }
        // Also evict from radix cache (remove LRU entry from trie)
        radixCache.evictLru();
        log.debug("[KvPrefixBlockPool] evicted LRU blockId={} (freed {}B, totalBytes={})",
                evictedId, cost, currentBytes);
    }
}
