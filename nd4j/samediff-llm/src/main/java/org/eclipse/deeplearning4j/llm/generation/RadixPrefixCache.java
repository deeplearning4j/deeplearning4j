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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Cross-request radix trie prefix cache with global LRU eviction.
 *
 * <p>Enables sharing of KV cache blocks across requests that share common
 * token prefixes (system prompts, few-shot examples, etc.). When a new
 * request arrives, the cache is searched for the longest matching prefix;
 * matching blocks can be reused directly, saving both computation and memory.</p>
 *
 * <h3>Data structure</h3>
 * <p>A radix trie where each node represents a block-aligned chunk of tokens.
 * Nodes are hashed using FNV-1a over their token content. A global LRU
 * tracks all registered prefixes for eviction when the cache is full.</p>
 *
 * <h3>Thread safety</h3>
 * <p>Uses a {@link ReentrantReadWriteLock} to allow concurrent lookups
 * while serializing registration and eviction operations.</p>
 *
 * <h3>Serialization</h3>
 * <p>The cache can be saved to and loaded from disk for persistence across
 * server restarts.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see PrefixLookupResult
 * @see KVCachePrefixTree
 */
@Slf4j
public class RadixPrefixCache {

    @Getter
    private final int blockSize;

    @Getter
    private final int maxCacheEntries;

    @Getter
    private final long bytesPerBlock;

    /** Root of the radix trie. */
    private final TrieNode root;

    /** Global LRU ordering of prefix entries (key = hash of full prefix). */
    private final LinkedHashMap<Long, TrieEntry> lruMap;

    /** Read-write lock for thread-safe access. */
    private final ReentrantReadWriteLock rwLock;

    /**
     * Internal trie node.
     */
    private static class TrieNode {
        /** Token block at this node (null for root). */
        final int[] tokens;
        /** Number of valid tokens in this block. */
        final int validCount;
        /** Block IDs associated with this node. */
        int blockId = -1;
        /** Children keyed by FNV-1a hash of their block tokens. */
        final Map<Long, TrieNode> children;
        /** LRU timestamp (nanoTime of last access). */
        long lastAccessNanos;

        TrieNode(int[] tokens, int validCount) {
            this.tokens = tokens;
            this.validCount = validCount;
            this.children = new HashMap<>();
            this.lastAccessNanos = System.nanoTime();
        }
    }

    /**
     * Entry in the global LRU map representing a registered prefix.
     */
    private static class TrieEntry {
        final int[] fullTokenIds;
        final int[] blockIds;
        long lastAccessNanos;

        TrieEntry(int[] fullTokenIds, int[] blockIds) {
            this.fullTokenIds = fullTokenIds;
            this.blockIds = blockIds;
            this.lastAccessNanos = System.nanoTime();
        }
    }

    /**
     * Create a radix prefix cache.
     *
     * @param blockSize       tokens per KV cache block
     * @param maxCacheEntries maximum number of registered prefixes
     * @param bytesPerBlock   memory bytes per KV block (for savings calculation)
     */
    public RadixPrefixCache(int blockSize, int maxCacheEntries, long bytesPerBlock) {
        if (blockSize < 1) {
            throw new IllegalArgumentException("blockSize must be >= 1, got " + blockSize);
        }
        if (maxCacheEntries < 1) {
            throw new IllegalArgumentException("maxCacheEntries must be >= 1, got " + maxCacheEntries);
        }
        this.blockSize = blockSize;
        this.maxCacheEntries = maxCacheEntries;
        this.bytesPerBlock = bytesPerBlock;
        this.root = new TrieNode(null, 0);
        this.lruMap = new LinkedHashMap<>(16, 0.75f, true);
        this.rwLock = new ReentrantReadWriteLock();
    }

    /**
     * Find the longest matching prefix and return shared block IDs.
     *
     * <p>Does not modify the trie structure. Updates LRU timestamps for
     * matched entries.</p>
     *
     * @param tokenIds the token sequence to look up
     * @return result containing matched token count and shared block IDs
     */
    public PrefixLookupResult lookup(int[] tokenIds) {
        rwLock.readLock().lock();
        try {
            List<Integer> matchedBlockIds = new ArrayList<>();
            int matchedTokens = 0;
            TrieNode current = root;

            int pos = 0;
            while (pos < tokenIds.length) {
                int blockEnd = Math.min(pos + blockSize, tokenIds.length);
                int validCount = blockEnd - pos;
                int[] blockTokens = new int[blockSize];
                System.arraycopy(tokenIds, pos, blockTokens, 0, validCount);

                long blockHash = hashBlock(blockTokens, validCount);
                TrieNode child = current.children.get(blockHash);

                if (child == null || !matchesTokens(child, blockTokens, validCount)) {
                    break;
                }

                child.lastAccessNanos = System.nanoTime();
                if (child.blockId >= 0) {
                    matchedBlockIds.add(child.blockId);
                }
                matchedTokens += validCount;
                current = child;
                pos = blockEnd;
            }

            if (matchedBlockIds.isEmpty()) {
                return PrefixLookupResult.noMatch();
            }

            int[] blockIds = matchedBlockIds.stream().mapToInt(Integer::intValue).toArray();
            long memorySaved = (long) blockIds.length * bytesPerBlock;

            log.trace("Prefix lookup matched {} tokens, {} blocks, saved {} bytes",
                    matchedTokens, blockIds.length, memorySaved);

            return new PrefixLookupResult(matchedTokens, blockIds, memorySaved);

        } finally {
            rwLock.readLock().unlock();
        }
    }

    /**
     * Register a completed request's prefix and block IDs.
     *
     * <p>Inserts trie nodes for each block-aligned chunk of the token sequence
     * and records the block IDs. If the cache exceeds the maximum number of
     * entries, the LRU entry is evicted.</p>
     *
     * @param tokenIds the token sequence
     * @param blockIds the physical block IDs for each block-aligned chunk
     */
    public void register(int[] tokenIds, int[] blockIds) {
        rwLock.writeLock().lock();
        try {
            // Evict if at capacity
            while (lruMap.size() >= maxCacheEntries) {
                evictLru();
            }

            TrieNode current = root;
            int blockIdx = 0;
            int pos = 0;

            while (pos < tokenIds.length && blockIdx < blockIds.length) {
                int blockEnd = Math.min(pos + blockSize, tokenIds.length);
                int validCount = blockEnd - pos;
                int[] blockTokens = new int[blockSize];
                System.arraycopy(tokenIds, pos, blockTokens, 0, validCount);

                long blockHash = hashBlock(blockTokens, validCount);
                TrieNode child = current.children.get(blockHash);

                if (child == null || !matchesTokens(child, blockTokens, validCount)) {
                    child = new TrieNode(blockTokens.clone(), validCount);
                    current.children.put(blockHash, child);
                }

                child.blockId = blockIds[blockIdx];
                child.lastAccessNanos = System.nanoTime();

                current = child;
                blockIdx++;
                pos = blockEnd;
            }

            // Register in global LRU
            long prefixHash = hashFullPrefix(tokenIds);
            TrieEntry entry = new TrieEntry(tokenIds.clone(), blockIds.clone());
            lruMap.put(prefixHash, entry);

            log.trace("Registered prefix: {} tokens, {} blocks", tokenIds.length, blockIds.length);

        } finally {
            rwLock.writeLock().unlock();
        }
    }

    /**
     * Evict the least-recently-used prefix entry.
     *
     * <p>Removes the LRU entry from the global map. Does not remove trie nodes
     * (they may be shared by other prefixes). Trie nodes with no children and
     * no active references are cleaned up lazily.</p>
     */
    public void evictLru() {
        rwLock.writeLock().lock();
        try {
            if (lruMap.isEmpty()) return;

            Map.Entry<Long, TrieEntry> lruEntry = lruMap.entrySet().iterator().next();
            lruMap.remove(lruEntry.getKey());

            log.trace("Evicted LRU prefix entry (hash={})", lruEntry.getKey());

        } finally {
            rwLock.writeLock().unlock();
        }
    }

    /**
     * Number of registered prefix entries.
     */
    public int size() {
        rwLock.readLock().lock();
        try {
            return lruMap.size();
        } finally {
            rwLock.readLock().unlock();
        }
    }

    /**
     * Clear all entries from the cache.
     */
    public void clear() {
        rwLock.writeLock().lock();
        try {
            lruMap.clear();
            root.children.clear();
            log.info("RadixPrefixCache cleared");
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    /**
     * Save the cache to disk in binary format.
     *
     * @param file the file to write to
     * @throws IOException if writing fails
     */
    public void saveToDisk(Path file) throws IOException {
        rwLock.readLock().lock();
        try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(file))) {
            // Header
            dos.writeInt(blockSize);
            dos.writeInt(maxCacheEntries);
            dos.writeLong(bytesPerBlock);

            // Entries
            dos.writeInt(lruMap.size());
            for (TrieEntry entry : lruMap.values()) {
                // Token IDs
                dos.writeInt(entry.fullTokenIds.length);
                for (int tokenId : entry.fullTokenIds) {
                    dos.writeInt(tokenId);
                }
                // Block IDs
                dos.writeInt(entry.blockIds.length);
                for (int blockId : entry.blockIds) {
                    dos.writeInt(blockId);
                }
            }

            log.info("Saved RadixPrefixCache to {}: {} entries", file, lruMap.size());
        } finally {
            rwLock.readLock().unlock();
        }
    }

    /**
     * Load a cache from disk.
     *
     * @param file the file to read from
     * @return the loaded cache
     * @throws IOException if reading fails
     */
    public static RadixPrefixCache loadFromDisk(Path file) throws IOException {
        try (DataInputStream dis = new DataInputStream(Files.newInputStream(file))) {
            int blockSize = dis.readInt();
            int maxCacheEntries = dis.readInt();
            long bytesPerBlock = dis.readLong();

            RadixPrefixCache cache = new RadixPrefixCache(blockSize, maxCacheEntries, bytesPerBlock);

            int numEntries = dis.readInt();
            for (int i = 0; i < numEntries; i++) {
                int tokenLen = dis.readInt();
                int[] tokenIds = new int[tokenLen];
                for (int j = 0; j < tokenLen; j++) {
                    tokenIds[j] = dis.readInt();
                }

                int blockLen = dis.readInt();
                int[] blockIds = new int[blockLen];
                for (int j = 0; j < blockLen; j++) {
                    blockIds[j] = dis.readInt();
                }

                cache.register(tokenIds, blockIds);
            }

            log.info("Loaded RadixPrefixCache from {}: {} entries", file, numEntries);
            return cache;
        }
    }

    @Override
    public String toString() {
        rwLock.readLock().lock();
        try {
            return String.format("RadixPrefixCache[entries=%d/%d, blockSize=%d]",
                    lruMap.size(), maxCacheEntries, blockSize);
        } finally {
            rwLock.readLock().unlock();
        }
    }

    // ========== Internal Helpers ==========

    /**
     * FNV-1a hash over a block of tokens.
     */
    private static long hashBlock(int[] tokens, int validCount) {
        long hash = 0xcbf29ce484222325L; // FNV offset basis
        for (int i = 0; i < validCount; i++) {
            hash ^= tokens[i];
            hash *= 0x100000001b3L; // FNV prime
        }
        return hash;
    }

    /**
     * FNV-1a hash over a full token prefix.
     */
    private static long hashFullPrefix(int[] tokenIds) {
        long hash = 0xcbf29ce484222325L;
        for (int tokenId : tokenIds) {
            hash ^= tokenId;
            hash *= 0x100000001b3L;
        }
        return hash;
    }

    /**
     * Check if a trie node's tokens match the given block tokens.
     */
    private static boolean matchesTokens(TrieNode node, int[] blockTokens, int validCount) {
        if (node.validCount != validCount) return false;
        if (node.tokens == null) return false;
        for (int i = 0; i < validCount; i++) {
            if (node.tokens[i] != blockTokens[i]) return false;
        }
        return true;
    }
}
