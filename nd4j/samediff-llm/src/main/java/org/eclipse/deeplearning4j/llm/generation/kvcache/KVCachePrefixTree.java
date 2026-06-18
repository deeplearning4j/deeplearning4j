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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Trie-based prefix tree for KV cache block sharing across sequences.
 *
 * <p>When multiple sequences share a common prompt prefix (e.g., system prompt,
 * few-shot examples), their KV cache blocks for the shared prefix are identical.
 * This class tracks token prefix to physical block ID mappings with reference
 * counting, enabling copy-on-write sharing of KV cache blocks.</p>
 *
 * <h3>How it works</h3>
 * <p>Each trie node represents a block-aligned token range. When a sequence is registered,
 * the prefix tree walks the token sequence block-by-block, matching or creating nodes.
 * Physical block IDs are stored at each node, and a reference count tracks how many
 * sequences share that block.</p>
 *
 * <pre>
 * Example with blockSize=4:
 *
 * Sequence A: [10, 20, 30, 40, 50, 60, 70, 80]
 * Sequence B: [10, 20, 30, 40, 90, 91, 92, 93]
 *
 * Trie:
 *   root
 *     └─ [10,20,30,40] → block 0 (refCount=2)  ← shared!
 *           ├─ [50,60,70,80] → block 1 (refCount=1)  ← seq A only
 *           └─ [90,91,92,93] → block 2 (refCount=1)  ← seq B only
 * </pre>
 *
 * <h3>Copy-on-Write</h3>
 * <p>When a sequence needs to modify a shared block (e.g., appending tokens to a
 * partially-filled shared block), the block must be copied first. The trie's
 * reference count determines whether a copy is needed (refCount > 1).</p>
 *
 * <h3>Integration with PagedKVCache</h3>
 * <p>This trie maps logical token prefixes to physical block IDs managed by
 * {@link PagedKVCache}. The paged cache handles allocation/deallocation; this
 * trie handles sharing and prefix matching.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see PagedKVCache
 */
@Slf4j
public class KVCachePrefixTree {

    /**
     * Node in the prefix trie. Each node corresponds to one block-sized chunk
     * of the token sequence.
     */
    static class TrieNode {
        /** The token IDs for this block (length = blockSize, possibly padded). */
        @Getter
        private final int[] blockTokens;

        /** The physical block ID in the KV cache pool. -1 if not yet assigned. */
        @Getter
        private int physicalBlockId;

        /** Number of sequences currently sharing this block. */
        private final AtomicInteger refCount;

        /** Children indexed by the hash of their block tokens. */
        private final Map<Long, TrieNode> children;

        /** Number of valid tokens in this block (may be < blockSize for last block). */
        @Getter
        private int validTokenCount;

        TrieNode(int[] blockTokens, int physicalBlockId, int validTokenCount) {
            this.blockTokens = blockTokens;
            this.physicalBlockId = physicalBlockId;
            this.validTokenCount = validTokenCount;
            this.refCount = new AtomicInteger(0);
            this.children = new HashMap<>();
        }

        /** Increment reference count and return new value. */
        int addRef() {
            return refCount.incrementAndGet();
        }

        /** Decrement reference count and return new value. */
        int release() {
            return refCount.decrementAndGet();
        }

        /** Current reference count. */
        int getRefCount() {
            return refCount.get();
        }

        /** Whether this block is shared (refCount > 1). */
        boolean isShared() {
            return refCount.get() > 1;
        }

        /** Set the physical block ID (used during copy-on-write). */
        void setPhysicalBlockId(int blockId) {
            this.physicalBlockId = blockId;
        }
    }

    /** Root of the trie (sentinel node, does not correspond to a block). */
    private final TrieNode root;

    /** Block size in tokens. Must match the PagedKVCache block size. */
    @Getter
    private final int blockSize;

    /** Total number of nodes in the trie. */
    private int nodeCount;

    /** Total number of shared blocks (blocks with refCount > 1). */
    private int sharedBlockCount;

    /**
     * Create a KV cache prefix tree.
     *
     * @param blockSize number of tokens per block, must match the associated
     *                  {@link PagedKVCache} block size
     */
    public KVCachePrefixTree(int blockSize) {
        if (blockSize < 1) {
            throw new IllegalArgumentException("blockSize must be >= 1, got " + blockSize);
        }
        this.blockSize = blockSize;
        this.root = new TrieNode(new int[0], -1, 0);
        this.nodeCount = 0;
        this.sharedBlockCount = 0;
    }

    /**
     * Register a token sequence and return its block mapping.
     *
     * <p>Walks the trie block-by-block, matching existing nodes where possible
     * (incrementing their reference counts) and creating new nodes where the
     * sequence diverges from existing prefixes.</p>
     *
     * @param tokenIds the full token sequence
     * @param blockAllocator function that allocates a new physical block ID
     * @return list of {@link BlockMapping} entries mapping logical positions to physical block IDs
     */
    public List<BlockMapping> registerSequence(int[] tokenIds, BlockAllocator blockAllocator) {
        List<BlockMapping> mappings = new ArrayList<>();
        TrieNode current = root;
        int pos = 0;

        while (pos < tokenIds.length) {
            int blockEnd = Math.min(pos + blockSize, tokenIds.length);
            int[] blockTokens = new int[blockSize];
            int validCount = blockEnd - pos;
            System.arraycopy(tokenIds, pos, blockTokens, 0, validCount);

            long blockHash = hashBlock(blockTokens, validCount);
            TrieNode child = current.children.get(blockHash);

            if (child != null && matchesBlock(child, blockTokens, validCount)) {
                // Existing prefix -- share this block
                int prevRef = child.getRefCount();
                child.addRef();
                if (prevRef == 1 && child.getRefCount() > 1) {
                    sharedBlockCount++;
                }
                log.trace("Shared block at pos {} (block {}), refCount={}",
                        pos, child.getPhysicalBlockId(), child.getRefCount());
            } else {
                // New prefix -- allocate a new block
                int physicalBlockId = blockAllocator.allocate();
                child = new TrieNode(blockTokens.clone(), physicalBlockId, validCount);
                child.addRef();
                current.children.put(blockHash, child);
                nodeCount++;
                log.trace("New block at pos {} (block {}), tokens={}", pos, physicalBlockId, validCount);
            }

            mappings.add(new BlockMapping(pos / blockSize, child.getPhysicalBlockId(),
                    child.isShared(), validCount));
            current = child;
            pos = blockEnd;
        }

        return mappings;
    }

    /**
     * Unregister a token sequence, decrementing reference counts.
     *
     * <p>Walks the trie to find matching nodes and decrements their reference counts.
     * Nodes with zero references are eligible for removal (and their physical blocks
     * can be freed).</p>
     *
     * @param tokenIds the token sequence to unregister
     * @return list of physical block IDs that are no longer referenced (can be freed)
     */
    public List<Integer> unregisterSequence(int[] tokenIds) {
        List<Integer> freedBlocks = new ArrayList<>();
        List<TrieNode> path = new ArrayList<>();
        List<Long> pathHashes = new ArrayList<>();
        List<TrieNode> parentPath = new ArrayList<>();

        TrieNode current = root;
        int pos = 0;

        // Walk the trie to find all matching nodes
        while (pos < tokenIds.length) {
            int blockEnd = Math.min(pos + blockSize, tokenIds.length);
            int[] blockTokens = new int[blockSize];
            int validCount = blockEnd - pos;
            System.arraycopy(tokenIds, pos, blockTokens, 0, validCount);

            long blockHash = hashBlock(blockTokens, validCount);
            TrieNode child = current.children.get(blockHash);

            if (child == null || !matchesBlock(child, blockTokens, validCount)) {
                break; // Sequence not fully registered
            }

            parentPath.add(current);
            pathHashes.add(blockHash);
            path.add(child);
            current = child;
            pos = blockEnd;
        }

        // Decrement ref counts from leaf to root, collecting freed blocks
        for (int i = path.size() - 1; i >= 0; i--) {
            TrieNode node = path.get(i);
            int prevRef = node.getRefCount();
            int newRef = node.release();

            if (prevRef > 1 && newRef <= 1) {
                sharedBlockCount--;
            }

            if (newRef <= 0) {
                // Node is unreferenced -- free the block and remove from parent
                freedBlocks.add(node.getPhysicalBlockId());
                TrieNode parent = parentPath.get(i);
                parent.children.remove(pathHashes.get(i));
                nodeCount--;
            }
        }

        return freedBlocks;
    }

    /**
     * Look up the shared block IDs for a token prefix.
     *
     * <p>Returns the physical block IDs for the longest matching prefix of the given
     * token sequence. Does not modify reference counts.</p>
     *
     * @param tokenIds the token sequence to look up
     * @return list of physical block IDs for the matching prefix (empty if no match)
     */
    public List<Integer> lookupPrefix(int[] tokenIds) {
        List<Integer> blockIds = new ArrayList<>();
        TrieNode current = root;
        int pos = 0;

        while (pos < tokenIds.length) {
            int blockEnd = Math.min(pos + blockSize, tokenIds.length);
            int[] blockTokens = new int[blockSize];
            int validCount = blockEnd - pos;
            System.arraycopy(tokenIds, pos, blockTokens, 0, validCount);

            long blockHash = hashBlock(blockTokens, validCount);
            TrieNode child = current.children.get(blockHash);

            if (child == null || !matchesBlock(child, blockTokens, validCount)) {
                break;
            }

            blockIds.add(child.getPhysicalBlockId());
            current = child;
            pos = blockEnd;
        }

        return blockIds;
    }

    /**
     * Check whether a block needs copy-on-write before modification.
     *
     * @param tokenPrefix the token prefix up to and including the block to check
     * @return true if the block is shared and must be copied before writing
     */
    public boolean needsCopyOnWrite(int[] tokenPrefix) {
        TrieNode current = root;
        int pos = 0;
        TrieNode lastNode = null;

        while (pos < tokenPrefix.length) {
            int blockEnd = Math.min(pos + blockSize, tokenPrefix.length);
            int[] blockTokens = new int[blockSize];
            int validCount = blockEnd - pos;
            System.arraycopy(tokenPrefix, pos, blockTokens, 0, validCount);

            long blockHash = hashBlock(blockTokens, validCount);
            TrieNode child = current.children.get(blockHash);

            if (child == null) {
                return false; // Block doesn't exist in trie
            }

            lastNode = child;
            current = child;
            pos = blockEnd;
        }

        return lastNode != null && lastNode.isShared();
    }

    /**
     * Get the total number of nodes in the trie.
     */
    public int getNodeCount() {
        return nodeCount;
    }

    /**
     * Get the number of blocks that are shared by multiple sequences.
     */
    public int getSharedBlockCount() {
        return sharedBlockCount;
    }

    /**
     * Get memory savings from block sharing.
     *
     * @param bytesPerBlock size of one KV cache block in bytes
     * @return approximate bytes saved through sharing
     */
    public long getMemorySavedBytes(long bytesPerBlock) {
        // Each shared block saves (refCount - 1) copies
        return countTotalSavedRefs(root) * bytesPerBlock;
    }

    @Override
    public String toString() {
        return String.format("KVCachePrefixTree[nodes=%d, shared=%d, blockSize=%d]",
                nodeCount, sharedBlockCount, blockSize);
    }

    // ========== Internal Helpers ==========

    /**
     * Hash a block of tokens for trie lookup.
     * Uses FNV-1a hash on the valid token portion.
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
     * Check if a trie node matches the given block tokens.
     */
    private static boolean matchesBlock(TrieNode node, int[] blockTokens, int validCount) {
        if (node.getValidTokenCount() != validCount) {
            return false;
        }
        int[] nodeTokens = node.getBlockTokens();
        for (int i = 0; i < validCount; i++) {
            if (nodeTokens[i] != blockTokens[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Count total saved references across all nodes (for memory savings calculation).
     */
    private long countTotalSavedRefs(TrieNode node) {
        long saved = 0;
        if (node.getRefCount() > 1) {
            saved += node.getRefCount() - 1;
        }
        for (TrieNode child : node.children.values()) {
            saved += countTotalSavedRefs(child);
        }
        return saved;
    }

    // ========== Supporting Types ==========

    /**
     * Mapping from a logical block index to a physical block ID.
     */
    public static class BlockMapping {
        /** Logical block index (0-based, position / blockSize). */
        @Getter
        private final int logicalBlockIndex;

        /** Physical block ID in the KV cache pool. */
        @Getter
        private final int physicalBlockId;

        /** Whether this block is shared with other sequences. */
        @Getter
        private final boolean shared;

        /** Number of valid tokens in this block. */
        @Getter
        private final int validTokenCount;

        public BlockMapping(int logicalBlockIndex, int physicalBlockId, boolean shared, int validTokenCount) {
            this.logicalBlockIndex = logicalBlockIndex;
            this.physicalBlockId = physicalBlockId;
            this.shared = shared;
            this.validTokenCount = validTokenCount;
        }

        @Override
        public String toString() {
            return String.format("BlockMapping[logical=%d, physical=%d, shared=%s, valid=%d]",
                    logicalBlockIndex, physicalBlockId, shared, validTokenCount);
        }
    }

    /**
     * Functional interface for allocating physical block IDs from the KV cache pool.
     */
    @FunctionalInterface
    public interface BlockAllocator {
        /**
         * Allocate a new physical block ID.
         *
         * @return the allocated block ID
         * @throws IllegalStateException if the pool is exhausted
         */
        int allocate();
    }
}
