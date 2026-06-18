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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * KV cache manager for beam search with copy-on-write block sharing.
 *
 * <p>During beam search, multiple beams share a common prompt prefix and diverge
 * only at the most recent tokens. Without sharing, each beam would need a full copy
 * of the KV cache, multiplying memory by the beam width.</p>
 *
 * <h3>Copy-on-Write (CoW)</h3>
 * <p>Beams share page tables via the {@link KVCachePrefixTree}. When a beam forks
 * (is selected as a parent for the next step), the child inherits the parent's page
 * table with shared reference counts. Physical block data is only copied when a shared
 * block needs modification (a new token appended to a shared last block).</p>
 *
 * <h3>Beam lifecycle</h3>
 * <ol>
 *   <li>{@link #initializeBeams(int[])} -- register prompt, create initial beam slots</li>
 *   <li>{@link #appendToken(int, int, INDArray, INDArray)} -- append decoded token to a beam</li>
 *   <li>{@link #reindexBeams(int[])} -- after beam selection, reassign parent page tables</li>
 *   <li>{@link #freeBeam(int)} -- release a pruned beam's private blocks</li>
 * </ol>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see PagedKVCache
 * @see KVCachePrefixTree
 */
@Slf4j
public class BeamKVCacheManager {

    @Getter private final PagedKVCache kvCache;
    @Getter private final KVCachePrefixTree prefixTree;
    @Getter private final int numBeams;

    /** Token IDs for each beam. */
    private final List<int[]> beamTokens;

    /** Whether each beam slot is active. */
    private final boolean[] beamActive;

    /**
     * Create a beam KV cache manager.
     *
     * @param kvCache    the underlying paged KV cache
     * @param prefixTree the prefix tree for block sharing
     * @param numBeams   the beam width
     */
    public BeamKVCacheManager(PagedKVCache kvCache, KVCachePrefixTree prefixTree, int numBeams) {
        if (kvCache == null) throw new IllegalArgumentException("kvCache must not be null");
        if (prefixTree == null) throw new IllegalArgumentException("prefixTree must not be null");
        if (numBeams < 1) throw new IllegalArgumentException("numBeams must be >= 1, got " + numBeams);
        if (numBeams > kvCache.getMaxBatchSize()) {
            throw new IllegalArgumentException("numBeams (" + numBeams + ") exceeds maxBatchSize ("
                    + kvCache.getMaxBatchSize() + ")");
        }

        this.kvCache = kvCache;
        this.prefixTree = prefixTree;
        this.numBeams = numBeams;
        this.beamTokens = new ArrayList<>(numBeams);
        this.beamActive = new boolean[numBeams];

        for (int i = 0; i < numBeams; i++) {
            beamTokens.add(new int[0]);
        }
    }

    /**
     * Initialize all beams with the prompt tokens.
     *
     * <p>Registers the prompt in the prefix tree and creates shared page tables
     * for all beam slots. All beams start with the same KV cache (shared blocks).</p>
     *
     * @param promptTokenIds the prompt token IDs
     */
    public void initializeBeams(int[] promptTokenIds) {
        if (promptTokenIds == null || promptTokenIds.length == 0) {
            throw new IllegalArgumentException("promptTokenIds must not be empty");
        }

        log.info("Initializing {} beams with prompt of {} tokens", numBeams, promptTokenIds.length);

        // Register prompt for the first beam (allocates physical blocks)
        KVCachePrefixTree.BlockAllocator allocator = () -> {
            // Use beam 0's slot for initial allocation
            return kvCache.allocateBlock();
        };

        List<KVCachePrefixTree.BlockMapping> mappings = prefixTree.registerSequence(promptTokenIds, allocator);

        // Set up page tables for all beams sharing the prompt blocks
        for (int beam = 0; beam < numBeams; beam++) {
            beamTokens.set(beam, promptTokenIds.clone());
            beamActive[beam] = true;

            // Register additional beams in prefix tree (increments ref counts on shared blocks)
            if (beam > 0) {
                prefixTree.registerSequence(promptTokenIds, allocator);
            }
        }

        log.info("Beams initialized: {} shared blocks, {} total blocks in prefix tree",
                prefixTree.getSharedBlockCount(), prefixTree.getNodeCount());
    }

    /**
     * Fork a beam: create a new beam as a copy of an existing parent beam.
     *
     * <p>This is O(1) because it only copies the page table reference and increments
     * block reference counts in the prefix tree. No physical block data is copied.</p>
     *
     * @param parentBeamIdx index of the parent beam to fork from
     * @param newBeamIdx    index of the new beam slot to populate
     */
    public void forkBeam(int parentBeamIdx, int newBeamIdx) {
        validateBeamIdx(parentBeamIdx);
        validateBeamIdx(newBeamIdx);

        if (!beamActive[parentBeamIdx]) {
            throw new IllegalStateException("Cannot fork from inactive beam " + parentBeamIdx);
        }

        // If the new slot was active, free it first
        if (beamActive[newBeamIdx]) {
            freeBeam(newBeamIdx);
        }

        // Copy token sequence
        int[] parentTokens = beamTokens.get(parentBeamIdx);
        beamTokens.set(newBeamIdx, parentTokens.clone());
        beamActive[newBeamIdx] = true;

        // Register in prefix tree (increments ref counts on shared blocks)
        KVCachePrefixTree.BlockAllocator allocator = kvCache::allocateBlock;
        prefixTree.registerSequence(parentTokens, allocator);

        log.debug("Forked beam {} -> {}, {} tokens, shared blocks={}",
                parentBeamIdx, newBeamIdx, parentTokens.length, prefixTree.getSharedBlockCount());
    }

    /**
     * Append a token to a beam, handling copy-on-write for shared blocks.
     *
     * <p>If the beam's last block is shared (reference count &gt; 1 in the prefix tree),
     * the block is copied to a new physical block before appending. If the block is
     * not shared, the token is appended directly.</p>
     *
     * @param beamIdx  index of the beam to append to
     * @param tokenId  the token ID being appended
     * @param newKey   new key tensor for this token, shape [1, 1, numKvHeads, headDim]
     * @param newValue new value tensor for this token (same shape as newKey)
     */
    public void appendToken(int beamIdx, int tokenId, INDArray newKey, INDArray newValue) {
        validateBeamIdx(beamIdx);
        if (!beamActive[beamIdx]) {
            throw new IllegalStateException("Cannot append to inactive beam " + beamIdx);
        }

        int[] currentTokens = beamTokens.get(beamIdx);
        int blockSize = prefixTree.getBlockSize();

        // Check if the last block needs copy-on-write
        if (currentTokens.length > 0 && prefixTree.needsCopyOnWrite(currentTokens)) {
            performCopyOnWrite(beamIdx, currentTokens);
        }

        // Append the new KV entry to the beam's slot in the paged cache
        kvCache.append(beamIdx, newKey, newValue);

        // Update token sequence
        int[] newTokens = Arrays.copyOf(currentTokens, currentTokens.length + 1);
        newTokens[currentTokens.length] = tokenId;
        beamTokens.set(beamIdx, newTokens);

        log.trace("Appended token {} to beam {}, new length={}", tokenId, beamIdx, newTokens.length);
    }

    /**
     * Reindex beams after beam selection.
     *
     * <p>After scoring and selecting the best beams, the beam indices may need to be
     * reassigned. For example, if beam 2 is selected as the continuation for slots 0 and 1,
     * their page tables need to point to beam 2's blocks.</p>
     *
     * @param beamIndices array of length {@code numBeams} where {@code beamIndices[i]} is the
     *                    source beam for the new beam {@code i}
     */
    public void reindexBeams(int[] beamIndices) {
        if (beamIndices.length != numBeams) {
            throw new IllegalArgumentException("beamIndices length (" + beamIndices.length
                    + ") must match numBeams (" + numBeams + ")");
        }

        // Save old token sequences
        int[][] oldTokens = new int[numBeams][];
        for (int i = 0; i < numBeams; i++) {
            oldTokens[i] = beamTokens.get(i);
        }

        // Unregister all old beams from prefix tree
        for (int i = 0; i < numBeams; i++) {
            if (beamActive[i] && oldTokens[i].length > 0) {
                List<Integer> freed = prefixTree.unregisterSequence(oldTokens[i]);
                for (int blockId : freed) {
                    kvCache.freeSequence(i);
                }
            }
        }

        // Re-register with new assignments
        KVCachePrefixTree.BlockAllocator allocator = kvCache::allocateBlock;
        for (int i = 0; i < numBeams; i++) {
            int sourceBeam = beamIndices[i];
            int[] sourceTokens = oldTokens[sourceBeam];
            beamTokens.set(i, sourceTokens.clone());
            beamActive[i] = true;

            if (sourceTokens.length > 0) {
                prefixTree.registerSequence(sourceTokens, allocator);
            }
        }

        log.debug("Reindexed beams: {}, shared blocks={}", Arrays.toString(beamIndices),
                prefixTree.getSharedBlockCount());
    }

    /**
     * Free a beam's private blocks and deactivate it.
     *
     * <p>Unregisters the beam from the prefix tree (decrementing reference counts)
     * and frees any blocks that become unreferenced.</p>
     *
     * @param beamIdx index of the beam to free
     */
    public void freeBeam(int beamIdx) {
        validateBeamIdx(beamIdx);
        if (!beamActive[beamIdx]) return;

        int[] tokens = beamTokens.get(beamIdx);
        if (tokens.length > 0) {
            List<Integer> freedBlocks = prefixTree.unregisterSequence(tokens);
            log.debug("Freed beam {}: {} blocks returned to pool", beamIdx, freedBlocks.size());
        }

        // Free the sequence slot in the KV cache
        kvCache.freeSequence(beamIdx);

        beamTokens.set(beamIdx, new int[0]);
        beamActive[beamIdx] = false;
    }

    /**
     * Get the page table for a beam as an INDArray (for passing to native attention ops).
     *
     * @param beamIdx beam index
     * @return int32 tensor with physical block IDs for the beam
     */
    public INDArray getBeamPageTable(int beamIdx) {
        validateBeamIdx(beamIdx);
        return kvCache.getPageTableArray(beamIdx);
    }

    /**
     * Get the current token sequence for a beam.
     *
     * @param beamIdx beam index
     * @return copy of the token ID array
     */
    public int[] getBeamTokens(int beamIdx) {
        validateBeamIdx(beamIdx);
        return beamTokens.get(beamIdx).clone();
    }

    /**
     * Check if a beam slot is active.
     *
     * @param beamIdx beam index
     * @return true if the beam is active
     */
    public boolean isBeamActive(int beamIdx) {
        validateBeamIdx(beamIdx);
        return beamActive[beamIdx];
    }

    /**
     * Perform copy-on-write on the last block of a beam's KV cache.
     *
     * <p>Copies the shared block's data to a new physical block so the beam can
     * append without modifying data shared with other beams.</p>
     */
    private void performCopyOnWrite(int beamIdx, int[] tokens) {
        int blockSize = prefixTree.getBlockSize();
        int lastBlockIdx = (tokens.length - 1) / blockSize;

        // Get the current physical block for the last logical block
        int[] pageTable = kvCache.getRawPageTable(beamIdx);
        if (lastBlockIdx >= pageTable.length) return;

        int oldBlockId = pageTable[lastBlockIdx];
        if (oldBlockId < 0) return;

        // Allocate a new block and copy all data from old to new
        int newBlockId = kvCache.allocateBlock();
        kvCache.copyBlock(oldBlockId, newBlockId);

        // Update the page table to point to the new block
        pageTable[lastBlockIdx] = newBlockId;

        log.debug("CoW: beam {} block {} copied {} -> {}", beamIdx, lastBlockIdx, oldBlockId, newBlockId);
    }

    private void validateBeamIdx(int beamIdx) {
        if (beamIdx < 0 || beamIdx >= numBeams) {
            throw new IllegalArgumentException("beamIdx out of range [0, " + numBeams + "): " + beamIdx);
        }
    }

    @Override
    public String toString() {
        int activeCount = 0;
        for (boolean active : beamActive) {
            if (active) activeCount++;
        }
        return String.format("BeamKVCacheManager[beams=%d/%d active, sharedBlocks=%d, %s]",
                activeCount, numBeams, prefixTree.getSharedBlockCount(), kvCache);
    }
}
