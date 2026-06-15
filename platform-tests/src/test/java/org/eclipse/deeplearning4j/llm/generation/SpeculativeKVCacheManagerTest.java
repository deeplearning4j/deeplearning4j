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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class SpeculativeKVCacheManagerTest extends BaseNd4jTestWithBackends {

    // ==================== Static KV: Zero-Copy Rollback ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStaticZeroCopyRollback(Nd4jBackend backend) {
        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();

        INDArray keyBuf = Nd4j.rand(DataType.FLOAT, 1, 2, 32, 4);
        INDArray valBuf = Nd4j.rand(DataType.FLOAT, 1, 2, 32, 4);
        Map<String, INDArray> kvBuffers = new HashMap<>();
        kvBuffers.put("key", keyBuf);
        kvBuffers.put("value", valBuf);

        int initialPos = 10;
        int checkpointId = manager.checkpoint(kvBuffers, initialPos);

        // Simulate speculative tokens being written at positions 10-14
        // (data is written to the same kvBuffers, no copy needed)

        // Rollback should return the checkpointed position
        int rolledBackPos = manager.rollback(checkpointId);
        assertEquals(initialPos, rolledBackPos, "Rollback should return the original cache position");

        // Checkpoint should be consumed
        assertFalse(manager.hasCheckpoint(checkpointId), "Checkpoint should be consumed after rollback");
        assertEquals(1, manager.getRollbackCount());

        keyBuf.close();
        valBuf.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStaticRollbackInvalidIdThrows(Nd4jBackend backend) {
        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();
        assertThrows(IllegalArgumentException.class, () -> manager.rollback(999));
    }

    // ==================== Paged KV: Rollback with Block Freeing ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPagedRollbackFreesBlocks(Nd4jBackend backend) {
        int maxBatch = 1, maxSeqLen = 256, numKvHeads = 2, headDim = 4, blockSize = 8;
        PagedKVCache cache = new PagedKVCache(maxBatch, maxSeqLen, numKvHeads, headDim,
                blockSize, DataType.FLOAT, 1.2);

        // Append initial tokens (fills 2 blocks: 16 tokens / 8 block_size)
        INDArray keys = Nd4j.rand(DataType.FLOAT, 16, numKvHeads, headDim);
        INDArray vals = Nd4j.rand(DataType.FLOAT, 16, numKvHeads, headDim);
        cache.append(0, keys, vals);

        int freeBlocksAtCheckpoint = cache.getNumFreeBlocks();
        int allocatedAtCheckpoint = cache.getNumAllocatedBlocks(0);

        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();
        int checkpointId = manager.checkpoint(cache, 0);

        // Append more speculative tokens (fills 1 more block)
        INDArray moreKeys = Nd4j.rand(DataType.FLOAT, 8, numKvHeads, headDim);
        INDArray moreVals = Nd4j.rand(DataType.FLOAT, 8, numKvHeads, headDim);
        cache.append(0, moreKeys, moreVals);

        int freeBlocksAfterSpec = cache.getNumFreeBlocks();
        assertTrue(freeBlocksAfterSpec < freeBlocksAtCheckpoint,
                "Speculative append should consume blocks");

        // Rollback should free the speculative blocks
        manager.rollbackPaged(checkpointId, cache, 0);

        assertEquals(freeBlocksAtCheckpoint, cache.getNumFreeBlocks(),
                "Rollback should restore free block count");
        assertEquals(allocatedAtCheckpoint, cache.getNumAllocatedBlocks(0),
                "Rollback should restore allocated block count");
        assertEquals(16, cache.getSequenceLength(0),
                "Rollback should restore sequence length");

        assertEquals(1, manager.getRollbackCount());

        cache.close();
        keys.close();
        vals.close();
        moreKeys.close();
        moreVals.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPagedRollbackSeqIdxMismatchThrows(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(2, 64, 2, 4, 8, DataType.FLOAT, 1.2);

        INDArray keys = Nd4j.rand(DataType.FLOAT, 4, 2, 4);
        INDArray vals = Nd4j.rand(DataType.FLOAT, 4, 2, 4);
        cache.append(0, keys, vals);

        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();
        int cpId = manager.checkpoint(cache, 0);

        // Try to rollback with wrong seqIdx
        assertThrows(IllegalArgumentException.class, () -> manager.rollbackPaged(cpId, cache, 1));

        cache.close();
        keys.close();
        vals.close();
    }

    // ==================== Full Accept ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFullAcceptStaticKv(Nd4jBackend backend) {
        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();

        Map<String, INDArray> kvBuffers = new HashMap<>();
        kvBuffers.put("key", Nd4j.zeros(DataType.FLOAT, 1, 2, 32, 4));

        int cpId = manager.checkpoint(kvBuffers, 10);
        assertTrue(manager.hasCheckpoint(cpId));

        // Accept 5 speculative tokens
        int newPos = manager.acceptTokens(cpId, 5);
        assertEquals(15, newPos, "Accepted position should be checkpoint pos + numAccepted");

        assertFalse(manager.hasCheckpoint(cpId), "Checkpoint consumed after accept");
        assertEquals(1, manager.getAcceptCount());
        assertEquals(0, manager.getActiveCheckpointCount());

        kvBuffers.get("key").close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFullAcceptPagedKv(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(1, 128, 2, 4, 8, DataType.FLOAT, 1.2);
        INDArray keys = Nd4j.rand(DataType.FLOAT, 10, 2, 4);
        INDArray vals = Nd4j.rand(DataType.FLOAT, 10, 2, 4);
        cache.append(0, keys, vals);

        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();
        int cpId = manager.checkpoint(cache, 0);

        int newPos = manager.acceptTokens(cpId, 3);
        assertEquals(10 + 3, newPos, "Accept on paged checkpoint should advance from context length");
        assertEquals(1, manager.getAcceptCount());

        cache.close();
        keys.close();
        vals.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAcceptInvalidCheckpointThrows(Nd4jBackend backend) {
        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();
        assertThrows(IllegalArgumentException.class, () -> manager.acceptTokens(999, 1));
    }

    // ==================== Full Rollback ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFullRollbackMultipleCheckpoints(Nd4jBackend backend) {
        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();

        Map<String, INDArray> kvBuffers = new HashMap<>();
        kvBuffers.put("key", Nd4j.zeros(DataType.FLOAT, 1, 2, 32, 4));

        int cp1 = manager.checkpoint(kvBuffers, 5);
        int cp2 = manager.checkpoint(kvBuffers, 10);
        int cp3 = manager.checkpoint(kvBuffers, 15);

        assertEquals(3, manager.getActiveCheckpointCount());

        // Rollback to earliest checkpoint
        int pos = manager.rollback(cp1);
        assertEquals(5, pos);
        assertEquals(2, manager.getActiveCheckpointCount());

        // Rollback second
        pos = manager.rollback(cp2);
        assertEquals(10, pos);
        assertEquals(1, manager.getActiveCheckpointCount());

        // Rollback third
        pos = manager.rollback(cp3);
        assertEquals(15, pos);
        assertEquals(0, manager.getActiveCheckpointCount());

        assertEquals(3, manager.getRollbackCount());
        assertEquals(3, manager.getCheckpointCount());

        kvBuffers.get("key").close();
    }

    // ==================== Tree Compaction ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTreeCompaction(Nd4jBackend backend) {
        int blockSize = 4;
        PagedKVCache cache = new PagedKVCache(1, 64, 2, 4, blockSize, DataType.FLOAT, 1.5);

        // Append 12 tokens (fills 3 blocks: 0-3, 4-7, 8-11)
        INDArray keys = Nd4j.rand(DataType.FLOAT, 12, 2, 4);
        INDArray vals = Nd4j.rand(DataType.FLOAT, 12, 2, 4);
        cache.append(0, keys, vals);

        assertEquals(3, cache.getNumAllocatedBlocks(0));
        int freeBlocksBefore = cache.getNumFreeBlocks();

        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();

        // Accept only positions in blocks 0 and 2 (positions 0,1 in block0 and 8,9 in block2)
        int[] acceptedPositions = {0, 1, 8, 9};
        manager.compactAcceptedPath(cache, 0, acceptedPositions);

        // Block 1 (covering positions 4-7) should have been freed
        // The accepted blocks are logical 0 and 2; logical 1 should be freed
        int freeBlocksAfter = cache.getNumFreeBlocks();
        assertTrue(freeBlocksAfter > freeBlocksBefore,
                "Compaction should free rejected branch blocks");

        // Sequence length should be updated to the last accepted position + 1
        assertEquals(10, cache.getSequenceLength(0));

        cache.close();
        keys.close();
        vals.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTreeCompactionEmptyAccepted(Nd4jBackend backend) {
        PagedKVCache cache = new PagedKVCache(1, 64, 2, 4, 4, DataType.FLOAT, 1.5);

        INDArray keys = Nd4j.rand(DataType.FLOAT, 4, 2, 4);
        INDArray vals = Nd4j.rand(DataType.FLOAT, 4, 2, 4);
        cache.append(0, keys, vals);

        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();

        // Empty or null accepted positions should be a no-op
        int seqLenBefore = cache.getSequenceLength(0);
        manager.compactAcceptedPath(cache, 0, new int[0]);
        assertEquals(seqLenBefore, cache.getSequenceLength(0));

        manager.compactAcceptedPath(cache, 0, null);
        assertEquals(seqLenBefore, cache.getSequenceLength(0));

        cache.close();
        keys.close();
        vals.close();
    }

    // ==================== Checkpoint Lifecycle ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCheckpointLifecycle(Nd4jBackend backend) {
        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();

        Map<String, INDArray> kvBuffers = new HashMap<>();
        kvBuffers.put("key", Nd4j.zeros(DataType.FLOAT, 1, 2, 16, 4));

        // Create
        int cpId = manager.checkpoint(kvBuffers, 5);
        assertTrue(manager.hasCheckpoint(cpId));
        assertEquals(1, manager.getActiveCheckpointCount());
        assertEquals(1, manager.getCheckpointCount());

        // Discard (not a rollback, no stats update for rollback/accept)
        manager.discardCheckpoint(cpId);
        assertFalse(manager.hasCheckpoint(cpId));
        assertEquals(0, manager.getActiveCheckpointCount());
        assertEquals(0, manager.getRollbackCount());
        assertEquals(0, manager.getAcceptCount());

        // Discarding non-existent is a no-op
        manager.discardCheckpoint(cpId);
        assertEquals(0, manager.getActiveCheckpointCount());

        kvBuffers.get("key").close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCheckpointIdsAreUnique(Nd4jBackend backend) {
        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();

        Map<String, INDArray> kvBuffers = new HashMap<>();
        kvBuffers.put("key", Nd4j.zeros(DataType.FLOAT, 1, 2, 16, 4));

        int id1 = manager.checkpoint(kvBuffers, 1);
        int id2 = manager.checkpoint(kvBuffers, 2);
        int id3 = manager.checkpoint(kvBuffers, 3);

        assertNotEquals(id1, id2);
        assertNotEquals(id2, id3);
        assertNotEquals(id1, id3);

        assertEquals(3, manager.getActiveCheckpointCount());

        kvBuffers.get("key").close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMixedStaticAndPagedCheckpoints(Nd4jBackend backend) {
        SpeculativeKVCacheManager manager = new SpeculativeKVCacheManager();

        // Static checkpoint
        Map<String, INDArray> kvBuffers = new HashMap<>();
        kvBuffers.put("key", Nd4j.zeros(DataType.FLOAT, 1, 2, 16, 4));
        int staticCpId = manager.checkpoint(kvBuffers, 5);

        // Paged checkpoint
        PagedKVCache cache = new PagedKVCache(1, 64, 2, 4, 8, DataType.FLOAT, 1.2);
        INDArray keys = Nd4j.rand(DataType.FLOAT, 4, 2, 4);
        INDArray vals = Nd4j.rand(DataType.FLOAT, 4, 2, 4);
        cache.append(0, keys, vals);
        int pagedCpId = manager.checkpoint(cache, 0);

        assertEquals(2, manager.getActiveCheckpointCount());
        assertTrue(manager.hasCheckpoint(staticCpId));
        assertTrue(manager.hasCheckpoint(pagedCpId));

        // Accept the static one
        int newPos = manager.acceptTokens(staticCpId, 3);
        assertEquals(8, newPos);
        assertEquals(1, manager.getActiveCheckpointCount());

        // Rollback the paged one
        manager.rollbackPaged(pagedCpId, cache, 0);
        assertEquals(0, manager.getActiveCheckpointCount());

        assertEquals(2, manager.getCheckpointCount());
        assertEquals(1, manager.getAcceptCount());
        assertEquals(1, manager.getRollbackCount());

        cache.close();
        kvBuffers.get("key").close();
        keys.close();
        vals.close();
    }
}
