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
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class KVCacheCheckpointTest extends BaseNd4jTestWithBackends {

    // ==================== KVCacheCheckpoint: fromStaticKv ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStaticCheckpointRoundtrip(Nd4jBackend backend) {
        int batch = 1, heads = 2, maxSeq = 16, headDim = 4;
        int cachePosition = 5;

        // Create KV buffers with known data
        INDArray keyBuf = Nd4j.linspace(1, batch * heads * maxSeq * headDim,
                (long) batch * heads * maxSeq * headDim, DataType.FLOAT).reshape(batch, heads, maxSeq, headDim);
        INDArray valBuf = Nd4j.linspace(100, 100 + batch * heads * maxSeq * headDim - 1,
                (long) batch * heads * maxSeq * headDim, DataType.FLOAT).reshape(batch, heads, maxSeq, headDim);

        Map<String, INDArray> kvBuffers = new HashMap<>();
        kvBuffers.put("present.0.key", keyBuf);
        kvBuffers.put("present.0.value", valBuf);

        KVCacheCheckpoint checkpoint = KVCacheCheckpoint.fromStaticKv(kvBuffers, cachePosition);

        // Verify metadata
        assertEquals(cachePosition, checkpoint.getCachePosition());
        assertEquals(0, checkpoint.getBasePosition());
        assertFalse(checkpoint.isIncremental());
        assertNull(checkpoint.getPageTableSnapshot());

        // Verify stored buffers are copies with correct shape
        Map<String, INDArray> stored = checkpoint.getKvBuffers();
        assertEquals(2, stored.size());
        assertArrayEquals(new long[]{batch, heads, cachePosition, headDim}, stored.get("present.0.key").shape());
        assertArrayEquals(new long[]{batch, heads, cachePosition, headDim}, stored.get("present.0.value").shape());

        // Verify the stored data matches the original valid portion
        INDArray expectedKey = keyBuf.get(
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, cachePosition),
                org.nd4j.linalg.indexing.NDArrayIndex.all());
        assertEquals(expectedKey, stored.get("present.0.key"));

        // Verify memory bytes
        assertTrue(checkpoint.getMemoryBytes() > 0);

        checkpoint.close();
        keyBuf.close();
        valBuf.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStaticCheckpointZeroPosition(Nd4jBackend backend) {
        INDArray keyBuf = Nd4j.zeros(DataType.FLOAT, 1, 2, 16, 4);
        Map<String, INDArray> kvBuffers = new HashMap<>();
        kvBuffers.put("key", keyBuf);

        KVCacheCheckpoint checkpoint = KVCacheCheckpoint.fromStaticKv(kvBuffers, 0);

        assertEquals(0, checkpoint.getCachePosition());
        assertTrue(checkpoint.getKvBuffers().isEmpty());
        assertEquals(0, checkpoint.getMemoryBytes());

        checkpoint.close();
        keyBuf.close();
    }

    // ==================== KVCacheCheckpoint: fromPagedKv ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPagedCheckpointRoundtrip(Nd4jBackend backend) {
        int maxBatch = 2, maxSeqLen = 128, numKvHeads = 2, headDim = 4, blockSize = 16;
        PagedKVCache cache = new PagedKVCache(maxBatch, maxSeqLen, numKvHeads, headDim,
                blockSize, DataType.FLOAT, 1.2);

        // Append some tokens to seq 0
        int numTokens = 20;
        INDArray keys = Nd4j.rand(DataType.FLOAT, numTokens, numKvHeads, headDim);
        INDArray vals = Nd4j.rand(DataType.FLOAT, numTokens, numKvHeads, headDim);
        cache.append(0, keys, vals);

        assertEquals(numTokens, cache.getSequenceLength(0));

        KVCacheCheckpoint checkpoint = KVCacheCheckpoint.fromPagedKv(cache, 0);

        // Verify metadata
        assertEquals(numTokens, checkpoint.getCachePosition());
        assertFalse(checkpoint.isIncremental());
        assertNotNull(checkpoint.getPageTableSnapshot());

        // 20 tokens / 16 block_size = 2 blocks
        int expectedBlocks = (numTokens + blockSize - 1) / blockSize;
        assertEquals(expectedBlocks, checkpoint.getPageTableSnapshot().length);

        // Verify block data was copied (2 blocks * 2 entries each = 4 entries)
        assertEquals(expectedBlocks * 2, checkpoint.getKvBuffers().size());
        assertTrue(checkpoint.getKvBuffers().containsKey("key_block_0"));
        assertTrue(checkpoint.getKvBuffers().containsKey("value_block_0"));

        assertTrue(checkpoint.getMemoryBytes() > 0);

        checkpoint.close();
        cache.close();
        keys.close();
        vals.close();
    }

    // ==================== KVCacheCheckpoint: incrementalDelta ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIncrementalDelta(Nd4jBackend backend) {
        int batch = 1, heads = 2, maxSeq = 32, headDim = 4;

        INDArray keyBuf = Nd4j.linspace(1, batch * heads * maxSeq * headDim,
                (long) batch * heads * maxSeq * headDim, DataType.FLOAT).reshape(batch, heads, maxSeq, headDim);

        Map<String, INDArray> kvBuffers = new HashMap<>();
        kvBuffers.put("key", keyBuf);

        // Base checkpoint at position 5
        KVCacheCheckpoint base = KVCacheCheckpoint.fromStaticKv(kvBuffers, 5);

        // Full checkpoint at position 10
        KVCacheCheckpoint full = KVCacheCheckpoint.fromStaticKv(kvBuffers, 10);

        // Create delta
        KVCacheCheckpoint delta = full.incrementalDelta(base);

        assertTrue(delta.isIncremental());
        assertEquals(10, delta.getCachePosition());
        assertEquals(5, delta.getBasePosition());

        // Delta should contain data from position 5 to 10 (5 positions)
        INDArray deltaKey = delta.getKvBuffers().get("key");
        assertNotNull(deltaKey);
        assertArrayEquals(new long[]{batch, heads, 5, headDim}, deltaKey.shape());

        delta.close();
        full.close();
        base.close();
        keyBuf.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIncrementalDeltaBaseAheadThrows(Nd4jBackend backend) {
        INDArray keyBuf = Nd4j.zeros(DataType.FLOAT, 1, 2, 32, 4);
        Map<String, INDArray> kvBuffers = new HashMap<>();
        kvBuffers.put("key", keyBuf);

        KVCacheCheckpoint small = KVCacheCheckpoint.fromStaticKv(kvBuffers, 5);
        KVCacheCheckpoint large = KVCacheCheckpoint.fromStaticKv(kvBuffers, 10);

        assertThrows(IllegalArgumentException.class, () -> small.incrementalDelta(large));

        small.close();
        large.close();
        keyBuf.close();
    }

    // ==================== KVCacheCheckpoint: disk persistence ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDiskPersistence(Nd4jBackend backend, @TempDir Path tempDir) throws Exception {
        int batch = 1, heads = 2, cachePos = 4, headDim = 4;
        INDArray keyBuf = Nd4j.rand(DataType.FLOAT, batch, heads, 16, headDim);
        INDArray valBuf = Nd4j.rand(DataType.FLOAT, batch, heads, 16, headDim);

        Map<String, INDArray> kvBuffers = new HashMap<>();
        kvBuffers.put("key", keyBuf);
        kvBuffers.put("value", valBuf);

        KVCacheCheckpoint original = KVCacheCheckpoint.fromStaticKv(kvBuffers, cachePos);

        Path file = tempDir.resolve("checkpoint.bin");
        original.saveToDisk(file);

        assertTrue(file.toFile().exists());
        assertTrue(file.toFile().length() > 0);

        KVCacheCheckpoint loaded = KVCacheCheckpoint.loadFromDisk(file);

        assertEquals(original.getCachePosition(), loaded.getCachePosition());
        assertEquals(original.getBasePosition(), loaded.getBasePosition());
        assertEquals(original.isIncremental(), loaded.isIncremental());
        assertEquals(original.getKvBuffers().size(), loaded.getKvBuffers().size());

        // Verify data matches
        for (String name : original.getKvBuffers().keySet()) {
            INDArray origArr = original.getKvBuffers().get(name);
            INDArray loadedArr = loaded.getKvBuffers().get(name);
            assertNotNull(loadedArr, "Missing buffer: " + name);
            assertArrayEquals(origArr.shape(), loadedArr.shape());
            assertEquals(origArr.dataType(), loadedArr.dataType());
            // Verify values match (element-wise comparison)
            double maxDiff = origArr.sub(loadedArr).amaxNumber().doubleValue();
            assertEquals(0.0, maxDiff, 1e-6, "Data mismatch for buffer: " + name);
        }

        loaded.close();
        original.close();
        keyBuf.close();
        valBuf.close();
    }

    // ==================== KVCacheCheckpointManager ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testManagerFIFOEviction(Nd4jBackend backend) {
        int maxCheckpoints = 3;
        try (KVCacheCheckpointManager manager = new KVCacheCheckpointManager(maxCheckpoints)) {
            INDArray buf = Nd4j.zeros(DataType.FLOAT, 1, 2, 16, 4);
            Map<String, INDArray> kvBuffers = new HashMap<>();
            kvBuffers.put("key", buf);

            // Create checkpoints up to capacity
            String id1 = manager.createCheckpoint(kvBuffers, 1);
            String id2 = manager.createCheckpoint(kvBuffers, 2);
            String id3 = manager.createCheckpoint(kvBuffers, 3);
            assertEquals(3, manager.size());

            // Adding a 4th should evict the oldest (id1)
            String id4 = manager.createCheckpoint(kvBuffers, 4);
            assertEquals(3, manager.size());
            assertNull(manager.getCheckpoint(id1), "Oldest checkpoint should have been evicted");
            assertNotNull(manager.getCheckpoint(id2));
            assertNotNull(manager.getCheckpoint(id4));

            // Verify list order
            List<String> ids = manager.listCheckpoints();
            assertEquals(3, ids.size());
            assertEquals(id2, ids.get(0));
            assertEquals(id3, ids.get(1));
            assertEquals(id4, ids.get(2));

            buf.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testManagerDeleteCheckpoint(Nd4jBackend backend) {
        try (KVCacheCheckpointManager manager = new KVCacheCheckpointManager(16)) {
            INDArray buf = Nd4j.zeros(DataType.FLOAT, 1, 2, 16, 4);
            Map<String, INDArray> kvBuffers = new HashMap<>();
            kvBuffers.put("key", buf);

            String id1 = manager.createCheckpoint(kvBuffers, 1);
            String id2 = manager.createCheckpoint(kvBuffers, 2);
            assertEquals(2, manager.size());

            manager.deleteCheckpoint(id1);
            assertEquals(1, manager.size());
            assertNull(manager.getCheckpoint(id1));
            assertNotNull(manager.getCheckpoint(id2));

            buf.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testManagerMemoryTracking(Nd4jBackend backend) {
        try (KVCacheCheckpointManager manager = new KVCacheCheckpointManager()) {
            assertEquals(0, manager.getTotalMemoryBytes());

            INDArray buf = Nd4j.ones(DataType.FLOAT, 1, 2, 8, 4);
            Map<String, INDArray> kvBuffers = new HashMap<>();
            kvBuffers.put("key", buf);

            String id = manager.createCheckpoint(kvBuffers, 4);
            long memAfterOne = manager.getTotalMemoryBytes();
            assertTrue(memAfterOne > 0, "Memory should be tracked after creating a checkpoint");

            // Adding a second should roughly double the memory (same buffer config)
            manager.createCheckpoint(kvBuffers, 4);
            long memAfterTwo = manager.getTotalMemoryBytes();
            assertEquals(memAfterOne * 2, memAfterTwo, memAfterOne * 0.1);

            // Deleting one should halve it
            manager.deleteCheckpoint(id);
            long memAfterDelete = manager.getTotalMemoryBytes();
            assertEquals(memAfterOne, memAfterDelete, memAfterOne * 0.1);

            buf.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testManagerDefaultMaxCheckpoints(Nd4jBackend backend) {
        try (KVCacheCheckpointManager manager = new KVCacheCheckpointManager()) {
            assertEquals(KVCacheCheckpointManager.DEFAULT_MAX_CHECKPOINTS, manager.getMaxCheckpoints());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testManagerInvalidMaxCheckpoints(Nd4jBackend backend) {
        assertThrows(IllegalArgumentException.class, () -> new KVCacheCheckpointManager(0));
        assertThrows(IllegalArgumentException.class, () -> new KVCacheCheckpointManager(-1));
    }
}
