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
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for KVCacheDiskOffloader and TieredKVCacheManager.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
class TestDiskKVCache {

    private static final int BLOCK_SIZE = 4;
    private static final int NUM_KV_HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final DataType DATA_TYPE = DataType.FLOAT;

    @TempDir
    Path tempDir;

    private KVCacheDiskOffloader diskOffloader;

    @BeforeEach
    void setUp() {
        diskOffloader = new KVCacheDiskOffloader(
                tempDir.resolve("kv_disk"),
                10,         // maxDiskBlocks
                BLOCK_SIZE,
                NUM_KV_HEADS,
                HEAD_DIM,
                DATA_TYPE,
                2           // ioThreads
        );
    }

    @AfterEach
    void tearDown() {
        if (diskOffloader != null) {
            diskOffloader.close();
        }
    }

    // ==================== KVCacheDiskOffloader Tests ====================

    @Test
    void testEvictAndRestoreRoundtrip() throws Exception {
        INDArray key = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);
        INDArray value = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);

        // Evict to disk
        CompletableFuture<Void> writeFuture = diskOffloader.evictToDisk(0, key, value);
        writeFuture.get(10, TimeUnit.SECONDS);

        assertTrue(diskOffloader.hasBlock(0), "Block should exist on disk after eviction");
        assertEquals(1, diskOffloader.getDiskBlockCount());

        // Restore from disk
        CompletableFuture<KVCacheHostOffloader.KVBlockPair> readFuture = diskOffloader.restoreFromDisk(0);
        KVCacheHostOffloader.KVBlockPair restored = readFuture.get(10, TimeUnit.SECONDS);

        assertNotNull(restored);
        assertNotNull(restored.getKey());
        assertNotNull(restored.getValue());

        // Verify data integrity
        assertEquals(key.shape().length, restored.getKey().shape().length);
        for (int i = 0; i < key.shape().length; i++) {
            assertEquals(key.shape()[i], restored.getKey().shape()[i],
                    "Key shape mismatch at dimension " + i);
        }

        // Verify actual values match
        INDArray keyDiff = key.sub(restored.getKey());
        double maxKeyError = keyDiff.amaxNumber().doubleValue();
        assertTrue(maxKeyError < 1e-5, "Key data should survive roundtrip, max error: " + maxKeyError);

        INDArray valDiff = value.sub(restored.getValue());
        double maxValError = valDiff.amaxNumber().doubleValue();
        assertTrue(maxValError < 1e-5, "Value data should survive roundtrip, max error: " + maxValError);
    }

    @Test
    void testHasBlockFalseForMissing() {
        assertFalse(diskOffloader.hasBlock(999));
    }

    @Test
    void testDiskBlockCount() throws Exception {
        assertEquals(0, diskOffloader.getDiskBlockCount());

        INDArray key = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);
        INDArray value = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);

        diskOffloader.evictToDisk(0, key, value).get(10, TimeUnit.SECONDS);
        assertEquals(1, diskOffloader.getDiskBlockCount());

        diskOffloader.evictToDisk(1, key, value).get(10, TimeUnit.SECONDS);
        assertEquals(2, diskOffloader.getDiskBlockCount());
    }

    @Test
    void testReleaseBlock() throws Exception {
        INDArray key = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);
        INDArray value = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);

        diskOffloader.evictToDisk(5, key, value).get(10, TimeUnit.SECONDS);
        assertTrue(diskOffloader.hasBlock(5));

        diskOffloader.releaseBlock(5);
        assertFalse(diskOffloader.hasBlock(5));
        assertEquals(0, diskOffloader.getDiskBlockCount());
    }

    @Test
    void testLruEvictionWhenAtCapacity() throws Exception {
        // Create offloader with max 3 disk blocks
        diskOffloader.close();
        diskOffloader = new KVCacheDiskOffloader(
                tempDir.resolve("kv_disk_lru"),
                3,
                BLOCK_SIZE,
                NUM_KV_HEADS,
                HEAD_DIM,
                DATA_TYPE,
                2
        );

        INDArray key = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);
        INDArray value = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);

        // Fill to capacity
        for (int i = 0; i < 3; i++) {
            diskOffloader.evictToDisk(i, key, value).get(10, TimeUnit.SECONDS);
        }
        assertEquals(3, diskOffloader.getDiskBlockCount());

        // Adding one more should evict the LRU (block 0)
        diskOffloader.evictToDisk(10, key, value).get(10, TimeUnit.SECONDS);
        assertEquals(3, diskOffloader.getDiskBlockCount());
        assertFalse(diskOffloader.hasBlock(0), "LRU block 0 should have been evicted");
        assertTrue(diskOffloader.hasBlock(10), "Newly added block should exist");
    }

    @Test
    void testMultipleEvictAndRestore() throws Exception {
        INDArray[] keys = new INDArray[5];
        INDArray[] values = new INDArray[5];

        for (int i = 0; i < 5; i++) {
            keys[i] = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM).mul(i + 1);
            values[i] = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM).mul(i + 1);
            diskOffloader.evictToDisk(i, keys[i], values[i]).get(10, TimeUnit.SECONDS);
        }

        assertEquals(5, diskOffloader.getDiskBlockCount());

        // Restore block 2 and verify
        KVCacheHostOffloader.KVBlockPair restored =
                diskOffloader.restoreFromDisk(2).get(10, TimeUnit.SECONDS);
        assertNotNull(restored);
        assertEquals(2, restored.getOriginalBlockId());
    }

    @Test
    void testGetMaxDiskBlocks() {
        assertEquals(10, diskOffloader.getMaxDiskBlocks());
    }

    @Test
    void testGetStats() throws Exception {
        INDArray key = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);
        INDArray value = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);

        diskOffloader.evictToDisk(0, key, value).get(10, TimeUnit.SECONDS);

        String stats = diskOffloader.getStats();
        assertNotNull(stats);
        assertTrue(stats.contains("KVCacheDiskOffloader"));
        assertTrue(stats.contains("1/10"));  // 1 block out of 10
    }

    @Test
    void testCloseIsIdempotent() {
        diskOffloader.close();
        // Second close should not throw
        diskOffloader.close();
        diskOffloader = null; // Prevent tearDown from closing again
    }

    // ==================== TieredKVCacheManager Tests ====================

    @Test
    void testTieredManagerTouchAndLruTracking() {
        // Create a small GPU cache
        PagedKVCache gpuCache = new PagedKVCache(1, 64, NUM_KV_HEADS, HEAD_DIM,
                BLOCK_SIZE, DATA_TYPE, 1.2);
        KVCacheHostOffloader hostOffloader = new KVCacheHostOffloader(
                BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, DATA_TYPE, 10, 0);

        TieredKVCacheManager manager = new TieredKVCacheManager(
                gpuCache, hostOffloader, diskOffloader);

        // Touch some GPU blocks
        manager.touchGpuBlock(0, 0);
        manager.touchGpuBlock(1, 0);
        manager.touchGpuBlock(2, 0);

        // Stats should not throw
        String stats = manager.getStats();
        assertNotNull(stats);
        assertTrue(stats.contains("TieredKVCache"));

        gpuCache.close();
        hostOffloader.close();
    }

    @Test
    void testTieredManagerEvictFromGpuToHost() {
        PagedKVCache gpuCache = new PagedKVCache(1, 64, NUM_KV_HEADS, HEAD_DIM,
                BLOCK_SIZE, DATA_TYPE, 1.2);
        KVCacheHostOffloader hostOffloader = new KVCacheHostOffloader(
                BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, DATA_TYPE, 10, 0);

        TieredKVCacheManager manager = new TieredKVCacheManager(
                gpuCache, hostOffloader, diskOffloader);

        // Write some data to GPU block 0 so it has content
        INDArray gpuKey = gpuCache.getKeyBlockPool().get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all());
        gpuKey.assign(Nd4j.ones(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM));

        manager.touchGpuBlock(0, 0);

        // Evict block 0 from GPU to host
        manager.evictFromGpu(0);

        assertTrue(hostOffloader.hasBlock(0), "Block should be in host after GPU eviction");

        gpuCache.close();
        hostOffloader.close();
    }

    @Test
    void testTieredManagerEvictFromHostToDisk() throws Exception {
        PagedKVCache gpuCache = new PagedKVCache(1, 64, NUM_KV_HEADS, HEAD_DIM,
                BLOCK_SIZE, DATA_TYPE, 1.2);
        KVCacheHostOffloader hostOffloader = new KVCacheHostOffloader(
                BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, DATA_TYPE, 10, 0);

        TieredKVCacheManager manager = new TieredKVCacheManager(
                gpuCache, hostOffloader, diskOffloader);

        // Put a block in host first
        INDArray key = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);
        INDArray value = Nd4j.rand(DATA_TYPE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);
        hostOffloader.evictBlock(5, key, value);

        assertTrue(hostOffloader.hasBlock(5));

        // Evict from host to disk
        manager.evictFromHost(5);

        // Give async disk write time to complete
        Thread.sleep(500);

        assertTrue(diskOffloader.hasBlock(5), "Block should be on disk after host eviction");
        assertFalse(hostOffloader.hasBlock(5), "Block should be removed from host after eviction to disk");

        gpuCache.close();
        hostOffloader.close();
    }

    @Test
    void testTieredManagerStatsFormat() {
        PagedKVCache gpuCache = new PagedKVCache(1, 64, NUM_KV_HEADS, HEAD_DIM,
                BLOCK_SIZE, DATA_TYPE, 1.2);
        KVCacheHostOffloader hostOffloader = new KVCacheHostOffloader(
                BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, DATA_TYPE, 10, 0);

        TieredKVCacheManager manager = new TieredKVCacheManager(
                gpuCache, hostOffloader, diskOffloader);

        String stats = manager.getStats();
        assertTrue(stats.startsWith("TieredKVCache["));
        assertTrue(stats.contains("GPU="));
        assertTrue(stats.contains("Host="));
        assertTrue(stats.contains("Disk="));

        gpuCache.close();
        hostOffloader.close();
    }

    @Test
    void testTieredManagerDefaultPressureThreshold() {
        PagedKVCache gpuCache = new PagedKVCache(1, 64, NUM_KV_HEADS, HEAD_DIM,
                BLOCK_SIZE, DATA_TYPE, 1.2);
        KVCacheHostOffloader hostOffloader = new KVCacheHostOffloader(
                BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, DATA_TYPE, 10, 0);

        TieredKVCacheManager manager = new TieredKVCacheManager(
                gpuCache, hostOffloader, diskOffloader);

        assertEquals(TieredKVCacheManager.DEFAULT_GPU_PRESSURE_THRESHOLD,
                manager.getGpuPressureThreshold(), 1e-9);

        gpuCache.close();
        hostOffloader.close();
    }
}
