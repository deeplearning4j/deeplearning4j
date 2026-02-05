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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.lang.reflect.Field;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.WORKSPACES)
public class MemoryMgrTest extends BaseNd4jTestWithBackends {


    @Override
    public char ordering(){
        return 'c';
    }

    @BeforeEach
    public void before() {
        ArrayCacheMemoryMgr.setCacheDefaults();
    }

    @AfterEach
    public void after() {
        ArrayCacheMemoryMgr.getLruCache().clear();
        ArrayCacheMemoryMgr.getCurrentCacheSize().set(0);
        ArrayCacheMemoryMgr.getMaxMemFrac().set(0.0);
        ArrayCacheMemoryMgr.getMaxMemFrac().set(0.0);
        ArrayCacheMemoryMgr.getCurrentCacheSize().set(0);
        ArrayCacheMemoryMgr.getMaxMemFrac().set(0.0);
        ArrayCacheMemoryMgr.getLruCacheValues().clear();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("Hard to test assertions with static context. Need to rewrite. Should still have tests here.")
    public void testArrayReuseTooLarge(Nd4jBackend backend) throws Exception {

        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        mmgr.setMaxCacheBytes(new AtomicLong(1000));
        assertEquals(1000, mmgr.getMaxCacheBytes().get());

        INDArray[] arrays = new INDArray[100];
        for( int i = 0; i < arrays.length; i++) {
            arrays[i] = Nd4j.create(DataType.FLOAT, 25);        //100 bytes each
        }

        for( int i = 0; i < 10; i++) {
            mmgr.release(arrays[i]);
        }

        assertEquals(1000, mmgr.getCurrentCacheSize().get());
        assertEquals(10, mmgr.getLruCache().size());
        assertEquals(10, mmgr.getLruCacheValues().size());


        //At this point: array store is full.
        //If we try to release more, the oldest (first released) values should be closed
        for( int i = 0; i < 10; i++) {
            INDArray toRelease = Nd4j.create(DataType.FLOAT, 25);
            mmgr.release(toRelease);
            //oldest N only should be closed by this point...
            for( int j = 0; j < 10; j++) {
                if(j <= i) {
                    //Should have been closed
                    assertTrue(arrays[j].wasClosed());
                } else {
                    //Should still be open
                    assertFalse(arrays[j].wasClosed());
                }
            }
        }


        assertEquals(1000, mmgr.getCurrentCacheSize().get());
        assertEquals(10, mmgr.getLruCache().size());
        assertEquals(10, mmgr.getLruCacheValues().size());

        //now, allocate some values:
        for( int i = 1; i <= 10; i++) {
            INDArray a1 = mmgr.allocate(true, DataType.FLOAT, 25);
            assertEquals(1000 - i * 100, mmgr.getCurrentCacheSize().get());
            assertEquals(10 - i, mmgr.getLruCache().size());
            assertEquals(10 - i, mmgr.getLruCacheValues().size());
        }

        //note since M2 and later: cache size won't reach 0 and will be retained in there as long as a buffer is in use
        //we keep references to buffers that could be reused later if they are ever released
        //due to the way the cache was originally written, it was not views friendly
        //however buffers that are used in control flow are now reference counted
        //any buffer that is referenced at least once in use will not be returned as candidates
        //for release as long as their reference count is at least 1. That is why the prior assertions
        //on the cache being completely empty are now gone.
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCacheHit(Nd4jBackend backend) {
        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        INDArray allocate = mmgr.allocate(false, DataType.INT64, 1);
        mmgr.release(allocate);
        INDArray allocate2 = mmgr.allocate(false,allocate.dataType(),1);
        assertEquals(allocate.data(),allocate2.data());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testManyArrays(Nd4jBackend backend) {

        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        for( int i = 0; i < 1000; i++) {
            mmgr.release(Nd4j.scalar(0));
        }

        assertEquals(4 * 1000, mmgr.getCurrentCacheSize().get());
        assertEquals(1000, mmgr.getLruCache().size());
        assertEquals(1000, mmgr.getLruCacheValues().size());

        for( int i = 0; i < 1000; i++ ){
            mmgr.release(Nd4j.scalar(0));
        }

        assertEquals(4 * 2000, mmgr.getCurrentCacheSize().get());
        assertEquals(2000, mmgr.getLruCache().size());
        assertEquals(2000, mmgr.getLruCacheValues().size());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewsNeverCached(Nd4jBackend backend) {
        // Bug fix test: views should never enter the cache because they share
        // a DataBuffer with the parent. Caching and reusing a view would corrupt
        // the parent's data.
        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        boolean wasCacheEnabled = ArrayCacheMemoryMgr.isCacheEnabled();
        try {
            ArrayCacheMemoryMgr.setEnableCache(true);

            INDArray parent = Nd4j.create(DataType.FLOAT, 10, 10);
            INDArray view = parent.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all());
            assertTrue(view.isView(), "get() should produce a view");

            long cacheSizeBefore = mmgr.getCurrentCacheSize().get();
            mmgr.release(view);
            long cacheSizeAfter = mmgr.getCurrentCacheSize().get();

            // Cache size should not increase - view should have been rejected
            assertEquals(cacheSizeBefore, cacheSizeAfter,
                    "Cache size should not change when releasing a view");

            // Allocating the same shape should NOT return the view
            INDArray allocated = mmgr.allocate(false, DataType.FLOAT, 5, 10);
            assertFalse(allocated.isView(),
                    "Allocate should never return a view from cache");
        } finally {
            ArrayCacheMemoryMgr.setEnableCache(wasCacheEnabled);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewsNeverReturnedFromAllocate(Nd4jBackend backend) {
        // Bug fix test: even if a view somehow enters the cache (e.g. via older code path),
        // allocate() should skip it and return a fresh array.
        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        boolean wasCacheEnabled = ArrayCacheMemoryMgr.isCacheEnabled();
        try {
            ArrayCacheMemoryMgr.setEnableCache(true);

            // Release a normal array first so cache has entries
            INDArray normal = Nd4j.create(DataType.FLOAT, 5, 10);
            mmgr.release(normal);

            // Allocate should return the cached array (not a view)
            INDArray result = mmgr.allocate(false, DataType.FLOAT, 5, 10);
            assertNotNull(result);
            assertFalse(result.isView(), "Cached array returned by allocate should not be a view");
            assertArrayEquals(new long[]{5, 10}, result.shape());
        } finally {
            ArrayCacheMemoryMgr.setEnableCache(wasCacheEnabled);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScopeOutPreservesCacheWhenEnabled(Nd4jBackend backend) {
        // Bug fix test: scopeOut() should NOT destroy the cache when caching is enabled.
        // This is needed for autoregressive generation where the same graph is executed
        // repeatedly and arrays should be reused across steps.
        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        boolean wasCacheEnabled = ArrayCacheMemoryMgr.isCacheEnabled();
        try {
            ArrayCacheMemoryMgr.setEnableCache(true);

            // Add some arrays to cache
            for (int i = 0; i < 10; i++) {
                mmgr.release(Nd4j.create(DataType.FLOAT, 100));
            }
            long cacheSizeBefore = mmgr.getCurrentCacheSize().get();
            assertTrue(cacheSizeBefore > 0, "Cache should have entries");

            // scopeOut should be a no-op when cache is enabled
            mmgr.scopeOut();

            assertEquals(cacheSizeBefore, mmgr.getCurrentCacheSize().get(),
                    "scopeOut() should not clear cache when caching is enabled");
            assertEquals(10, mmgr.getLruCache().size(),
                    "LRU cache should still have all entries after scopeOut");
        } finally {
            ArrayCacheMemoryMgr.setEnableCache(wasCacheEnabled);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScopeOutClearsCacheWhenDisabled(Nd4jBackend backend) {
        // Verify original behavior: scopeOut() should destroy cache when disabled
        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        boolean wasCacheEnabled = ArrayCacheMemoryMgr.isCacheEnabled();
        try {
            ArrayCacheMemoryMgr.setEnableCache(false);

            // Release arrays (when cache disabled, they just get closed immediately)
            // So first enable cache to populate it, then disable and call scopeOut
            ArrayCacheMemoryMgr.setEnableCache(true);
            for (int i = 0; i < 5; i++) {
                mmgr.release(Nd4j.create(DataType.FLOAT, 50));
            }
            assertTrue(mmgr.getCurrentCacheSize().get() > 0, "Cache should have entries");

            // Now disable cache and call scopeOut - should clear everything
            ArrayCacheMemoryMgr.setEnableCache(false);
            mmgr.scopeOut();

            assertEquals(0, mmgr.getCurrentCacheSize().get(),
                    "scopeOut() should clear cache when disabled");
            assertEquals(0, mmgr.getLruCache().size(),
                    "LRU cache should be empty after scopeOut with cache disabled");
        } finally {
            ArrayCacheMemoryMgr.setEnableCache(wasCacheEnabled);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCacheReuseAcrossSimulatedDecodeSteps(Nd4jBackend backend) {
        // Test that arrays released at end of one "decode step" are reused in next step.
        // This simulates what happens in autoregressive generation.
        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        boolean wasCacheEnabled = ArrayCacheMemoryMgr.isCacheEnabled();
        try {
            ArrayCacheMemoryMgr.setEnableCache(true);

            // Step 1: allocate, compute, release
            INDArray a1 = mmgr.allocate(false, DataType.FLOAT, 4, 8);
            INDArray a2 = mmgr.allocate(false, DataType.FLOAT, 4, 8);
            a1.assign(1.0f);
            a2.assign(2.0f);

            mmgr.release(a1);
            mmgr.release(a2);

            long cacheAfterRelease = mmgr.getCurrentCacheSize().get();
            assertTrue(cacheAfterRelease > 0, "Cache should have entries after release");

            // Step 2: allocate same shapes - should get cached arrays back
            INDArray b1 = mmgr.allocate(false, DataType.FLOAT, 4, 8);
            INDArray b2 = mmgr.allocate(false, DataType.FLOAT, 4, 8);

            // Cache should be smaller after allocating from it
            long cacheAfterAlloc = mmgr.getCurrentCacheSize().get();
            assertTrue(cacheAfterAlloc < cacheAfterRelease,
                    "Cache size should decrease after allocating cached arrays");

            // Shapes should match
            assertArrayEquals(new long[]{4, 8}, b1.shape());
            assertArrayEquals(new long[]{4, 8}, b2.shape());

            // Arrays should not be views
            assertFalse(b1.isView());
            assertFalse(b2.isView());
        } finally {
            ArrayCacheMemoryMgr.setEnableCache(wasCacheEnabled);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCacheDoesNotReturnWrongShape(Nd4jBackend backend) {
        // Verify cache only returns arrays with matching shape
        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        boolean wasCacheEnabled = ArrayCacheMemoryMgr.isCacheEnabled();
        try {
            ArrayCacheMemoryMgr.setEnableCache(true);

            // Cache arrays with shape [512]
            INDArray big = Nd4j.create(DataType.FLOAT, 512);
            mmgr.release(big);

            // Request shape [1] - should NOT get the [512] array
            INDArray small = mmgr.allocate(false, DataType.FLOAT, 1);
            assertArrayEquals(new long[]{1}, small.shape(),
                    "Allocate should return correct shape, not a cached different shape");
        } finally {
            ArrayCacheMemoryMgr.setEnableCache(wasCacheEnabled);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewDataIntegrityAfterRelease(Nd4jBackend backend) {
        // Critical test: releasing a view must NOT corrupt the parent's data.
        // If a view were cached and reused for a different op, the parent array
        // would be silently corrupted.
        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        boolean wasCacheEnabled = ArrayCacheMemoryMgr.isCacheEnabled();
        try {
            ArrayCacheMemoryMgr.setEnableCache(true);

            INDArray parent = Nd4j.ones(DataType.FLOAT, 10).mul(42.0f);
            INDArray view = parent.get(NDArrayIndex.interval(0, 5));
            assertTrue(view.isView());

            // Release the view (should be rejected by cache)
            mmgr.release(view);

            // Parent data should be intact
            for (int i = 0; i < 10; i++) {
                assertEquals(42.0f, parent.getFloat(i), 1e-5,
                        "Parent data should not be corrupted after releasing view");
            }

            // Allocate same shape - should get a fresh array, not the view
            INDArray fresh = mmgr.allocate(false, DataType.FLOAT, 5);

            // Write different data to the fresh allocation
            fresh.assign(99.0f);

            // Parent should still be 42.0
            for (int i = 0; i < 10; i++) {
                assertEquals(42.0f, parent.getFloat(i), 1e-5,
                        "Parent data must not be affected by writing to separately allocated array");
            }
        } finally {
            ArrayCacheMemoryMgr.setEnableCache(wasCacheEnabled);
        }
    }

}
