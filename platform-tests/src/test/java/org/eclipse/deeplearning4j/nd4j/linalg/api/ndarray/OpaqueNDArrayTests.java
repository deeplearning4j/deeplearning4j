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
package org.eclipse.deeplearning4j.nd4j.linalg.api.ndarray;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueNDArray;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
public class OpaqueNDArrayTests extends BaseNd4jTestWithBackends {

    @Test
    public void testBasicConversion() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2).castTo(DataType.FLOAT);
        try (OpaqueNDArray opaque = OpaqueNDArray.fromINDArrayUncached(arr)) {
            INDArray arr2 = OpaqueNDArray.toINDArray(opaque);
            assertEquals(arr, arr2);
        }

        INDArray view = arr.get(NDArrayIndex.all(), NDArrayIndex.interval(0,1));
        try (OpaqueNDArray opaqueView = OpaqueNDArray.fromINDArrayUncached(view)) {
            INDArray view2 = OpaqueNDArray.toINDArray(opaqueView);
            assertEquals(view, view2);
        }

        INDArray arr3 = arr.castTo(DataType.INT32);
        try (OpaqueNDArray opaque3 = OpaqueNDArray.fromINDArrayUncached(arr3)) {
            INDArray arr4 = OpaqueNDArray.toINDArray(opaque3);
            assertEquals(arr3, arr4);
        }

        INDArray castBack = arr3.castTo(DataType.FLOAT);
        try (OpaqueNDArray opaqueCastBack = OpaqueNDArray.fromINDArrayUncached(castBack)) {
            INDArray castBack2 = OpaqueNDArray.toINDArray(opaqueCastBack);
            assertEquals(castBack, castBack2);
            assertEquals(castBack, arr);
        }
    }

    @Test
    public void testOpaqueNDArrayMemoryLeak() throws Exception {
        log.info("Starting OpaqueNDArray memory leak test");
        
        // Get baseline
        NativeOps nativeOps = Nd4j.getNativeOps();
        long beforeCount = nativeOps.getOpaqueNDArrayLeakCount();
        long beforeBytes = nativeOps.getOpaqueNDArrayLeakBytes();
        long beforePhysical = Pointer.physicalBytes();
        
        log.info("Before test - Count: {}, Bytes: {}, Physical: {}", 
                 beforeCount, beforeBytes, beforePhysical);
        
        // Create and destroy many arrays
        final int iterations = 1000;
        for (int i = 0; i < iterations; i++) {
            INDArray arr = Nd4j.create(DataType.FLOAT, 100, 100);
            
            try (OpaqueNDArray opaque = OpaqueNDArray.fromINDArrayUncached(arr)) {
                assertNotNull(opaque.buffer());
                assertTrue(opaque.length() > 0);
            } // Should auto-cleanup via close()
            
            arr.close();
        }
        
        // Force cleanup cycles
        System.gc();
        Thread.sleep(2000);
        System.gc();
        Thread.sleep(1000);
        
        long afterCount = nativeOps.getOpaqueNDArrayLeakCount();
        long afterBytes = nativeOps.getOpaqueNDArrayLeakBytes();
        long afterPhysical = Pointer.physicalBytes();
        
        log.info("After test - Count: {}, Bytes: {}, Physical: {}", 
                 afterCount, afterBytes, afterPhysical);
        
        long leakedCount = afterCount - beforeCount;
        long leakedBytes = afterBytes - beforeBytes;
        long leakedPhysical = afterPhysical - beforePhysical;
        
        log.info("Leaked - Count: {}, Bytes: {} ({} MB), Physical: {} ({} MB)",
                 leakedCount, leakedBytes, leakedBytes / (1024.0 * 1024.0),
                 leakedPhysical, leakedPhysical / (1024.0 * 1024.0));
        
        // Allow small tolerance for transient allocations
        assertTrue(leakedCount < 10, 
                   "Too many OpaqueNDArrays leaked: " + leakedCount);
        assertTrue(leakedBytes < 1_000_000, 
                   "Too much memory leaked: " + leakedBytes + " bytes (" + 
                   (leakedBytes / (1024.0 * 1024.0)) + " MB)");
    }

    @Test
    public void testOpaqueNDArrayCloseExplicit() {
        NativeOps nativeOps = Nd4j.getNativeOps();
        long beforeCount = nativeOps.getOpaqueNDArrayLeakCount();
        
        INDArray arr = Nd4j.create(DataType.FLOAT, 50, 50);
        OpaqueNDArray opaque = OpaqueNDArray.fromINDArrayUncached(arr);
        
        assertNotNull(opaque);
        assertFalse(opaque.isNull());
        
        long duringCount = nativeOps.getOpaqueNDArrayLeakCount();
        assertTrue(duringCount > beforeCount, "Count should increase after allocation");
        
        // Explicit close
        opaque.close();
        
        // Verify deallocator was called
        assertNotNull(opaque.getDeallocator());
        assertTrue(opaque.getDeallocator().isDeallocated());
        
        arr.close();
    }

    @Test
    public void testOpaqueNDArrayCachedVsUncached() {
        INDArray arr = Nd4j.create(DataType.FLOAT, 10, 10);
        
        // Cached version (via getOrCreateOpaqueNDArray)
        OpaqueNDArray cached1 = OpaqueNDArray.fromINDArray(arr);
        OpaqueNDArray cached2 = OpaqueNDArray.fromINDArray(arr);
        
        // Should return same instance when cached
        assertSame(cached1, cached2, "Cached OpaqueNDArray should be same instance");
        
        // Uncached version
        try (OpaqueNDArray uncached1 = OpaqueNDArray.fromINDArrayUncached(arr);
             OpaqueNDArray uncached2 = OpaqueNDArray.fromINDArrayUncached(arr)) {
            
            // Should create new instances
            assertNotSame(uncached1, uncached2, "Uncached OpaqueNDArray should be different instances");
            assertNotSame(cached1, uncached1, "Cached and uncached should be different");
        }
        
        arr.close();
    }

    @Test
    public void testOpContext() throws Exception {
        NativeOps nativeOps = Nd4j.getNativeOps();
        OpContext context = Nd4j.getExecutioner().buildContext();
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2).castTo(DataType.FLOAT);
        context.setInputArray(0,arr);
        OpaqueNDArray inputArrayResult = nativeOps.getInputArrayNative(context.contextPointer(), 0);
        INDArray converted = OpaqueNDArray.toINDArray(inputArrayResult);
        assertEquals(arr,converted);

        context.setDArguments(DataType.FLOAT);
        long dataTypeResult = nativeOps.dataTypeNativeAt(context.contextPointer(), 0);
        final long expectedDataType = 5;
        assertEquals(expectedDataType, dataTypeResult, "Data type at index 0 did not match the expected value");

        context.setBArguments(true);
        boolean bArgResult = nativeOps.bArgAtNative(context.contextPointer(), 0);
        final boolean expectedBArg = true;
        assertEquals(expectedBArg, bArgResult, "Boolean arg at index 0 did not match the expected value");

        context.setIArguments(42L);
        long iArgResult = nativeOps.iArgumentAtNative(context.contextPointer(), 0);
        final long expectedIArg = 42L;
        assertEquals(expectedIArg, iArgResult, "Integer arg at index 0 did not match the expected value");

        long numDResult = nativeOps.numDNative(context.contextPointer());
        final long expectedNumD = 1L;
        assertEquals(expectedNumD, numDResult, "Number of D arguments did not match the expected value");

        long numBResult = nativeOps.numBNative(context.contextPointer());
        final long expectedNumB = 1L;
        assertEquals(expectedNumB, numBResult, "Number of B arguments did not match the expected value");

        context.setOutputArray(0,arr);
        long numOutputsResult = nativeOps.numOutputsNative(context.contextPointer());
        final long expectedNumOutputs = 1L;
        assertEquals(expectedNumOutputs, numOutputsResult, "Number of outputs did not match the expected value");

        long numInputsResult = nativeOps.numInputsNative(context.contextPointer());
        final long expectedNumInputs = 1;
        assertEquals(expectedNumInputs, numInputsResult, "Number of inputs did not match the expected value");

        context.setTArguments(3.14);
        double tArgResult = nativeOps.tArgumentNative(context.contextPointer(), 0);
        final double expectedTArg = 3.14;
        assertEquals(expectedTArg, tArgResult, 0.001, "T argument at index 0 did not match the expected value");

        long numTArgsResult = nativeOps.numTArgumentsNative(context.contextPointer());
        final long expectedNumTArgs = 1L;
        assertEquals(expectedNumTArgs, numTArgsResult, "Number of T arguments did not match the expected value");
        
        context.close();
        arr.close();
    }

    @Test
    public void testLargeArrayCleanup() throws Exception {
        log.info("Testing large array cleanup");
        
        NativeOps nativeOps = Nd4j.getNativeOps();
        long beforeBytes = nativeOps.getOpaqueNDArrayLeakBytes();
        
        // Create a large array (40 MB)
        INDArray largeArray = Nd4j.create(DataType.FLOAT, 10_000_000);
        
        try (OpaqueNDArray opaque = OpaqueNDArray.fromINDArrayUncached(largeArray)) {
            long duringBytes = nativeOps.getOpaqueNDArrayLeakBytes();
            long allocated = duringBytes - beforeBytes;
            
            log.info("Allocated {} bytes ({} MB)", allocated, allocated / (1024.0 * 1024.0));
            assertTrue(allocated > 35_000_000, "Should allocate at least 35 MB");
        } // Should cleanup here
        
        largeArray.close();
        
        // Force cleanup
        System.gc();
        Thread.sleep(1000);
        
        long afterBytes = nativeOps.getOpaqueNDArrayLeakBytes();
        long remaining = afterBytes - beforeBytes;
        
        log.info("Remaining after cleanup: {} bytes ({} MB)", 
                 remaining, remaining / (1024.0 * 1024.0));
        
        // Should have cleaned up most of it
        assertTrue(remaining < 5_000_000, 
                   "Should cleanup large array, but " + remaining + " bytes remain");
    }
}
