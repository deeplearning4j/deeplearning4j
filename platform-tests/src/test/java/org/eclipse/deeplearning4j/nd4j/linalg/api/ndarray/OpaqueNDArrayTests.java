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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
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
@Tag(TagNames.SMOKE)
public class OpaqueNDArrayTests extends BaseNd4jTestWithBackends {

    @Test
    public void equalsWithEpsReadsThirdExtraArgument() {
        NativeOps nativeOps = Nd4j.getNativeOps();
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray x = Nd4j.createFromArray(1.0, 2.0, 4.0).castTo(type);
            INDArray y = Nd4j.createFromArray(1.05, 2.05, 4.05).castTo(type);
            for (double epsilon : new double[]{0.1, 0.01}) {
                INDArray z = Nd4j.scalar(type, -77);
                INDArray extraArray = Nd4j.createFromArray(0.0, 0.0, epsilon).castTo(type);
                Pointer extra = reduce3ExtraPointer(extraArray);
                nativeOps.clearLastError();
                try (OpaqueNDArray xOpaque = OpaqueNDArray.fromINDArrayUncached(x);
                     OpaqueNDArray yOpaque = OpaqueNDArray.fromINDArrayUncached(y);
                     OpaqueNDArray zOpaque = OpaqueNDArray.fromINDArrayUncached(z)) {
                    // Native Reduce3 EqualsWithEps is op 4; slots 0/1 are scratch, slot 2 is epsilon.
                    nativeOps.execReduce3Scalar(null, 4, xOpaque, extra, yOpaque, zOpaque);
                    assertEquals(0, nativeOps.lastErrorCode());
                    assertEquals(epsilon == 0.1 ? 1 : 0, z.getDouble(0), 0);
                }
            }
        }
    }

    private Pointer reduce3ExtraPointer(INDArray extraArray) {
        // The raw CUDA Reduce3 ABI consumes device parameters; CPU/Vulkan consume host parameters.
        if ("CUDA".equals(Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend"))) {
            extraArray.data().opaqueBuffer().syncToSpecial();
            return Nd4j.getNativeOps().dbSpecialBuffer(extraArray.data().opaqueBuffer());
        }
        return extraArray.data().addressPointer();
    }

    @Test
    public void equalsWithEpsDimensionalAndAllPairsPreserveEpsilon() {
        NativeOps nativeOps = Nd4j.getNativeOps();
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (char order : new char[]{'c', 'f'}) {
                INDArray x = Nd4j.createFromArray(new double[][]{{1, 2, 4}, {10, 20, 40}})
                        .castTo(type).dup(order);
                INDArray y = Nd4j.createFromArray(new double[][]{{1.05, 2.05, 4.05}, {10.05, 20.05, 40.05}})
                        .castTo(type).dup(order == 'c' ? 'f' : 'c');
                INDArray dimensions = Nd4j.createFromArray(1L);
                for (double epsilon : new double[]{0.1, 0.01}) {
                    INDArray extraArray = Nd4j.createFromArray(0.0, 0.0, epsilon).castTo(type);
                    Pointer extra = reduce3ExtraPointer(extraArray);
                    for (boolean allPairs : new boolean[]{false, true}) {
                        INDArray owner = allPairs
                                ? Nd4j.create(type, new long[]{3, 5}, order).assign(-77)
                                : Nd4j.create(type, 5).assign(-77);
                        INDArray z = allPairs
                                ? owner.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 2, 5))
                                : owner.get(NDArrayIndex.interval(1, 2, 5));
                        nativeOps.clearLastError();
                        try (OpaqueNDArray xOpaque = OpaqueNDArray.fromINDArrayUncached(x);
                             OpaqueNDArray yOpaque = OpaqueNDArray.fromINDArrayUncached(y);
                             OpaqueNDArray zOpaque = OpaqueNDArray.fromINDArrayUncached(z);
                             OpaqueNDArray dimsOpaque = OpaqueNDArray.fromINDArrayUncached(dimensions)) {
                            if (allPairs) nativeOps.execReduce3All(null, 4, xOpaque, yOpaque, zOpaque, dimsOpaque, extra);
                            else nativeOps.execReduce3Tad(null, 4, xOpaque, extra, yOpaque, zOpaque, dimsOpaque);
                            assertEquals(0, nativeOps.lastErrorCode());
                            double diagonal = epsilon == 0.1 ? 1 : 0;
                            if (allPairs) {
                                for (int i = 0; i < 2; i++) {
                                    for (int j = 0; j < 2; j++) {
                                        assertEquals(i == j ? diagonal : 0, owner.getDouble(i + 1, 2 * j + 1), 0);
                                    }
                                }
                                for (int i = 0; i < 3; i++) {
                                    for (int j = 0; j < 5; j++) {
                                        if (i == 0 || j % 2 == 0) assertEquals(-77, owner.getDouble(i, j), 0);
                                    }
                                }
                            } else {
                                assertEquals(diagonal, owner.getDouble(1), 0);
                                assertEquals(diagonal, owner.getDouble(3), 0);
                                assertEquals(-77, owner.getDouble(0), 0);
                                assertEquals(-77, owner.getDouble(2), 0);
                                assertEquals(-77, owner.getDouble(4), 0);
                            }
                        }
                    }
                }
            }
        }
    }

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


}
