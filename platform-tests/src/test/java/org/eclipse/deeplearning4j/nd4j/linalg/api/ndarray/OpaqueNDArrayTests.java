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

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueNDArray;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class OpaqueNDArrayTests extends BaseNd4jTestWithBackends {

    @Test
    public void testBasicConversion() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2).castTo(DataType.FLOAT);
        OpaqueNDArray opaque = OpaqueNDArray.fromINDArray(arr);
        INDArray arr2 = OpaqueNDArray.toINDArray(opaque);
        assertEquals(arr,arr2);


        INDArray view = arr.get(NDArrayIndex.all(), NDArrayIndex.interval(0,1));
        OpaqueNDArray opaqueView = OpaqueNDArray.fromINDArray(view);
        INDArray view2 = OpaqueNDArray.toINDArray(opaqueView);
        assertEquals(view,view2);

        INDArray arr3 = arr.castTo(DataType.INT32);
        OpaqueNDArray opaque3 = OpaqueNDArray.fromINDArray(arr3);
        INDArray arr4 = OpaqueNDArray.toINDArray(opaque3);
        assertEquals(arr3,arr4);

        INDArray castBack = arr3.castTo(DataType.FLOAT);
        OpaqueNDArray opaqueCastBack = OpaqueNDArray.fromINDArray(castBack);
        INDArray castBack2 = OpaqueNDArray.toINDArray(opaqueCastBack);
        assertEquals(castBack,castBack2);
        assertEquals(castBack,arr);


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
        // Test dataTypeNativeAt
        long dataTypeResult = nativeOps.dataTypeNativeAt(context.contextPointer(), 0);
        final long expectedDataType = 5; // Example value, replace with actual expected data type
        assertEquals(expectedDataType, dataTypeResult, "Data type at index 0 did not match the expected value");

        context.setBArguments(true);
        // Test bArgAtNative
        boolean bArgResult = nativeOps.bArgAtNative(context.contextPointer(), 0);
        final boolean expectedBArg = true; // Example value, replace with actual expected boolean argument
        assertEquals(expectedBArg, bArgResult, "Boolean arg at index 0 did not match the expected value");

        context.setIArguments(42L);
        // Test iArgumentAtNative
        long iArgResult = nativeOps.iArgumentAtNative(context.contextPointer(), 0);
        final long expectedIArg = 42L; // Example value, replace with actual expected integer argument
        assertEquals(expectedIArg, iArgResult, "Integer arg at index 0 did not match the expected value");

        // Test numDNative
        long numDResult = nativeOps.numDNative(context.contextPointer());
        final long expectedNumD = 1L; // Example value, replace with actual expected number of D arguments
        assertEquals(expectedNumD, numDResult, "Number of D arguments did not match the expected value");

        // Test numBNative
        long numBResult = nativeOps.numBNative(context.contextPointer());
        final long expectedNumB = 1L; // Example value, replace with actual expected number of B arguments
        assertEquals(expectedNumB, numBResult, "Number of B arguments did not match the expected value");

        context.setOutputArray(0,arr);
        // Test numOutputsNative
        long numOutputsResult = nativeOps.numOutputsNative(context.contextPointer());
        final long expectedNumOutputs = 1L; // Example value, replace with actual expected number of outputs
        assertEquals(expectedNumOutputs, numOutputsResult, "Number of outputs did not match the expected value");

        // Test numInputsNative
        long numInputsResult = nativeOps.numInputsNative(context.contextPointer());
        final long expectedNumInputs = 1; // Example value, replace with actual expected number of inputs
        assertEquals(expectedNumInputs, numInputsResult, "Number of inputs did not match the expected value");

        context.setTArguments(3.14);
        // Test tArgumentNative
        double tArgResult = nativeOps.tArgumentNative(context.contextPointer(), 0);
        final double expectedTArg = 3.14; // Example value, replace with actual expected T argument
        assertEquals(expectedTArg, tArgResult, 0.001, "T argument at index 0 did not match the expected value");

        // Test numTArgumentsNative
        long numTArgsResult = nativeOps.numTArgumentsNative(context.contextPointer());
        final long expectedNumTArgs = 1L; // Example value, replace with actual expected number of T arguments
        assertEquals(expectedNumTArgs, numTArgsResult, "Number of T arguments did not match the expected value");
        context.close();
    }


}
