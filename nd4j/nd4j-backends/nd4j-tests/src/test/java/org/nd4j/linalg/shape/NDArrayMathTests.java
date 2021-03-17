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

package org.nd4j.linalg.shape;

import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.NDArrayMath;

import static org.junit.jupiter.api.Assertions.assertEquals;


/**
 * @author Adam Gibson
 */

public class NDArrayMathTests extends BaseNd4jTestWithBackends {



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorPerSlice(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(2, 2, 2, 2);
        assertEquals(4, NDArrayMath.vectorsPerSlice(arr));

        INDArray matrix = Nd4j.create(2, 2);
        assertEquals(2, NDArrayMath.vectorsPerSlice(matrix));

        INDArray arrSliceZero = arr.slice(0);
        assertEquals(4, NDArrayMath.vectorsPerSlice(arrSliceZero));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatricesPerSlice(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(2, 2, 2, 2);
        assertEquals(2, NDArrayMath.matricesPerSlice(arr));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLengthPerSlice(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(2, 2, 2, 2);
        val lengthPerSlice = NDArrayMath.lengthPerSlice(arr);
        assertEquals(8, lengthPerSlice);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void toffsetForSlice(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(3, 2, 2);
        int slice = 1;
        assertEquals(4, NDArrayMath.offsetForSlice(arr, slice));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMapOntoVector(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(3, 2, 2);
        assertEquals(NDArrayMath.mapIndexOntoVector(2, arr), 4);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNumVectors(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(3, 2, 2);
        assertEquals(4, NDArrayMath.vectorsPerSlice(arr));
        INDArray matrix = Nd4j.create(2, 2);
        assertEquals(2, NDArrayMath.vectorsPerSlice(matrix));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOffsetForSlice(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(2, 2, 2, 2);
        int[] dimensions = {0, 1};
        INDArray permuted = arr.permute(2, 3, 0, 1);
        int[] test = {0, 0, 1, 1};
        for (int i = 0; i < permuted.tensorsAlongDimension(dimensions); i++) {
            assertEquals(test[i], NDArrayMath.sliceOffsetForTensor(i, permuted, new int[] {2, 2}));
        }

        val arrTensorsPerSlice = NDArrayMath.tensorsPerSlice(arr, new int[] {2, 2});
        assertEquals(2, arrTensorsPerSlice);

        INDArray arr2 = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 2, 2);
        int[] assertions = {0, 1, 2};
        for (int i = 0; i < assertions.length; i++) {
            assertEquals(assertions[i], NDArrayMath.sliceOffsetForTensor(i, arr2, new int[] {2, 2}));
        }



        val tensorsPerSlice = NDArrayMath.tensorsPerSlice(arr2, new int[] {2, 2});
        assertEquals(1, tensorsPerSlice);


        INDArray otherTest = Nd4j.linspace(1, 144, 144, DataType.DOUBLE).reshape(6, 3, 2, 2, 2);
//        System.out.println(otherTest);
        INDArray baseArr = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape(2, 2, 2);
        for (int i = 0; i < baseArr.tensorsAlongDimension(0, 1); i++) {
//            System.out.println(NDArrayMath.sliceOffsetForTensor(i, baseArr, new int[] {2, 2}));
            NDArrayMath.sliceOffsetForTensor(i, baseArr, new int[] {2, 2});
        }


    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOddDimensions(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(3, 2, 2);
        val numMatrices = NDArrayMath.matricesPerSlice(arr);
        assertEquals(1, numMatrices);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTotalVectors(Nd4jBackend backend) {
        INDArray arr2 = Nd4j.create(2, 2, 2, 2);
        assertEquals(8, NDArrayMath.numVectors(arr2));
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
