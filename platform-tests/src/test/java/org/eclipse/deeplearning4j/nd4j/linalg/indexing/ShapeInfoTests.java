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

package org.eclipse.deeplearning4j.nd4j.linalg.indexing;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

@Tag(TagNames.NDARRAY_INDEXING)
@NativeTag
public class ShapeInfoTests  extends BaseNd4jTestWithBackends  {


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNDArrays(Nd4jBackend backend) {
        LongShapeDescriptor longShapeDescriptor = LongShapeDescriptor.builder()
                .shape(new long[]{2, 2})
                .stride(new long[]{2, 1})
                .order('c')
                .offset(0)
                .ews(1)
                .extras(ArrayOptionsHelper.composeTypicalChecks(
                        DataType.FLOAT))
                .build();
        INDArray arr = Nd4j.create(longShapeDescriptor);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBasicArrayCreation(Nd4jBackend backend) {
        LongShapeDescriptor descriptor = LongShapeDescriptor.builder()
                .shape(new long[]{2, 2})
                .stride(new long[]{2, 1})
                .order('c')
                .offset(0)
                .ews(1)
                .extras(ArrayOptionsHelper.composeTypicalChecks(DataType.FLOAT))
                .build();

        INDArray arr = Nd4j.create(descriptor);

        assertArrayEquals(new long[]{2, 2}, arr.shape());
        assertArrayEquals(new long[]{2, 1}, arr.stride());
        assertEquals(DataType.FLOAT, arr.dataType());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayWithOffset(Nd4jBackend backend) {
        LongShapeDescriptor descriptor = LongShapeDescriptor.builder()
                .shape(new long[]{2, 2})
                .stride(new long[]{2, 1})
                .order('c')
                .offset(2)
                .ews(1)
                .extras(ArrayOptionsHelper.composeTypicalChecks(DataType.FLOAT))
                .build();

        INDArray arr = Nd4j.create(descriptor);

        assertArrayEquals(new long[]{2, 2}, arr.shape());
        assertEquals(4, arr.length());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayWithDifferentStrides(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new float[]{1, 2, 3, 0, 0, 0, 0, 0, 0}, new int[]{3, 3}, 'f');
        assertEquals(1, arr.getFloat(0, 0));
        assertEquals(2, arr.getFloat(1, 0));
        assertEquals(3, arr.getFloat(2, 0));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayReshape(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(6).reshape(2, 3);
        INDArray reshaped = arr.reshape(3, 2);

        assertArrayEquals(new long[]{3, 2}, reshaped.shape());
        assertEquals(0, reshaped.getFloat(0, 0));
        assertEquals(1, reshaped.getFloat(0, 1));
        assertEquals(4, reshaped.getFloat(2, 0));
        assertEquals(5, reshaped.getFloat(2, 1));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArraySlicing(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(16).reshape(4, 4);
        INDArray slice = arr.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 3));

        assertArrayEquals(new long[]{2, 2}, slice.shape());
        assertEquals(5, slice.getFloat(0, 0));
        assertEquals(10, slice.getFloat(1, 1));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewOffsets1D(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(10);

        INDArray view1 = arr.get(NDArrayIndex.interval(2, 7));
        assertEquals(5, view1.length());
        assertEquals(2, view1.getDouble(0), 1e-5);
        assertEquals(6, view1.getDouble(4), 1e-5);

        INDArray view2 = arr.get(NDArrayIndex.interval(5, 2, 9));
        assertEquals(2, view2.length());
        assertEquals(5, view2.getDouble(0), 1e-5);
        assertEquals(7, view2.getDouble(1), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewOffsets2D(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(24).reshape(4, 6);

        INDArray view1 = arr.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(2, 5));
        assertArrayEquals(new long[]{2, 3}, view1.shape());
        assertEquals(8, view1.getDouble(0, 0), 1e-5);
        assertEquals(16, view1.getDouble(1, 2), 1e-5);

        INDArray view2 = arr.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 5));
        assertArrayEquals(new long[]{4, 2}, view2.shape());
        assertEquals(1, view2.getDouble(0, 0), 1e-5);
        assertEquals(21, view2.getDouble(3, 1), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewOffsets3D(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(60).reshape(3, 4, 5);

        INDArray view1 = arr.get(NDArrayIndex.point(1), NDArrayIndex.interval(1, 3), NDArrayIndex.all());
        assertArrayEquals(new long[]{2, 5}, view1.shape());
        assertEquals(25, view1.getDouble(0, 0), 1e-5);
        assertEquals(34, view1.getDouble(1, 4), 1e-5);

        INDArray view2 = arr.get(NDArrayIndex.all(), NDArrayIndex.point(2), NDArrayIndex.interval(1, 2, 5));
        assertArrayEquals(new long[]{3, 2}, view2.shape());
        assertEquals(11, view2.getDouble(0, 0), 1e-5);
        assertEquals(53, view2.getDouble(2, 1), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewOffsets4D(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(120).reshape(2, 3, 4, 5);

        INDArray view1 = arr.get(NDArrayIndex.point(1), NDArrayIndex.interval(1, 3), NDArrayIndex.all(), NDArrayIndex.interval(1, 4));
        assertArrayEquals(new long[]{2, 4, 3}, view1.shape());
        assertEquals(81, view1.getDouble(0, 0, 0), 1e-5);
        assertEquals(118, view1.getDouble(1, 3, 2), 1e-5);

        INDArray view2 = arr.get(NDArrayIndex.all(), NDArrayIndex.point(2), NDArrayIndex.interval(1, 3), NDArrayIndex.interval(0, 2, 4,true));
        assertArrayEquals(new long[]{2, 2, 3}, view2.shape());
        assertEquals(45, view2.getDouble(0, 0, 0), 1e-5);
        assertEquals(112, view2.getDouble(1, 1, 1), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewOffsets5D(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(240).reshape(2, 3, 4, 5, 2);

        INDArray view1 = arr.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 3), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.point(1));
        assertArrayEquals(new long[]{2, 2, 5}, view1.shape());
        assertEquals(61, view1.getDouble(0, 0, 0), 1e-5);
        assertEquals(229, view1.getDouble(1, 1, 4), 1e-5);

        INDArray view2 = arr.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 4), NDArrayIndex.interval(1, 4), NDArrayIndex.all());
        assertArrayEquals(new long[]{3, 2, 3, 2}, view2.shape());
        assertEquals(122, view2.getDouble(0, 0, 0, 0), 1e-5);
        assertEquals(227, view2.getDouble(2, 1, 2, 1), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewOffsets6D(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(720).reshape(2, 3, 4, 5, 2, 3);

        INDArray view1 = arr.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.interval(1, 3),
                NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 3));
        assertArrayEquals(new long[]{3, 2, 2, 2}, view1.shape());
        assertEquals(402, view1.getDouble(0, 0, 0, 0), 1e-5);
        assertEquals(677, view1.getDouble(2, 1, 1, 1), 1e-5);

        INDArray view2 = arr.get(NDArrayIndex.all(), NDArrayIndex.point(2), NDArrayIndex.all(),
                NDArrayIndex.interval(1, 4), NDArrayIndex.point(1), NDArrayIndex.all());
        assertArrayEquals(new long[]{2, 4, 3, 3}, view2.shape());
        assertEquals(249, view2.getDouble(0, 0, 0, 0), 1e-5);
        assertEquals(713, view2.getDouble(1, 3, 2, 2), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMixedDataTypeViews(Nd4jBackend backend) {
        INDArray arrFloat = Nd4j.arange(24).reshape(4, 6).castTo(DataType.FLOAT);
        INDArray arrDouble = Nd4j.arange(24).reshape(4, 6).castTo(DataType.DOUBLE);
        INDArray arrLong = Nd4j.arange(24).reshape(4, 6).castTo(DataType.LONG);

        INDArray viewFloat = arrFloat.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(2, 5));
        INDArray viewDouble = arrDouble.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(2, 5));
        INDArray viewLong = arrLong.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(2, 5));

        assertEquals(8.0f, viewFloat.getFloat(0, 0), 1e-5);
        assertEquals(16.0f, viewFloat.getFloat(1, 2), 1e-5);
        assertEquals(8.0, viewDouble.getDouble(0, 0), 1e-5);
        assertEquals(16.0, viewDouble.getDouble(1, 2), 1e-5);
        assertEquals(8L, viewLong.getLong(0, 0));
        assertEquals(16L, viewLong.getLong(1, 2));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNestedViews(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(120).reshape(4, 5, 6);
        INDArray view1 = arr.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(), NDArrayIndex.interval(2, 5));
        INDArray view2 = view1.get(NDArrayIndex.point(1), NDArrayIndex.interval(1, 4));

        assertArrayEquals(new long[]{3, 3}, view2.shape());
        assertEquals(68, view2.getDouble(0, 0), 1e-5);
        assertEquals(82, view2.getDouble(2, 2), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMixedDatatypeOperations(Nd4jBackend backend) {
        Nd4j.getEnvironment().setMaxMasterThreads(1);
        Nd4j.getEnvironment().setVerbose(true);
        Nd4j.getEnvironment().setDebug(true);
        INDArray arrFloat = Nd4j.arange(24).reshape(4, 6).castTo(DataType.FLOAT);
        INDArray arrInt = Nd4j.arange(24);
        arrInt = arrInt.reshape(4, 6);
        arrInt = arrInt.castTo(DataType.INT);
        INDArray viewFloat = arrFloat.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(2, 5));
        INDArray viewInt = arrInt.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(2, 5));
        INDArray castedBack = viewInt.castTo(DataType.FLOAT);
        assertEquals(castedBack, viewFloat);
        INDArray result = viewFloat.add(viewInt);
        assertEquals(DataType.FLOAT, result.dataType());
        assertEquals(16.0f, result.getFloat(0, 0), 1e-5);
        assertEquals(32.0f, result.getFloat(1, 2), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNonContiguousViews(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(60).reshape(5, 4, 3);
        INDArray view = arr.get(NDArrayIndex.interval(0, 2, 4,true), NDArrayIndex.all(), NDArrayIndex.interval(1, 3));

        assertArrayEquals(new long[]{3, 4, 2}, view.shape());
        assertArrayEquals(new long[]{24, 3, 1}, view.stride());
        assertEquals(1, view.getDouble(0, 0, 0), 1e-5);
        assertEquals(59, view.getDouble(2, 3, 1), 1e-5);
    }

    // Additional test to cover EWS (Element Wise Stride)
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEWS(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new float[9], new int[]{3, 3}, 'c');
        assertEquals(1, arr.elementWiseStride());
    }

    // Additional test to cover data type conversion
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDataTypeConversion(Nd4jBackend backend) {
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);
        INDArray arrFloat = Nd4j.arange(24).reshape(4, 6).castTo(DataType.FLOAT);
        INDArray arrDouble = arrFloat.castTo(DataType.DOUBLE);

        assertEquals(DataType.FLOAT, arrFloat.dataType());
        assertEquals(DataType.DOUBLE, arrDouble.dataType());
        assertEquals(arrFloat.getFloat(2, 3), (float)arrDouble.getDouble(2, 3), 1e-5);
    }

    // Additional test to cover more complex indexing
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testComplexIndexing(Nd4jBackend backend) {
        INDArray arr = Nd4j.arange(120).reshape(2, 3, 4, 5);
        INDArray view = arr.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.interval(1, 3), NDArrayIndex.interval(2, 2, 5));

        assertArrayEquals(new long[]{3, 2, 2}, view.shape());
        assertEquals(67, view.getDouble(0, 0, 0), 1e-5);
        assertEquals(114, view.getDouble(2, 1, 1), 1e-5);

        // Additional assertions to match the entire view
        assertEquals(69, view.getDouble(0, 0, 1), 1e-5);
        assertEquals(72, view.getDouble(0, 1, 0), 1e-5);
        assertEquals(74, view.getDouble(0, 1, 1), 1e-5);
        assertEquals(87, view.getDouble(1, 0, 0), 1e-5);
        assertEquals(89, view.getDouble(1, 0, 1), 1e-5);
        assertEquals(92, view.getDouble(1, 1, 0), 1e-5);
        assertEquals(94, view.getDouble(1, 1, 1), 1e-5);
        assertEquals(107, view.getDouble(2, 0, 0), 1e-5);
        assertEquals(109, view.getDouble(2, 0, 1), 1e-5);
        assertEquals(112, view.getDouble(2, 1, 0), 1e-5);
    }
}
