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

package org.nd4j.imports;

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.graph.FlatArray;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

@Slf4j

public class ByteOrderTests  extends BaseNd4jTestWithBackends {


    @AfterEach
    public void tearDown() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testByteArrayOrder1(Nd4jBackend backend) {
        val ndarray = Nd4j.create(DataType.FLOAT, 2).assign(1);

        assertEquals(DataType.FLOAT, ndarray.data().dataType());

        val array = ndarray.data().asBytes();

        assertEquals(8, array.length);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testByteArrayOrder2(Nd4jBackend backend) {
        val original = Nd4j.linspace(1, 25, 25, DataType.FLOAT).reshape(5, 5);
        val bufferBuilder = new FlatBufferBuilder(0);

        int array = original.toFlatArray(bufferBuilder);
        bufferBuilder.finish(array);

        val flatArray = FlatArray.getRootAsFlatArray(bufferBuilder.dataBuffer());

        val restored = Nd4j.createFromFlatArray(flatArray);

        assertEquals(original, restored);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testByteArrayOrder3(Nd4jBackend backend) {
        val original = Nd4j.linspace(1, 25, 25, DataType.FLOAT).reshape('f', 5, 5);
        val bufferBuilder = new FlatBufferBuilder(0);

        int array = original.toFlatArray(bufferBuilder);
        bufferBuilder.finish(array);

        val flatArray = FlatArray.getRootAsFlatArray(bufferBuilder.dataBuffer());

        val restored = Nd4j.createFromFlatArray(flatArray);

        assertEquals(original, restored);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShapeStridesOf1(Nd4jBackend backend) {
        val buffer = new int[]{2, 5, 5, 5, 1, 0, 1, 99};

        val shape = Shape.shapeOf(buffer);
        val strides = Shape.stridesOf(buffer);

        assertArrayEquals(new int[]{5, 5}, shape);
        assertArrayEquals(new int[]{5, 1}, strides);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShapeStridesOf2(Nd4jBackend backend) {
        val buffer = new int[]{3, 5, 5, 5, 25, 5, 1, 0, 1, 99};

        val shape = Shape.shapeOf(buffer);
        val strides = Shape.stridesOf(buffer);

        assertArrayEquals(new int[]{5, 5, 5}, shape);
        assertArrayEquals(new int[]{25, 5, 1}, strides);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarEncoding(Nd4jBackend backend) {
        val scalar = Nd4j.scalar(2.0f);

        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(0);
        val fb = scalar.toFlatArray(bufferBuilder);
        bufferBuilder.finish(fb);
        val db = bufferBuilder.dataBuffer();

        val flat = FlatArray.getRootAsFlatArray(db);


        val restored = Nd4j.createFromFlatArray(flat);

        assertEquals(scalar, restored);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorEncoding_1(Nd4jBackend backend) {
        val scalar = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5});

        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(0);
        val fb = scalar.toFlatArray(bufferBuilder);
        bufferBuilder.finish(fb);
        val db = bufferBuilder.dataBuffer();

        val flat = FlatArray.getRootAsFlatArray(db);

        val restored = Nd4j.createFromFlatArray(flat);

        assertEquals(scalar, restored);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorEncoding_2(Nd4jBackend backend) {
        val scalar = Nd4j.createFromArray(new double[]{1, 2, 3, 4, 5});

        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(0);
        val fb = scalar.toFlatArray(bufferBuilder);
        bufferBuilder.finish(fb);
        val db = bufferBuilder.dataBuffer();

        val flat = FlatArray.getRootAsFlatArray(db);

        val restored = Nd4j.createFromFlatArray(flat);

        assertEquals(scalar, restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStringEncoding_1(Nd4jBackend backend) {
        val strings = Arrays.asList("alpha", "beta", "gamma");
        val vector = Nd4j.create(strings, 3);

        val bufferBuilder = new FlatBufferBuilder(0);

        val fb = vector.toFlatArray(bufferBuilder);
        bufferBuilder.finish(fb);
        val db = bufferBuilder.dataBuffer();

        val flat = FlatArray.getRootAsFlatArray(db);

        val restored = Nd4j.createFromFlatArray(flat);

        assertEquals(vector, restored);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
