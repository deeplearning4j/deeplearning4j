/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.imports;

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.graph.FlatArray;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@Slf4j
@RunWith(Parameterized.class)
public class ByteOrderTests  extends BaseNd4jTest {

    public ByteOrderTests(Nd4jBackend backend) {
        super(backend);
    }

    @After
    public void tearDown() throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    @Test
    public void testByteArrayOrder1() {
        val ndarray = Nd4j.create(2).assign(1);

        assertEquals(DataBuffer.Type.FLOAT, ndarray.data().dataType());

        val array = ndarray.data().asBytes();

        assertEquals(8, array.length);
    }

    @Test
    public void testByteArrayOrder2() {
        val original = Nd4j.linspace(1, 25, 25).reshape(5, 5);
        val bufferBuilder = new FlatBufferBuilder(0);

        int array = original.toFlatArray(bufferBuilder);
        bufferBuilder.finish(array);

        val flatArray = FlatArray.getRootAsFlatArray(bufferBuilder.dataBuffer());

        val restored = Nd4j.createFromFlatArray(flatArray);

        assertEquals(original, restored);
    }


    @Test
    public void testByteArrayOrder3() {
        val original = Nd4j.linspace(1, 25, 25).reshape('f', 5, 5);
        val bufferBuilder = new FlatBufferBuilder(0);

        int array = original.toFlatArray(bufferBuilder);
        bufferBuilder.finish(array);

        val flatArray = FlatArray.getRootAsFlatArray(bufferBuilder.dataBuffer());

        val restored = Nd4j.createFromFlatArray(flatArray);

        assertEquals(original, restored);
    }

    @Test
    public void testShapeStridesOf1() {
        val buffer = new int[]{2, 5, 5, 5, 1, 0, 1, 99};

        val shape = Shape.shapeOf(buffer);
        val strides = Shape.stridesOf(buffer);

        assertArrayEquals(new int[]{5, 5}, shape);
        assertArrayEquals(new int[]{5, 1}, strides);
    }

    @Test
    public void testShapeStridesOf2() {
        val buffer = new int[]{3, 5, 5, 5, 25, 5, 1, 0, 1, 99};

        val shape = Shape.shapeOf(buffer);
        val strides = Shape.stridesOf(buffer);

        assertArrayEquals(new int[]{5, 5, 5}, shape);
        assertArrayEquals(new int[]{25, 5, 1}, strides);
    }

    @Test
    public void testScalarEncoding() {
        val scalar = Nd4j.trueScalar(2.0f);

        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(0);
        val fb = scalar.toFlatArray(bufferBuilder);
        bufferBuilder.finish(fb);
        val db = bufferBuilder.dataBuffer();

        val flat = FlatArray.getRootAsFlatArray(db);


        val restored = Nd4j.createFromFlatArray(flat);

        assertEquals(scalar, restored);
    }


    @Test
    public void testVectorEncoding_1() {
        val scalar = Nd4j.trueVector(new float[]{1, 2, 3, 4, 5});

        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(0);
        val fb = scalar.toFlatArray(bufferBuilder);
        bufferBuilder.finish(fb);
        val db = bufferBuilder.dataBuffer();

        val flat = FlatArray.getRootAsFlatArray(db);

        val restored = Nd4j.createFromFlatArray(flat);

        assertEquals(scalar, restored);
    }

    @Test
    public void testVectorEncoding_2() {
        val scalar = Nd4j.trueVector(new double[]{1, 2, 3, 4, 5});

        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(0);
        val fb = scalar.toFlatArray(bufferBuilder);
        bufferBuilder.finish(fb);
        val db = bufferBuilder.dataBuffer();

        val flat = FlatArray.getRootAsFlatArray(db);

        val restored = Nd4j.createFromFlatArray(flat);

        assertEquals(scalar, restored);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
