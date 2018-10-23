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

package org.nd4j.linalg.mixed;

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.graph.FlatArray;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.reduce.bool.IsInf;
import org.nd4j.linalg.api.ops.impl.reduce.bool.IsNaN;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldEqualTo;
import org.nd4j.linalg.api.ops.impl.transforms.strict.OldSoftMax;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Basic tests for mixed data types
 * @author raver119@gmail.com
 */
@Slf4j
public class MixedDataTypesTests {

    @Test
    public void testBasicCreation_1() throws Exception {
        val array = Nd4j.create(DataType.LONG, 3, 3);

        assertNotNull(array);
        assertEquals(9, array.length());
        assertEquals(DataType.LONG, array.dataType());
        assertEquals(DataType.LONG, ArrayOptionsHelper.dataType(array.shapeInfoJava()));
    }

    @Test
    public void testBasicCreation_2() throws Exception {
        val array = Nd4j.create(DataType.SHORT, 3, 3);

        assertNotNull(array);
        assertEquals(9, array.length());
        assertEquals(DataType.SHORT, array.dataType());
        assertEquals(DataType.SHORT, ArrayOptionsHelper.dataType(array.shapeInfoJava()));
    }

    @Test
    public void testBasicCreation_3() throws Exception {
        val array = Nd4j.create(DataType.HALF, 3, 3);

        assertNotNull(array);
        assertEquals(9, array.length());
        assertEquals(DataType.HALF, array.dataType());
        assertEquals(DataType.HALF, ArrayOptionsHelper.dataType(array.shapeInfoJava()));
    }

    @Test
    public void testBasicCreation_4() throws Exception {
        val scalar = Nd4j.trueScalar(1.0);
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.DOUBLE, scalar.dataType());
        assertEquals(1.0, scalar.getDouble(0), 1e-5);
    }

    @Test
    public void testBasicCreation_5() throws Exception {
        val scalar = Nd4j.trueScalar(new Integer(1));
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.INT, scalar.dataType());
        assertEquals(1.0, scalar.getInt(0), 1e-5);
    }

    @Test
    public void testBasicCreation_6() throws Exception {
        val scalar = Nd4j.trueScalar(1);
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.INT, scalar.dataType());
        assertEquals(1.0, scalar.getInt(0), 1e-5);
    }

    @Test
    public void testBasicCreation_7() throws Exception {
        val scalar = Nd4j.trueScalar(1L);
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.LONG, scalar.dataType());
        assertEquals(1, scalar.getInt(0));
    }

    @Test
    public void testBasicOps_1() throws Exception {
        val exp = new int[]{1,1,1,1,1,1,1,1,1};
        val array = Nd4j.create(DataType.INT, 3, 3);
        array.assign(1);

        val vector = array.data().asInt();
        assertArrayEquals(exp, vector);
    }

    @Test
    public void testBasicOps_2() throws Exception {
        val exp = new int[]{1,1,1,1,1,1,1,1,1};
        val arrayX = Nd4j.create(DataType.INT, 3, 3);
        val arrayY = Nd4j.create(new int[]{1,1,1,1,1,1,1,1,1}, new long[]{3, 3}, DataType.INT);

        arrayX.addi(arrayY);

        val vector = arrayX.data().asInt();
        assertArrayEquals(exp, vector);
    }

    @Test
    public void testBasicOps_3() throws Exception {
        val exp = new int[]{1,1,1,1,1,1,1,1,1};
        val arrayX = Nd4j.create(DataType.INT, 3, 3);
        val arrayY = Nd4j.create(new int[]{1,1,1,1,1,1,1,1,1}, new long[]{3, 3}, DataType.LONG);

        arrayX.addi(arrayY);

        val vector = arrayX.data().asInt();
        assertArrayEquals(exp, vector);
    }

    @Test
    public void testBasicOps_4() throws Exception {
        val arrayX = Nd4j.create(new int[]{7,8,7,9,1,1,1,1,1}, new long[]{3, 3}, DataType.LONG);

        val result = arrayX.maxNumber();
        val l = result.longValue();

        assertEquals(9L, l);
    }

    @Test
    public void testBasicOps_5() throws Exception {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);

        val result = arrayX.meanNumber().floatValue();

        assertEquals(2.5f, result, 1e-5);
    }

    @Test
    public void testBasicOps_6() throws Exception {
        val arrayX = Nd4j.create(new int[]{1, 0, 0, 4}, new  long[]{4}, DataType.INT);

        val z = Nd4j.getExecutioner().exec(new CountNonZero(arrayX)).z();

        assertEquals(DataType.LONG, z.dataType());
        val result = z.getInt(0);

        assertEquals(2, result);
    }

    @Test
    public void testBasicOps_7() throws Exception {
        val arrayX = Nd4j.create(new float[]{1, 0, Float.NaN, 4}, new  long[]{4}, DataType.FLOAT);

        val z = Nd4j.getExecutioner().exec(new IsInf(arrayX)).z();

        assertEquals(DataType.BOOL, z.dataType());
        val result = z.getInt(0);

        val z2 = Nd4j.getExecutioner().exec(new IsNaN(arrayX)).z();
        assertEquals(DataType.BOOL, z2.dataType());
        val result2 = z2.getInt(0);

        assertEquals(1, result2);
    }

    @Test
    public void testBasicOps_8() throws Exception {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new int[]{1, 0, 0, 4}, new  long[]{4}, DataType.INT);
        val exp = new long[]{1, 0, 0, 1};

        val result = Nd4j.getExecutioner().exec(new OldEqualTo(arrayX, arrayY)).z();
        val arr = result.data().asLong();

        assertArrayEquals(exp, arr);
    }

    @Test
    public void testBasicOps_9() throws Exception {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val exp = new long[]{1, 0, 0, 1};

        val op = new CosineSimilarity(arrayX, arrayY);
        val result = Nd4j.getExecutioner().exec(op).z();
        assertEquals(DataType.FLOAT, result.dataType());
        val arr = result.getDouble(0);

        assertEquals(1.0, arr, 1e-5);
    }

    @Test
    public void testMethods_1() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val exp = Nd4j.create(new int[]{2, 4, 6, 8}, new  long[]{4}, DataType.INT);

        val arrayZ = arrayX.add(arrayY);
        assertEquals(DataType.INT, arrayZ.dataType());
        assertEquals(exp, arrayZ);
    }

    @Test
    public void testMethods_2() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new double[]{1, 2, 3, 4}, new  long[]{4}, DataType.DOUBLE);
        val exp = Nd4j.create(new double[]{2, 4, 6, 8}, new  long[]{4}, DataType.DOUBLE);

        val arrayZ = arrayX.add(arrayY);

        assertEquals(DataType.DOUBLE, arrayZ.dataType());
        assertEquals(exp, arrayZ);
    }

    @Test
    public void testMethods_3() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new double[]{0.5, 0.5, 0.5, 0.5}, new  long[]{4}, DataType.DOUBLE);
        val exp = Nd4j.create(new double[]{1.5, 2.5, 3.5, 4.5}, new  long[]{4}, DataType.DOUBLE);

        val arrayZ = arrayX.add(arrayY);

        assertEquals(DataType.DOUBLE, arrayZ.dataType());
        assertEquals(exp, arrayZ);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTypesValidation_1() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.LONG);
        val arrayY = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val exp = new long[]{1, 0, 0, 1};

        val op = new CosineSimilarity(arrayX, arrayY);
        val result = Nd4j.getExecutioner().exec(op).z();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTypesValidation_2() throws Exception {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new int[]{1, 0, 0, 4}, new  long[]{4}, DataType.LONG);
        val exp = new long[]{1, 0, 0, 1};

        val result = Nd4j.getExecutioner().exec(new OldEqualTo(arrayX, arrayY)).z();
        val arr = result.data().asLong();

        assertArrayEquals(exp, arr);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTypesValidation_3() throws Exception {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);

        val result = Nd4j.getExecutioner().exec(new OldSoftMax(arrayX)).z();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTypesValidation_4() throws Exception {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new int[]{1, 0, 0, 4}, new  long[]{4}, DataType.DOUBLE);

        arrayX.addi(arrayY);
    }


    @Test
    public void testFlatSerde_1() throws Exception {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);

        val builder = new FlatBufferBuilder(512);
        val flat = arrayX.toFlatArray(builder);
        builder.finish(flat);
        val db = builder.dataBuffer();

        val flatb = FlatArray.getRootAsFlatArray(db);

        val restored = Nd4j.createFromFlatArray(flatb);

        assertEquals(arrayX, restored);
    }

    @Test
    public void testFlatSerde_2() throws Exception {
        val arrayX = Nd4j.create(new long[]{1, 2, 3, 4}, new  long[]{4}, DataType.LONG);

        val builder = new FlatBufferBuilder(512);
        val flat = arrayX.toFlatArray(builder);
        builder.finish(flat);
        val db = builder.dataBuffer();

        val flatb = FlatArray.getRootAsFlatArray(db);

        val restored = Nd4j.createFromFlatArray(flatb);

        assertEquals(arrayX, restored);
    }

    @Test
    public void testFlatSerde_3() throws Exception {
        val arrayX = Nd4j.create(new boolean[]{true, false, true, true}, new  long[]{4}, DataType.BOOL);

        val builder = new FlatBufferBuilder(512);
        val flat = arrayX.toFlatArray(builder);
        builder.finish(flat);
        val db = builder.dataBuffer();

        val flatb = FlatArray.getRootAsFlatArray(db);

        val restored = Nd4j.createFromFlatArray(flatb);

        assertEquals(arrayX, restored);
    }

}
