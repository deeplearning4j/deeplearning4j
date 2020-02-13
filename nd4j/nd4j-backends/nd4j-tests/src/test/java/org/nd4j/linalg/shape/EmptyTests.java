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

package org.nd4j.linalg.shape;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.bool.All;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.*;

@Slf4j
@RunWith(Parameterized.class)
public class EmptyTests extends BaseNd4jTest {

    DataType initialType;

    public EmptyTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }


    @Test
    public void testEmpyArray_1() {
        val array = Nd4j.empty();

        assertNotNull(array);
        assertTrue(array.isEmpty());

        assertFalse(array.isScalar());
        assertFalse(array.isVector());
        assertFalse(array.isRowVector());
        assertFalse(array.isColumnVector());
        assertFalse(array.isCompressed());
        assertFalse(array.isSparse());

        assertFalse(array.isAttached());

        assertEquals(Nd4j.dataType(), array.dataType());
    }


    @Test
    public void testEmptyDtype_1() {
        val array = Nd4j.empty(DataType.INT);

        assertTrue(array.isEmpty());
        assertEquals(DataType.INT, array.dataType());
    }

    @Test
    public void testEmptyDtype_2() {
        val array = Nd4j.empty(DataType.LONG);

        assertTrue(array.isEmpty());
        assertEquals(DataType.LONG, array.dataType());
    }

    @Test
    public void testConcat_1() {
        val row1 = Nd4j.create(new double[]{1, 1, 1, 1}, new long[]{1, 4});
        val row2 = Nd4j.create(new double[]{2, 2, 2, 2}, new long[]{1, 4});
        val row3 = Nd4j.create(new double[]{3, 3, 3, 3}, new long[]{1, 4});

        val exp = Nd4j.create(new double[]{1, 1, 1, 1,    2, 2, 2, 2,   3, 3, 3, 3}, new int[]{3, 4});

        val op = DynamicCustomOp.builder("concat")
                .addInputs(row1, row2, row3)
                .addIntegerArguments(0)
                .build();

        Nd4j.getExecutioner().exec(op);

        val z = op.getOutputArgument(0);

        assertEquals(exp, z);
    }

    @Test
    public void testEmptyReductions(){

        INDArray empty = Nd4j.empty(DataType.FLOAT);
        try {
            empty.sumNumber();
        } catch (Exception e){
            assertTrue(e.getMessage().contains("empty"));
        }

        try {
            empty.varNumber();
        } catch (Exception e){
            assertTrue(e.getMessage().contains("empty"));
        }

        try {
            empty.stdNumber();
        } catch (Exception e){
            assertTrue(e.getMessage().contains("empty"));
        }

        try {
            empty.meanNumber();
        } catch (Exception e){
            assertTrue(e.getMessage().contains("empty"));
        }
    }

    @Test
    public void testGetEmpty(){
        INDArray empty = Nd4j.empty(DataType.FLOAT);
        try {
            empty.getFloat(0);
        } catch (Exception e){
            assertTrue(e.getMessage().contains("empty"));
        }

        try {
            empty.getDouble(0);
        } catch (Exception e){
            assertTrue(e.getMessage().contains("empty"));
        }

        try {
            empty.getLong(0);
        } catch (Exception e){
            assertTrue(e.getMessage().contains("empty"));
        }
    }

    @Test
    public void testEmptyWithShape_1() {
        val array = Nd4j.create(DataType.FLOAT, 2, 0, 3);

        assertNotNull(array);
        assertEquals(DataType.FLOAT, array.dataType());
        assertEquals(0, array.length());
        assertTrue(array.isEmpty());
        assertArrayEquals(new long[]{2, 0, 3}, array.shape());
        assertArrayEquals(new long[]{0, 0, 0}, array.stride());
    }

    @Test
    public void testEmptyWithShape_2(){
        val array = Nd4j.create(DataType.FLOAT, 0);

        assertNotNull(array);
        assertEquals(DataType.FLOAT, array.dataType());
        assertEquals(0, array.length());
        assertTrue(array.isEmpty());
        assertArrayEquals(new long[]{0}, array.shape());
        assertArrayEquals(new long[]{0}, array.stride());
        assertEquals(1, array.rank());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEmptyWithShape_3() {
        val array = Nd4j.create(DataType.FLOAT, 2, 0, 3);
        array.tensorAlongDimension(0, 2);
    }

    @Test
    public void testEmptyWithShape_4(){
        val array = Nd4j.create(DataType.FLOAT, 0, 3);

        assertNotNull(array);
        assertEquals(DataType.FLOAT, array.dataType());
        assertEquals(0, array.length());
        assertTrue(array.isEmpty());
        assertArrayEquals(new long[]{0, 3}, array.shape());
        assertArrayEquals(new long[]{0, 0}, array.stride());
        assertEquals(2, array.rank());
        assertEquals(0, array.rows());
        assertEquals(3, array.columns());
        assertEquals(0, array.size(0));
        assertEquals(3, array.size(1));
        assertEquals(0, array.stride(0));
        assertEquals(0, array.stride(1));
    }

    @Test
    public void testEmptyReduction_1() {
        val x = Nd4j.create(DataType.FLOAT, 2, 0, 3);
        val e = Nd4j.create(DataType.FLOAT, 2, 1, 3).assign(0);

        val reduced = x.sum(true, 1);

        assertArrayEquals(e.shape(), reduced.shape());
        assertEquals(e, reduced);
    }

    @Test
    public void testEmptyReduction_2() {
        val x = Nd4j.create(DataType.FLOAT, 2, 0, 3);
        val e = Nd4j.create(DataType.FLOAT, 2, 3).assign(0);

        val reduced = x.sum(false, 1);

        assertArrayEquals(e.shape(), reduced.shape());
        assertEquals(e, reduced);
    }


    @Test
    public void testEmptyReduction_3() {
        val x = Nd4j.create(DataType.FLOAT, 2, 0);
        val e = Nd4j.create(DataType.FLOAT, 0);

        val reduced = x.argMax(0);

        assertArrayEquals(e.shape(), reduced.shape());
        assertEquals(e, reduced);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEmptyReduction_4() {
        val x = Nd4j.create(DataType.FLOAT, 2, 0);
        val e = Nd4j.create(DataType.FLOAT, 0);

        val reduced = x.argMax(1);

        assertArrayEquals(e.shape(), reduced.shape());
        assertEquals(e, reduced);
    }

    @Test
    public void testEmptyCreateMethods(){
        DataType dt = DataType.FLOAT;
        assertArrayEquals(new long[]{0}, Nd4j.create(0).shape());
        assertArrayEquals(new long[]{0,0}, Nd4j.create(0,0).shape());
        assertArrayEquals(new long[]{0,0,0}, Nd4j.create(0,0,0).shape());
        assertArrayEquals(new long[]{0}, Nd4j.create(0L).shape());
        assertArrayEquals(new long[]{0}, Nd4j.create(dt, 0L).shape());

        assertArrayEquals(new long[]{0}, Nd4j.zeros(0).shape());
        assertArrayEquals(new long[]{0,0}, Nd4j.zeros(0,0).shape());
        assertArrayEquals(new long[]{0,0,0}, Nd4j.zeros(0,0,0).shape());
        assertArrayEquals(new long[]{0,0,0}, Nd4j.zeros(new int[]{0,0,0}, 'f').shape());
        assertArrayEquals(new long[]{0}, Nd4j.zeros(0L).shape());
        assertArrayEquals(new long[]{0}, Nd4j.zeros(dt, 0L).shape());

        assertArrayEquals(new long[]{0}, Nd4j.ones(0).shape());
        assertArrayEquals(new long[]{0,0}, Nd4j.ones(0,0).shape());
        assertArrayEquals(new long[]{0,0,0}, Nd4j.ones(0,0,0).shape());
        assertArrayEquals(new long[]{0}, Nd4j.ones(0L).shape());
        assertArrayEquals(new long[]{0}, Nd4j.ones(dt, 0L).shape());

        assertArrayEquals(new long[]{0}, Nd4j.valueArrayOf(0, 1.0).shape());
        assertArrayEquals(new long[]{0}, Nd4j.valueArrayOf(0,1.0).shape());
        assertArrayEquals(new long[]{0,0}, Nd4j.valueArrayOf(0,0,1.0).shape());
        assertArrayEquals(new long[]{1,0}, Nd4j.valueArrayOf(new long[]{1,0}, 1.0).shape());
        assertArrayEquals(new long[]{1,0}, Nd4j.valueArrayOf(new long[]{1,0}, 1.0f).shape());
        assertArrayEquals(new long[]{1,0}, Nd4j.valueArrayOf(new long[]{1,0}, 1L).shape());
        assertArrayEquals(new long[]{1,0}, Nd4j.valueArrayOf(new long[]{1,0}, 1).shape());

        assertArrayEquals(new long[]{0}, Nd4j.createUninitialized(0).shape());
        assertArrayEquals(new long[]{0,0}, Nd4j.createUninitialized(0,0).shape());
        assertArrayEquals(new long[]{0,0}, Nd4j.createUninitialized(dt, 0,0).shape());

        assertArrayEquals(new long[]{0,0}, Nd4j.zerosLike(Nd4j.ones(0,0)).shape());
        assertArrayEquals(new long[]{0,0}, Nd4j.onesLike(Nd4j.ones(0,0)).shape());
        assertArrayEquals(new long[]{0,0}, Nd4j.ones(0,0).like().shape());
        assertArrayEquals(new long[]{0,0}, Nd4j.ones(0,0).ulike().shape());
    }

    @Test
    public void testEqualShapesEmpty(){
        assertTrue(Nd4j.create(0).equalShapes(Nd4j.create(0)));
        assertFalse(Nd4j.create(0).equalShapes(Nd4j.create(1, 0)));
    }

    @Test
    public void testEmptyWhere() {
        val mask = Nd4j.createFromArray(false,     false,     false,     false,     false);
        val result = Nd4j.where(mask, null, null);

        assertTrue(result[0].isEmpty());
        assertNotNull(result[0].shapeInfoDataBuffer().asLong());
    }

    @Test
    public void testAllEmptyReduce(){
        INDArray x = Nd4j.createFromArray(true, true, true);
        val all = new All(x);
        all.setEmptyReduce(true);   //For TF compatibility - empty array for axis (which means no-op - and NOT all array reduction)
        INDArray out = Nd4j.exec(all);
        assertEquals(x, out);
    }

    @Test
    public void testEmptyNoop() {
        val output = Nd4j.empty(DataType.LONG);

        val op = DynamicCustomOp.builder("noop")
                .addOutputs(output)
                .build();

        Nd4j.exec(op);
    }

    @Test
    public void testEmptyConstructor_1() {
        val x = Nd4j.create(new double[0]);
        assertTrue(x.isEmpty());
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
