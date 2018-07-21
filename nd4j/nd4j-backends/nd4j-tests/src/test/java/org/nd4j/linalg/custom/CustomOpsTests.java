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

package org.nd4j.linalg.custom;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.custom.ScatterUpdate;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpStatus;
import org.nd4j.linalg.api.ops.random.compat.RandomStandardNormal;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * This class holds various CustomOps tests
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CustomOpsTests {

    @Test
    public void testNonInplaceOp1() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);
        val arrayZ = Nd4j.create(10, 10);

        arrayX.assign(3.0);
        arrayY.assign(1.0);

        val exp = Nd4j.create(10,10).assign(4.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .addInputs(arrayX, arrayY)
                .addOutputs(arrayZ)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayZ);
    }

    /**
     * This test works inplace, but without inplace declaration
     */
    @Test
    public void testNonInplaceOp2() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);

        arrayX.assign(3.0);
        arrayY.assign(1.0);

        val exp = Nd4j.create(10,10).assign(4.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .addInputs(arrayX, arrayY)
                .addOutputs(arrayX)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayX);
    }

    @Test
    @Ignore // it's noop, we dont care anymore
    public void testNoOp1() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(5, 3);

        arrayX.assign(3.0);
        arrayY.assign(1.0);

        val expX = Nd4j.create(10,10).assign(3.0);
        val expY = Nd4j.create(5,3).assign(1.0);

        CustomOp op = DynamicCustomOp.builder("noop")
                .addInputs(arrayX, arrayY)
                .addOutputs(arrayX, arrayY)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(expX, arrayX);
        assertEquals(expY, arrayY);
    }

    @Test
    public void testFloor() throws Exception {
        val arrayX = Nd4j.create(10, 10);

        arrayX.assign(3.0);

        val exp = Nd4j.create(10,10).assign(3.0);

        CustomOp op = DynamicCustomOp.builder("floor")
                .addInputs(arrayX)
                .addOutputs(arrayX)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayX);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testInplaceOp1() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);

        arrayX.assign(4.0);
        arrayY.assign(2.0);

        val exp = Nd4j.create(10,10).assign(6.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .addInputs(arrayX, arrayY)
                .callInplace(true)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayX);
    }

    @Test
    public void testNoneInplaceOp3() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);

        arrayX.assign(4.0);
        arrayY.assign(2.0);

        val exp = Nd4j.create(10,10).assign(6.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .addInputs(arrayX, arrayY)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, op.getOutputArgument(0));
    }


    @Test
    public void testMergeMax1() throws Exception {
        val array0 = Nd4j.create(new double[] {1, 0, 0, 0, 0});
        val array1 = Nd4j.create(new double[] {0, 2, 0, 0, 0});
        val array2 = Nd4j.create(new double[] {0, 0, 3, 0, 0});
        val array3 = Nd4j.create(new double[] {0, 0, 0, 4, 0});
        val array4 = Nd4j.create(new double[] {0, 0, 0, 0, 5});

        val z = Nd4j.create(5);
        val exp = Nd4j.create(new double[]{1, 2, 3, 4, 5});

        CustomOp op = DynamicCustomOp.builder("mergemax")
                .addInputs(array0, array1, array2, array3, array4)
                .addOutputs(z)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, z);
    }

    @Test
    public void testMergeMaxF() throws Exception {

        val array0 = Nd4j.rand('f', 5, 2).add(1); //some random array with +ve numbers
        val array1 = array0.dup('f').add(5);
        array1.put(0, 0, 0); //array1 is always bigger than array0 except at 0,0

        //expected value of maxmerge
        val exp = array1.dup('f');
        exp.putScalar(0, 0, array0.getDouble(0, 0));

        val zF = Nd4j.zeros(array0.shape(), 'f');
        CustomOp op = DynamicCustomOp.builder("mergemax")
                .addInputs(array0, array1)
                .addOutputs(zF)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, zF);
    }

    @Test
    public void testMergeMaxMixedOrder_Subtract() {
        val exp = Nd4j.create(new int[] {2, 2}, 'c').assign(5.0);
        Nd4j.getExecutioner().commit();;

        val array0 = Nd4j.create(new int[] {2, 2}, 'f'); //some random array with +ve numbers
        val array1 = array0.dup('c').addi(5.0);

        Nd4j.getExecutioner().commit();

        assertEquals(exp, array1);
    }

    @Test
    public void testMergeMaxSameOrder_Subtract() {
        val exp = Nd4j.create(new int[] {2, 2}, 'c').assign(5.0);
        Nd4j.getExecutioner().commit();;

        val array0 = Nd4j.create(new int[] {2, 2}, 'c'); //some random array with +ve numbers
        val array1 = array0.dup('c').addi(5);

        assertEquals(exp, array1);
    }

    @Test
    public void testMergeMaxMixedOrder() {
        val array0 = Nd4j.rand('f', 5, 2).addi(1); //some random array with +ve numbers
        val array1 = array0.dup('c').addi(5);
        array1.put(0, 0, 0); //array1 is always bigger than array0 except at 0,0

        //expected value of maxmerge
        val exp = array1.dup();
        exp.putScalar(0, 0, array0.getDouble(0, 0));

        val zF = Nd4j.zeros(array0.shape() ,'f');
        CustomOp op = DynamicCustomOp.builder("mergemax")
                .addInputs(array0, array1)
                .addOutputs(zF)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, zF);
    }


    @Test
    public void testOutputShapes1() {
        val array0 = Nd4j.rand('f', 5, 2).addi(1); //some random array with +ve numbers
        val array1 = array0.dup().addi(5);
        array1.put(0, 0, 0); //array1 is always bigger than array0 except at 0,0

        //expected value of maxmerge
        val exp = array1.dup();
        exp.putScalar(0, 0, array0.getDouble(0, 0));

        CustomOp op = DynamicCustomOp.builder("mergemax")
                .addInputs(array0, array1)
                .build();

        val shapes = Nd4j.getExecutioner().calculateOutputShape(op);

        assertEquals(1, shapes.size());
        assertArrayEquals(new long[]{5, 2}, shapes.get(0));
    }


    @Test
    public void testScatterUpdate1() throws Exception {
        val matrix = Nd4j.create(5, 5);
        val updates = Nd4j.create(2, 5).assign(1.0);
        int[] dims = new int[]{1};
        int[] indices = new int[]{1, 3};

        val exp0 = Nd4j.create(1, 5).assign(0);
        val exp1 = Nd4j.create(1, 5).assign(1);

        ScatterUpdate op = new ScatterUpdate(matrix, updates, indices, dims, ScatterUpdate.UpdateOp.ADD);
        Nd4j.getExecutioner().exec(op);

        log.info("Matrix: {}", matrix);
        assertEquals(exp0, matrix.getRow(0));
        assertEquals(exp1, matrix.getRow(1));
        assertEquals(exp0, matrix.getRow(2));
        assertEquals(exp1, matrix.getRow(3));
        assertEquals(exp0, matrix.getRow(4));
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testScatterUpdate2() throws Exception {
        val matrix = Nd4j.create(5, 5);
        val updates = Nd4j.create(2, 5).assign(1.0);
        int[] dims = new int[]{0};
        int[] indices = new int[]{0, 1};

        val exp0 = Nd4j.create(1, 5).assign(0);
        val exp1 = Nd4j.create(1, 5).assign(1);

        ScatterUpdate op = new ScatterUpdate(matrix, updates, indices, dims, ScatterUpdate.UpdateOp.ADD);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testScatterUpdate3() throws Exception {
        val matrix = Nd4j.create(5, 5);
        val updates = Nd4j.create(2, 5).assign(1.0);
        int[] dims = new int[]{1};
        int[] indices = new int[]{0, 6};

        val exp0 = Nd4j.create(1, 5).assign(0);
        val exp1 = Nd4j.create(1, 5).assign(1);

        ScatterUpdate op = new ScatterUpdate(matrix, updates, indices, dims, ScatterUpdate.UpdateOp.ADD);
    }

    @Test
    public void testOpStatus1() throws Exception {
        assertEquals(OpStatus.ND4J_STATUS_OK, OpStatus.byNumber(0));
    }

    @Test
    public void testRandomStandardNormal_1() {
        if (Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA)
            return;

        val shape = Nd4j.create(new float[] {5, 10});
        val op = new RandomStandardNormal(shape);

        Nd4j.getExecutioner().exec(op);

        assertEquals(1, op.outputArguments().length);
        val output = op.getOutputArgument(0);

        assertArrayEquals(new long[]{5, 10}, output.shape());
    }

    @Test
    public void testRandomStandardNormal_2() {
        if (Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA)
            return;

        val shape = new long[]{5, 10};
        val op = new RandomStandardNormal(shape);

        Nd4j.getExecutioner().exec(op);

        assertEquals(1, op.outputArguments().length);
        val output = op.getOutputArgument(0);

        assertArrayEquals(new long[]{5, 10}, output.shape());
    }
}
