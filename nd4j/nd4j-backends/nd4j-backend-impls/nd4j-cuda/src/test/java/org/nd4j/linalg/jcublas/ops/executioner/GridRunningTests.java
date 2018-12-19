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

package org.nd4j.linalg.jcublas.ops.executioner;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * These tests are meant to run with GridExecutioner as current one
 * @author raver119@gmail.com
 */
public class GridRunningTests {

    @Test
    public void testScalarPassing1() throws Exception {
        INDArray array = Nd4j.create(5);
        INDArray exp = Nd4j.create(new float[]{6f, 6f, 6f, 6f, 6f});

        CudaGridExecutioner executioner = (CudaGridExecutioner) Nd4j.getExecutioner();

        ScalarAdd opA = new ScalarAdd(array, 1f);

        ScalarAdd opB = new ScalarAdd(array, 2f);

        ScalarAdd opC = new ScalarAdd(array, 3f);

        Nd4j.getExecutioner().exec(opA);
        assertEquals(1, executioner.getQueueLength());
        Nd4j.getExecutioner().exec(opB);
        assertEquals(1, executioner.getQueueLength());
        Nd4j.getExecutioner().exec(opC);
        assertEquals(1, executioner.getQueueLength());

        assertEquals(exp, array);

        assertEquals(0, executioner.getQueueLength());
    }

    @Test
    public void testScalarPassing2() throws Exception {
        INDArray array = Nd4j.create(5);
        INDArray exp = Nd4j.create(new float[]{6f, 6f, 6f, 6f, 6f});

        CudaGridExecutioner executioner = (CudaGridExecutioner) Nd4j.getExecutioner();

        ScalarAdd opA = new ScalarAdd(array, 1f);

        ScalarAdd opB = new ScalarAdd(array, 2f);

        ScalarAdd opC = new ScalarAdd(array, 3f);

        INDArray res1 = Nd4j.getExecutioner().execAndReturn(opA);
        assertEquals(1, executioner.getQueueLength());
        INDArray res2 = Nd4j.getExecutioner().execAndReturn(opB);
        assertEquals(1, executioner.getQueueLength());
        INDArray res3 = Nd4j.getExecutioner().execAndReturn(opC);
        assertEquals(1, executioner.getQueueLength());

        assertEquals(exp, array);

        assertEquals(0, executioner.getQueueLength());

        assertTrue(res1 == res2);
        assertTrue(res3 == res2);
    }

    @Test
    public void testMul_Scalar1() throws Exception {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        INDArray x = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        INDArray y = Nd4j.create(10).assign(0.000003);

        x.muli(y);
        x.divi(0.0000022);

        ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

        INDArray eX = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

        INDArray eY = Nd4j.create(10).assign(0.000003);
        ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

        eX.muli(eY);
        ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();
        System.out.println("Data before divi2: " + Arrays.toString(eX.data().asDouble()));

        eX.divi(0.0000022);
        ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

        System.out.println("Data1: " + Arrays.toString(x.data().asDouble()));
        System.out.println("Data2: " + Arrays.toString(eX.data().asDouble()));

        assertEquals(eX, x);
    }
}
