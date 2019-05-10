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

package org.nd4j.linalg.blas;


import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.Collections;

import static org.junit.Assert.*;

@Slf4j
@RunWith(Parameterized.class)
public class BlasTests extends BaseNd4jTest {

    public BlasTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void simpleTest() {
        INDArray m1 = Nd4j.create(new double[][]{{1.0}, {2.0}, {3.0}, {4.0}});

        m1 = m1.reshape(2, 2);

        INDArray m2 = Nd4j.create(new double[][]{{1.0, 2.0, 3.0, 4.0},});
        m2 = m2.reshape(2, 2);
        m2.setOrder('f');

        //mmul gives the correct result
        INDArray correctResult;
        correctResult = m1.mmul(m2);
        System.out.println("================");
        System.out.println(m1);
        System.out.println(m2);
        System.out.println(correctResult);
        System.out.println("================");
        INDArray newResult = Nd4j.create(DataType.DOUBLE, correctResult.shape(), 'c');
        m1.mmul(m2, newResult);
        assertEquals(correctResult, newResult);

        //But not so mmuli (which is somewhat mixed)
        INDArray target = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        target = m1.mmuli(m2, m1);
        assertEquals(true, target.equals(correctResult));
        assertEquals(true, m1.equals(correctResult));
    }


    @Test
    public void testGemmInvalid1() {
        final INDArray a = Nd4j.rand(3, 4);
        final INDArray b = Nd4j.rand(4, 5);

        final INDArray target = Nd4j.zeros(new int[]{2, 3, 5}, 'f');
        final INDArray view = target.tensorAlongDimension(0, 1, 2);

        try {
            Nd4j.gemm(a, b, view, false, false, 1.0, 0.0);
            fail("Expected exception");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("view"));
        }
    }

    @Test
    public void testGemmInvalid2() {
        final INDArray a = Nd4j.rand(4, 3);
        final INDArray b = Nd4j.rand(4, 5);

        final INDArray target = Nd4j.zeros(3, 5, 'c');

        try {
            Nd4j.gemm(a, b, target, true, false, 1.0, 0.0);
            fail("Expected exception");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("view"));
        }
    }

    @Test
    public void testGemmInvalid3() {
        final INDArray a = Nd4j.rand(4, 3);
        final INDArray b = Nd4j.rand(4, 5);

        final INDArray target = Nd4j.zeros(new int[]{2, 3, 5}, 'f');
        final INDArray view = target.tensorAlongDimension(0, 1, 2);

        try {
            Nd4j.gemm(a, b, view, true, false, 1.0, 0.0);
            fail("Expected exception");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("view"));
        }
    }

    @Test
    public void testGemm1() {
        final INDArray a = Nd4j.rand(4, 3);
        final INDArray b = Nd4j.rand(4, 5);

        final INDArray result = a.transpose().mmul(b);
        final INDArray result2 = Nd4j.gemm(a, b, true, false);

        assertEquals(result, result2);
    }

    @Test
    public void testGemm2() {
        final INDArray a = Nd4j.rand(4, 3);
        final INDArray b = Nd4j.rand(4, 5);

        final INDArray target = Nd4j.zeros(new int[]{2, 3, 5}, 'f');
        final INDArray view = target.tensorAlongDimension(0, 1, 2);

        a.transpose().mmuli(b, view);

        final INDArray result = a.transpose().mmul(b);

        assertEquals(result, view);
    }

    @Test
    public void testGemm3() {
        final INDArray a = Nd4j.rand(4, 3);
        final INDArray b = Nd4j.rand(4, 5);

        final INDArray target = Nd4j.zeros(new int[]{2, 3, 5}, 'c');
        final INDArray view = target.tensorAlongDimension(0, 1, 2);

        a.transpose().mmuli(b, view);

        final INDArray result = a.transpose().mmul(b);

        assertEquals(result, view);
    }


    @Test
    public void testMmuli1() {
        final INDArray activations = Nd4j.createUninitialized(new long[]{1, 3, 1}, 'f');
        final INDArray z = activations.tensorAlongDimension(0, 1, 2);

        Nd4j.getRandom().setSeed(12345);
        final INDArray a = Nd4j.rand(3, 4);
        final INDArray b = Nd4j.rand(4, 1);

        INDArray ab = a.mmul(b);
        a.mmul(b, z);
        assertEquals(ab, z);
    }

    @Test
    public void testMmuli2() {
        final INDArray activations = Nd4j.createUninitialized(new long[]{2, 3, 1}, 'f');
        final INDArray z = activations.tensorAlongDimension(0, 1, 2);

        Nd4j.getRandom().setSeed(12345);
        final INDArray a = Nd4j.rand(3, 4);
        final INDArray b = Nd4j.rand(4, 1);

        INDArray ab = a.mmul(b);
        a.mmul(b, z);
        assertEquals(ab, z);
    }

    @Test
    public void testMmuli3(){
        final INDArray activations = Nd4j.createUninitialized(new long[]{1, 3, 2}, 'f');
        final INDArray z = activations.tensorAlongDimension(0, 1, 2);

        final INDArray a = Nd4j.rand(3, 4);
        final INDArray b = Nd4j.rand(4, 2);

        INDArray ab = a.mmul(b);
        a.mmul(b, z);
        assertEquals(ab, z);
    }

    @Test
    public void test_Fp16_Mmuli_1(){
        final INDArray activations = Nd4j.createUninitialized(DataType.HALF, new long[]{1, 3, 2}, 'f');
        final INDArray z = activations.tensorAlongDimension(0, 1, 2);

        final INDArray a = Nd4j.rand(DataType.HALF, 3, 4);
        final INDArray b = Nd4j.rand(DataType.HALF,4, 2);

        INDArray ab = a.mmul(b);
        a.mmul(b, z);
        assertEquals(ab, z);
    }

    @Test
    public void test_Fp16_Mmuli_2(){
        val a = Nd4j.create(DataType.HALF, 32, 768);
        val b = Nd4j.create(DataType.HALF, 768);

        val c = a.mmul(b);
    }

    @Test
    @Ignore
    public void testHalfPrecision() {
        val a = Nd4j.create(DataType.HALF, 64, 768);
        val b = Nd4j.create(DataType.HALF, 768, 1024);
        val c = Nd4j.create(DataType.HALF, new long[]{64, 1024}, 'f');

        val durations = new ArrayList<Long>();
        val iterations = 100;
        for (int e = 0; e < iterations; e++) {
            val timeStart = System.currentTimeMillis();
            a.mmuli(b, c);
            val timeEnd = System.currentTimeMillis();
            durations.add(timeEnd - timeStart);
        }

        Collections.sort(durations);

        log.info("Median time: {} ms", durations.get(durations.size() / 2));
    }

    @Test
    public void testMmuli4(){
        try {
            Nd4j.rand(1, 3).mmuli(Nd4j.rand(3, 1), Nd4j.createUninitialized(new int[]{10, 10, 1}));
            fail("Expected exception");
        } catch (IllegalStateException e){
            assertTrue(e.getMessage().contains("shape"));
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
