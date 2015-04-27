/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.buffer.test;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 2/21/15.
 */
public abstract class DoubleDataBufferTest {

    @Before
    public void before() {
        Nd4j.dtype = DataBuffer.DOUBLE;
    }

    @Test
    public void testGetSet() {
        double[] d1 = new double[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        double[] d2 = d.asDouble();
        assertArrayEquals(d1, d2, 1e-1f);
        d.destroy();

    }


    @Test
    public void testDup() {
        double[] d1 = new double[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        DataBuffer d2 = d.dup();
        assertEquals(d, d2);
        d.destroy();
        d2.destroy();
    }

    @Test
    public void testPut() {
        double[] d1 = new double[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        d.put(0, 0.0);
        double[] result = new double[]{0, 2, 3, 4};
        d1 = d.asDouble();
        assertArrayEquals(d1, result, 1e-1f);
        d.destroy();
    }


    @Test
    public void testGetRange() {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        double[] get = buffer.getDoublesAt(0, 3);
        double[] data = new double[]{1, 2, 3};
        assertArrayEquals(get, data, 1e-1f);


        double[] get2 = buffer.asDouble();
        double[] allData = buffer.getDoublesAt(0, buffer.length());
        assertArrayEquals(get2, allData, 1e-1f);
        buffer.destroy();


    }


    @Test
    public void testGetOffsetRange() {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        double[] get = buffer.getDoublesAt(1, 3);
        double[] data = new double[]{2, 3, 4};
        assertArrayEquals(get, data, 1e-1f);


        double[] allButLast = new double[]{2, 3, 4, 5};

        double[] allData = buffer.getDoublesAt(1, buffer.length());
        assertArrayEquals(allButLast, allData, 1e-1f);
        buffer.destroy();


    }

    @Test
    public void testAssign() {
        INDArray oneTwo = Nd4j.create(new double[]{1, 2});
        INDArray threeFour = Nd4j.create(new double[]{3, 4});
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4);
        INDArray test = Nd4j.create(4);
        test.data().assign(oneTwo.data(), threeFour.data());
        assertEquals(oneThroughFour, test);
    }


}
