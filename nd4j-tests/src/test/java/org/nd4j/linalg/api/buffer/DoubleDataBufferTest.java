/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.buffer;

import io.netty.buffer.ByteBuf;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.ByteBuffer;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Double data buffer tests
 *
 * This tests the double buffer data type
 * Put all buffer related tests here
 *
 * @author Adam Gibson
 */
public  class DoubleDataBufferTest extends BaseNd4jTest {

    public DoubleDataBufferTest(Nd4jBackend backend) {
        super(backend);
    }



    public DoubleDataBufferTest(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public DoubleDataBufferTest(String name) {
        super(name);
    }

    public DoubleDataBufferTest() {
    }

    @Before
    public void before() {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
    }

    @Test
    public void testGetSet() throws Exception {
        double[] d1 = new double[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        double[] d2 = d.asDouble();
        assertArrayEquals(d1, d2, 1e-1f);

    }


    @Test
    public void testDup() throws Exception {
        double[] d1 = new double[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        DataBuffer d2 = d.dup();
        assertEquals(d, d2);
    }

    @Test
    public void testPut() throws Exception {
        double[] d1 = new double[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        d.put(0, 0.0);
        double[] result = new double[]{0, 2, 3, 4};
        d1 = d.asDouble();
        assertArrayEquals(d1, result, 1e-1f);
    }


    @Test
    public void testGetRange() throws Exception {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        double[] get = buffer.getDoublesAt(0, 3);
        double[] data = new double[]{1, 2, 3};
        assertArrayEquals(get, data, 1e-1f);


        double[] get2 = buffer.asDouble();
        double[] allData = buffer.getDoublesAt(0, buffer.length());
        assertArrayEquals(get2, allData, 1e-1f);


    }


    @Test
    public void testGetOffsetRange() throws Exception {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        double[] get = buffer.getDoublesAt(1, 3);
        double[] data = new double[]{2, 3, 4};
        assertArrayEquals(get, data, 1e-1f);


        double[] allButLast = new double[]{2, 3, 4, 5};

        double[] allData = buffer.getDoublesAt(1, buffer.length());
        assertArrayEquals(allButLast, allData, 1e-1f);

    }

    @Test
    public void testAssign() {
        DataBuffer assertion = Nd4j.createBuffer(new double[]{1, 2, 3});
        DataBuffer one = Nd4j.createBuffer(new double[]{1});
        DataBuffer twoThree = Nd4j.createBuffer(new double[]{2,3});
        DataBuffer blank = Nd4j.createBuffer(new double[]{0, 0, 0});
        blank.assign(one,twoThree);
        assertEquals(assertion, blank);
    }



    @Override
    public char ordering() {
        return 'c';
    }
}
