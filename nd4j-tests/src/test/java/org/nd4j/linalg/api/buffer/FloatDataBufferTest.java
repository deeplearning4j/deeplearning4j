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


import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 2/14/15.
 */
public  class FloatDataBufferTest  extends BaseNd4jTest {

    public FloatDataBufferTest(Nd4jBackend backend) {
        super(backend);
    }

    public FloatDataBufferTest(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public FloatDataBufferTest(String name) {
        super(name);
    }

    public FloatDataBufferTest() {
    }

    @Before
    public void before() {
        Nd4j.dtype = DataBuffer.Type.FLOAT;
    }

    @Test
    public void testGetSet() throws Exception {
        float[] d1 = new float[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        float[] d2 = d.asFloat();
        assertArrayEquals(d1, d2, 1e-1f);

    }

    @Test
    public void testDup() throws Exception {
        float[] d1 = new float[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        DataBuffer d2 = d.dup();
        assertEquals(d, d2);
    }

    @Test
    public void testPut() throws Exception {
        float[] d1 = new float[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        d.put(0, 0.0);
        float[] result = new float[]{0, 2, 3, 4};
        d1 = d.asFloat();
        assertArrayEquals(d1, result, 1e-1f);
    }


    @Test
    public void testGetRange() throws Exception {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        float[] get = buffer.getFloatsAt(0, 3);
        float[] data = new float[]{1, 2, 3};
        assertArrayEquals(get, data, 1e-1f);


        float[] get2 = buffer.asFloat();
        float[] allData = buffer.getFloatsAt(0, buffer.length());
        assertArrayEquals(get2, allData, 1e-1f);


    }


    @Test
    public void testGetOffsetRange() throws Exception {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        float[] get = buffer.getFloatsAt(1, 3);
        float[] data = new float[]{2, 3, 4};
        assertArrayEquals(get, data, 1e-1f);


        float[] allButLast = new float[]{2, 3, 4, 5};

        float[] allData = buffer.getFloatsAt(1, buffer.length());
        assertArrayEquals(allButLast, allData, 1e-1f);


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

    @Override
    public char ordering() {
        return 'c';
    }

}
