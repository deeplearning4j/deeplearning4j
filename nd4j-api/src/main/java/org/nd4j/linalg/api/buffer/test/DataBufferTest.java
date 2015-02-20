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


import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.ArrayOps;
import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactories;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 2/14/15.
 */
public abstract class DataBufferTest {


    @Test
    public void testGetSet() {
        double[] d1 = new double[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        double[] d2 = d.asDouble();
        assertArrayEquals(d1, d2, 1e-1);
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
        assertArrayEquals(d1, result, 1e-1);
        d.destroy();
    }

    @Test
    public void testApply() {
        INDArray ones = Nd4j.valueArrayOf(5, 2.0);
        ones.toString();
        DataBuffer buffer = ones.data();
        //square
        ElementWiseOp op = new ArrayOps()
                .from(ones).op(ElementWiseOpFactories.pow()).extraArgs(new Object[]{2})
                .build();
        buffer.apply(op);
        INDArray four = Nd4j.valueArrayOf(5, 4.0);
        DataBuffer d = four.data();
        assertEquals(buffer, d);
        buffer.destroy();
        d.destroy();


    }

    @Test
    public void testGetRange() {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        double[] get = buffer.getDoublesAt(0, 3);
        double[] data = new double[]{1, 2, 3};
        assertArrayEquals(get, data, 1e-1);


        double[] get2 = buffer.asDouble();
        double[] allData = buffer.getDoublesAt(0, buffer.length());
        assertArrayEquals(get2, allData, 1e-1);
        buffer.destroy();


    }


    @Test
    public void testGetOffsetRange() {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        double[] get = buffer.getDoublesAt(1, 3);
        double[] data = new double[]{2, 3, 4};
        assertArrayEquals(get, data, 1e-1);


        double[] allButLast = new double[]{2, 3, 4, 5};

        double[] allData = buffer.getDoublesAt(1, buffer.length());
        assertArrayEquals(allButLast, allData, 1e-1);
        buffer.destroy();


    }

    @Test
    public void testBufferElementWiseOperations() {
        DataBuffer buffer = Nd4j.ones(5).data();
        buffer.addi(1.0);
        float[] data = buffer.asFloat();
        for(int i = 0; i < data.length; i++)
              assertEquals(2.0,data[i],1e-1);
        buffer.subi(1.0);
        data = buffer.asFloat();
        for(int i = 0; i < data.length; i++)
            assertEquals(1.0,data[i],1e-1);
        buffer.muli(1.0);
        data = buffer.asFloat();
        for(int i = 0; i < data.length; i++)
            assertEquals(1.0,data[i],1e-1);

        buffer.divi(1.0);
        data = buffer.asFloat();
        for(int i = 0; i < data.length; i++)
            assertEquals(1.0,data[i],1e-1);


        buffer.destroy();
        buffer = Nd4j.ones(5).data();

        DataBuffer buffer2 = Nd4j.linspace(1,5,5).data();
        float[] data3 = buffer2.asFloat();
        buffer.muli(buffer2);
        data = buffer.asFloat();
        for(int i = 0; i < data3.length; i++)
            assertEquals(data[i],data3[i],1e-1);



        buffer.destroy();
        buffer2.destroy();


    }



}
