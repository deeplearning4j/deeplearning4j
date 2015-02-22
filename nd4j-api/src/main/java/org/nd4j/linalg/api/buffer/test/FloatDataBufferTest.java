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
import org.nd4j.linalg.ops.ArrayOps;
import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactories;
import org.nd4j.linalg.util.ArrayUtil;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 2/14/15.
 */
public abstract class FloatDataBufferTest {

     @Before
    public void before() {
        Nd4j.dtype = DataBuffer.FLOAT;
    }

    @Test
    public void testGetSet() {
        float[] d1 = new float[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        float[] d2 = d.asFloat();
        assertArrayEquals(d1, d2, 1e-1f);
        d.destroy();

    }

    @Test
    public void testDup() {
        float[] d1 = new float[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        DataBuffer d2 = d.dup();
        assertEquals(d, d2);
        d.destroy();
        d2.destroy();
    }

    @Test
    public void testPut() {
        float[] d1 = new float[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        d.put(0, 0.0);
        float[] result = new float[]{0, 2, 3, 4};
        d1 = d.asFloat();
        assertArrayEquals(d1, result, 1e-1f);
        d.destroy();
    }

    @Test
    public void testApply() {
        INDArray ones = Nd4j.valueArrayOf(5, 2.0f);
        DataBuffer buffer = ones.data();

        buffer.apply(ElementWiseOpFactories.pow().create(new Object[]{2.0}));
        INDArray four = Nd4j.valueArrayOf(5, 4.0f);
        DataBuffer d = four.data();
        assertEquals(buffer, d);
        buffer.destroy();
        d.destroy();


    }

    @Test
    public void testElementWiseOp() {
        DataBuffer buffer = Nd4j.ones(5).data();
        float[] data = ArrayUtil.copy(buffer.asFloat());
        buffer.apply(ElementWiseOpFactories.negative().create());
        float[] newData = buffer.asFloat();
        //negative
        for(int i = 0; i < data.length; i++) {
            assertEquals(-data[i],newData[i],1e-1);
        }
        //now back to positive
        buffer.apply(ElementWiseOpFactories.abs().create());
        newData = buffer.asFloat();
        for(int i = 0; i < data.length; i++) {
            assertEquals(data[i],newData[i],1e-1);
        }

        buffer.apply(ElementWiseOpFactories.pow().create(new Object[]{new Float(2)}));
        newData = buffer.asFloat();
        for(int i = 0; i < data.length; i++) {
            assertEquals(data[i],newData[i],1e-1);
        }

        buffer.destroy();



    }

    @Test
    public void testPow() {
        DataBuffer buffer = Nd4j.ones(5).data();
        float[] data = buffer.asFloat();
        float[] newData = buffer.asFloat();
        buffer.apply(ElementWiseOpFactories.pow().create(new Object[]{new Float(2)}));
        newData = buffer.asFloat();
        for(int i = 0; i < data.length; i++) {
            assertEquals(data[i],newData[i],1e-1);
        }
        buffer.destroy();
    }


    @Test
    public void testGetRange() {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        float[] get = buffer.getFloatsAt(0, 3);
        float[] data = new float[]{1, 2, 3};
        assertArrayEquals(get, data, 1e-1f);


        float[] get2 = buffer.asFloat();
        float[] allData = buffer.getFloatsAt(0, buffer.length());
        assertArrayEquals(get2, allData, 1e-1f);
        buffer.destroy();


    }


    @Test
    public void testGetOffsetRange() {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        float[] get = buffer.getFloatsAt(1, 3);
        float[] data = new float[]{2, 3, 4};
        assertArrayEquals(get, data, 1e-1f);


        float[] allButLast = new float[]{2, 3, 4, 5};

        float[] allData = buffer.getFloatsAt(1, buffer.length());
        assertArrayEquals(allButLast, allData, 1e-1f);
        buffer.destroy();


    }

    @Test
    public void testBufferElementWiseOperations() {
        DataBuffer buffer = Nd4j.ones(5).data();
        buffer.addi(1.0);
        float[] data = buffer.asFloat();
        for(int i = 0; i < data.length; i++)
              assertEquals(2.0,data[i],1e-1f);
        buffer.subi(1.0);
        data = buffer.asFloat();
        for(int i = 0; i < data.length; i++)
            assertEquals(1.0,data[i],1e-1f);
        buffer.muli(1.0);
        data = buffer.asFloat();
        for(int i = 0; i < data.length; i++)
            assertEquals(1.0,data[i],1e-1f);

        buffer.divi(1.0);
        data = buffer.asFloat();
        for(int i = 0; i < data.length; i++)
            assertEquals(1.0,data[i],1e-1f);


        buffer.destroy();
        buffer = Nd4j.ones(5).data();

        DataBuffer buffer2 = Nd4j.linspace(1,5,5).data();
        float[] data3 = buffer2.asFloat();
        buffer.muli(buffer2);
        data = buffer.asFloat();
        for(int i = 0; i < data3.length; i++)
            assertEquals(data[i],data3[i],1e-1f);



        buffer.destroy();
        buffer2.destroy();


    }



}
