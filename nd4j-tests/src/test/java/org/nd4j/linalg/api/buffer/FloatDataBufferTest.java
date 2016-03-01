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

import static org.junit.Assert.*;

import io.netty.buffer.ByteBuf;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.SerializationUtils;

import java.io.*;
import java.nio.ByteBuffer;

import static org.junit.Assert.assertArrayEquals;

/**
 * Float data buffer tests
 *
 * This tests the float buffer data type
 * Put all buffer related tests here
 *
 * @author Adam Gibson
 */
public  class FloatDataBufferTest  extends BaseNd4jTest {

    public FloatDataBufferTest(Nd4jBackend backend) {
        super(backend);
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
        assertArrayEquals(getFailureMessage(),d1, d2, 1e-1f);

    }



    @Test
    public void testSerialization() {
        DataBuffer buf = Nd4j.createBuffer(5);
        String fileName = "buf.ser";
        File file = new File(fileName);
        file.deleteOnExit();
        SerializationUtils.saveObject(buf, file);
        DataBuffer buf2 = SerializationUtils.readObject(file);
        assertEquals(buf, buf2);

        Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;
        buf = Nd4j.createBuffer(5);
        file.deleteOnExit();
        SerializationUtils.saveObject(buf, file);
        buf2 = SerializationUtils.readObject(file);
        assertEquals(buf, buf2);
    }

    @Test
    public void testDup() throws Exception {
        float[] d1 = new float[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        DataBuffer d2 = d.dup();
        assertEquals(getFailureMessage(), d, d2);
    }

    @Test
    public void testToNio() {
        DataBuffer buff = Nd4j.createBuffer(new double[]{1, 2, 3, 4});
        assertEquals(4,buff.length());
        if(buff.allocationMode() == DataBuffer.AllocationMode.HEAP)
            return;
        ByteBuffer nio = buff.asNio();
        ByteBuf netty = buff.asNetty();
        assertEquals(16,netty.capacity());
        assertEquals(16,nio.capacity());

    }

    @Test
    public void testPut() throws Exception {
        float[] d1 = new float[]{1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        d.put(0, 0.0);
        float[] result = new float[]{0, 2, 3, 4};
        d1 = d.asFloat();
        assertArrayEquals(getFailureMessage(), d1, result, 1e-1f);
    }


    @Test
    public void testGetRange() throws Exception {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        float[] get = buffer.getFloatsAt(0, 3);
        float[] data = new float[]{1, 2, 3};
        assertArrayEquals(getFailureMessage(), get, data, 1e-1f);


        float[] get2 = buffer.asFloat();
        float[] allData = buffer.getFloatsAt(0, buffer.length());
        assertArrayEquals(getFailureMessage(),get2, allData, 1e-1f);


    }


    @Test
    public void testGetOffsetRange() throws Exception {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        float[] get = buffer.getFloatsAt(1, 3);
        float[] data = new float[]{2, 3, 4};
        assertArrayEquals(getFailureMessage(),get, data, 1e-1f);


        float[] allButLast = new float[]{2, 3, 4, 5};

        float[] allData = buffer.getFloatsAt(1, buffer.length());
        assertArrayEquals(getFailureMessage(),allButLast, allData, 1e-1f);


    }


    @Test
    public void testAsBytes() {
        INDArray arr = Nd4j.create(5);
        byte[] d = arr.data().asBytes();
        assertEquals(getFailureMessage(),4 * 5,d.length);
        INDArray rand = Nd4j.rand(3, 3);
        rand.data().asBytes();

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

    @Test
    public void testReadWrite() throws Exception {
        DataBuffer assertion = Nd4j.createBuffer(new double[]{1, 2, 3});
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        assertion.write(dos);

        DataBuffer clone = assertion.dup();
        assertion.read(new DataInputStream(new ByteArrayInputStream(bos.toByteArray())));
        assertEquals(assertion,clone);
    }

    @Test
    public void testOffset() {
        DataBuffer create = Nd4j.createBuffer(new float[]{1,2,3,4},2);
        assertEquals(2,create.length());
        assertEquals(4,create.underlyingLength());
        assertEquals(2,create.offset());
        assertEquals(3,create.getDouble(0),1e-1);
        assertEquals(4,create.getDouble(1),1e-1);

    }

    @Override
    public char ordering() {
        return 'c';
    }

}
