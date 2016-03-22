package org.nd4j.linalg.jcublas.buffer;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class CudaFloatDataBufferTest {

    @Test
    public void getDouble() throws Exception {
        DataBuffer buffer = Nd4j.createBuffer(new float[]{1f,2f,3f,4f});

        assertEquals("CudaFloatDataBuffer", buffer.getClass().getSimpleName());


        System.out.println("Starting check...");

        assertEquals(1.0f, buffer.getFloat(0), 0.001f);
        assertEquals(2.0f, buffer.getFloat(1), 0.001f);
        assertEquals(3.0f, buffer.getFloat(2), 0.001f);
        assertEquals(4.0f, buffer.getFloat(3), 0.001f);

        System.out.println("Data: " + buffer);
    }

    @Test
    public void testPut() throws Exception {
        DataBuffer buffer = Nd4j.createBuffer(new float[]{1f,2f,3f,4f});
        buffer.put(2, 16f);

        assertEquals(16.0f, buffer.getFloat(2), 0.001f);

        System.out.println("Data: " + buffer);
    }

    @Test
    public void testNdArrayView() throws Exception {
        INDArray array = Nd4j.create(new float[] {1f,2f,3f,4f});

        assertEquals(1.0f, array.getFloat(0), 0.001f);
        assertEquals(2.0f, array.getFloat(1), 0.001f);
        assertEquals(3.0f, array.getFloat(2), 0.001f);
        assertEquals(4.0f, array.getFloat(3), 0.001f);

    }
}