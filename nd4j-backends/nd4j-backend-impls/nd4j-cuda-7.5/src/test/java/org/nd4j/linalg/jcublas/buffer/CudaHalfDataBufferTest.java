package org.nd4j.linalg.jcublas.buffer;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.factory.CudaDataBufferFactory;

/**
 * @author raver119@gmail.com
 */
public class CudaHalfDataBufferTest {

    @Test
    public void testConversion1() throws Exception {
        DataBuffer bufferOriginal = new CudaFloatDataBuffer(new float[]{1f, 2f, 3f, 4f, 5f});

        CudaDataBufferFactory factory = new CudaDataBufferFactory();

        DataBuffer bufferHalfs = factory.convertToHalfs(bufferOriginal);


    }
}
