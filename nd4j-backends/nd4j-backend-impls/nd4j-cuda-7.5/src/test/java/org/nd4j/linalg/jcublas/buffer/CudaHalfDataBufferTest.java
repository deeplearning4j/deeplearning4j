package org.nd4j.linalg.jcublas.buffer;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.factory.CudaDataBufferFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;

/**
 * @author raver119@gmail.com
 */
public class CudaHalfDataBufferTest {
    private static Logger logger = LoggerFactory.getLogger(CudaHalfDataBufferTest.class);

    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration().enableDebug(true);
    }

    @Test
    public void testConversion1() throws Exception {
        DataBuffer bufferOriginal = new CudaFloatDataBuffer(new float[]{1f, 2f, 3f, 4f, 5f});

        CudaDataBufferFactory factory = new CudaDataBufferFactory();

        DataBuffer bufferHalfs = factory.convertToHalfs(bufferOriginal);

        DataBuffer bufferRestored = factory.restoreFromHalfs(bufferHalfs);


        logger.info("Buffer original: {}", Arrays.toString(bufferOriginal.asFloat()));
        logger.info("Buffer restored: {}", Arrays.toString(bufferRestored.asFloat()));

        assertArrayEquals(bufferOriginal.asFloat(), bufferRestored.asFloat(), 0.01f);
    }
}
