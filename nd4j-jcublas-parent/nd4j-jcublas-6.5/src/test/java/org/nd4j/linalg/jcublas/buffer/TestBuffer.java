package org.nd4j.linalg.jcublas.buffer;

import jcuda.jcublas.JCublas;
import org.junit.After;
import org.junit.Before;
import org.nd4j.linalg.api.buffer.test.DataBufferTest;
import org.nd4j.linalg.jcublas.JCublasNDArrayFactory;

/**
 * Created by agibsonccc on 2/14/15.
 */
public class TestBuffer extends DataBufferTest {
    @Before
    public void before() {
        JCublas.cublasInit();

    }



}
