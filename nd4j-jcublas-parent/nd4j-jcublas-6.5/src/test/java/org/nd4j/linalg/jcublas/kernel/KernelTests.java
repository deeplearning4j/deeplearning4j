package org.nd4j.linalg.jcublas.kernel;

import jcuda.driver.CUfunction;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.io.IOException;

/**
 * Created by agibsonccc on 2/17/15.
 */
public class KernelTests {


    @Test
    public  void testKernelLoading() throws IOException {
        KernelFunctions.load("addi.cu", DataBuffer.FLOAT);
        CUfunction function = KernelFunctions.loadFunction(0, KernelFunctions.load("addi.cu", DataBuffer.FLOAT),"add");
    }

}
