package org.nd4j.linalg.cpu.nativecpu;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.CBLAS;
import org.nd4j.linalg.cpu.nativecpu.util.LibUtils;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;
/**
 * @author Adam Gibson
 */
public class LoopTest {
    static {
        LibUtils.loadLibrary("libnd4j");
    }
    @Test
    public void testLoop() {
        INDArray linspace = Nd4j.linspace(1,4,4);
        System.out.println(System.getProperty("java.library.path"));
        float sum = CBLAS.sasum(4,linspace.data().asNioFloat(),1);
        assertEquals(10,sum,1e-1);

    }


    @Test
    public void testColumnSumDouble() {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnVar = twoByThree.sum(0);
        INDArray assertion = Nd4j.create(new float[]{44850.0f, 45000.0f, 45150.0f, 45300.0f});
        assertEquals(assertion, columnVar);

    }

}
