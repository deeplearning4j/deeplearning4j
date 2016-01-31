package org.nd4j.linalg.cpu.javacpp;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.CBLAS;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;
/**
 * @author Adam Gibson
 */
public class LoopTest {
    @Test
    public void testLoop() {
        INDArray linspace = Nd4j.linspace(1,4,4);
        System.out.println(System.getProperty("java.library.path"));
        float sum = CBLAS.sasum(4,linspace.data().asNioFloat(),1);
        assertEquals(10,sum,1e-1);

    }


}
