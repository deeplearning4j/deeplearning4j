package org.nd4j.linalg.cpu.nativecpu;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.CBLAS;
import org.nd4j.linalg.cpu.nativecpu.util.LibUtils;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

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
    public void testMultiDimSum() {
        double[] data = new double[]{10, 26, 42};
        INDArray assertion = Nd4j.create(data);
        for(int i = 0; i < data.length; i++) {
            assertEquals(data[i],assertion.getDouble(i),1e-1);
        }

        INDArray twoTwoByThree = Nd4j.linspace(1,12,12).reshape(2, 2, 3);
        twoTwoByThree.toString();
        INDArray tensor = twoTwoByThree.tensorAlongDimension(2, 0, 1);
        INDArray multiSum = twoTwoByThree.sum(0, 1);
        assertEquals(assertion,multiSum);

    }

    @Test
    public void testArrCreationShape() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        for(int i = 0; i < 2; i++)
            assertEquals(2,arr.size(i));
        int[] stride = ArrayUtil.calcStrides(new int[]{2, 2});
        for(int i = 0; i < stride.length; i++) {
            assertEquals(stride[i],arr.stride(i));
        }

        assertArrayEquals(new int[]{2,2},arr.shape());
        assertArrayEquals(new int[]{2,1},arr.stride());
    }

    @Test
    public void testColumnSumDouble() {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        String s = twoByThree.toString();
        INDArray columnVar = twoByThree.sum(0);
        INDArray assertion = Nd4j.create(new float[]{44850.0f, 45000.0f, 45150.0f, 45300.0f});
        assertEquals(assertion, columnVar);

    }

}
