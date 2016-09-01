package jcuda.jcublas.ops;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class SporadicTests {

    @Test
    public void testIsMax1() throws Exception {
        int[] shape = new int[]{2,2};
        int length = 4;
        int alongDimension = 0;

        INDArray arrC = Nd4j.linspace(1,length, length).reshape('c',shape);
        Nd4j.getExecutioner().execAndReturn(new IsMax(arrC, alongDimension));

        //System.out.print(arrC);
        assertEquals(0.0, arrC.getDouble(0), 0.1);
        assertEquals(0.0, arrC.getDouble(1), 0.1);
        assertEquals(1.0, arrC.getDouble(2), 0.1);
        assertEquals(1.0, arrC.getDouble(3), 0.1);
    }
}
