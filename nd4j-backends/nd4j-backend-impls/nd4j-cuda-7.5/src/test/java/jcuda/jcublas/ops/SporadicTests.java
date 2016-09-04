package jcuda.jcublas.ops;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class SporadicTests {

    @Before
    public void setUp() throws Exception {
        CudaEnvironment.getInstance().getConfiguration().enableDebug(false).setVerbose(true);
    }

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

    @Test
    public void randomStrangeTest() {
        DataBuffer.Type type = Nd4j.dataType();
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        int a=9;
        int b=2;
        int[] shapes = new int[a];
        for (int i = 0; i < a; i++) {
            shapes[i] = b;
        }
        INDArray c = Nd4j.linspace(1, (int) (100 * 1 + 1 + 2), (int) Math.pow(b, a)).reshape(shapes);
        c=c.sum(0);
        double[] d = c.data().asDouble();
        System.out.println("d: " + Arrays.toString(d));

        DataTypeUtil.setDTypeForContext(type);
    }
}
