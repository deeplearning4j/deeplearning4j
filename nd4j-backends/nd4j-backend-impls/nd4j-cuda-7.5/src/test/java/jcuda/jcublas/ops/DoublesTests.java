package jcuda.jcublas.ops;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Min;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class DoublesTests {

    @Before
    public void setUp() throws Exception {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        CudaEnvironment.getInstance().getConfiguration().enableDebug(true).allowMultiGPU(false);
    }

    @Test
    public void testDoubleAxpy1() throws Exception {
//        Nd4j.getConstantHandler().getConstantBuffer(new double[]{1.0f});
//        Nd4j.getConstantHandler().getConstantBuffer(new double[]{0.0});


        INDArray array1 = Nd4j.create(new double[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f });
        INDArray array2 = Nd4j.create(new double[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f });

        //array1.muli(1.0);
        //array2.muli(1.0);


        long time1 = System.nanoTime();
        Nd4j.getBlasWrapper().axpy(new Double(0.75), array1, array2);
        long time2 = System.nanoTime();
        System.out.println("AXPY execution time: [" + (time2 - time1) + "] ns");

        assertEquals(1.767578125, array2.getDouble(0), 0.001);
        assertEquals(1.767578125, array2.getDouble(1), 0.001);
    }
}
