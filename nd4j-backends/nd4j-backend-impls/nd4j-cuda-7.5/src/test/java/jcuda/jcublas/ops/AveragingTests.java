package jcuda.jcublas.ops;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * @author raver119@gmail.com
 */
public class AveragingTests {

    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration().enableDebug(true).setVerbose(true);
    }

    @Test
    public void testSingleDeviceAveraging() throws Exception {
        INDArray array1 = Nd4j.valueArrayOf(500, 1.0);
        INDArray array2 = Nd4j.valueArrayOf(500, 2.0);
        INDArray array3 = Nd4j.valueArrayOf(500, 3.0);

        INDArray arrayMean = Nd4j.average(array1, array2, array3);


        assertNotEquals(null, arrayMean);

        assertEquals(2.0f, arrayMean.getFloat(12), 0.1f);
        assertEquals(2.0f, arrayMean.getFloat(150), 0.1f);
        assertEquals(2.0f, arrayMean.getFloat(475), 0.1f);
    }

}
