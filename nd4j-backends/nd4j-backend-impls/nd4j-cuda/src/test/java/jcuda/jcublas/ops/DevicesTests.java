package jcuda.jcublas.ops;

import org.junit.Test;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author raver119@gmail.com
 */
public class DevicesTests {

    @Test
    public void testOtherDevice() {
        CudaEnvironment.getInstance().getConfiguration().useDevices(1);

        INDArray array = Nd4j.create(1000000);
        for (int i = 0; i < 1000000; i++) {
            array.addi(10f);
        }
    }
}
