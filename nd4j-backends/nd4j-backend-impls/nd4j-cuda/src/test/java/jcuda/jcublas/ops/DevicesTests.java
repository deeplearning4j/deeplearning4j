package jcuda.jcublas.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * This unit should be run manually only, because it relies on init-time variables, and they can't be changed in runtime.
 *
 * @author raver119@gmail.com
 */
@Ignore
public class DevicesTests {

    @Test
    public void testOtherDevice1() {
        CudaEnvironment.getInstance().getConfiguration().useDevices(1, 2);

        INDArray array = Nd4j.create(1000000);
        for (int i = 0; i < 10000; i++) {
            array.addi(10f);
        }

        assertEquals(1, AtomicAllocator.getInstance().getAllocationPoint(array).getDeviceId());
    }

    @Test
    public void testOtherDevice2() {
        CudaEnvironment.getInstance().getConfiguration().useDevices(0);

        INDArray array = Nd4j.create(1000000);
        for (int i = 0; i < 10000; i++) {
            array.addi(10f);
        }

        assertEquals(0, AtomicAllocator.getInstance().getAllocationPoint(array).getDeviceId());
    }
}
