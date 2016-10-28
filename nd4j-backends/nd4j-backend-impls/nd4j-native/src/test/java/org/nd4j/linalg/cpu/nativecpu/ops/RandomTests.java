package org.nd4j.linalg.cpu.nativecpu.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BoundedDistribution;
import org.nd4j.linalg.cpu.nativecpu.rng.CpuNativeRandom;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertTrue;

/**
 * Tests for NativeRandom with respect to backend
 *
 * @author raver119@gmail.com
 */
public class RandomTests {

    @Test
    public void testDistribution() {
        CpuNativeRandom random1 = new CpuNativeRandom(119, 10000000);

        INDArray z1 = Nd4j.create(1000);
        BoundedDistribution distribution = new BoundedDistribution(z1, 1.0, 2.0);
        Nd4j.getExecutioner().exec(distribution, random1);

        System.out.println("Data: " + z1);
        for (int e = 0; e < z1.length(); e++) {
            double val = z1.getDouble(e);
            assertTrue(val >= 1.0 && val <= 2.0);
        }
    }
}
