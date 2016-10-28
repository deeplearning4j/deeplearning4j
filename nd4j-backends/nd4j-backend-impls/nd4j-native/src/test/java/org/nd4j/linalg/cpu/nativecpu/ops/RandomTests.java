package org.nd4j.linalg.cpu.nativecpu.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BoundedDistribution;
import org.nd4j.linalg.api.ops.random.impl.Linspace;
import org.nd4j.linalg.cpu.nativecpu.rng.CpuNativeRandom;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Tests for NativeRandom with respect to backend
 *
 * @author raver119@gmail.com
 */
public class RandomTests {
    private static CpuNativeRandom random = new CpuNativeRandom(119, 10000000);

    @Test
    public void testDistribution() throws Exception {
        CpuNativeRandom random1 = new CpuNativeRandom(119, 100000);
        CpuNativeRandom random2 = new CpuNativeRandom(119, 100000);

        INDArray z1 = Nd4j.create(200);
        INDArray z2 = Nd4j.create(200);
        BoundedDistribution distribution = new BoundedDistribution(z1, 1.0, 2.0);
        Nd4j.getExecutioner().exec(distribution, random1);
        BoundedDistribution distribution2 = new BoundedDistribution(z2, 1.0, 2.0);
        Nd4j.getExecutioner().exec(distribution2, random2);

        System.out.println("Data: " + z1);
        System.out.println("Data: " + z2);
        for (int e = 0; e < z1.length(); e++) {
            double val = z1.getDouble(e);
            assertTrue(val >= 1.0 && val <= 2.0);
        }

        //assertEquals(z1, z2);
    }


    @Test
    public void testLinspace1() throws Exception {
        INDArray z1 = Nd4j.linspace(1, 100, 200);

        Linspace linspace = new Linspace(1, 100, 200);
        Nd4j.getExecutioner().exec(linspace, random);

        INDArray z2 = linspace.z();

        assertEquals(z1, z2);
    }
}
