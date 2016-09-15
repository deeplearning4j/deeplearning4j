package org.nd4j.linalg.jcublas.ops.executioner;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * These tests are meant to run with GridExecutioner as current one
 * @author raver119@gmail.com
 */
public class GridRunningTests {

    @Test
    public void testScalarPassing1() throws Exception {
        INDArray array = Nd4j.create(5);
        INDArray exp = Nd4j.create(new float[]{6f, 6f, 6f, 6f, 6f});

        CudaGridExecutioner executioner = (CudaGridExecutioner) Nd4j.getExecutioner();

        ScalarAdd opA = new ScalarAdd(array, 1f);

        ScalarAdd opB = new ScalarAdd(array, 2f);

        ScalarAdd opC = new ScalarAdd(array, 3f);

        Nd4j.getExecutioner().exec(opA);
        assertEquals(1, executioner.getQueueLength());
        Nd4j.getExecutioner().exec(opB);
        assertEquals(1, executioner.getQueueLength());
        Nd4j.getExecutioner().exec(opC);
        assertEquals(1, executioner.getQueueLength());

        assertEquals(exp, array);

        assertEquals(0, executioner.getQueueLength());
    }

    @Test
    public void testScalarPassing2() throws Exception {
        INDArray array = Nd4j.create(5);
        INDArray exp = Nd4j.create(new float[]{6f, 6f, 6f, 6f, 6f});

        CudaGridExecutioner executioner = (CudaGridExecutioner) Nd4j.getExecutioner();

        ScalarAdd opA = new ScalarAdd(array, 1f);

        ScalarAdd opB = new ScalarAdd(array, 2f);

        ScalarAdd opC = new ScalarAdd(array, 3f);

        INDArray res1 = Nd4j.getExecutioner().execAndReturn(opA);
        assertEquals(1, executioner.getQueueLength());
        INDArray res2 = Nd4j.getExecutioner().execAndReturn(opB);
        assertEquals(1, executioner.getQueueLength());
        INDArray res3 = Nd4j.getExecutioner().execAndReturn(opC);
        assertEquals(1, executioner.getQueueLength());

        assertEquals(exp, array);

        assertEquals(0, executioner.getQueueLength());

        assertTrue(res1 == res2);
        assertTrue(res3 == res2);
    }
}
