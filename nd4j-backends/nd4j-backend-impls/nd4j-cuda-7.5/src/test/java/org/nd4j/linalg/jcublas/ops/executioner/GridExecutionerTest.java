package org.nd4j.linalg.jcublas.ops.executioner;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class GridExecutionerTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void isMatchingMetaOp1() throws Exception {
        GridExecutioner executioner = new GridExecutioner();

        INDArray array = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        ScalarAdd opB = new ScalarAdd(array, 10f);

        executioner.exec(opA);
        assertTrue(executioner.isMatchingMetaOp(opB));
    }

    @Test
    public void isMatchingMetaOp2() throws Exception {
        GridExecutioner executioner = new GridExecutioner();

        INDArray array = Nd4j.create(10);
        INDArray array2 = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        ScalarAdd opB = new ScalarAdd(array2, 10f);

        executioner.exec(opA);
        assertFalse(executioner.isMatchingMetaOp(opB));
    }

    @Test
    public void isMatchingMetaOp3() throws Exception {
        GridExecutioner executioner = new GridExecutioner();

        INDArray array = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        Max opB = new Max(array);

        executioner.exec(opA);
        assertFalse(executioner.isMatchingMetaOp(opB));
    }

}