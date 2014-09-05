package org.nd4j.linalg.jblas;

import static org.junit.Assert.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * NDArray creation tests
 *
 * @author Adam Gibson
 */
public class TestNDArrayCreation {

    private static Logger log = LoggerFactory.getLogger(TestNDArrayCreation.class);

    @Test
    public void testCreation() {
        INDArray arr = Nd4j.create(1, 1);
        assertTrue(arr.isScalar());

        INDArray arr2 = Nd4j.scalar(0d, 0);
        assertEquals(arr,arr2);
        arr = Nd4j.create(1, 1);

    }







}
