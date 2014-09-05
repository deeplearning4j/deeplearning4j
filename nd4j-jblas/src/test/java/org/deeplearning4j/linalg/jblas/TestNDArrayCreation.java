package org.deeplearning4j.linalg.jblas;

import static org.junit.Assert.*;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
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
        INDArray arr = NDArrays.create(1,1);
        assertTrue(arr.isScalar());

        INDArray arr2 = NDArrays.scalar(0d,0);
        assertEquals(arr,arr2);
        arr = NDArrays.create(1,1);

    }







}
