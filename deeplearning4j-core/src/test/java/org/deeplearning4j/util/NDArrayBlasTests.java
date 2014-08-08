package org.deeplearning4j.util;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.junit.Test;
import static org.junit.Assert.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * NDArray Blas Tests
 *
 * @author Adam Gibson
 */
public class NDArrayBlasTests {

    private static Logger log = LoggerFactory.getLogger(NDArrayBlasTests.class);


    @Test
    public void testIaMax() {
        NDArray n = new NDArray(new double[]{1,2,3,4},new int[]{4});
        ComplexNDArray complex = new ComplexNDArray(new double[]{1,0,2,0,3,0,4,0},new int[]{4});
        int maxIndex = 3;
        assertEquals(maxIndex,NDArrayBlas.iamax(n));
        assertEquals(maxIndex,NDArrayBlas.iamax(complex));

    }







}
