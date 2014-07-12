package org.deeplearning4j.nn;

import static org.junit.Assert.*;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * NDArrayTests
 * @author Adam Gibson
 */
public class NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayTests.class);

    @Test
    public void testBasicOps() {
        NDArray n = new NDArray(DoubleMatrix.ones(27).data,new int[]{3,3,3});
        assertEquals(27,n.length);
        n.checkDimensions(n.addi(1));
        assertEquals(54,n.sum(),1e-1);
        NDArray a = n.slice(2);
        assertEquals(true,Arrays.equals(new int[]{3,3},a.shape()));

    }

    @Test
    public void testTranspose() {
        NDArray n = new NDArray(DoubleMatrix.ones(100).data,new int[]{5,5,4});
        NDArray transpose = n.transpose();
        assertEquals(n.length,transpose.length);
        assertEquals(true,Arrays.equals(new int[]{4,5,5},transpose.shape()));

    }

    @Test
    public void testPermute() {
        NDArray n = new NDArray(DoubleMatrix.rand(20).data,new int[]{5,4});
        NDArray transpose = n.transpose();
        NDArray permute = n.permute(new int[]{1,0});
        assertEquals(permute,transpose);
        assertEquals(transpose.length,permute.length,1e-1);
    }





}
