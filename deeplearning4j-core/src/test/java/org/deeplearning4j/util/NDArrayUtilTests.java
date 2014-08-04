package org.deeplearning4j.util;


import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;

import org.deeplearning4j.nn.linalg.NDArray;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 *
 * @author Adam Gibson
 */
public class NDArrayUtilTests {

    private static Logger log = LoggerFactory.getLogger(NDArrayUtil.class);

    @Test
    public void testCenter() {
        NDArray a = new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{2,4});
        NDArray centered = NDArrayUtil.center(a,new int[]{2,2});
        NDArray assertion = new NDArray(new double[]{2,3,6,7},new int[]{2,2});
        assertEquals(assertion,centered);

    }

    @Test
    public void testPadWithZeros() {
        NDArray ret = new NDArray(new int[]{2,5});
        NDArray test = NDArrayUtil.padWithZeros(ret,new int[]{2,5,5});
        assertEquals(true, Arrays.equals(new int[]{2,5,5},test.shape()));
        assertEquals(ret.sum(),test.sum(),1e-1);
    }

    @Test
    public void testTruncate() {
        NDArray ret = new NDArray(new double[]{1,2,3,4},new int[]{2,2});
        //axis 0 or column wise
        NDArray truncated = NDArrayUtil.truncate(ret,1,0);
        NDArray answer = new NDArray(new double[]{1,2},new int[]{2});
        assertEquals(answer,truncated);
        //axis 1 or row wise
        NDArray answer0 = new NDArray(new double[]{1,3},new int[]{2});
        NDArray truncated0 = NDArrayUtil.truncate(ret,1,1);
        assertEquals(answer0,truncated0);


        NDArray arr2 = new NDArray(DoubleMatrix.linspace(1, 24, 24).data,new int[]{4,3,2});
        NDArray dimension1 = new NDArray(new double[]{1,2,3,4,7,8,9,10,13,14,15,16,19,20,21,22},new int[]{4,2,2});
        NDArray truncatedTest = NDArrayUtil.truncate(arr2,2,1);
        assertEquals(dimension1, truncatedTest);


        NDArray arr3 = new NDArray(DoubleMatrix.linspace(1,30,30).data,new int[]{3,5,2});
        NDArray truncatedArr3 = new NDArray(new double[]{1,2,3,4,11,12,13,14,21,22,23,24},new int[]{3,2,2});
        NDArray truncatedArr3Test = NDArrayUtil.truncate(arr3,2,1);
        assertEquals(truncatedArr3,truncatedArr3Test);
    }

}
