package org.deeplearning4j.util;


import static org.junit.Assert.*;

import org.deeplearning4j.nn.linalg.NDArray;
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

    }

}
