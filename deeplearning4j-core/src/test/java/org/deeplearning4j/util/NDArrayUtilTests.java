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
        NDArray ret = new NDArray(new int[]{2,1});
        NDArray truncated = NDArrayUtil.truncate(ret,new int[]{2});
        assertEquals(true,Arrays.equals(new int[]{2},truncated.shape()));
    }

}
