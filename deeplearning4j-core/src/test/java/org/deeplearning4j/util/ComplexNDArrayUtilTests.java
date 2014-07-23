package org.deeplearning4j.util;


import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

/**
 *
 * @author Adam Gibson
 */
public class ComplexNDArrayUtilTests {

    private static Logger log = LoggerFactory.getLogger(ComplexNDArrayUtilTests.class);


    @Test
    public void testPadWithZeros() {
        ComplexNDArray ret = new ComplexNDArray(new int[]{2,5});
        ComplexNDArray test = ComplexNDArrayUtil.padWithZeros(ret,new int[]{2,5,5});
        assertEquals(true, Arrays.equals(new int[]{2,5,5},test.shape()));
        assertEquals(ret.sum().real(),test.sum().real(),1e-1);
    }

    @Test
    public void testTruncate() {
        ComplexNDArray ret = new ComplexNDArray(new int[]{2,1});
        ComplexNDArray truncated = ComplexNDArrayUtil.truncate(ret,new int[]{2});
        assertEquals(true,Arrays.equals(new int[]{2},truncated.shape()));
    }

}
