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
        ComplexNDArray ret = new ComplexNDArray(new double[]{1,0,2,0,3,0,4,0},new int[]{2,2});
        ret.toString();
        //axis 0 or column wise
        ComplexNDArray truncated = ComplexNDArrayUtil.truncate(ret,new int[]{2},0);
        ComplexNDArray answer = new ComplexNDArray(new double[]{1,0,2,0},new int[]{2});
        assertEquals(answer,truncated);
        //axis 1 or row wise
        ComplexNDArray answer0 = new ComplexNDArray(new double[]{1,0,3,0},new int[]{2});
        ComplexNDArray truncated0 = ComplexNDArrayUtil.truncate(ret,new int[]{2},1);
        assertEquals(answer0,truncated0);

    }





}
