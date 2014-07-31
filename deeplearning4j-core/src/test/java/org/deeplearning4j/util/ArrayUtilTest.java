package org.deeplearning4j.util;

import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 *
 */
public class ArrayUtilTest {

    private static Logger log = LoggerFactory.getLogger(ArrayUtilTest.class);

    @Test
    public void testRange() {
        int[] range = ArrayUtil.range(0,2);
        int[] test = {0,1};
        assertEquals(true, Arrays.equals(test,range));

        int[] test2 = {-1,0};
        int[] range2 = ArrayUtil.range(-1,1);
        assertEquals(true, Arrays.equals(test2,range2));

    }

    @Test
    public void testStrides() {
        int[] shape = {5,4,3};
        int[] cStyleStride = {12,3,1};
        int[] fortranStyleStride = {1,5,20};
        int[] fortranStyleTest = ArrayUtil.calcStridesFortran(shape);
        int[] cStyleTest = ArrayUtil.calcStrides(shape);
        assertEquals(true,Arrays.equals(cStyleStride,cStyleTest));
        assertEquals(true,Arrays.equals(fortranStyleStride,fortranStyleTest));

        int[] shape2 = {2,2};
        int[] cStyleStride2 = {2,1};
        int[] fortranStyleStride2 = {1,2};
        int[] cStyleTest2 = ArrayUtil.calcStrides(shape2);
        int[] fortranStyleTest2 = ArrayUtil.calcStridesFortran(shape2);
        assertEquals(true,Arrays.equals(cStyleStride2,cStyleTest2));
        assertEquals(true,Arrays.equals(fortranStyleStride2,fortranStyleTest2));



    }


}
