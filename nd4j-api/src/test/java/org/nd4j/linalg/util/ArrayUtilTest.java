package org.nd4j.linalg.util;

import org.junit.Test;

/**
 * Created by agibsonccc on 9/6/14.
 */
public class ArrayUtilTest {

    @Test
    public void testCalcStridesFortran() {
       int[] strides = ArrayUtil.calcStridesFortran(new int[]{1,2,2});
    }


}
