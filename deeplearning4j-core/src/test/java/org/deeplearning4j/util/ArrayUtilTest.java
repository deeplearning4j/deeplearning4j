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


}
