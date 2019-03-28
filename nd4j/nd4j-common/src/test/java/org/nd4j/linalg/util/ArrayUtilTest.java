package org.nd4j.linalg.util;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class ArrayUtilTest {

    @Test
    public void testInvertPermutationInt(){
        assertArrayEquals(
                new int[]{ 2, 4, 3, 0, 1 },
                ArrayUtil.invertPermutation(3, 4, 0, 2, 1)
                );
    }

    @Test
    public void testInvertPermutationLong(){
        assertArrayEquals(
                new long[]{ 2, 4, 3, 0, 1 },
                ArrayUtil.invertPermutation(3L, 4L, 0L, 2L, 1L)
        );
    }

}
