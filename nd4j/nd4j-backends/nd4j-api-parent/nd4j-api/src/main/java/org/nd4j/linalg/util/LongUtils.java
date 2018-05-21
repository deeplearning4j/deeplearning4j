package org.nd4j.linalg.util;

/**
 * @author raver119@gmail.com
 */
public class LongUtils {

    public static int[] toInts(long[] array) {
        int[] ret = new int[array.length];
        for (int e = 0; e < array.length; e++) {
            ret[e] = (int) array[e];
        }

        return ret;
    }

    public static long[] toLongs(int[] array) {
        long[] ret = new long[array.length];
        for (int e = 0; e < array.length; e++) {
            ret[e] = array[e];
        }

        return ret;
    }
}
