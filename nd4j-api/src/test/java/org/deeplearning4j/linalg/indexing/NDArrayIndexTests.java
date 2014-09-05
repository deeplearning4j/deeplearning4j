package org.deeplearning4j.linalg.indexing;

import static org.junit.Assert.*;

import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Some basic tests for the NDArrayIndex
 *
 * @author Adam Gibson
 */
public class NDArrayIndexTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayIndexTests.class);

    @Test
    public void testInterval() {
        int[] interval = NDArrayIndex.interval(0,2).indices();
        assertTrue(Arrays.equals(interval,new int[]{0,1}));
        int[] interval2 = NDArrayIndex.interval(1,3).indices();
        assertEquals(2,interval2.length);

    }

}
