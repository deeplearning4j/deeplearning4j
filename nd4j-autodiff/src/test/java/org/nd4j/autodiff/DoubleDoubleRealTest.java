package org.nd4j.autodiff;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class DoubleDoubleRealTest {

    @Test
    public void testEquals() throws Exception {
        DoubleDoubleReal first = new DoubleDoubleReal(12);
        DoubleDoubleReal second = new DoubleDoubleReal(12);
        DoubleDoubleReal third = new DoubleDoubleReal(11);

        assertTrue(first.equals(second));
        assertFalse(first.equals(third));
    }

    @Test
    public void testCompareToLessThan() throws Exception {
        DoubleDoubleReal val1 = new DoubleDoubleReal(10);
        DoubleDoubleReal val2 = new DoubleDoubleReal(20);

        assertTrue(val1.compareTo(val2) < 0);
    }

    @Test
    public void testCompareToGreaterThan() throws Exception {
        DoubleDoubleReal val1 = new DoubleDoubleReal(7);
        DoubleDoubleReal val2 = new DoubleDoubleReal(-4);

        assertTrue(val1.compareTo(val2) > 0);
    }

    @Test
    public void testCompareToEqual() throws Exception {
        DoubleDoubleReal val1 = new DoubleDoubleReal(5);
        DoubleDoubleReal val2 = new DoubleDoubleReal(5);

        assertTrue(val1.compareTo(val2) == 0);
    }
}
