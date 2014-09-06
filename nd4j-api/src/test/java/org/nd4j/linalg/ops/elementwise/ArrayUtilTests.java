package org.nd4j.linalg.ops.elementwise;

import org.nd4j.linalg.util.ArrayUtil;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Adam Gibson
 */
public class ArrayUtilTests {
    private static Logger log = LoggerFactory.getLogger(ArrayUtilTests.class);

    @Test(expected = AssertionError.class)
    public void testAssertSquareError() {
         double[][] d = new double[][]{
                 {1,2,3},{2,1}
         };
        ArrayUtil.assertSquare(d);
    }



}
