package org.deeplearning4j.util;

import static org.junit.Assert.*;

import org.jblas.DoubleMatrix;
import org.junit.Test;

public class MatrixUtilTests {


    @Test
    public void testReverse() {
        DoubleMatrix zeros = DoubleMatrix.zeros(1,3);
        DoubleMatrix reverse = MatrixUtil.reverse(zeros);
        assertEquals(true, zeros.rows == reverse.rows);
        assertEquals(true, zeros.columns == reverse.columns);
    }

}
