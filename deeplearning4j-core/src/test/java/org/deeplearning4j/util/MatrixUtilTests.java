package org.deeplearning4j.util;

import static org.junit.Assert.*;

import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MatrixUtilTests {

    private static Logger log = LoggerFactory.getLogger(MatrixUtilTests.class);




    @Test
    public void testRot90() {
        DoubleMatrix zeros = DoubleMatrix.rand(1, 3);
        log.info("Starting " + zeros);
        MatrixUtil.rot90(zeros);
        log.info("Ending " + zeros);

    }

    @Test
    public void testUpSample() {
        DoubleMatrix d = DoubleMatrix.ones(28,28);
        DoubleMatrix scale = new DoubleMatrix(2,1);
        MatrixUtil.assign(scale,2);
        MatrixUtil.upSample(d, scale);
    }

    @Test
    public void testCumSum() {
        DoubleMatrix test = new DoubleMatrix(new double[][]{
                {1,2,3},
                {4,5,6}
        });

        DoubleMatrix cumSum = MatrixUtil.cumsum(test);

        DoubleMatrix solution = new DoubleMatrix(new double[][]{
                {1,2,3},
                {5,7,9}
        });

        assertEquals(solution,cumSum);

    }

    @Test
    public void testReverse() {
        DoubleMatrix zeros = DoubleMatrix.zeros(1,3);
        DoubleMatrix reverse = MatrixUtil.reverse(zeros);
        assertEquals(true, zeros.rows == reverse.rows);
        assertEquals(true, zeros.columns == reverse.columns);
    }

}
