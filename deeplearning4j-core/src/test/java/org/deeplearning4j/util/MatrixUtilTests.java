package org.deeplearning4j.util;

import static org.junit.Assert.*;

import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MatrixUtilTests {

    private static Logger log = LoggerFactory.getLogger(MatrixUtilTests.class);



    @Test
    public void test1DFilter() {
        DoubleMatrix x = new DoubleMatrix(new double[]{
                0.166968,
                0.064888,
                0.428329,
                0.864608,
                0.154120,
                0.881056,
                0.608056,
                0.575432,
                0.197161,
                0.047612
        });

        DoubleMatrix A = new DoubleMatrix(new double[]{
                1,
                2.7804e-4,
                3.2223e-4,
                2.4192e-4,
                9.5975e-4,
                8.5936e-4,
                7.5952e-4,
                2.6142e-4,
                4.7416e-4,
                1.3154e-4,
                1.8972e-4
        });

        log.info("A  " + A);
        DoubleMatrix B = new DoubleMatrix(new double[]{
                0.0063109 ,
                .3856189,
                .9307907,
                .1634197,
                0.9245115
        });
        log.info("B " + B);

        DoubleMatrix test = MatrixUtil.oneDimensionalDigitalFilter(B,A,x);
        DoubleMatrix answer = new DoubleMatrix(new double[]{
                0.0010537,
                0.0647952,
                0.1831190,
                0.2582388,
                0.8978864,
                0.9993074,
                1.0234692,
                1.8814360,
                1.0732472,
                1.5226931
        });
        assertEquals(answer,test);
    }

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
