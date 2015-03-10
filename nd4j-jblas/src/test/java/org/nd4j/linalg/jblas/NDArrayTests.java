/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.jblas;


import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;


/**
 * NDArrayTests
 *
 * @author Adam Gibson
 */
public class NDArrayTests extends org.nd4j.linalg.api.test.NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayTests.class);


    @Test
    public void testMatrixVector() {
        double[][] data = new double[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 8}
        };


        Nd4j.factory().setOrder('f');
        double[] mmul = {1, 2, 3, 4};

        DoubleMatrix d = new DoubleMatrix(data);
        INDArray d2 = Nd4j.create(data);
        assertEquals(d.rows, d2.rows());
        assertEquals(d.columns, d2.columns());
        verifyElements(d, d2);

        INDArray toMmulD2 = Nd4j.create(mmul).transpose();
        DoubleMatrix toMmulD = new DoubleMatrix(mmul);


        assertEquals(d.rows, d2.rows());
        assertEquals(d.columns, d2.columns());

        assertEquals(toMmulD.rows, toMmulD2.rows());
        assertEquals(toMmulD.columns, toMmulD2.columns());

        DoubleMatrix mmulResultD = d.mmul(toMmulD);
        INDArray mmulResultD2 = d2.mmul(toMmulD2);

        verifyElements(mmulResultD, mmulResultD2);


        Nd4j.factory().setOrder('c');


    }

    @Test
    public void testTransposeMmul() {
        super.testTransposeMmul();
        Nd4j.factory().setOrder('f');

        //note that transpose() and transposei() are equivalent here
        INDArray a = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        DoubleMatrix aj = DoubleMatrix.linspace(1, 6, 6).reshape(2, 3);
        INDArray aT = a.transposei();
        DoubleMatrix aTj = aj.transpose();
        double[][] result = new double[][]{
                {1, 2}, {3, 4}, {5, 6}
        };
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                assertEquals(result[i][j], aT.getDouble(i, j), 1e-1);
            }
        }

        INDArray testMMul = a.mmul(aT);
        double[][] result2 = new double[][]{
                {35, 44}, {44, 56}
        };


        DoubleMatrix testMmulJ = aj.mmul(aTj);

        for (int i = 0; i < result2.length; i++) {
            for (int j = 0; j < result2[i].length; j++) {
                assertEquals(result2[i][j], testMMul.getDouble(i, j), 1e-1);
            }
        }

        for (int i = 0; i < result2.length; i++) {
            for (int j = 0; j < result2[i].length; j++) {
                assertEquals(result2[i][j], testMmulJ.get(i, j), 1e-1);
            }
        }

    }

    @Test
    public void testAxpy() {
        Nd4j.factory().setOrder('f');
        Nd4j.dtype = DataBuffer.FLOAT;
        INDArray row = Nd4j.create(new float[]{1, 2, 3});
        float alpha = 1f;
        INDArray matrix = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});
        INDArray row1 = matrix.slice(1);
        Nd4j.getBlasWrapper().axpy(alpha, row, row1);
        INDArray answer = Nd4j.create(new float[]{3, 6, 9});
        assertEquals(answer, row1);
    }


    @Test
    public void testFortranRavel() {
        double[][] data = new double[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 8}
        };

        INDArray toRavel = Nd4j.create(data);
        Nd4j.factory().setOrder('f');
        INDArray toRavelF = Nd4j.create(data);
        INDArray ravel = toRavel.ravel();
        INDArray ravelF = toRavelF.ravel();
        assertEquals(ravel, ravelF);
        Nd4j.factory().setOrder('c');

    }


    @Test
    public void testNorm1() {
        DoubleMatrix norm1 = DoubleMatrix.linspace(1, 8, 8).reshape(2, 4);
        INDArray norm1NDArray = Nd4j.linspace(1, 8, 8).reshape(2, 4);
        assertEquals(norm1.norm1(), norm1NDArray.norm1(Integer.MAX_VALUE).getFloat(0), 1e-1);
    }


    @Test
    public void testFortranReshapeMatrix() {
        double[][] data = new double[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 8}
        };

        Nd4j.factory().setOrder('f');

        DoubleMatrix d = new DoubleMatrix(data);
        INDArray d2 = Nd4j.create(data);
        assertEquals(d.rows, d2.rows());
        assertEquals(d.columns, d2.columns());
        verifyElements(d, d2);


        DoubleMatrix reshapedD = d.reshape(4, 2);
        INDArray reshapedD2 = d2.reshape(4, 2);
        verifyElements(reshapedD, reshapedD2);
        Nd4j.factory().setOrder('c');


    }


    @Test
    public void testDot() {
        INDArray linspace = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row1 = linspace.getRow(0);
        INDArray row2 = linspace.getRow(1);
        double dot = Nd4j.getBlasWrapper().dot(row1, row2);
        assertEquals(11, dot, 1e-1);


    }


    @Test
    public void testFortranCreation() {
        double[][] data = new double[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 8}
        };


        Nd4j.factory().setOrder('f');
        float[][] mmul = {{1, 2, 3, 4}, {5, 6, 7, 8}};

        INDArray d2 = Nd4j.create(data);
        verifyElements(mmul, d2);
    }


    @Test
    public void testMatrixMatrix() {
        double[][] data = new double[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 8}
        };


        Nd4j.factory().setOrder('f');
        double[][] mmul = {{1, 2, 3, 4}, {5, 6, 7, 8}};

        DoubleMatrix d = new DoubleMatrix(data).reshape(4, 2);
        INDArray d2 = Nd4j.create(data).reshape(4, 2);
        assertEquals(d.rows, d2.rows());
        assertEquals(d.columns, d2.columns());
        verifyElements(d, d2);

        INDArray toMmulD2 = Nd4j.create(mmul);
        DoubleMatrix toMmulD = new DoubleMatrix(mmul);

        DoubleMatrix mmulResultD = d.mmul(toMmulD);
        INDArray mmulResultD2 = d2.mmul(toMmulD2);
        verifyElements(mmulResultD, mmulResultD2);


        Nd4j.factory().setOrder('c');
    }

    @Test
    public void testVectorVector() {
        DoubleMatrix d = new DoubleMatrix(2, 1);
        d.data = new double[]{1, 2};
        DoubleMatrix d2 = new DoubleMatrix(1, 2);
        d2.data = new double[]{3, 4};

        INDArray d3 = Nd4j.create(new double[]{1, 2}).reshape(2, 1);
        INDArray d4 = Nd4j.create(new double[]{3, 4});

        assertEquals(d.rows, d3.rows());
        assertEquals(d.columns, d3.columns());

        assertEquals(d2.rows, d4.rows());
        assertEquals(d2.columns, d4.columns());

        DoubleMatrix resultMatrix = d.mmul(d2);


        INDArray resultNDArray = d3.mmul(d4);
        verifyElements(resultMatrix, resultNDArray);

    }


    @Test
    public void testVector() {
        Nd4j.factory().setOrder('f');

        DoubleMatrix dJblas = DoubleMatrix.linspace(1, 4, 4);
        INDArray d = Nd4j.linspace(1, 4, 4);
        verifyElements(dJblas, d);
        Nd4j.factory().setOrder('c');


    }

    @Test
    public void testRowVectorOps() {
        if (Nd4j.factory().order() == NDArrayFactory.C) {
            INDArray twoByTwo = Nd4j.create(new double[]{1, 3, 2, 4}, new int[]{2, 2});
            INDArray toAdd = Nd4j.create(new double[]{1, 2}, new int[]{2});
            twoByTwo.addiRowVector(toAdd);
            INDArray assertion = Nd4j.create(new double[]{2, 5, 3, 6}, new int[]{2, 2});
            assertEquals(assertion, twoByTwo);

        }


    }

    @Test
    public void testColumnVectorOps() {
        if (Nd4j.factory().order() == NDArrayFactory.C) {
            INDArray twoByTwo = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
            INDArray toAdd = Nd4j.create(new float[]{1, 2}, new int[]{2, 1});
            twoByTwo.addiColumnVector(toAdd);
            INDArray assertion = Nd4j.create(new float[]{2, 3, 5, 6}, new int[]{2, 2});
            assertEquals(assertion, twoByTwo);


        }


    }

    @Test
    public void testReshapeCompatibility() {
        Nd4j.factory().setOrder('f');
        DoubleMatrix oneThroughFourJblas = DoubleMatrix.linspace(1, 4, 4).reshape(2, 2);
        DoubleMatrix fiveThroughEightJblas = DoubleMatrix.linspace(5, 8, 4).reshape(2, 2);
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray fiveThroughEight = Nd4j.linspace(5, 8, 4).reshape(2, 2);
        verifyElements(oneThroughFourJblas, oneThroughFour);
        verifyElements(fiveThroughEightJblas, fiveThroughEight);
        Nd4j.factory().setOrder('c');

    }

    @Test
    public void testRowSumCompat() {
        Nd4j.factory().setOrder('f');
        DoubleMatrix rowsJblas = DoubleMatrix.linspace(1, 8, 8).reshape(2, 4);
        INDArray rows = Nd4j.linspace(1, 8, 8).reshape(2, 4);
        verifyElements(rowsJblas, rows);

        INDArray rowSums = rows.sum(1);
        DoubleMatrix jblasRowSums = rowsJblas.rowSums();
        verifyElements(jblasRowSums, rowSums);


        float[][] data = new float[][]{
                {1, 2}, {3, 4}
        };

        INDArray rowSumsData = Nd4j.create(data);
        Nd4j.factory().setOrder('c');
        INDArray rowSumsCOrder = Nd4j.create(data);
        assertEquals(rowSumsData, rowSumsCOrder);
        INDArray rowSumsDataSum = rowSumsData.sum(1);
        INDArray rowSumsCOrderSum = rowSumsCOrder.sum(1);
        assertEquals(rowSumsDataSum, rowSumsCOrderSum);
        INDArray assertion = Nd4j.create(new float[]{3, 7});
        assertEquals(assertion, rowSumsCOrderSum);
        assertEquals(assertion, rowSumsDataSum);
    }


    protected void verifyElements(float[][] d, INDArray d2) {
        for (int i = 0; i < d2.rows(); i++) {
            for (int j = 0; j < d2.columns(); j++) {
                float test1 = d[i][j];
                float test2 = d2.getFloat(i, j);
                assertEquals(test1, test2, 1e-6);
            }
        }
    }


    protected void verifyElements(DoubleMatrix d, INDArray d2) {
        if (d.isVector() && d2.isVector())
            for (int j = 0; j < d2.length(); j++) {
                float test1 = (float) d.get(j);
                float test2 = d2.getFloat(j);
                assertEquals(test1, test2, 1e-6);
            }

        else {
            for (int i = 0; i < d.rows; i++) {
                for (int j = 0; j < d.columns; j++) {
                    float test1 = (float) d.get(i, j);
                    float test2 = d2.getFloat(i, j);
                    assertEquals(test1, test2, 1e-6);
                }
            }
        }

    }

}
