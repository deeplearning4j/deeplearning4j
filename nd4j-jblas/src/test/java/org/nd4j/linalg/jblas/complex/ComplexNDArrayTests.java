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

package org.nd4j.linalg.jblas.complex;

import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;


/**
 * Tests for a complex ndarray
 *
 * @author Adam Gibson
 */
public class ComplexNDArrayTests extends org.nd4j.linalg.api.test.ComplexNDArrayTests {

    private static Logger log = LoggerFactory.getLogger(ComplexNDArrayTests.class);


    @Test
    public void testPut() {
        Nd4j.factory().setOrder('f');
        ComplexDoubleMatrix c = new ComplexDoubleMatrix(2, 2, 1, 0, 2, 0, 3, 0, 4, 0);
        IComplexNDArray c2 = Nd4j.createComplex(new float[]{1, 0, 2, 0, 3, 0, 4, 0}, new int[]{2, 2});
        verifyElements(c, c2);

        c.put(1, 1, new ComplexDouble(4, 6));
        c2.putScalar(1, 1, Nd4j.createDouble(4, 6));
        verifyElements(c, c2);


    }

    @Test
    public void testComplexMult() {
        INDArray ones = Nd4j.ones(10);
        IComplexNDArray complex = Nd4j.complexOnes(10);
        assertEquals(ones.mul(Nd4j.createDouble(1, 0)), complex);
        INDArray fives = Nd4j.valueArrayOf(new int[]{10}, 5);
        IComplexNDArray ten = Nd4j.complexValueOf(10, 10);
        IComplexNDArray tenTest = fives.muli(Nd4j.createDouble(2, 0));
        assertEquals(ten, tenTest);


    }


    @Test
    public void testOrdering() {
        DoubleMatrix linspace = DoubleMatrix.linspace(1, 10, 10);
        ComplexDoubleMatrix c = new ComplexDoubleMatrix(linspace);

        INDArray linspace2 = Nd4j.linspace(1, 10, 10).transpose();
        IComplexNDArray c2 = Nd4j.createComplex(linspace2);

        assertEquals(c.rows, c2.rows());
        assertEquals(c.columns, c2.columns());

        verifyElements(linspace, linspace2);
        verifyElements(c, c2);

        IComplexNDArray transpose = c2.transpose();
        ComplexDoubleMatrix c3 = c.transpose();
        assertEquals(c3.rows, transpose.rows());
        assertEquals(c3.columns, transpose.columns());
        verifyElements(c3, transpose);

        assertEquals(c3.rows, transpose.rows());
        assertEquals(c3.columns, transpose.columns());


        ComplexDoubleMatrix mmul = c3.mmul(c);
        IComplexNDArray mmulNDArray = transpose.mmul(c2);
        verifyElements(mmul, mmulNDArray);


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

        ComplexDoubleMatrix complexOneThroughForJblas = new ComplexDoubleMatrix(oneThroughFourJblas);
        ComplexDoubleMatrix complexFiveThroughEightJblas = new ComplexDoubleMatrix(fiveThroughEightJblas);

        IComplexNDArray complexOneThroughFour = Nd4j.createComplex(oneThroughFour);
        IComplexNDArray complexFiveThroughEight = Nd4j.createComplex(fiveThroughEight);
        verifyElements(complexOneThroughForJblas, complexOneThroughFour);
        verifyElements(complexFiveThroughEightJblas, complexFiveThroughEight);

        Nd4j.factory().setOrder('c');

    }


    @Test
    public void testTwoByTwoMmulJblas() {
        Nd4j.factory().setOrder('f');
        ComplexDoubleMatrix oneThroughForJblas = new ComplexDoubleMatrix(DoubleMatrix.linspace(1, 4, 4).reshape(2, 2));
        ComplexDoubleMatrix fiveThroughEightJblas = new ComplexDoubleMatrix(DoubleMatrix.linspace(5, 8, 4).reshape(2, 2));
        ComplexDoubleMatrix jBlasResult = oneThroughForJblas.mmul(fiveThroughEightJblas);


        INDArray plainOneThroughFour = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray plainFiveThroughEight = Nd4j.linspace(5, 8, 4).reshape(2, 2);

        IComplexNDArray oneThroughFour = Nd4j.createComplex(plainOneThroughFour);
        IComplexNDArray fiveThroughEight = Nd4j.createComplex(plainFiveThroughEight);
        verifyElements(oneThroughForJblas, oneThroughFour);
        verifyElements(fiveThroughEightJblas, fiveThroughEight);


        IComplexNDArray test = oneThroughFour.mmul(fiveThroughEight);
        verifyElements(jBlasResult, test);
    }


    protected void verifyElements(double[][] d, INDArray d2) {
        for (int i = 0; i < d2.rows(); i++) {
            for (int j = 0; j < d2.columns(); j++) {
                double test1 = d[i][j];
                double test2 = d2.getFloat(i, j);
                assertEquals(test1, test2, 1e-6);
            }
        }
    }


    protected void verifyElements(DoubleMatrix d, INDArray d2) {
        for (int i = 0; i < d.rows; i++) {
            for (int j = 0; j < d.columns; j++) {
                double test1 = d.get(i, j);
                double test2 = d2.getFloat(i, j);
                assertEquals(test1, test2, 1e-6);
            }
        }
    }

    protected void verifyElements(ComplexDoubleMatrix d, IComplexNDArray d2) {
        for (int i = 0; i < d.rows; i++) {
            for (int j = 0; j < d.columns; j++) {
                ComplexDouble test1 = d.get(i, j);
                IComplexNumber test2 = d2.getComplex(i, j);
                assertEquals(test1.real(), test2.realComponent().doubleValue(), 1e-6);
                assertEquals(test1.imag(), test2.imaginaryComponent().doubleValue(), 1e-6);

            }
        }
    }


}
