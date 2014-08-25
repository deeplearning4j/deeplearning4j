package org.deeplearning4j.linalg.jblas;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;


/**
 * Tests for a complex ndarray
 *
 * @author Adam Gibson
 */
public class ComplexNDArrayTests extends org.deeplearning4j.linalg.api.test.ComplexNDArrayTests {

    private static Logger log = LoggerFactory.getLogger(ComplexNDArrayTests.class);

    @Test
    public void testOrdering() {
        DoubleMatrix linspace = DoubleMatrix.linspace(1,10,10);
        ComplexDoubleMatrix c = new ComplexDoubleMatrix(linspace);

        INDArray linspace2 = NDArrays.linspace(1,10,10).transpose();
        IComplexNDArray c2 = NDArrays.createComplex(linspace2);

        assertEquals(c.rows,c2.rows());
        assertEquals(c.columns,c2.columns());

        verifyElements(linspace,linspace2);
        verifyElements(c,c2);

        IComplexNDArray transpose = c2.transpose();
        ComplexDoubleMatrix c3 = c.transpose();
        assertEquals(c3.rows,transpose.rows());
        assertEquals(c3.columns,transpose.columns());
        verifyElements(c3,transpose);

        assertEquals(c3.rows,transpose.rows());
        assertEquals(c3.columns,transpose.columns());


        ComplexDoubleMatrix mmul = c3.mmul(c);
        IComplexNDArray mmulNDArray = transpose.mmul(c2);
        verifyElements(mmul,mmulNDArray);


    }


    protected void verifyElements(double[][] d,INDArray d2) {
        for(int i = 0; i < d2.rows(); i++) {
            for(int j = 0; j < d2.columns(); j++) {
                double test1 =  d[i][j];
                double test2 = (double) d2.getScalar(i,j).element();
                assertEquals(test1,test2,1e-6);
            }
        }
    }


    protected void verifyElements(DoubleMatrix d,INDArray d2) {
        for(int i = 0; i < d.rows; i++) {
            for(int j = 0; j < d.columns; j++) {
                double test1 = d.get(i,j);
                double test2 = (double) d2.getScalar(i,j).element();
                assertEquals(test1,test2,1e-6);
            }
        }
    }

    protected void verifyElements(ComplexDoubleMatrix d,IComplexNDArray d2) {
        for(int i = 0; i < d.rows; i++) {
            for(int j = 0; j < d.columns; j++) {
                ComplexDouble test1 = d.get(i,j);
                IComplexNumber test2 =  d2.getComplex(i, j);
                assertEquals(test1.real(),test2.realComponent().doubleValue(),1e-6);
                assertEquals(test1.imag(),test2.imaginaryComponent().doubleValue(),1e-6);

            }
        }
    }


}
