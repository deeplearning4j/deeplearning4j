package org.nd4j.linalg.jblas.eigen;

import static org.junit.Assert.*;

import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 9/30/14.
 */
public class EigenTests extends org.nd4j.linalg.eigen.EigenTests {
    @Test
    public void testDoubleMatrixEigen() {
        ComplexDoubleMatrix[] eigen = org.jblas.Eigen.eigenvectors(DoubleMatrix.linspace(1,4,4).reshape(2, 2));

        IComplexNDArray[] otherEigen1 = Eigen.eigenvectors(Nd4j.ones(4).reshape(2,2));
        IComplexNDArray real = Nd4j.createComplex(new float[]{-0.37228132f, 0,5.37228132f,0});
        assertEquals(otherEigen1[0],real);
        IComplexNDArray real2 = Nd4j.createComplex(new float[]{0.70710678f,0, 0.70710678f,0,-0.70710678f,0,0.70710678f,0},new int[]{2,2});
        assertEquals(otherEigen1[1],real2);
    }


}
