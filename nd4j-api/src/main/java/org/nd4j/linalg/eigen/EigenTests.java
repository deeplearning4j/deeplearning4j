package org.nd4j.linalg.eigen;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 9/30/14.
 */
public abstract class EigenTests {

    private static Logger log = LoggerFactory.getLogger(EigenTests.class);


    @Test
    public void testEigen() {
        INDArray linspace = Nd4j.linspace(1,4,4).reshape(2,2);
        IComplexNDArray solution = Nd4j.createComplex(new float[]{-0.37228132f,0,0,0,0,0,5.37228132f,0},new int[]{2,2});
        IComplexNDArray[] eigen = Eigen.eigenvectors(linspace);
        assertEquals(eigen[0],solution);

    }

}
