package org.deeplearning4j.linalg.jblas.util;

import static org.junit.Assert.*;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.jblas.DoubleMatrix;

import org.junit.Test;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.DataInputStream;
import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class JblasSerdeTests {
    private static Logger log = LoggerFactory.getLogger(JblasSerdeTests.class);

    @Test
    public void testBinary() throws Exception {
        DoubleMatrix d = new DoubleMatrix();
        ClassPathResource c = new ClassPathResource("/test-matrix.ser");
        d.in(new DataInputStream(c.getInputStream()));
        INDArray assertion = JblasSerde.readJblasBinary(new DataInputStream(c.getInputStream()));
        assertTrue(Arrays.equals(new int[]{d.rows,d.columns},assertion.shape()));
        for(int i = 0; i < d.rows; i++) {
            for(int j = 0; j < d.columns; j++) {
                assertEquals(d.get(i,j),(double) assertion.getScalar(i,j).element(),1e-1);
            }
        }
    }

}
