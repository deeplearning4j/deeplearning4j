package org.deeplearning4j.nn.learning;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.nn.Tensor;
import static org.junit.Assert.*;

import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Adam Gibson
 */
public class TensorTests {

    private static Logger log = LoggerFactory.getLogger(TensorTests.class);

    @Test
    public void testSetGetSlice() {
        Tensor t = Tensor.rand(2,3,3,4.0,6.0);
        assertEquals(18,t.length);
    }

    @Test
    public void getSetTest() {
        Tensor t = new Tensor(1,1,1);
        t.set(0,0,0,2.0);
        assertEquals(true,2.0 == t.get(0,0,0));
    }



    @Test
    public void testAsMatrix() {
        Tensor t = Tensor.rand(2,3,3,4.0,6.0);
        DoubleMatrix matrix = t.toMatrix();
        assertEquals(true,matrix.rows == 6);
        assertEquals(true, matrix.columns == 3);
    }

    @Test
    public void testCreate() {
        Tensor t = Tensor.rand(2,3,3,4.0,6.0);
        DoubleMatrix matrix = t.toMatrix();
        assertEquals(t,Tensor.create(matrix,3));
    }

    @Test
    public void tensorTests() {
        RandomGenerator rng = new MersenneTwister(123);
        Tensor t = Tensor.rand(2,3,3,4.0,6.0);
        log.info("Original tensor " + t);
        assertEquals(t, t.permute(new int[]{1, 2, 3}));
        assertEquals(t.transpose(),t.permute(new int[]{2,1,3}));
        log.info("T " + t.permute(new int[]{3, 2, 1}));
        assertEquals(2, t.permute(new int[]{2, 3, 1}).slices());
        log.info("T " + t.permute(new int[]{2, 3, 1}));
        log.info("T 2 " + t.permute(new int[]{1, 3, 2}));
        log.info("T 2 " + t.permute(new int[]{3,1,2}));


    }

}
