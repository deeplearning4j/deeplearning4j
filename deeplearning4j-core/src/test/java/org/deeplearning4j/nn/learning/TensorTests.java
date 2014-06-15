package org.deeplearning4j.nn.learning;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.nn.FourDTensor;
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
    public void testFourDTensorCreation() {
        FourDTensor fd = new FourDTensor(2,2,2,2);
        assertEquals(2,fd.numTensors());
        //1 2 x 3 x 3 matrix
        assertEquals(8,fd.getTensor(1).length);

        assertEquals(16,fd.length);
        assertEquals(8,fd.getTensor(0).length);
        assertEquals(4,fd.getSliceOfTensor(1,1).length);


        FourDTensor fd2 = new FourDTensor(2,3,2,2);
        assertEquals(2,fd2.numTensors());
        //1 2 x 3 x 3 matrix
        assertEquals(12,fd2.getTensor(1).length);

        assertEquals(24,fd2.length);
        assertEquals(12,fd2.getTensor(0).length);
        assertEquals(6,fd2.getSliceOfTensor(1,1).length);


        fd.put(1,1,1,1,0.5);
        assertEquals("Was " + fd.get(1,1,1,1),true, 0.5 == fd.get(1,1,1,1));

        DoubleMatrix slice = DoubleMatrix.rand(2,2);
        fd.put(0,0,slice);
        assertEquals(slice,fd.getSliceOfTensor(0,0));
    }

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


        Tensor t2 = new Tensor(2,2,2);
        t2.set(0,0,0,1);
        assertEquals(true,1 == t2.get(0,0,0));

    }



    @Test
    public void testAsMatrix() {
        Tensor t = Tensor.rand(2,3,3,4.0,6.0);
        DoubleMatrix matrix = t.toMatrix();
        assertEquals(true,matrix.rows == 6);
        assertEquals(true, matrix.columns == 3);
    }

    @Test
    public void testTensorRowSums() {
        Tensor t = Tensor.rand(2,3,3,4.0,6.0);
        Tensor rowSums = t.sliceRowSums();
        assertEquals(t.slices(),rowSums.slices());
        assertEquals(rowSums.rows(),1);
    }

    @Test
    public void testTensorColumnSums() {
        Tensor t = Tensor.rand(2,3,3,4.0,6.0);
        Tensor columnSums = t.sliceColumnSums();
        assertEquals(t.slices(),columnSums.slices());
        assertEquals(columnSums.columns(),t.columns());
    }


    @Test
    public void testCreate() {
        Tensor t = Tensor.rand(2,3,3,4.0,6.0);
        DoubleMatrix matrix = t.toMatrix();
        assertEquals(t.length,matrix.length);
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
