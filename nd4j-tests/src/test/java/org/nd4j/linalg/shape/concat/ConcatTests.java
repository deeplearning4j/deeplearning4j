package org.nd4j.linalg.shape.concat;
import static org.junit.Assert.*;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class ConcatTests extends BaseNd4jTest {

    public ConcatTests(Nd4jBackend backend) {
        super(backend);
    }





    @Test
    public void testConcat() {
        INDArray A = Nd4j.linspace(1, 8, 8).reshape(2, 2, 2);
        INDArray B = Nd4j.linspace(1, 12, 12).reshape(3, 2, 2);
        INDArray concat = Nd4j.concat(0, A, B);
        assertTrue(Arrays.equals(new int[]{5, 2, 2}, concat.shape()));

    }

    @Test
    public void testConcatHorizontally() {
        INDArray rowVector = Nd4j.ones(5);
        INDArray other = Nd4j.ones(5);
        INDArray concat = Nd4j.hstack(other, rowVector);
        assertEquals(rowVector.rows(), concat.rows());
        assertEquals(rowVector.columns() * 2, concat.columns());

    }


    @Test
    public void testVStackColumn() {
        INDArray linspaced = Nd4j.linspace(1,3,3).reshape(3, 1);
        INDArray stacked = linspaced.dup();
        INDArray assertion = Nd4j.create(new double[]{1, 2, 3, 1, 2, 3}, new int[]{6, 1});
        INDArray test =  Nd4j.vstack(linspaced, stacked);
        assertEquals(assertion, test);
    }



    @Test
    public void testConcatScalars() {
        INDArray first = Nd4j.arange(0,1).reshape(1,1);
        INDArray second = Nd4j.arange(0,1).reshape(1, 1);
        INDArray firstRet = Nd4j.concat(0, first, second);
        assertTrue(firstRet.isColumnVector());
        INDArray secondRet = Nd4j.concat(1, first, second);
        assertTrue(secondRet.isRowVector());


    }


    @Test
    public void testConcatMatrices() {
        INDArray a = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray b = a.dup();


        INDArray concat1 = Nd4j.concat(1, a, b);
        INDArray oneAssertion = Nd4j.create(new double[][]{{1, 3, 1, 3}, {2, 4, 2, 4}});
        assertEquals(oneAssertion,concat1);

        INDArray concat = Nd4j.concat(0, a, b);
        INDArray zeroAssertion = Nd4j.create(new double[][]{{1, 3}, {2, 4}, {1, 3}, {2, 4}});
        assertEquals(zeroAssertion, concat);
    }


    @Test
    public void testConcatColVectorAndMatrix() {
      
        INDArray colVector = Nd4j.create(new double[]{1, 2, 3, 1, 2, 3}, new int[]{6, 1});
        INDArray matrix = Nd4j.create(new double[]{4, 5, 6, 4, 5, 6}, new int[]{2, 3});

        INDArray assertion = Nd4j.create(new double[]{1, 2, 3, 1, 2, 3, 4, 5}, new int[]{8, 1});

        INDArray concat = Nd4j.vstack(colVector, matrix);
        assertEquals(assertion,concat);
        
    }

    @Test
    public void testConcatRowVectorAndMatrix() {

        INDArray rowVector = Nd4j.create(new double[]{1, 2, 3, 1, 2, 3}, new int[]{1, 6});
        INDArray matrix = Nd4j.create(new double[]{4, 5, 6, 4, 5, 6}, new int[]{3, 2});

        INDArray assertion = Nd4j.create(new double[]{1, 2, 3, 1, 2, 3, 4, 5}, new int[]{1, 8});

        INDArray concat = Nd4j.hstack(rowVector, matrix);
        assertEquals(assertion, concat);
      
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
