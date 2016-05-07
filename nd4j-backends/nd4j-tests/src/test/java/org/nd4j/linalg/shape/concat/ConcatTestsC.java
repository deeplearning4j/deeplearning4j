package org.nd4j.linalg.shape.concat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

import static org.junit.Assert.*;


/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class ConcatTestsC extends BaseNd4jTest {

    public ConcatTestsC(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testConcatVertically() {
        INDArray rowVector = Nd4j.ones(5);
        INDArray other = Nd4j.ones(5);
        INDArray concat = Nd4j.vstack(other, rowVector);
        assertEquals(rowVector.rows() * 2, concat.rows());
        assertEquals(rowVector.columns(), concat.columns());

        INDArray arr2 = Nd4j.create(5,5);
        INDArray slice1 = arr2.slice(0);
        INDArray slice2 = arr2.slice(1);
        INDArray arr3 = Nd4j.create(2, 5);
        INDArray vstack = Nd4j.vstack(slice1, slice2);
        assertEquals(arr3,vstack);

        INDArray col1 = arr2.getColumn(0);
        INDArray col2 = arr2.getColumn(1);
        INDArray vstacked = Nd4j.vstack(col1,col2);
        assertEquals(Nd4j.create(10,1),vstacked);



    }


    @Test
    public void testConcatScalars() {
        INDArray first = Nd4j.arange(0,1).reshape(1, 1);
        INDArray second = Nd4j.arange(0,1).reshape(1, 1);
        INDArray firstRet = Nd4j.concat(0, first, second);
        assertTrue(firstRet.isColumnVector());
        INDArray secondRet = Nd4j.concat(1, first, second);
        assertTrue(secondRet.isRowVector());
    }

    @Test
    public void testConcatScalars1() {
        INDArray first = Nd4j.scalar(1);
        INDArray second = Nd4j.scalar(2);
        INDArray third = Nd4j.scalar(3);

        INDArray result = Nd4j.concat(0, first, second, third);

        assertEquals(1f, result.getFloat(0), 0.01f);
        assertEquals(2f, result.getFloat(1), 0.01f);
        assertEquals(3f, result.getFloat(2), 0.01f);
    }

    @Test
    public void testConcatVectors1() {
        INDArray first = Nd4j.ones(10);
        INDArray second = Nd4j.ones(10);
        INDArray third = Nd4j.ones(10);

        INDArray result = Nd4j.concat(0, first, second, third);

        assertEquals(3, result.rows());
        assertEquals(10, result.columns());

        System.out.println(result);

        for (int x = 0; x < 30; x++) {
            assertEquals(1f, result.getFloat(x), 0.001f);
        }
    }

    @Test
    public void testConcatMatrices() {
        INDArray a = Nd4j.linspace(1,4,4).reshape(2, 2);
        INDArray b = a.dup();


        INDArray concat1 = Nd4j.concat(1, a, b);
        INDArray oneAssertion = Nd4j.create(new double[][]{{1, 2, 1, 2}, {3, 4, 3, 4}});

        System.out.println("Assertion: " + Arrays.toString(oneAssertion.data().asFloat()));
        System.out.println("Result: " + Arrays.toString(concat1.data().asFloat()));

        assertEquals(oneAssertion,concat1);

        INDArray concat = Nd4j.concat(0, a, b);
        INDArray zeroAssertion = Nd4j.create(new double[][]{{1, 2}, {3, 4}, {1, 2}, {3, 4}});
        assertEquals(zeroAssertion, concat);
    }

    @Test
    public void testAssign() {
        INDArray vector = Nd4j.linspace(1, 5, 5);
        vector.assign(1);
        assertEquals(Nd4j.ones(5),vector);
        INDArray twos = Nd4j.ones(2, 2);
        INDArray rand = Nd4j.rand(2, 2);
        twos.assign(rand);
        assertEquals(rand,twos);

        INDArray tensor = Nd4j.rand((long) 3, 3, 3, 3);
        INDArray ones = Nd4j.ones(3, 3, 3);
        assertTrue(Arrays.equals(tensor.shape(), ones.shape()));
        ones.assign(tensor);
        assertEquals(tensor,ones);
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
    public void testConcatRowVectors() {
        INDArray rowVector = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6}, new int[]{1, 6});
        INDArray matrix = Nd4j.create(new double[]{7, 8, 9, 10, 11, 12}, new int[]{1, 6});

        INDArray assertion1 = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new int[]{1, 12});
        INDArray assertion0 = Nd4j.create(new double[][]{{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}});

        INDArray concat1 = Nd4j.hstack(rowVector, matrix);
        INDArray concat0 = Nd4j.vstack(rowVector, matrix);
        assertEquals(assertion1, concat1);
        assertEquals(assertion0, concat0);
    }


    @Test
    public void testConcat3d() {
        INDArray first = Nd4j.linspace(1, 24, 24).reshape('c', 2, 3, 4);
        INDArray second = Nd4j.linspace(24, 36, 12).reshape('c', 1, 3, 4);
        INDArray third = Nd4j.linspace(36, 48, 12).reshape('c', 1, 3, 4);

        //Concat, dim 0
        INDArray exp = Nd4j.create(2 + 1 + 1, 3, 4);
        exp.put(new INDArrayIndex[]{NDArrayIndex.interval(0, 2), NDArrayIndex.all(), NDArrayIndex.all()}, first);
        exp.put(new INDArrayIndex[]{NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all()}, second);
        exp.put(new INDArrayIndex[]{NDArrayIndex.point(3), NDArrayIndex.all(), NDArrayIndex.all()}, third);

        INDArray concat0 = Nd4j.concat(0, first, second, third);

        assertEquals(exp, concat0);

        //Concat, dim 1
        second = Nd4j.linspace(24, 32, 8).reshape('c', 2, 1, 4);
        third = Nd4j.linspace(32, 48, 16).reshape('c', 2, 2, 4);
        exp = Nd4j.create(2, 3 + 1 + 2, 4);
        exp.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, 3), NDArrayIndex.all()}, first);
        exp.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(3), NDArrayIndex.all()}, second);
        exp.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(4, 6), NDArrayIndex.all()}, third);

        INDArray concat1 = Nd4j.concat(1, first, second, third);

        assertEquals(exp, concat1);

        //Concat, dim 2
        second = Nd4j.linspace(24, 36, 12).reshape('c', 2, 3, 2);
        third = Nd4j.linspace(36, 42, 6).reshape('c', 2, 3, 1);
        exp = Nd4j.create(2, 3, 4 + 2 + 1);

        exp.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)}, first);
        exp.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(4, 6)}, second);
        exp.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(6)}, third);

        INDArray concat2 = Nd4j.concat(2, first, second, third);

        assertEquals(exp, concat2);
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
