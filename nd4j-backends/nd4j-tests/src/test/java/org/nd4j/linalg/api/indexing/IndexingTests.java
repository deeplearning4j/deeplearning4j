package org.nd4j.linalg.api.indexing;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class IndexingTests extends BaseNd4jTest {


    public IndexingTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testGetScalar() {
        INDArray arr = Nd4j.linspace(1, 5, 5);
        INDArray d = arr.get(NDArrayIndex.point(1));
        assertTrue(d.isScalar());
        assertEquals(2.0, d.getDouble(0), 1e-1);

    }

    @Test
    public void testNewAxis() {
        INDArray arr = Nd4j.rand(new int[] {4, 2, 3});
        INDArray view = arr.get(NDArrayIndex.newAxis(), NDArrayIndex.all(), NDArrayIndex.point(1));
        System.out.println(view);
    }

    @Test
    public void testVectorIndexing() {
        INDArray x = Nd4j.linspace(0, 10, 11);
        int[] index = new int[] {5, 8, 9};
        INDArray columnsTest = x.getColumns(index);
        assertEquals(Nd4j.create(new double[] {5, 8, 9}), columnsTest);
        int[] index2 = new int[] {2, 2, 4}; //retrieve the same columns twice
        INDArray columnsTest2 = x.getColumns(index2);
        assertEquals(Nd4j.create(new double[] {2, 2, 4}), columnsTest2);

    }

    @Test
    public void testGetRowsColumnsMatrix() {
        INDArray arr = Nd4j.linspace(1, 24, 24).reshape(4, 6);
        INDArray firstAndSecondColumnsAssertion = Nd4j.create(new double[][] {{1, 5}, {2, 6}, {3, 7}, {4, 8}});

        System.out.println(arr);
        INDArray firstAndSecondColumns = arr.getColumns(0, 1);
        assertEquals(firstAndSecondColumnsAssertion, firstAndSecondColumns);

        INDArray firstAndSecondRows = Nd4j.create(new double[][] {{1.00, 5.00, 9.00, 13.00, 17.00, 21.00},
                        {1.00, 5.00, 9.00, 13.00, 17.00, 21.00}, {2.00, 6.00, 10.00, 14.00, 18.00, 22.00}});

        INDArray rows = arr.getRows(new int[] {0, 0, 1});
        assertEquals(firstAndSecondRows, rows);
    }



    @Test
    public void testSlicing() {
        INDArray arange = Nd4j.arange(1, 17).reshape(4, 4);
        INDArray slice1Assert = Nd4j.create(new double[] {2, 6, 10, 14});
        INDArray slice1Test = arange.slice(1);
        assertEquals(slice1Assert, slice1Test);
    }

    @Test
    public void testArangeMul() {
        INDArray arange = Nd4j.arange(1, 17).reshape('f', 4, 4);
        INDArrayIndex index = NDArrayIndex.interval(0, 2);
        INDArray get = arange.get(index, index);
        INDArray zeroPointTwoFive = Nd4j.ones(2, 2).mul(0.25);
        INDArray mul = get.mul(zeroPointTwoFive);
        INDArray assertion = Nd4j.create(new double[][] {{0.25, 1.25}, {0.5, 1.5}}, 'f');
        assertEquals(assertion, mul);

    }

    @Test
    public void testGetIndicesVector() {
        INDArray line = Nd4j.linspace(1, 4, 4);
        INDArray test = Nd4j.create(new float[] {2, 3});
        INDArray result = line.get(NDArrayIndex.point(0), NDArrayIndex.interval(1, 3));
        assertEquals(test, result);
    }



    @Override
    public char ordering() {
        return 'f';
    }
}
