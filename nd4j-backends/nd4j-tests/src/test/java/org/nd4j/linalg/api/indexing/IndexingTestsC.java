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
import org.nd4j.linalg.indexing.SpecifiedIndex;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class IndexingTestsC extends BaseNd4jTest {


    public IndexingTestsC(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testGetPointRowVector() {
        INDArray arr = Nd4j.linspace(1, 1000, 1000);

        INDArray arr2 = arr.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, 100));

        assertEquals(100, arr2.length()); //Returning: length 0
        assertEquals(arr2, Nd4j.linspace(1, 100, 100));
    }

    @Test
    public void testSpecifiedIndexVector() {
        INDArray rootMatrix = Nd4j.linspace(1, 16, 16).reshape(4, 4);
        INDArray threeD = Nd4j.linspace(1, 16, 16).reshape(2, 2, 2, 2);
        INDArray get = rootMatrix.get(NDArrayIndex.all(), new SpecifiedIndex(0, 2));
        INDArray assertion = Nd4j.create(new double[][] {{1, 3}, {5, 7}, {9, 11}, {13, 15}});

        assertEquals(assertion, get);

        INDArray assertion2 = Nd4j.create(new double[][] {{1, 3, 4}, {5, 7, 8}, {9, 11, 12}, {13, 15, 16}});
        INDArray get2 = rootMatrix.get(NDArrayIndex.all(), new SpecifiedIndex(0, 2, 3));

        assertEquals(assertion2, get2);

    }


    @Test
    public void testPutRowIndexing() {
        INDArray arr = Nd4j.ones(1, 10);
        INDArray row = Nd4j.create(1, 10);

        arr.putRow(0, row); //OK
        arr.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.all()}, row); //Exception
        assertEquals(arr, row);
    }

    @Test
    public void testVectorIndexing2() {
        INDArray wholeVector = Nd4j.linspace(1, 5, 5).get(NDArrayIndex.interval(1, 2, 3, true));
        INDArray assertion = Nd4j.create(new double[] {2, 4});
        assertEquals(assertion, wholeVector);
        INDArray wholeVectorTwo = Nd4j.linspace(1, 5, 5).get(NDArrayIndex.interval(1, 2, 4, true));
        assertEquals(assertion, wholeVectorTwo);
        INDArray wholeVectorThree = Nd4j.linspace(1, 5, 5).get(NDArrayIndex.interval(1, 2, 4, false));
        assertEquals(assertion, wholeVectorThree);
        INDArray threeFiveAssertion = Nd4j.create(new double[] {3, 5});
        INDArray threeFive = Nd4j.linspace(1, 5, 5).get(NDArrayIndex.interval(2, 2, 4, true));
        assertEquals(threeFiveAssertion, threeFive);
    }


    @Test
    public void testOffsetsC() {
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        assertEquals(3, NDArrayIndex.offset(arr, 1, 1));
        assertEquals(3, NDArrayIndex.offset(arr, NDArrayIndex.point(1), NDArrayIndex.point(1)));

        INDArray arr2 = Nd4j.linspace(1, 6, 6).reshape(3, 2);
        assertEquals(3, NDArrayIndex.offset(arr2, 1, 1));
        assertEquals(3, NDArrayIndex.offset(arr2, NDArrayIndex.point(1), NDArrayIndex.point(1)));
        assertEquals(6, NDArrayIndex.offset(arr2, 2, 2));
        assertEquals(6, NDArrayIndex.offset(arr2, NDArrayIndex.point(2), NDArrayIndex.point(2)));



    }

    @Test
    public void testIndexFor() {
        int[] shape = {1, 2};
        INDArrayIndex[] indexes = NDArrayIndex.indexesFor(shape);
        for (int i = 0; i < indexes.length; i++) {
            assertEquals(shape[i], indexes[i].offset());
        }
    }

    @Test
    public void testGetScalar() {
        INDArray arr = Nd4j.linspace(1, 5, 5);
        INDArray d = arr.get(NDArrayIndex.point(1));
        assertTrue(d.isScalar());
        assertEquals(2.0, d.getDouble(0), 1e-1);

    }

    @Test
    public void testVectorIndexing() {
        INDArray arr = Nd4j.linspace(1, 10, 10);
        INDArray assertion = Nd4j.create(new double[] {2, 3, 4, 5});
        INDArray viewTest = arr.get(NDArrayIndex.point(0), NDArrayIndex.interval(1, 5));
        assertEquals(assertion, viewTest);
    }

    @Test
    public void testNegativeIndices() {
        INDArray test = Nd4j.create(10, 10, 10);
        test.putScalar(new int[] {0, 0, -1}, 1.0);
        assertEquals(1.0, test.getScalar(0, 0, -1).sumNumber());
    }

    @Test
    public void testGetIndices2d() throws Exception {
        INDArray twoByTwo = Nd4j.linspace(1, 6, 6).reshape(3, 2);
        INDArray firstRow = twoByTwo.getRow(0);
        INDArray secondRow = twoByTwo.getRow(1);
        INDArray firstAndSecondRow = twoByTwo.getRows(new int[] {1, 2});
        INDArray firstRowViaIndexing = twoByTwo.get(NDArrayIndex.interval(0, 1));
        assertEquals(firstRow, firstRowViaIndexing);
        INDArray secondRowViaIndexing = twoByTwo.get(NDArrayIndex.point(1));
        assertEquals(secondRow, secondRowViaIndexing);

        INDArray firstAndSecondRowTest = twoByTwo.get(NDArrayIndex.interval(1, 3));
        assertEquals(firstAndSecondRow, firstAndSecondRowTest);

        INDArray individualElement = twoByTwo.get(NDArrayIndex.interval(1, 2), NDArrayIndex.interval(1, 2));
        assertEquals(Nd4j.create(new float[] {4}), individualElement);
    }

    @Test
    public void testGetRow() {
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.linspace(0, 14, 15).reshape(3, 5);
        int[] toGet = {0, 1};
        INDArray out = in.getRows(toGet);
        assertEquals(in.getRow(0), out.getRow(0));
        assertEquals(in.getRow(1), out.getRow(1));

        int[] toGet2 = {0, 1, 2, 0, 1, 2};
        INDArray out2 = in.getRows(toGet2);
        for (int i = 0; i < toGet2.length; i++) {
            assertEquals(in.getRow(toGet2[i]), out2.getRow(i));
        }
    }


    @Test
    public void testGetRowEdgeCase() {
        INDArray rowVec = Nd4j.linspace(1, 5, 5);
        INDArray get = rowVec.getRow(0); //Returning shape [1,1]

        assertArrayEquals(new int[] {1, 5}, get.shape());
        assertEquals(rowVec, get);
    }

    @Test
    public void testGetColumnEdgeCase() {
        INDArray colVec = Nd4j.linspace(1, 5, 5).transpose();
        INDArray get = colVec.getColumn(0); //Returning shape [1,1]

        assertArrayEquals(new int[] {5, 1}, get.shape());
        assertEquals(colVec, get);
    }

    @Test
    public void testConcatColumns() {
        INDArray input1 = Nd4j.zeros(2, 1);
        INDArray input2 = Nd4j.ones(2, 1);
        INDArray concat = Nd4j.concat(1, input1, input2);
        INDArray assertion = Nd4j.create(new double[][] {{0, 1}, {0, 1}});
        assertEquals(assertion, concat);
    }

    @Test
    public void testGetIndicesVector() {
        INDArray line = Nd4j.linspace(1, 4, 4);
        INDArray test = Nd4j.create(new float[] {2, 3});
        INDArray result = line.get(NDArrayIndex.point(0), NDArrayIndex.interval(1, 3));
        assertEquals(test, result);
    }

    @Test
    public void testArangeMul() {
        INDArray arange = Nd4j.arange(1, 17).reshape(4, 4);
        INDArrayIndex index = NDArrayIndex.interval(0, 2);
        INDArray get = arange.get(index, index);
        INDArray ones = Nd4j.ones(2, 2).mul(0.25);
        INDArray mul = get.mul(ones);
        INDArray assertion = Nd4j.create(new double[][] {{0.25, 0.5}, {1.25, 1.5}});
        assertEquals(assertion, mul);

    }


    @Override
    public char ordering() {
        return 'c';
    }
}
