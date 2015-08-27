package org.nd4j.linalg.api.indexing;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * @author Adam Gibson
 */
public class IndexingTestsC extends BaseNd4jTest {

    public IndexingTestsC(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public IndexingTestsC(Nd4jBackend backend) {
        super(backend);
    }

    public IndexingTestsC() {
    }

    public IndexingTestsC(String name) {
        super(name);
    }

    @Test
    public void testOffsetsC() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        assertEquals(3, NDArrayIndex.offset(arr,1,1));
        assertEquals(3,NDArrayIndex.offset(arr,NDArrayIndex.point(1),NDArrayIndex.point(1)));

        INDArray arr2 = Nd4j.linspace(1,6,6).reshape(3,2);
        assertEquals(3, NDArrayIndex.offset(arr2,1,1));
        assertEquals(3, NDArrayIndex.offset(arr2, NDArrayIndex.point(1), NDArrayIndex.point(1)));
        assertEquals(6, NDArrayIndex.offset(arr2,2,2));
        assertEquals(6, NDArrayIndex.offset(arr2, NDArrayIndex.point(2), NDArrayIndex.point(2)));



    }

    @Test
    public void testIndexFor() {
        int[] shape = {1,2};
        INDArrayIndex[] indexes = NDArrayIndex.indexesFor(shape);
        for(int i = 0; i < indexes.length; i++) {
            assertEquals(shape[i],indexes[i].offset());
        }
    }

    @Test
    public void testGetScalar() {
        INDArray arr = Nd4j.linspace(1,5,5);
        INDArray d = arr.get(NDArrayIndex.point(1));
        assertTrue(d.isScalar());
        assertEquals(2.0,d.getDouble(0));

    }


    @Test
    public void testGetIndices2d() throws Exception {
        INDArray twoByTwo = Nd4j.linspace(1, 6, 6).reshape(3, 2);
        INDArray firstRow = twoByTwo.getRow(0);
        INDArray secondRow = twoByTwo.getRow(1);
        INDArray firstAndSecondRow = twoByTwo.getRows(new int[]{1, 2});
        INDArray firstRowViaIndexing = twoByTwo.get(NDArrayIndex.interval(0, 1));
        assertEquals(firstRow, firstRowViaIndexing);
        INDArray secondRowViaIndexing = twoByTwo.get(NDArrayIndex.point(1));
        assertEquals(secondRow, secondRowViaIndexing);

        INDArray firstAndSecondRowTest = twoByTwo.get(NDArrayIndex.interval(1, 3));
        assertEquals(firstAndSecondRow, firstAndSecondRowTest);

        INDArray individualElement = twoByTwo.get(NDArrayIndex.interval(1, 2), NDArrayIndex.interval(1, 2));
        assertEquals(Nd4j.create(new float[]{4}), individualElement);


    }


    @Test
    public void testConcatColumns() {
        INDArray input1 = Nd4j.zeros(2, 1);
        INDArray input2 = Nd4j.ones(2, 1);
        INDArray concat = Nd4j.concat(1, input1, input2);
        INDArray assertion = Nd4j.create(new double[][]{{0,1}, {0,1}});
        assertEquals(assertion,concat);
    }

    @Test
    public void testGetIndicesVector() {
        INDArray line = Nd4j.linspace(1, 4, 4);
        INDArray test = Nd4j.create(new float[]{2, 3});
        INDArray result = line.get(NDArrayIndex.point(0), NDArrayIndex.interval(1, 3));
        assertEquals(test, result);
    }

    @Test
    public void testArangeMul() {
        INDArray arange = Nd4j.arange(1,17).reshape(4, 4);
        INDArrayIndex index = NDArrayIndex.interval(0, 2);
        INDArray get = arange.get(index, index);
        INDArray ones = Nd4j.ones(2,2).mul(0.25);
        INDArray mul = get.mul(ones);
        INDArray assertion = Nd4j.create(new double[][]{
                {0.25, 0.5},
                {1.25, 1.5}
        });
        assertEquals(assertion, mul);

    }


    @Override
    public char ordering() {
        return 'c';
    }
}
