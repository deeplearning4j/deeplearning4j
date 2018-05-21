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
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class IndexingTestsC extends BaseNd4jTest {


    public IndexingTestsC(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testNegativeBounds() {
       INDArray arr = Nd4j.linspace(1,10,10).reshape(2,5);
       INDArrayIndex interval = NDArrayIndex.interval(0,1,-2,arr.size(1));
       INDArray get = arr.get(NDArrayIndex.all(),interval);
       INDArray assertion = Nd4j.create(new double[][]{
               {1,2,3},
               {6,7,8}
       });
       assertEquals(assertion,get);
    }

    @Test
    public void testNewAxis() {
        INDArray arr = Nd4j.linspace(1, 12, 12).reshape(3, 2, 2);
        INDArray get = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), newAxis(), newAxis());
        long[] shapeAssertion = {3, 2, 1, 1, 2};
        assertArrayEquals(shapeAssertion, get.shape());
    }


    @Test
    public void broadcastBug() throws Exception {
        INDArray a = Nd4j.create(new double[] {1.0, 2.0, 3.0, 4.0}, new int[] {2, 2});
        final INDArray col = a.get(NDArrayIndex.all(), NDArrayIndex.point(0));

        final INDArray aBad = col.broadcast(2, 2);
        final INDArray aGood = col.dup().broadcast(2, 2);
        System.out.println(aBad);
        System.out.println(aGood);
        assertTrue(Transforms.abs(aGood.sub(aBad).div(aGood)).maxNumber().doubleValue() < 0.01);
    }


    @Test
    public void testIntervalsIn3D() {
        INDArray arr = Nd4j.arange(8).reshape(2, 2, 2);
        INDArray assertion = Nd4j.create(new double[][] {{4, 5}, {6, 7}}).reshape(1, 2, 2);
        INDArray rest = arr.get(interval(1, 2), interval(0, 2), interval(0, 2));
        assertEquals(assertion, rest);

    }

    @Test
    public void testSmallInterval() {
        INDArray arr = Nd4j.arange(8).reshape(2, 2, 2);
        INDArray assertion = Nd4j.create(new double[][] {{4, 5}, {6, 7}}).reshape(1, 2, 2);
        INDArray rest = arr.get(interval(1, 2), all(), all());
        assertEquals(assertion, rest);

    }

    @Test
    public void testAllWithNewAxisAndInterval() {
        INDArray arr = Nd4j.linspace(1, 24, 24).reshape(4, 2, 3);
        INDArray assertion2 = Nd4j.create(new double[][] {{7, 8, 9},}).reshape(1, 1, 3);

        INDArray get2 = arr.get(NDArrayIndex.point(1), newAxis(), NDArrayIndex.interval(0, 1));
        assertEquals(assertion2, get2);
    }

    @Test
    public void testAllWithNewAxisInMiddle() {
        INDArray arr = Nd4j.linspace(1, 24, 24).reshape(4, 2, 3);
        INDArray assertion2 = Nd4j.create(new double[][] {{7, 8, 9}, {10, 11, 12}}).reshape(1, 2, 3);

        INDArray get2 = arr.get(NDArrayIndex.point(1), newAxis(), all());
        assertEquals(assertion2, get2);
    }

    @Test
    public void testAllWithNewAxis() {
        INDArray arr = Nd4j.linspace(1, 24, 24).reshape(4, 2, 3);
        INDArray get = arr.get(newAxis(), all(), point(1));
        INDArray assertion = Nd4j.create(new double[][] {{4, 5, 6}, {10, 11, 12}, {16, 17, 18}, {22, 23, 24}})
                        .reshape(1, 4, 3);
        assertEquals(assertion, get);

    }

    @Test
    public void testIndexingWithMmul() {
        INDArray a = Nd4j.linspace(1, 9, 9).reshape(3, 3);
        INDArray b = Nd4j.linspace(1, 5, 5);
        System.out.println(b);
        INDArray view = a.get(all(), NDArrayIndex.interval(0, 1));
        INDArray c = view.mmul(b);
        INDArray assertion = a.get(all(), NDArrayIndex.interval(0, 1)).dup().mmul(b);
        assertEquals(assertion, c);
    }

    @Test
    public void testPointPointInterval() {
        INDArray wholeArr = Nd4j.linspace(1, 36, 36).reshape(4, 3, 3);
        INDArray get = wholeArr.get(point(0), interval(1, 3), interval(1, 3));
        INDArray assertion = Nd4j.create(new double[][] {{5, 6}, {8, 9}});

        assertEquals(assertion, get);
    }

    @Test
    public void testIntervalLowerBound() {
        INDArray wholeArr = Nd4j.linspace(1, 24, 24).reshape(4, 2, 3);
        INDArray subarray = wholeArr.get(interval(1, 3), new SpecifiedIndex(new int[] {0}),
                        new SpecifiedIndex(new int[] {0, 2}));
        INDArray assertion = Nd4j.create(new double[][] {{7, 9}, {13, 15}});

        assertEquals(assertion, subarray);

    }


    @Test
    public void testGetPointRowVector() {
        INDArray arr = Nd4j.linspace(1, 1000, 1000);

        INDArray arr2 = arr.get(point(0), interval(0, 100));

        assertEquals(100, arr2.length()); //Returning: length 0
        assertEquals(arr2, Nd4j.linspace(1, 100, 100));
    }

    @Test
    public void testSpecifiedIndexVector() {
        INDArray rootMatrix = Nd4j.linspace(1, 16, 16).reshape(4, 4);
        INDArray threeD = Nd4j.linspace(1, 16, 16).reshape(2, 2, 2, 2);
        INDArray get = rootMatrix.get(all(), new SpecifiedIndex(0, 2));
        INDArray assertion = Nd4j.create(new double[][] {{1, 3}, {5, 7}, {9, 11}, {13, 15}});

        assertEquals(assertion, get);

        INDArray assertion2 = Nd4j.create(new double[][] {{1, 3, 4}, {5, 7, 8}, {9, 11, 12}, {13, 15, 16}});
        INDArray get2 = rootMatrix.get(all(), new SpecifiedIndex(0, 2, 3));

        assertEquals(assertion2, get2);

    }


    @Test
    public void testPutRowIndexing() {
        INDArray arr = Nd4j.ones(1, 10);
        INDArray row = Nd4j.create(1, 10);

        arr.putRow(0, row); //OK
        arr.put(new INDArrayIndex[] {point(0), all()}, row); //Exception
        assertEquals(arr, row);
    }

    @Test
    public void testVectorIndexing2() {
        INDArray wholeVector = Nd4j.linspace(1, 5, 5).get(interval(1, 2, 3, true));
        INDArray assertion = Nd4j.create(new double[] {2, 4});
        assertEquals(assertion, wholeVector);
        INDArray wholeVectorTwo = Nd4j.linspace(1, 5, 5).get(interval(1, 2, 4, true));
        assertEquals(assertion, wholeVectorTwo);
        INDArray wholeVectorThree = Nd4j.linspace(1, 5, 5).get(interval(1, 2, 4, false));
        assertEquals(assertion, wholeVectorThree);
        INDArray threeFiveAssertion = Nd4j.create(new double[] {3, 5});
        INDArray threeFive = Nd4j.linspace(1, 5, 5).get(interval(2, 2, 4, true));
        assertEquals(threeFiveAssertion, threeFive);
    }


    @Test
    public void testOffsetsC() {
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        assertEquals(3, NDArrayIndex.offset(arr, 1, 1));
        assertEquals(3, NDArrayIndex.offset(arr, point(1), point(1)));

        INDArray arr2 = Nd4j.linspace(1, 6, 6).reshape(3, 2);
        assertEquals(3, NDArrayIndex.offset(arr2, 1, 1));
        assertEquals(3, NDArrayIndex.offset(arr2, point(1), point(1)));
        assertEquals(6, NDArrayIndex.offset(arr2, 2, 2));
        assertEquals(6, NDArrayIndex.offset(arr2, point(2), point(2)));



    }

    @Test
    public void testIndexFor() {
        long[] shape = {1, 2};
        INDArrayIndex[] indexes = NDArrayIndex.indexesFor(shape);
        for (int i = 0; i < indexes.length; i++) {
            assertEquals(shape[i], indexes[i].offset());
        }
    }

    @Test
    public void testGetScalar() {
        INDArray arr = Nd4j.linspace(1, 5, 5);
        INDArray d = arr.get(point(1));
        assertTrue(d.isScalar());
        assertEquals(2.0, d.getDouble(0), 1e-1);

    }

    @Test
    public void testVectorIndexing() {
        INDArray arr = Nd4j.linspace(1, 10, 10);
        INDArray assertion = Nd4j.create(new double[] {2, 3, 4, 5});
        INDArray viewTest = arr.get(point(0), interval(1, 5));
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
        INDArray firstRowViaIndexing = twoByTwo.get(interval(0, 1));
        assertEquals(firstRow, firstRowViaIndexing);
        INDArray secondRowViaIndexing = twoByTwo.get(point(1));
        assertEquals(secondRow, secondRowViaIndexing);

        INDArray firstAndSecondRowTest = twoByTwo.get(interval(1, 3));
        assertEquals(firstAndSecondRow, firstAndSecondRowTest);

        INDArray individualElement = twoByTwo.get(interval(1, 2), interval(1, 2));
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

        assertArrayEquals(new long[] {1, 5}, get.shape());
        assertEquals(rowVec, get);
    }

    @Test
    public void testGetColumnEdgeCase() {
        INDArray colVec = Nd4j.linspace(1, 5, 5).transpose();
        INDArray get = colVec.getColumn(0); //Returning shape [1,1]

        assertArrayEquals(new long[] {5, 1}, get.shape());
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
        INDArray result = line.get(point(0), interval(1, 3));
        assertEquals(test, result);
    }

    @Test
    public void testArangeMul() {
        INDArray arange = Nd4j.arange(1, 17).reshape(4, 4);
        INDArrayIndex index = interval(0, 2);
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
