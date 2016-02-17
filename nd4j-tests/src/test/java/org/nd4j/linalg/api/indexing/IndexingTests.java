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
public class IndexingTests extends BaseNd4jTest {

    public IndexingTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public IndexingTests(Nd4jBackend backend) {
        super(backend);
    }

    public IndexingTests() {
    }

    public IndexingTests(String name) {
        super(name);
    }

    @Test
    public void testGetScalar() {
        INDArray arr = Nd4j.linspace(1,5,5);
        INDArray d = arr.get(NDArrayIndex.point(1));
        assertTrue(d.isScalar());
        assertEquals(2.0,d.getDouble(0));

    }




    @Test
    public void testSlicing() {
        INDArray arange = Nd4j.arange(1,17).reshape(4, 4);
        INDArray slice1Assert = Nd4j.create(new double[]{2,6,10,14});
        INDArray slice1Test = arange.slice(1);
        assertEquals(slice1Assert, slice1Test);
    }

    @Test
    public void testArangeMul() {
        INDArray arange = Nd4j.arange(1,17).reshape('f',4, 4);
        INDArrayIndex index = NDArrayIndex.interval(0, 2);
        INDArray get = arange.get(index, index);
        INDArray zeroPointTwoFive = Nd4j.ones(2,2).mul(0.25);
        INDArray mul = get.mul(zeroPointTwoFive);
        INDArray assertion = Nd4j.create(new double[][]{
                {0.25, 1.25},
                {0.5, 1.5}
        },'f');
        assertEquals(assertion, mul);

    }

    @Test
    public void testGetIndicesVector() {
        INDArray line = Nd4j.linspace(1, 4, 4);
        INDArray test = Nd4j.create(new float[]{2, 3});
        INDArray result = line.get(NDArrayIndex.point(0), NDArrayIndex.interval(1, 3));
        assertEquals(test, result);
    }



    @Override
    public char ordering() {
        return 'f';
    }
}
