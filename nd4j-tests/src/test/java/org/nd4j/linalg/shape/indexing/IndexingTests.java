package org.nd4j.linalg.shape.indexing;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * @author Adam Gibson
 */
public class IndexingTests extends BaseNd4jTest  {

    public IndexingTests() {
    }

    public IndexingTests(String name) {
        super(name);
    }

    public IndexingTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public IndexingTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testTensorGet() {
        INDArray threeTwoTwo = Nd4j.linspace(1, 12, 12).reshape(3,2,2);
        /*
        * [[[  1.,   7.],
        [  4.,  10.]],

       [[  2.,   8.],
        [  5.,  11.]],

       [[  3.,   9.],
        [  6.,  12.]]])
       */

        INDArray firstAssertion = Nd4j.create(new double[]{1,7});
        INDArray firstTest = threeTwoTwo.get(new NDArrayIndex(0), new NDArrayIndex(0), NDArrayIndex.all());
        assertEquals(firstAssertion,firstTest);
        INDArray secondAssertion = Nd4j.create(new double[]{3,9});
        INDArray secondTest = threeTwoTwo.get(new NDArrayIndex(2), new NDArrayIndex(0), NDArrayIndex.all());
        assertEquals(secondAssertion, secondTest);



    }

    @Test
    public void testGetRows() {
        INDArray arr = Nd4j.linspace(1,9,9).reshape(3,3);
        INDArray testAssertion = Nd4j.create(new double[][]{
                {5, 8},
                {6, 9}
        });

        INDArray test = arr.get(new NDArrayIndex(1, 3), new NDArrayIndex(1, 3));
        assertEquals(testAssertion, test);

    }

    @Test
    public void testFirstColumn() {
        INDArray arr = Nd4j.create(new double[][]{
                {5, 6},
                {7, 8}
        });

        INDArray assertion = Nd4j.create(new double[]{5,7});
        INDArray test = arr.get(NDArrayIndex.all(), new NDArrayIndex(0));
        assertEquals(assertion,test);
    }



    @Test
    public void testLinearIndex() {
        INDArray linspace = Nd4j.linspace(1,4,4).reshape(2,2);
        for(int i = 0; i < linspace.length(); i++) {
            assertEquals(i + 1,linspace.getDouble(i),1e-1);
        }
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
