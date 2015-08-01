package org.nd4j.linalg.shape.indexing;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * @author Adam Gibson
 */
public class IndexingTestsC extends BaseNd4jTest {

    @Test
    public void testExecSubArray() {
        INDArray nd = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});

        INDArray sub = nd.get(NDArrayIndex.all(), new NDArrayIndex(0,1));
        Nd4j.getExecutioner().exec(new ScalarAdd(sub, 2));
        assertEquals(getFailureMessage(), Nd4j.create(new double[][]{
                {3, 4}, {6, 7}
        }), sub);

    }



    @Override
    public char ordering() {
        return 'c';
    }
}
