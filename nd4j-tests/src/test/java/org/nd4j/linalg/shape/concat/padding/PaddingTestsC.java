package org.nd4j.linalg.shape.concat.padding;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class PaddingTestsC extends BaseNd4jTest {
    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testPrepend() {
        INDArray linspace = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray assertion = Nd4j.create(new double[][]{
                {1, 1, 1, 1, 2},
                {1, 1, 1, 3, 4}
        });

        INDArray prepend = Nd4j.prepend(linspace, 3, 1.0, -1);
        assertEquals(assertion,prepend);

        INDArray linspaced = Nd4j.linspace(1,16,16).reshape(2,2,2,2);
        INDArray prepend2 = Nd4j.prepend(linspaced, 2, 0.0, 2);
        INDArray tensorAssertion = Nd4j.create(new double[]{0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 9, 10, 11, 12, 0, 0, 0, 0, 13, 14, 15, 16}, new int[]{2, 2, 4, 2});
        assertEquals(tensorAssertion,prepend2);

    }
}
