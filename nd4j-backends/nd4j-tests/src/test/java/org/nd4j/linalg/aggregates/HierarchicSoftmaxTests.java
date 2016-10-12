package org.nd4j.linalg.aggregates;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class HierarchicSoftmaxTests extends BaseNd4jTest {


    public HierarchicSoftmaxTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testHSGradient1() throws Exception {
        INDArray syn0 = Nd4j.create(10, 10);
        INDArray syn1 = Nd4j.create(10, 10);

    }


    @Override
    public char ordering() {
        return 'c';
    }
}
