package org.nd4j.linalg.api.linearview;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class LinearViewTests extends BaseNd4jTest {
    public LinearViewTests() {
        super();
    }

    public LinearViewTests(String name) {
        super(name);
    }

    public LinearViewTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public LinearViewTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testLinearViewAlignment() {
        INDArray twoToFour = Nd4j.create(new double[][]{
                {1, 2},
                {3, 4}
        });
        INDArray linear = twoToFour.linearView();
        assertEquals(Nd4j.create(new double[]{1, 2, 3, 4}), linear);
    }

    @Test
    public void testLinearViewGetAndPut() throws Exception {
        INDArray test = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray linear = test.linearView();
        linear.putScalar(2, 6);
        linear.putScalar(3, 7);
        assertEquals(6, linear.getFloat(2), 1e-1);
        assertEquals(7, linear.getFloat(3), 1e-1);
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
