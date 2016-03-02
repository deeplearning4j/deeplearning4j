package org.nd4j.linalg.api.string;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.string.NDArrayStrings;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class TestFormatting extends BaseNd4jTest {

    public TestFormatting(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testTwoByTwo() {
        INDArray arr = Nd4j.create(2, 2,2,2);
        System.out.println(new NDArrayStrings().format(arr));

    }

    @Override
    public char ordering() {
        return 'f';
    }
}
