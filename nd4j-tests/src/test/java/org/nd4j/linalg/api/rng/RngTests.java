package org.nd4j.linalg.api.rng;
import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class RngTests extends BaseNd4jTest {
    public RngTests() {
    }

    public RngTests(String name) {
        super(name);
    }

    public RngTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public RngTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testRngConstitency() {
        Nd4j.getRandom().setSeed(123);
        INDArray arr = Nd4j.rand(1,5);
        Nd4j.getRandom().setSeed(123);
        INDArray arr2 = Nd4j.rand(1,5);
        assertEquals(arr,arr2);
    }


    @Override
    public char ordering() {
        return 'f';
    }

}
