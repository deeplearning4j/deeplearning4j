package org.nd4j.linalg.api.rng;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class RngTests extends BaseNd4jTest {
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
