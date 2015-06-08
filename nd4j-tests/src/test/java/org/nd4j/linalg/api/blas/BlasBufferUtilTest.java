package org.nd4j.linalg.api.blas;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsoncccc on 6/8/15.
 */
public class BlasBufferUtilTest extends BaseNd4jTest {

    @Test
    public void testDoubles() {
        //0 offset
        INDArray test = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray row1 = test.getRow(1);
        float[] data = {1f,3f,2f,4f};
    }

    @Test
    public void testFloats() {
        INDArray test = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray row1 = test.getRow(1);
        double[] data = {1,3,2,4};



    }

    @Override
    public char ordering() {
        return 'f';
    }
}
