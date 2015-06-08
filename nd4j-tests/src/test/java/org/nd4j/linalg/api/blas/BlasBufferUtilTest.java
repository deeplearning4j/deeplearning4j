package org.nd4j.linalg.api.blas;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Created by agibsoncccc on 6/8/15.
 */
public class BlasBufferUtilTest extends BaseNd4jTest {

    @Test
    public void testFloat() {
        //0 offset
        INDArray test = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray row1 = test.getRow(1);
        float[] dataAssertion = {2f,4f};
        float[] testData = BlasBufferUtil.getFloatData(row1);
        assertTrue(Arrays.equals(dataAssertion, testData));
    }

    @Test
    public void testDouble() {
        INDArray test = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray row1 = test.getRow(1);
        double[] dataAssertion = {2,4};
        double[] testData = BlasBufferUtil.getDoubleData(row1);
        assertTrue(Arrays.equals(dataAssertion, testData));
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
