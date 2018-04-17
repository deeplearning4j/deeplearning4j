package org.nd4j.linalg.util;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.*;

/**
 * @author Hamdi Douss
 */
@RunWith(Parameterized.class)
public class NDArrayUtilTest extends BaseNd4jTest {

    public NDArrayUtilTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testMatrixConversion() {
        int[][] nums = {{1, 2}, {3, 4}, {5, 6}};
        INDArray result = NDArrayUtil.toNDArray(nums);
        assertArrayEquals(new int[]{2,3}, result.shape());
    }

    @Test
    public void testVectorConversion() {
        int[] nums = {1, 2, 3, 4};
        INDArray result = NDArrayUtil.toNDArray(nums);
        assertArrayEquals(new int[]{1, 4}, result.shape());
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
