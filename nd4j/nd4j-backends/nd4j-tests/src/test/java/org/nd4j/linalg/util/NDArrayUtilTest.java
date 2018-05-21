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
        assertArrayEquals(new long[]{2,3}, result.shape());
    }

    @Test
    public void testVectorConversion() {
        int[] nums = {1, 2, 3, 4};
        INDArray result = NDArrayUtil.toNDArray(nums);
        assertArrayEquals(new long[]{1, 4}, result.shape());
    }


    @Test
    public void testFlattenArray1() {
        float[][][] arrX = new float[2][2][2];

        float[] arrZ = ArrayUtil.flatten(arrX);

        assertEquals(8, arrZ.length);
    }

    @Test
    public void testFlattenArray2() {
        float[][][] arrX = new float[5][4][3];

        float[] arrZ = ArrayUtil.flatten(arrX);

        assertEquals(60, arrZ.length);
    }


    @Test
    public void testFlattenArray3() {
        float[][][] arrX = new float[5][2][3];

        float[] arrZ = ArrayUtil.flatten(arrX);

        assertEquals(30, arrZ.length);
    }

    @Test
    public void testFlattenArray4() {
        float[][][][] arrX = new float[5][2][3][3];

        float[] arrZ = ArrayUtil.flatten(arrX);

        assertEquals(90, arrZ.length);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
