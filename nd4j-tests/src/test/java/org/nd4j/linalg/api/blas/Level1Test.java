package org.nd4j.linalg.api.blas;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class Level1Test extends BaseNd4jTest {

    @Test
    public void testDot() {
        INDArray vec1 = Nd4j.create(new float[]{1, 2, 3, 4});
        INDArray vec2 = Nd4j.create(new float[]{1, 2, 3, 4});
        assertEquals(30, Nd4j.getBlasWrapper().dot(vec1, vec2), 1e-1);

        INDArray matrix = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row = matrix.getRow(1);
        assertEquals(25, Nd4j.getBlasWrapper().dot(row, row), 1e-1);

    }

    @Override
    public char ordering() {
        return 'f';
    }
}
