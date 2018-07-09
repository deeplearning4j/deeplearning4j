package org.nd4j.linalg.api.blas;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * @author Audrey Loeffel
 */
@Ignore // temporary ignored
@RunWith(Parameterized.class)
public class SparseCOOLevel2Test extends BaseNd4jTest {

    // matrix = [[1, 2], [0, 0]]
    private double[] data = {1, 2};
    private int[][] indexes = new int[][] {{0, 0}, {0, 1}};
    private long[] shape = {2, 2};

    public SparseCOOLevel2Test(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testGemv() {
        INDArray array1 = Nd4j.createSparseCOO(data, indexes, shape);
        INDArray array2 = Nd4j.linspace(1, 2, 2).reshape(2, 1);

        INDArray array3 = array1.mmul(array2); // should be [5, 0]
        assertEquals(2, array3.length());
        assertEquals(5, array3.getFloat(0), 1e-5);
        assertEquals(0, array3.getFloat(1), 1e-5);
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
