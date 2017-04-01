package org.nd4j.linalg.ops.copy;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 2/12/16.
 */
@RunWith(Parameterized.class)
public class CopyTest extends BaseNd4jTest {
    public CopyTest(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testCopy() {
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray dup = arr.dup();
        assertEquals(arr, dup);
    }

    @Test
    public void testDup() {

        for (int x = 0; x < 100; x++) {
            INDArray orig = Nd4j.linspace(1, 4, 4);
            INDArray dup = orig.dup();
            assertEquals(orig, dup);

            INDArray matrix = Nd4j.create(new float[] {1, 2, 3, 4}, new int[] {2, 2});
            INDArray dup2 = matrix.dup();
            assertEquals(matrix, dup2);

            INDArray row1 = matrix.getRow(1);
            INDArray dupRow = row1.dup();
            assertEquals(row1, dupRow);


            INDArray columnSorted = Nd4j.create(new float[] {2, 1, 4, 3}, new int[] {2, 2});
            INDArray dup3 = columnSorted.dup();
            assertEquals(columnSorted, dup3);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
