package org.nd4j.linalg.api.blas;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * @author Audrey Loeffel
 */
@RunWith(Parameterized.class)
public class SparseLevel1Test extends BaseNd4jTest{

    public SparseLevel1Test(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void shouldComputeDot() {
        double[] data = {1,2,4} ;
        int[] col = {0,1,3};
        int[] pointerB = {0};
        int[] pointerE = {4};
        int[] shape = {1, 4};

        INDArray sparseVec = Nd4j.createSparseCSR(data, col, pointerB, pointerE, shape);
        INDArray vec = Nd4j.create( new double[] {1 ,2, 3, 4});
        assertEquals(21, Nd4j.getBlasWrapper().dot(sparseVec, vec), 1e-1);
    }
}
