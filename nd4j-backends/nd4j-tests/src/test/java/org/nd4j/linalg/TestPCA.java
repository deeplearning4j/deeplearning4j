
package org.nd4j.linalg.dimensionalityreduction;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by rcorbish
 */
@RunWith(Parameterized.class)
public class TestPCA extends BaseNd4jTest {


    public TestPCA(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testFactorDims() {
        int m = 13;
        int n = 4;

        double f[] = new double[] {7, 1, 11, 11, 7, 11, 3, 1, 2, 21, 1, 11, 10, 26, 29, 56, 31, 52, 55, 71, 31, 54, 47,
                        40, 66, 68, 6, 15, 8, 8, 6, 9, 17, 22, 18, 4, 23, 9, 8, 60, 52, 20, 47, 33, 22, 6, 44, 22, 26,
                        34, 12, 12};

        INDArray A = Nd4j.create(f, new int[] {m, n}, 'f');

        INDArray A1 = A.dup('f');
        INDArray Factor = org.nd4j.linalg.dimensionalityreduction.PCA.pca_factor(A1, 3, true);
        A1 = A.subiRowVector(A.mean(0));

        INDArray Reduced = A1.mmul(Factor);
        INDArray Reconstructed = Reduced.mmul(Factor.transpose());
        INDArray Diff = Reconstructed.sub(A1);
        for (int i = 0; i < m * n; i++) {
            assertEquals("Reconstructed matrix is very different from the original.", 0.0, Diff.getDouble(i), 1.0);
        }
    }


    @Test
    public void testFactorVariance() {
        int m = 13;
        int n = 4;

        double f[] = new double[] {7, 1, 11, 11, 7, 11, 3, 1, 2, 21, 1, 11, 10, 26, 29, 56, 31, 52, 55, 71, 31, 54, 47,
                        40, 66, 68, 6, 15, 8, 8, 6, 9, 17, 22, 18, 4, 23, 9, 8, 60, 52, 20, 47, 33, 22, 6, 44, 22, 26,
                        34, 12, 12};

        INDArray A = Nd4j.create(f, new int[] {m, n}, 'f');

        INDArray A1 = A.dup('f');
        INDArray Factor1 = org.nd4j.linalg.dimensionalityreduction.PCA.pca_factor(A1, 0.95, true);
        A1 = A.subiRowVector(A.mean(0));
        INDArray Reduced1 = A1.mmul(Factor1);
        INDArray Reconstructed1 = Reduced1.mmul(Factor1.transpose());
        INDArray Diff1 = Reconstructed1.sub(A1);
        for (int i = 0; i < m * n; i++) {
            assertEquals("Reconstructed matrix is very different from the original.", 0.0, Diff1.getDouble(i), 0.1);
        }
        INDArray A2 = A.dup('f');
        INDArray Factor2 = org.nd4j.linalg.dimensionalityreduction.PCA.pca_factor(A2, 0.50, true);
        assertTrue("Variance differences should change factor sizes.", Factor1.columns() > Factor2.columns());
    }


    @Override
    public char ordering() {
        return 'f';
    }
}

