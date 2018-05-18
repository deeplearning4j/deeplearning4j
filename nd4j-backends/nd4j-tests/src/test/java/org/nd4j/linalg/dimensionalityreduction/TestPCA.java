
package org.nd4j.linalg.dimensionalityreduction;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.string.NDArrayStrings;

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


    /**
     * Test new PCA routines, added by Luke Czapla
     */
    @Test
    public void testPCA() {
        INDArray m = Nd4j.randn(10000, 16);
        // 10000 random correlated samples of 16 features to analyze
        m.getColumn(0).muli(4.84);
        m.getColumn(1).muli(4.84);
        m.getColumn(2).muli(4.09);
        m.getColumn(1).addi(m.getColumn(2).div(2.0));
        m.getColumn(2).addi(34.286);
        m.getColumn(1).addi(m.getColumn(4));
        m.getColumn(4).subi(m.getColumn(5).div(2.0));
        m.getColumn(5).addi(3.4);
        m.getColumn(6).muli(6.0);
        m.getColumn(7).muli(0.2);
        m.getColumn(8).muli(2.0);
        m.getColumn(9).muli(6.0);
        m.getColumn(9).addi(m.getColumn(6).mul(1.0));
        m.getColumn(10).muli(0.2);
        m.getColumn(11).muli(2.0);
        m.getColumn(12).muli(0.2);
        m.getColumn(13).muli(4.0);
        m.getColumn(14).muli(3.2);
        m.getColumn(14).addi(m.getColumn(2).mul(1.0)).subi(m.getColumn(13).div(2.0));
        m.getColumn(15).muli(1.0);
        m.getColumn(13).subi(12.0);
        m.getColumn(15).addi(30.0);

        PCA myPCA = new PCA(m);
        INDArray reduced70 = myPCA.reducedBasis(0.70);
        INDArray reduced99 = myPCA.reducedBasis(0.99);
        assertTrue("Major variance differences should change number of basis vectors",
                        reduced99.columns() > reduced70.columns());
        INDArray reduced100 = myPCA.reducedBasis(1.0);
        assertTrue("100% variance coverage should include all eigenvectors", reduced100.columns() == m.columns());
        NDArrayStrings ns = new NDArrayStrings(5);
        System.out.println("Eigenvectors:\n" + ns.format(myPCA.getEigenvectors()));
        System.out.println("Eigenvalues:\n" + ns.format(myPCA.getEigenvalues()));
        double variance = 0.0;

        // FIXME: int cast
        // sample 1000 of the randomly generated samples with the reduced basis set
        for (int i = 0; i < 1000; i++)
            variance += myPCA.estimateVariance(m.getRow(i), (int) reduced70.columns());
        variance /= 1000.0;
        System.out.println("Fraction of variance using 70% variance with " + reduced70.columns() + " columns: "
                        + variance);
        assertTrue("Variance does not cover intended 70% variance", variance > 0.70);
        // create "dummy" data with the same exact trends
        INDArray testSample = myPCA.generateGaussianSamples(10000);
        PCA analyzePCA = new PCA(testSample);
        assertTrue("Means do not agree accurately enough",
                        myPCA.getMean().equalsWithEps(analyzePCA.getMean(), 0.2 * myPCA.getMean().columns()));
        assertTrue("Covariance is not reproduced accurately enough", myPCA.getCovarianceMatrix().equalsWithEps(
                        analyzePCA.getCovarianceMatrix(), 1.0 * analyzePCA.getCovarianceMatrix().length()));
        assertTrue("Eigenvalues are not close enough", myPCA.getEigenvalues().equalsWithEps(analyzePCA.getEigenvalues(),
                        0.5 * myPCA.getEigenvalues().columns()));
        assertTrue("Eigenvectors are not close enough", myPCA.getEigenvectors()
                        .equalsWithEps(analyzePCA.getEigenvectors(), 0.1 * analyzePCA.getEigenvectors().length()));
        System.out.println("Original cov:\n" + ns.format(myPCA.getCovarianceMatrix()) + "\nDummy cov:\n"
                        + ns.format(analyzePCA.getCovarianceMatrix()));
        INDArray testSample2 = analyzePCA.convertBackToFeatures(analyzePCA.convertToComponents(testSample));
        assertTrue("Transformation does not work.", testSample.equalsWithEps(testSample2, 1e-5 * testSample.length()));
    }


    @Override
    public char ordering() {
        return 'f';
    }

}

