/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dimensionalityreduction;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.string.NDArrayStrings;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;


public class TestPCA extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFactorDims(Nd4jBackend backend) {
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
            assertEquals(0.0, Diff.getDouble(i), 1.0,"Reconstructed matrix is very different from the original.");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFactorSVDTransposed(Nd4jBackend backend) {
        int m = 4;
        int n = 13;

        double f[] = new double[] {7, 1, 11, 11, 7, 11, 3, 1, 2, 21, 1, 11, 10, 26, 29, 56, 31, 52, 55, 71, 31, 54, 47,
                40, 66, 68, 6, 15, 8, 8, 6, 9, 17, 22, 18, 4, 23, 9, 8, 60, 52, 20, 47, 33, 22, 6, 44, 22, 26,
                34, 12, 12};

        INDArray A = Nd4j.create(f, new long[] {m, n}, 'f');

        INDArray A1 = A.dup('f');
        INDArray factor = org.nd4j.linalg.dimensionalityreduction.PCA.pca_factor(A1, 3, true);
        A1 = A.subiRowVector(A.mean(0));

        INDArray reduced = A1.mmul(factor);
        INDArray reconstructed = reduced.mmul(factor.transpose());
        INDArray diff = reconstructed.sub(A1);
        for (int i = 0; i < m * n; i++) {
            assertEquals(0.0, diff.getDouble(i), 1.0,"Reconstructed matrix is very different from the original.");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFactorVariance(Nd4jBackend backend) {
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
            assertEquals( 0.0, Diff1.getDouble(i), 0.1,"Reconstructed matrix is very different from the original.");
        }
        INDArray A2 = A.dup('f');
        INDArray Factor2 = org.nd4j.linalg.dimensionalityreduction.PCA.pca_factor(A2, 0.50, true);
        assertTrue(Factor1.columns() > Factor2.columns(),"Variance differences should change factor sizes.");
    }


    /**
     * Test new PCA routines, added by Luke Czapla
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPCA(Nd4jBackend backend) {
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
        assertTrue(  reduced99.columns() > reduced70.columns(),"Major variance differences should change number of basis vectors");
        INDArray reduced100 = myPCA.reducedBasis(1.0);
        assertTrue(reduced100.columns() == m.columns(),"100% variance coverage should include all eigenvectors");
        NDArrayStrings ns = new NDArrayStrings(5);
//        System.out.println("Eigenvectors:\n" + ns.format(myPCA.getEigenvectors()));
//        System.out.println("Eigenvalues:\n" + ns.format(myPCA.getEigenvalues()));
        double variance = 0.0;

        // sample 1000 of the randomly generated samples with the reduced basis set
        for (long i = 0; i < 1000; i++)
            variance += myPCA.estimateVariance(m.getRow(i), reduced70.columns());
        variance /= 1000.0;
        System.out.println("Fraction of variance using 70% variance with " + reduced70.columns() + " columns: " + variance);
        assertTrue(variance > 0.70,"Variance does not cover intended 70% variance");
        // create "dummy" data with the same exact trends
        INDArray testSample = myPCA.generateGaussianSamples(10000);
        PCA analyzePCA = new PCA(testSample);
        assertTrue( myPCA.getMean().equalsWithEps(analyzePCA.getMean(), 0.2 * myPCA.getMean().columns()),"Means do not agree accurately enough");
        assertTrue(myPCA.getCovarianceMatrix().equalsWithEps(
                analyzePCA.getCovarianceMatrix(), 1.0 * analyzePCA.getCovarianceMatrix().length()),"Covariance is not reproduced accurately enough");
        assertTrue( myPCA.getEigenvalues().equalsWithEps(analyzePCA.getEigenvalues(),
                0.5 * myPCA.getEigenvalues().columns()),"Eigenvalues are not close enough");
        assertTrue(myPCA.getEigenvectors()
                .equalsWithEps(analyzePCA.getEigenvectors(), 0.1 * analyzePCA.getEigenvectors().length()),"Eigenvectors are not close enough");
//        System.out.println("Original cov:\n" + ns.format(myPCA.getCovarianceMatrix()) + "\nDummy cov:\n"
//                        + ns.format(analyzePCA.getCovarianceMatrix()));
        INDArray testSample2 = analyzePCA.convertBackToFeatures(analyzePCA.convertToComponents(testSample));
        assertTrue( testSample.equalsWithEps(testSample2, 1e-5 * testSample.length()),"Transformation does not work.");
    }


    @Override
    public char ordering() {
        return 'f';
    }

}

