/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.inverse;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.nd4j.linalg.primitives.Pair;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by agibsoncccc on 12/7/15.
 */
@RunWith(Parameterized.class)
public class TestInvertMatrices extends BaseNd4jTest {


    public TestInvertMatrices(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testInverse() {
        RealMatrix matrix = new Array2DRowRealMatrix(new double[][] {{1, 2}, {3, 4}});

        RealMatrix inverse = MatrixUtils.inverse(matrix);
        INDArray arr = InvertMatrix.invert(Nd4j.linspace(1, 4, 4).reshape(2, 2), false);
        for (int i = 0; i < inverse.getRowDimension(); i++) {
            for (int j = 0; j < inverse.getColumnDimension(); j++) {
                assertEquals(arr.getDouble(i, j), inverse.getEntry(i, j), 1e-1);
            }
        }
    }

    @Test
    public void testInverseComparison() {

        List<Pair<INDArray, String>> list = NDArrayCreationUtil.getAllTestMatricesWithShape(10, 10, 12345);

        for (Pair<INDArray, String> p : list) {
            INDArray orig = p.getFirst();
            orig.assign(Nd4j.rand(orig.shape()));
            INDArray inverse = InvertMatrix.invert(orig, false);
            RealMatrix rm = CheckUtil.convertToApacheMatrix(orig);
            RealMatrix rmInverse = new LUDecomposition(rm).getSolver().getInverse();

            INDArray expected = CheckUtil.convertFromApacheMatrix(rmInverse);
            assertTrue(p.getSecond(), CheckUtil.checkEntries(expected, inverse, 1e-3, 1e-4));
        }
    }

    @Test
    public void testInvalidMatrixInversion() {
        try {
            InvertMatrix.invert(Nd4j.create(5, 4), false);
            fail("No exception thrown for invalid input");
        } catch (Exception e) {
        }

        try {
            InvertMatrix.invert(Nd4j.create(5, 5, 5), false);
            fail("No exception thrown for invalid input");
        } catch (Exception e) {
        }

        try {
            InvertMatrix.invert(Nd4j.create(1, 5), false);
            fail("No exception thrown for invalid input");
        } catch (Exception e) {
        }
    }

    /**
     * Example from: <a href="https://www.wolframalpha.com/input/?i=invert+matrix+((1,2),(3,4),(5,6))">here</a>
     */
    @Test
    public void testLeftPseudoInvert() {
        INDArray X = Nd4j.create(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        INDArray expectedLeftInverse = Nd4j.create(new double[][]{{-16, -4, 8}, {13, 4, -5}}).mul(1 / 12d);
        INDArray leftInverse = InvertMatrix.pLeftInvert(X, false);
        assertEquals(expectedLeftInverse, leftInverse);

        final INDArray identity3x3 = Nd4j.create(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        final INDArray identity2x2 = Nd4j.create(new double[][]{{1, 0}, {0, 1}});
        final double precision = 1e-5;

        // right inverse
        final INDArray rightInverseCheck = X.mmul(leftInverse);
        // right inverse must not hold since X rows are not linear independent (x_3 + x_1 = 2*x_2)
        assertFalse(rightInverseCheck.equalsWithEps(identity3x3, precision));

        // left inverse must hold since X columns are linear independent
        final INDArray leftInverseCheck = leftInverse.mmul(X);
        assertTrue(leftInverseCheck.equalsWithEps(identity2x2, precision));

        // general condition X = X * X^-1 * X
        final INDArray generalCond = X.mmul(leftInverse).mmul(X);
        assertTrue(X.equalsWithEps(generalCond, precision));
        checkMoorePenroseConditions(X, leftInverse, precision);
    }

    /**
     * Check the Moore-Penrose conditions for pseudo-matrices.
     *
     * @param A Initial matrix
     * @param B Pseudo-Inverse of {@code A}
     * @param precision Precision when comparing matrix elements
     */
    private void checkMoorePenroseConditions(INDArray A, INDArray B, double precision) {
        // ABA=A (AB need not be the general identity matrix, but it maps all column vectors of A to themselves)
        assertTrue(A.equalsWithEps(A.mmul(B).mmul(A), precision));
        // BAB=B (B is a weak inverse for the multiplicative semigroup)
        assertTrue(B.equalsWithEps(B.mmul(A).mmul(B), precision));
        // (AB)^T=AB (AB is Hermitian)
        assertTrue((A.mmul(B)).transpose().equalsWithEps(A.mmul(B), precision));
        // (BA)^T=BA (BA is also Hermitian)
        assertTrue((B.mmul(A)).transpose().equalsWithEps(B.mmul(A), precision));
    }

    /**
     * Example from: <a href="https://www.wolframalpha.com/input/?i=invert+matrix+((1,2),(3,4),(5,6))^T">here</a>
     */
    @Test
    public void testRightPseudoInvert() {
        INDArray X = Nd4j.create(new double[][]{{1, 2}, {3, 4}, {5, 6}}).transpose();
        INDArray expectedRightInverse = Nd4j.create(new double[][]{{-16, 13}, {-4, 4}, {8, -5}}).mul(1 / 12d);
        INDArray rightInverse = InvertMatrix.pRightInvert(X, false);
        assertEquals(expectedRightInverse, rightInverse);

        final INDArray identity3x3 = Nd4j.create(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        final INDArray identity2x2 = Nd4j.create(new double[][]{{1, 0}, {0, 1}});
        final double precision = 1e-5;

        // left inverse
        final INDArray leftInverseCheck = rightInverse.mmul(X);
        // left inverse must not hold since X columns are not linear independent (x_3 + x_1 = 2*x_2)
        assertFalse(leftInverseCheck.equalsWithEps(identity3x3, precision));

        // left inverse must hold since X rows are linear independent
        final INDArray rightInverseCheck = X.mmul(rightInverse);
        assertTrue(rightInverseCheck.equalsWithEps(identity2x2, precision));

        // general condition X = X * X^-1 * X
        final INDArray generalCond = X.mmul(rightInverse).mmul(X);
        assertTrue(X.equalsWithEps(generalCond, precision));
        checkMoorePenroseConditions(X, rightInverse, precision);
    }

    /**
     * Try to compute the right pseudo inverse of a matrix without full row rank (x1 = 2*x2)
     */
    @Test(expected = IllegalArgumentException.class)
    public void testRightPseudoInvertWithNonFullRowRank() {
        INDArray X = Nd4j.create(new double[][]{{1, 2}, {3, 6}, {5, 10}}).transpose();
        INDArray rightInverse = InvertMatrix.pRightInvert(X, false);
    }

    /**
     * Try to compute the left pseudo inverse of a matrix without full column rank (x1 = 2*x2)
     */
    @Test(expected = IllegalArgumentException.class)
    public void testLeftPseudoInvertWithNonFullColumnRank() {
        INDArray X = Nd4j.create(new double[][]{{1, 2}, {3, 6}, {5, 10}});
        INDArray leftInverse = InvertMatrix.pLeftInvert(X, false);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
