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

package org.eclipse.deeplearning4j.nd4j.linalg.blas;


import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@NativeTag
public class BlasTests extends BaseNd4jTestWithBackends {


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void pcaFactorTest(Nd4jBackend backend) {
        // PCA eigenvector signs and ordering can vary by backend/LAPACK implementation,
        // so we verify mathematical properties rather than hardcoded values.
        double[][] inputArray = { { 1.0, 2.0, 3.0 }, { 11.0, 12.0, 13.0 }, { 21.0, 22.0, 23.0 }, { 31.0, 32.0, 33.0 } };
        INDArray inputMatrix = Nd4j.create(inputArray);
        int nColumns = inputMatrix.columns();
        INDArray factor = PCA.pca_factor(inputMatrix, nColumns, false);

        // Factor must be square (nColumns x nColumns)
        assertEquals(nColumns, factor.rows());
        assertEquals(nColumns, factor.columns());

        // The factor columns (or rows, depending on backend) must be orthonormal.
        // For a square orthogonal matrix Q, both Q^T*Q = I and Q*Q^T = I.
        INDArray ftf = factor.transpose().mmul(factor);
        // Each diagonal element should be ≈ 1, off-diagonal ≈ 0
        for (int i = 0; i < nColumns; i++) {
            for (int j = 0; j < nColumns; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, ftf.getDouble(i, j), 1e-4,
                        "factor^T * factor should be identity at [" + i + "," + j + "]");
            }
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void pcaTest(Nd4jBackend backend) {
        // PCA eigenvector signs and ordering can vary by backend/LAPACK implementation,
        // so we verify mathematical properties rather than hardcoded values.
        double[][] inputArray = { { 1.0, 2.0, 3.0 }, { 11.0, 12.0, 13.0 }, { 21.0, 22.0, 23.0 }, { 31.0, 32.0, 33.0 } };
        INDArray inputMatrix = Nd4j.create(inputArray);
        int nColumns = inputMatrix.columns();

        // pca() returns the projected data; pca_factor() returns the transformation matrix.
        // Both should complete without error.
        INDArray ret = PCA.pca(inputMatrix, nColumns, true);
        INDArray factor = PCA.pca_factor(inputMatrix, nColumns, false);

        // Factor must be square (nColumns x nColumns)
        assertEquals(nColumns, factor.rows());
        assertEquals(nColumns, factor.columns());

        // The factor columns (or rows, depending on backend) must be orthonormal.
        // For a square orthogonal matrix Q, Q^T*Q = I.
        INDArray ftf = factor.transpose().mmul(factor);
        for (int i = 0; i < nColumns; i++) {
            for (int j = 0; j < nColumns; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, ftf.getDouble(i, j), 1e-4,
                        "factor^T * factor should be identity at [" + i + "," + j + "]");
            }
        }

        // The projected data must have the same number of rows as the input and nColumns columns
        assertEquals(inputMatrix.rows(), ret.rows());
        assertEquals(nColumns, ret.columns());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void simpleTest(Nd4jBackend backend) {
        INDArray m1 = Nd4j.create(new double[][]{{1.0}, {2.0}, {3.0}, {4.0}});

        m1 = m1.reshape(2, 2);

        INDArray m2 = Nd4j.create(new double[][]{{1.0, 2.0, 3.0, 4.0},});
        m2 = m2.reshape(2, 2);
        m2.setOrder('f');

        //mmul gives the correct result
        INDArray correctResult;
        correctResult = m1.mmul(m2);
        System.out.println("================");
        System.out.println(m1);
        System.out.println(m2);
        System.out.println(correctResult);
        System.out.println("================");
        INDArray newResult = Nd4j.create(DataType.DOUBLE, correctResult.shape(), 'c');
        m1.mmul(m2, newResult);
        assertEquals(correctResult, newResult);

        //But not so mmuli (which is somewhat mixed)
        INDArray target = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        target = m1.mmuli(m2, m1);
        assertEquals(true, target.equals(correctResult));
        assertEquals(true, m1.equals(correctResult));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemmInvalid1(Nd4jBackend backend) {
        // gemm now silently handles view targets via a temp array copy-back,
        // so writing into a view is valid and must produce the correct result.
        final INDArray a = Nd4j.rand(3, 4);
        final INDArray b = Nd4j.rand(4, 5);

        final INDArray target = Nd4j.zeros(new int[]{2, 3, 5}, 'f');
        final INDArray view = target.tensorAlongDimension(0, 1, 2);

        Nd4j.gemm(a, b, view, false, false, 1.0, 0.0);

        // Verify the result written into the view equals a direct mmul
        final INDArray expected = a.mmul(b);
        assertEquals(expected, view);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemmInvalid3(Nd4jBackend backend) {
        // gemm now silently handles view targets via a temp array copy-back,
        // so writing into a view is valid and must produce the correct result.
        // With transposeA=true: a(4x3)^T = (3x4) * b(4x5) = (3x5), matching view shape.
        final INDArray a = Nd4j.rand(4, 3);
        final INDArray b = Nd4j.rand(4, 5);

        final INDArray target = Nd4j.zeros(new int[]{2, 3, 5}, 'f');
        final INDArray view = target.tensorAlongDimension(0, 1, 2);

        Nd4j.gemm(a, b, view, true, false, 1.0, 0.0);

        // Verify the result written into the view equals a.transpose().mmul(b)
        final INDArray expected = a.transpose().mmul(b);
        assertEquals(expected, view);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemm1(Nd4jBackend backend) {
        final INDArray a = Nd4j.rand(4, 3);
        final INDArray b = Nd4j.rand(4, 5);

        final INDArray result = a.transpose().mmul(b);
        final INDArray result2 = Nd4j.gemm(a, b, true, false);

        assertEquals(result, result2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemm2(Nd4jBackend backend) {
        final INDArray a = Nd4j.rand(4, 3);
        final INDArray b = Nd4j.rand(4, 5);

        final INDArray target = Nd4j.zeros(new int[]{2, 3, 5}, 'f');
        final INDArray view = target.tensorAlongDimension(0, 1, 2);

        a.transpose().mmuli(b, view);

        final INDArray result = a.transpose().mmul(b);

        assertEquals(result, view);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemm3(Nd4jBackend backend) {
        final INDArray a = Nd4j.rand(4, 3);
        final INDArray b = Nd4j.rand(4, 5);

        final INDArray target = Nd4j.zeros(new int[]{2, 3, 5}, 'c');
        final INDArray view = target.tensorAlongDimension(0, 1, 2);

        a.transpose().mmuli(b, view);

        final INDArray result = a.transpose().mmul(b);

        assertEquals(result, view);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmuli1(Nd4jBackend backend) {
        final INDArray activations = Nd4j.createUninitialized(new long[]{1, 3, 1}, 'f');
        final INDArray z = activations.tensorAlongDimension(0, 1, 2);

        Nd4j.getRandom().setSeed(12345);
        final INDArray a = Nd4j.rand(3, 4);
        final INDArray b = Nd4j.rand(4, 1);

        INDArray ab = a.mmul(b);
        a.mmul(b, z);
        assertEquals(ab, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmuli2(Nd4jBackend backend) {
        final INDArray activations = Nd4j.createUninitialized(new long[]{2, 3, 1}, 'f');
        final INDArray z = activations.tensorAlongDimension(0, 1, 2);

        Nd4j.getRandom().setSeed(12345);
        final INDArray a = Nd4j.rand(3, 4);
        final INDArray b = Nd4j.rand(4, 1);

        INDArray ab = a.mmul(b);
        a.mmul(b, z);
        assertEquals(ab, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmuli3(Nd4jBackend backend) {
        final INDArray activations = Nd4j.createUninitialized(new long[]{1, 3, 2}, 'f');
        final INDArray z = activations.tensorAlongDimension(0, 1, 2);

        final INDArray a = Nd4j.rand(3, 4);
        final INDArray b = Nd4j.rand(4, 2);

        INDArray ab = a.mmul(b);
        a.mmul(b, z);
        assertEquals(ab, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test_Fp16_Mmuli_1(Nd4jBackend backend){
        final INDArray activations = Nd4j.createUninitialized(DataType.HALF, new long[]{1, 3, 2}, 'f');
        final INDArray z = activations.tensorAlongDimension(0, 1, 2);

        final INDArray a = Nd4j.rand(DataType.HALF, 3, 4);
        final INDArray b = Nd4j.rand(DataType.HALF,4, 2);

        INDArray ab = a.mmul(b);
        a.mmul(b, z);
        assertEquals(ab, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test_Fp16_Mmuli_2(Nd4jBackend backend){
        val a = Nd4j.create(DataType.HALF, 32, 768);
        val b = Nd4j.create(DataType.HALF, 768);

        val c = a.mmul(b);
    }

    @Disabled
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHalfPrecision(Nd4jBackend backend) {
        val a = Nd4j.create(DataType.HALF, 64, 768);
        val b = Nd4j.create(DataType.HALF, 768, 1024);
        val c = Nd4j.create(DataType.HALF, new long[]{64, 1024}, 'f');

        val durations = new ArrayList<Long>();
        val iterations = 100;
        for (int e = 0; e < iterations; e++) {
            val timeStart = System.currentTimeMillis();
            a.mmuli(b, c);
            val timeEnd = System.currentTimeMillis();
            durations.add(timeEnd - timeStart);
        }

        Collections.sort(durations);

        log.info("Median time: {} ms", durations.get(durations.size() / 2));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmuli4(Nd4jBackend backend){
        try {
            Nd4j.rand(1, 3).mmuli(Nd4j.rand(3, 1), Nd4j.createUninitialized(new int[]{10, 10, 1}));
            fail("Expected exception");
        } catch (IllegalStateException e){
            assertTrue(e.getMessage().contains("shape"));
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
