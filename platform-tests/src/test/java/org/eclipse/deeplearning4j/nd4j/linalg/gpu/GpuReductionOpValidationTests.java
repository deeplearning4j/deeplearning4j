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

package org.eclipse.deeplearning4j.nd4j.linalg.gpu;

import org.eclipse.deeplearning4j.tests.extensions.BackendTest;
import org.eclipse.deeplearning4j.tests.extensions.MultiBackendTestConfiguration;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * GPU-specific validation tests for reduction operations.
 * Tests max, min, prod, std, var, norm1, norm2 operations on CUDA backend.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@BackendTest(backends = {MultiBackendTestConfiguration.Backend.CUDA}, description = "GPU reduction operation validation")
public class GpuReductionOpValidationTests {

    // ========== MAX REDUCTION TESTS ==========

    @Test
    public void testMaxGlobalReduction2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 5, 3},
                {7, 2, 9},
                {4, 8, 6}
        });

        INDArray maxResult = arr.max();
        assertTrue(maxResult.isScalar());
        assertEquals(9.0, maxResult.getDouble(0), 1e-6);
    }

    @Test
    public void testMaxAlongDimensions2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 5, 3},
                {7, 2, 9},
                {4, 8, 6}
        });

        // Max along dimension 0 (columns)
        INDArray maxDim0 = arr.max(0);
        assertArrayEquals(new long[]{3}, maxDim0.shape());
        assertEquals(7.0, maxDim0.getDouble(0), 1e-6);
        assertEquals(8.0, maxDim0.getDouble(1), 1e-6);
        assertEquals(9.0, maxDim0.getDouble(2), 1e-6);

        // Max along dimension 1 (rows)
        INDArray maxDim1 = arr.max(1);
        assertArrayEquals(new long[]{3}, maxDim1.shape());
        assertEquals(5.0, maxDim1.getDouble(0), 1e-6);
        assertEquals(9.0, maxDim1.getDouble(1), 1e-6);
        assertEquals(8.0, maxDim1.getDouble(2), 1e-6);
    }

    @Test
    public void testMaxGlobalReduction3D() {
        INDArray arr = Nd4j.create(new double[][][] {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}},
                {{9, 10}, {11, 12}}
        });

        INDArray maxResult = arr.max();
        assertTrue(maxResult.isScalar());
        assertEquals(12.0, maxResult.getDouble(0), 1e-6);
    }

    @Test
    public void testMaxAlongDimensions3D() {
        INDArray arr = Nd4j.create(new double[][][] {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}},
                {{9, 10}, {11, 12}}
        });

        // Max along dimension 0
        INDArray maxDim0 = arr.max(0);
        assertArrayEquals(new long[]{2, 2}, maxDim0.shape());
        assertEquals(9.0, maxDim0.getDouble(0, 0), 1e-6);
        assertEquals(12.0, maxDim0.getDouble(1, 1), 1e-6);
    }

    // ========== MIN REDUCTION TESTS ==========

    @Test
    public void testMinGlobalReduction2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 5, 3},
                {7, 2, 9},
                {4, 8, 6}
        });

        INDArray minResult = arr.min();
        assertTrue(minResult.isScalar());
        assertEquals(1.0, minResult.getDouble(0), 1e-6);
    }

    @Test
    public void testMinAlongDimensions2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 5, 3},
                {7, 2, 9},
                {4, 8, 6}
        });

        // Min along dimension 0 (columns)
        INDArray minDim0 = arr.min(0);
        assertArrayEquals(new long[]{3}, minDim0.shape());
        assertEquals(1.0, minDim0.getDouble(0), 1e-6);
        assertEquals(2.0, minDim0.getDouble(1), 1e-6);
        assertEquals(3.0, minDim0.getDouble(2), 1e-6);

        // Min along dimension 1 (rows)
        INDArray minDim1 = arr.min(1);
        assertArrayEquals(new long[]{3}, minDim1.shape());
        assertEquals(1.0, minDim1.getDouble(0), 1e-6);
        assertEquals(2.0, minDim1.getDouble(1), 1e-6);
        assertEquals(4.0, minDim1.getDouble(2), 1e-6);
    }

    @Test
    public void testMinGlobalReduction3D() {
        INDArray arr = Nd4j.create(new double[][][] {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}},
                {{9, 10}, {11, 12}}
        });

        INDArray minResult = arr.min();
        assertTrue(minResult.isScalar());
        assertEquals(1.0, minResult.getDouble(0), 1e-6);
    }

    // ========== PROD REDUCTION TESTS ==========

    @Test
    public void testProdGlobalReduction2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });

        INDArray prodResult = arr.prod();
        assertTrue(prodResult.isScalar());
        // 1*2*3*4*5*6 = 720
        assertEquals(720.0, prodResult.getDouble(0), 1e-6);
    }

    @Test
    public void testProdAlongDimensions2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });

        // Prod along dimension 0 (columns)
        INDArray prodDim0 = arr.prod(0);
        assertArrayEquals(new long[]{3}, prodDim0.shape());
        assertEquals(4.0, prodDim0.getDouble(0), 1e-6);   // 1*4
        assertEquals(10.0, prodDim0.getDouble(1), 1e-6);  // 2*5
        assertEquals(18.0, prodDim0.getDouble(2), 1e-6);  // 3*6

        // Prod along dimension 1 (rows)
        INDArray prodDim1 = arr.prod(1);
        assertArrayEquals(new long[]{2}, prodDim1.shape());
        assertEquals(6.0, prodDim1.getDouble(0), 1e-6);   // 1*2*3
        assertEquals(120.0, prodDim1.getDouble(1), 1e-6); // 4*5*6
    }

    @Test
    public void testProdGlobalReduction3D() {
        INDArray arr = Nd4j.create(new double[][][] {
                {{1, 2}, {1, 2}},
                {{1, 2}, {1, 2}}
        });

        INDArray prodResult = arr.prod();
        assertTrue(prodResult.isScalar());
        // 1*2*1*2*1*2*1*2 = 16
        assertEquals(16.0, prodResult.getDouble(0), 1e-6);
    }

    // ========== STD (STANDARD DEVIATION) REDUCTION TESTS ==========

    @Test
    public void testStdGlobalReduction2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {2, 4, 4, 4, 5, 5, 7, 9}
        });

        INDArray stdResult = arr.std(true); // bias corrected
        assertTrue(stdResult.isScalar());
        // Known std of this data = 2.0
        assertEquals(2.0, stdResult.getDouble(0), 1e-5);
    }

    @Test
    public void testStdAlongDimensions2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });

        // Std along dimension 1 (rows)
        INDArray stdDim1 = arr.std(true, 1);
        assertArrayEquals(new long[]{2}, stdDim1.shape());

        // Verify std values are positive
        assertTrue(stdDim1.getDouble(0) > 0);
        assertTrue(stdDim1.getDouble(1) > 0);
    }

    @Test
    public void testStdUnbiased() {
        INDArray arr = Nd4j.create(new double[] {1, 2, 3, 4, 5});

        INDArray stdBiased = arr.std(false);    // Population std
        INDArray stdUnbiased = arr.std(true);   // Sample std

        assertTrue(stdBiased.isScalar());
        assertTrue(stdUnbiased.isScalar());

        // Sample std should be larger than population std
        assertTrue(stdUnbiased.getDouble(0) > stdBiased.getDouble(0));
    }

    // ========== VAR (VARIANCE) REDUCTION TESTS ==========

    @Test
    public void testVarGlobalReduction2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {2, 4, 4, 4, 5, 5, 7, 9}
        });

        INDArray varResult = arr.var(true); // bias corrected
        assertTrue(varResult.isScalar());
        // Variance = std^2 = 4.0
        assertEquals(4.0, varResult.getDouble(0), 1e-5);
    }

    @Test
    public void testVarAlongDimensions2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });

        // Var along dimension 1 (rows)
        INDArray varDim1 = arr.var(true, 1);
        assertArrayEquals(new long[]{2}, varDim1.shape());

        // Both rows have same variance pattern
        assertEquals(varDim1.getDouble(0), varDim1.getDouble(1), 1e-6);
    }

    @Test
    public void testVarStdRelationship() {
        INDArray arr = Nd4j.create(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

        INDArray std = arr.std(true);
        INDArray var = arr.var(true);

        // var = std^2
        assertEquals(var.getDouble(0), Math.pow(std.getDouble(0), 2), 1e-5);
    }

    // ========== NORM1 (L1 NORM / MANHATTAN) REDUCTION TESTS ==========

    @Test
    public void testNorm1GlobalReduction2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, -2, 3},
                {-4, 5, -6}
        });

        INDArray norm1Result = arr.norm1();
        assertTrue(norm1Result.isScalar());
        // |1| + |-2| + |3| + |-4| + |5| + |-6| = 21
        assertEquals(21.0, norm1Result.getDouble(0), 1e-6);
    }

    @Test
    public void testNorm1AlongDimensions2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, -2, 3},
                {-4, 5, -6}
        });

        // Norm1 along dimension 1 (rows)
        INDArray norm1Dim1 = arr.norm1(1);
        assertArrayEquals(new long[]{2}, norm1Dim1.shape());
        assertEquals(6.0, norm1Dim1.getDouble(0), 1e-6);   // |1| + |-2| + |3| = 6
        assertEquals(15.0, norm1Dim1.getDouble(1), 1e-6);  // |-4| + |5| + |-6| = 15
    }

    // ========== NORM2 (L2 NORM / EUCLIDEAN) REDUCTION TESTS ==========

    @Test
    public void testNorm2GlobalReduction2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {3, 4}
        });

        INDArray norm2Result = arr.norm2();
        assertTrue(norm2Result.isScalar());
        // sqrt(3^2 + 4^2) = 5
        assertEquals(5.0, norm2Result.getDouble(0), 1e-6);
    }

    @Test
    public void testNorm2AlongDimensions2D() {
        INDArray arr = Nd4j.create(new double[][] {
                {3, 4},
                {5, 12}
        });

        // Norm2 along dimension 1 (rows)
        INDArray norm2Dim1 = arr.norm2(1);
        assertArrayEquals(new long[]{2}, norm2Dim1.shape());
        assertEquals(5.0, norm2Dim1.getDouble(0), 1e-6);   // sqrt(3^2 + 4^2) = 5
        assertEquals(13.0, norm2Dim1.getDouble(1), 1e-6);  // sqrt(5^2 + 12^2) = 13
    }

    @Test
    public void testNorm2UnitVector() {
        // Unit vector should have norm2 = 1
        INDArray unitVector = Nd4j.create(new double[] {1.0/Math.sqrt(3), 1.0/Math.sqrt(3), 1.0/Math.sqrt(3)});

        INDArray norm2Result = unitVector.norm2();
        assertEquals(1.0, norm2Result.getDouble(0), 1e-5);
    }

    // ========== MIXED DATA TYPE TESTS ==========

    @Test
    public void testReductionsFloat32() {
        INDArray arr = Nd4j.create(DataType.FLOAT, 3, 3);
        arr.assign(Nd4j.linspace(1, 9, 9, DataType.FLOAT).reshape(3, 3));

        INDArray sum = arr.sum();
        INDArray max = arr.max();
        INDArray min = arr.min();

        assertEquals(DataType.FLOAT, sum.dataType());
        assertEquals(45.0f, sum.getFloat(0), 1e-5);
        assertEquals(9.0f, max.getFloat(0), 1e-5);
        assertEquals(1.0f, min.getFloat(0), 1e-5);
    }

    @Test
    public void testReductionsFloat64() {
        INDArray arr = Nd4j.create(DataType.DOUBLE, 3, 3);
        arr.assign(Nd4j.linspace(1, 9, 9, DataType.DOUBLE).reshape(3, 3));

        INDArray sum = arr.sum();
        INDArray max = arr.max();
        INDArray min = arr.min();

        assertEquals(DataType.DOUBLE, sum.dataType());
        assertEquals(45.0, sum.getDouble(0), 1e-10);
        assertEquals(9.0, max.getDouble(0), 1e-10);
        assertEquals(1.0, min.getDouble(0), 1e-10);
    }

    @Test
    public void testReductionsFloat16() {
        // Only run if float16 is supported
        try {
            INDArray arr = Nd4j.create(DataType.FLOAT16, 3, 3);
            arr.assign(Nd4j.linspace(1, 9, 9, DataType.FLOAT16).reshape(3, 3));

            INDArray sum = arr.sum();
            INDArray max = arr.max();
            INDArray min = arr.min();

            assertEquals(DataType.FLOAT16, sum.dataType());
            assertEquals(45.0f, sum.getFloat(0), 0.5f); // Larger tolerance for fp16
            assertEquals(9.0f, max.getFloat(0), 0.1f);
            assertEquals(1.0f, min.getFloat(0), 0.1f);
        } catch (Exception e) {
            // Float16 may not be supported on all GPUs
            System.out.println("Float16 test skipped: " + e.getMessage());
        }
    }

    // ========== LARGE ARRAY TESTS ==========

    @Test
    public void testLargeArraySum() {
        int size = 1_000_000;
        INDArray arr = Nd4j.ones(DataType.DOUBLE, size);

        INDArray sum = arr.sum();
        assertEquals((double) size, sum.getDouble(0), 1e-6);
    }

    @Test
    public void testLargeArrayMax() {
        int size = 1_000_000;
        INDArray arr = Nd4j.linspace(1, size, size, DataType.DOUBLE);

        INDArray max = arr.max();
        assertEquals((double) size, max.getDouble(0), 1e-6);
    }

    @Test
    public void testLargeArrayMean() {
        int size = 1_000_000;
        INDArray arr = Nd4j.linspace(1, size, size, DataType.DOUBLE);

        INDArray mean = arr.mean();
        // Mean of 1 to N is (N+1)/2
        assertEquals((size + 1.0) / 2.0, mean.getDouble(0), 1e-3);
    }

    // ========== VIEW AND STRIDE TESTS ==========

    @Test
    public void testViewReductions() {
        INDArray arr = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 4);
        INDArray view = arr.get(
                org.nd4j.linalg.indexing.NDArrayIndex.interval(1, 3),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(1, 4)
        );

        // View is 2x3 matrix
        assertArrayEquals(new long[]{2, 3}, view.shape());

        INDArray sum = view.sum();
        INDArray max = view.max();
        INDArray min = view.min();

        // Calculate expected values manually
        // Original: 1,2,3,4 / 5,6,7,8 / 9,10,11,12
        // View: 6,7,8 / 10,11,12
        assertEquals(54.0, sum.getDouble(0), 1e-6);  // 6+7+8+10+11+12
        assertEquals(12.0, max.getDouble(0), 1e-6);
        assertEquals(6.0, min.getDouble(0), 1e-6);
    }

    @Test
    public void testNonContiguousReduction() {
        INDArray arr = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 4);

        // Get every other row
        INDArray nonContig = arr.get(
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 2, 3),
                org.nd4j.linalg.indexing.NDArrayIndex.all()
        );

        // Should be rows 0 and 2: [1,2,3,4] and [9,10,11,12]
        INDArray sum = nonContig.sum();
        assertEquals(52.0, sum.getDouble(0), 1e-6);  // 1+2+3+4+9+10+11+12
    }

    // ========== EDGE CASE TESTS ==========

    @Test
    public void testScalarArrayReduction() {
        INDArray scalar = Nd4j.scalar(42.0);

        assertEquals(42.0, scalar.sum().getDouble(0), 1e-6);
        assertEquals(42.0, scalar.mean().getDouble(0), 1e-6);
        assertEquals(42.0, scalar.max().getDouble(0), 1e-6);
        assertEquals(42.0, scalar.min().getDouble(0), 1e-6);
        assertEquals(42.0, scalar.prod().getDouble(0), 1e-6);
    }

    @Test
    public void testSingleElementArrayReduction() {
        INDArray arr1D = Nd4j.create(new double[]{5.0});
        INDArray arr2D = Nd4j.create(new double[][]{{7.0}});
        INDArray arr3D = Nd4j.create(new double[][][]{{{9.0}}});

        assertEquals(5.0, arr1D.sum().getDouble(0), 1e-6);
        assertEquals(7.0, arr2D.max().getDouble(0), 1e-6);
        assertEquals(9.0, arr3D.min().getDouble(0), 1e-6);
    }

    @Test
    public void testNegativeValuesReduction() {
        INDArray arr = Nd4j.create(new double[] {-5, -2, -8, -1, -9});

        assertEquals(-25.0, arr.sum().getDouble(0), 1e-6);
        assertEquals(-5.0, arr.mean().getDouble(0), 1e-6);
        assertEquals(-1.0, arr.max().getDouble(0), 1e-6);
        assertEquals(-9.0, arr.min().getDouble(0), 1e-6);
    }

    @Test
    public void testMixedSignValuesReduction() {
        INDArray arr = Nd4j.create(new double[] {-5, 2, -8, 1, 9});

        assertEquals(-1.0, arr.sum().getDouble(0), 1e-6);
        assertEquals(9.0, arr.max().getDouble(0), 1e-6);
        assertEquals(-8.0, arr.min().getDouble(0), 1e-6);
    }
}
