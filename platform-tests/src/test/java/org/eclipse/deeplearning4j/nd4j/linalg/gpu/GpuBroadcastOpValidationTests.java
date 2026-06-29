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

import org.eclipse.deeplearning4j.tests.extensions.Backend;
import org.eclipse.deeplearning4j.tests.extensions.BackendTest;
import org.eclipse.deeplearning4j.tests.extensions.MultiBackendTestConfiguration;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.*;

/**
 * GPU-specific validation tests for broadcasting operations.
 * Tests various broadcast scenarios on CUDA backend with cuDNN helper.
 *
 * @author Adam Gibson
 */
@BackendTest(backends = {Backend.CUDA},
             helpers = {"cudnn"},
             description = "GPU broadcast operation validation")
public class GpuBroadcastOpValidationTests {

    // ========== BASIC ARITHMETIC BROADCAST TESTS ==========

    @Test
    public void testScalarBroadcastAdd() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });
        double scalar = 10.0;

        INDArray result = arr.add(scalar);

        assertEquals(11.0, result.getDouble(0, 0), 1e-6);
        assertEquals(16.0, result.getDouble(1, 2), 1e-6);
    }

    @Test
    public void testScalarBroadcastMul() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });
        double scalar = 2.0;

        INDArray result = arr.mul(scalar);

        assertEquals(2.0, result.getDouble(0, 0), 1e-6);
        assertEquals(12.0, result.getDouble(1, 2), 1e-6);
    }

    @Test
    public void testRowVectorBroadcastAdd() {
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });
        INDArray row = Nd4j.create(new double[] {10, 20, 30});

        INDArray result = matrix.addRowVector(row);

        assertEquals(11.0, result.getDouble(0, 0), 1e-6);
        assertEquals(22.0, result.getDouble(0, 1), 1e-6);
        assertEquals(17.0, result.getDouble(2, 0), 1e-6);
        assertEquals(39.0, result.getDouble(2, 2), 1e-6);
    }

    @Test
    public void testColumnVectorBroadcastAdd() {
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });
        INDArray col = Nd4j.create(new double[] {10, 20, 30}).reshape(3, 1);

        INDArray result = matrix.addColumnVector(col);

        assertEquals(11.0, result.getDouble(0, 0), 1e-6);
        assertEquals(13.0, result.getDouble(0, 2), 1e-6);
        assertEquals(27.0, result.getDouble(1, 0), 1e-6);
        assertEquals(39.0, result.getDouble(2, 2), 1e-6);
    }

    // ========== SHAPE BROADCAST TESTS ==========

    @Test
    public void testBroadcast1Dto2D() {
        INDArray arr2d = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });
        INDArray arr1d = Nd4j.create(new double[] {10, 20, 30});

        INDArray result = arr2d.add(arr1d);

        assertArrayEquals(new long[]{2, 3}, result.shape());
        assertEquals(11.0, result.getDouble(0, 0), 1e-6);
        assertEquals(36.0, result.getDouble(1, 2), 1e-6);
    }

    @Test
    public void testBroadcast2Dto3D() {
        INDArray arr3d = Nd4j.create(DataType.DOUBLE, 2, 3, 4);
        arr3d.assign(1.0);

        INDArray arr2d = Nd4j.create(DataType.DOUBLE, 3, 4);
        arr2d.assign(10.0);

        INDArray result = arr3d.add(arr2d);

        assertArrayEquals(new long[]{2, 3, 4}, result.shape());
        assertEquals(11.0, result.getDouble(0, 0, 0), 1e-6);
        assertEquals(11.0, result.getDouble(1, 2, 3), 1e-6);
    }

    @Test
    public void testBroadcastUnitDimensions() {
        // (2, 1, 3) + (1, 4, 3) -> (2, 4, 3)
        INDArray arr1 = Nd4j.ones(DataType.DOUBLE, 2, 1, 3);
        INDArray arr2 = Nd4j.ones(DataType.DOUBLE, 1, 4, 3).mul(2);

        INDArray result = arr1.add(arr2);

        assertArrayEquals(new long[]{2, 4, 3}, result.shape());
        assertEquals(3.0, result.getDouble(0, 0, 0), 1e-6);  // 1 + 2
        assertEquals(3.0, result.getDouble(1, 3, 2), 1e-6);  // 1 + 2
    }

    // ========== ALL ARITHMETIC OPERATIONS ==========

    @Test
    public void testBroadcastSub() {
        INDArray arr = Nd4j.create(new double[][] {{10, 20, 30}, {40, 50, 60}});
        INDArray vec = Nd4j.create(new double[] {1, 2, 3});

        INDArray result = arr.sub(vec);

        assertEquals(9.0, result.getDouble(0, 0), 1e-6);
        assertEquals(57.0, result.getDouble(1, 2), 1e-6);
    }

    @Test
    public void testBroadcastDiv() {
        INDArray arr = Nd4j.create(new double[][] {{10, 20, 30}, {40, 50, 60}});
        INDArray vec = Nd4j.create(new double[] {2, 5, 10});

        INDArray result = arr.div(vec);

        assertEquals(5.0, result.getDouble(0, 0), 1e-6);
        assertEquals(6.0, result.getDouble(1, 2), 1e-6);
    }

    @Test
    public void testBroadcastRsub() {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        INDArray vec = Nd4j.create(new double[] {10, 20, 30});

        INDArray result = arr.rsubRowVector(vec);  // vec - arr

        assertEquals(9.0, result.getDouble(0, 0), 1e-6);   // 10 - 1
        assertEquals(24.0, result.getDouble(1, 2), 1e-6);  // 30 - 6
    }

    @Test
    public void testBroadcastRdiv() {
        INDArray arr = Nd4j.create(new double[][] {{2, 4, 5}, {10, 20, 25}});
        INDArray vec = Nd4j.create(new double[] {100, 100, 100});

        INDArray result = arr.rdivRowVector(vec);  // vec / arr

        assertEquals(50.0, result.getDouble(0, 0), 1e-6);  // 100 / 2
        assertEquals(4.0, result.getDouble(1, 2), 1e-6);   // 100 / 25
    }

    // ========== COMPARISON BROADCAST TESTS ==========

    @Test
    public void testBroadcastLessThan() {
        INDArray arr = Nd4j.create(new double[][] {{1, 5, 3}, {4, 2, 6}});
        INDArray vec = Nd4j.create(new double[] {3, 3, 3});

        INDArray result = arr.lt(vec);

        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(1.0, result.getDouble(0, 0), 1e-6);  // 1 < 3 = true
        assertEquals(0.0, result.getDouble(0, 1), 1e-6);  // 5 < 3 = false
        assertEquals(0.0, result.getDouble(1, 2), 1e-6);  // 6 < 3 = false
    }

    @Test
    public void testBroadcastGreaterThan() {
        INDArray arr = Nd4j.create(new double[][] {{1, 5, 3}, {4, 2, 6}});
        INDArray vec = Nd4j.create(new double[] {3, 3, 3});

        INDArray result = arr.gt(vec);

        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(0.0, result.getDouble(0, 0), 1e-6);  // 1 > 3 = false
        assertEquals(1.0, result.getDouble(0, 1), 1e-6);  // 5 > 3 = true
        assertEquals(1.0, result.getDouble(1, 2), 1e-6);  // 6 > 3 = true
    }

    @Test
    public void testBroadcastEquals() {
        INDArray arr = Nd4j.create(new double[][] {{1, 3, 3}, {3, 2, 3}});
        INDArray vec = Nd4j.create(new double[] {3, 3, 3});

        INDArray result = arr.eq(vec);

        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(0.0, result.getDouble(0, 0), 1e-6);  // 1 == 3 = false
        assertEquals(1.0, result.getDouble(0, 1), 1e-6);  // 3 == 3 = true
        assertEquals(1.0, result.getDouble(1, 2), 1e-6);  // 3 == 3 = true
    }

    // ========== IN-PLACE BROADCAST TESTS ==========

    @Test
    public void testBroadcastAddi() {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        INDArray vec = Nd4j.create(new double[] {10, 20, 30});

        arr.addiRowVector(vec);

        assertEquals(11.0, arr.getDouble(0, 0), 1e-6);
        assertEquals(36.0, arr.getDouble(1, 2), 1e-6);
    }

    @Test
    public void testBroadcastMuli() {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        INDArray vec = Nd4j.create(new double[] {2, 3, 4});

        arr.muliRowVector(vec);

        assertEquals(2.0, arr.getDouble(0, 0), 1e-6);
        assertEquals(24.0, arr.getDouble(1, 2), 1e-6);
    }

    // ========== HIGHER DIMENSIONAL BROADCAST TESTS ==========

    @Test
    public void testBroadcast3DRowVector() {
        INDArray arr3d = Nd4j.create(new double[][][] {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}}
        });
        INDArray vec = Nd4j.create(new double[] {10, 20});

        INDArray result = arr3d.addRowVector(vec);

        assertArrayEquals(new long[]{2, 2, 2}, result.shape());
        assertEquals(11.0, result.getDouble(0, 0, 0), 1e-6);
        assertEquals(28.0, result.getDouble(1, 1, 1), 1e-6);
    }

    @Test
    public void testBroadcast4D() {
        // (2, 3, 4, 5) + (4, 5) -> (2, 3, 4, 5)
        INDArray arr4d = Nd4j.ones(DataType.DOUBLE, 2, 3, 4, 5);
        INDArray arr2d = Nd4j.ones(DataType.DOUBLE, 4, 5).mul(10);

        INDArray result = arr4d.add(arr2d);

        assertArrayEquals(new long[]{2, 3, 4, 5}, result.shape());
        assertEquals(11.0, result.getDouble(0, 0, 0, 0), 1e-6);
        assertEquals(11.0, result.getDouble(1, 2, 3, 4), 1e-6);
    }

    // ========== DATA TYPE BROADCAST TESTS ==========

    @Test
    public void testBroadcastFloat32() {
        INDArray arr = Nd4j.create(DataType.FLOAT, 3, 3);
        arr.assign(1.0f);
        INDArray vec = Nd4j.create(DataType.FLOAT, 3);
        vec.assign(10.0f);

        INDArray result = arr.addRowVector(vec);

        assertEquals(DataType.FLOAT, result.dataType());
        assertEquals(11.0f, result.getFloat(0, 0), 1e-5);
    }

    @Test
    public void testBroadcastFloat64() {
        INDArray arr = Nd4j.create(DataType.DOUBLE, 3, 3);
        arr.assign(1.0);
        INDArray vec = Nd4j.create(DataType.DOUBLE, 3);
        vec.assign(10.0);

        INDArray result = arr.addRowVector(vec);

        assertEquals(DataType.DOUBLE, result.dataType());
        assertEquals(11.0, result.getDouble(0, 0), 1e-10);
    }

    @Test
    public void testBroadcastMixedTypes() {
        // Float32 matrix + Float64 vector
        INDArray arr = Nd4j.create(DataType.FLOAT, 3, 3);
        arr.assign(1.0f);
        INDArray vec = Nd4j.create(DataType.DOUBLE, 3);
        vec.assign(10.0);

        // Cast to same type before operation
        INDArray result = arr.castTo(DataType.DOUBLE).addRowVector(vec);

        assertEquals(DataType.DOUBLE, result.dataType());
        assertEquals(11.0, result.getDouble(0, 0), 1e-6);
    }

    // ========== VIEW/SLICE BROADCAST TESTS ==========

    @Test
    public void testBroadcastOnView() {
        INDArray arr = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 4);
        INDArray view = arr.get(NDArrayIndex.interval(0, 2), NDArrayIndex.interval(1, 4));
        // view is 2x3: [[2,3,4], [6,7,8]]

        INDArray vec = Nd4j.create(new double[] {10, 20, 30});
        INDArray result = view.addRowVector(vec);

        assertArrayEquals(new long[]{2, 3}, result.shape());
        assertEquals(12.0, result.getDouble(0, 0), 1e-6);  // 2 + 10
        assertEquals(38.0, result.getDouble(1, 2), 1e-6);  // 8 + 30
    }

    @Test
    public void testBroadcastOnTranspose() {
        INDArray arr = Nd4j.create(new double[][] {{1, 2}, {3, 4}, {5, 6}});
        INDArray transposed = arr.transpose();  // 2x3
        INDArray vec = Nd4j.create(new double[] {10, 20, 30});

        INDArray result = transposed.addRowVector(vec);

        assertArrayEquals(new long[]{2, 3}, result.shape());
        assertEquals(11.0, result.getDouble(0, 0), 1e-6);  // 1 + 10
        assertEquals(36.0, result.getDouble(1, 2), 1e-6);  // 6 + 30
    }

    // ========== LARGE ARRAY BROADCAST TESTS ==========

    @Test
    public void testLargeArrayBroadcast() {
        int rows = 10000;
        int cols = 1000;

        INDArray arr = Nd4j.ones(DataType.DOUBLE, rows, cols);
        INDArray vec = Nd4j.ones(DataType.DOUBLE, cols).mul(10);

        INDArray result = arr.addRowVector(vec);

        assertEquals(11.0, result.getDouble(0, 0), 1e-6);
        assertEquals(11.0, result.getDouble(rows - 1, cols - 1), 1e-6);
    }

    @Test
    public void testLargeArray3DBroadcast() {
        int d0 = 100;
        int d1 = 100;
        int d2 = 100;

        INDArray arr = Nd4j.ones(DataType.DOUBLE, d0, d1, d2);
        INDArray vec = Nd4j.ones(DataType.DOUBLE, d2).mul(5);

        INDArray result = arr.addRowVector(vec);

        assertEquals(6.0, result.getDouble(0, 0, 0), 1e-6);
        assertEquals(6.0, result.getDouble(d0 - 1, d1 - 1, d2 - 1), 1e-6);
    }

    // ========== EDGE CASE TESTS ==========

    @Test
    public void testBroadcastWithScalarArray() {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        INDArray scalar = Nd4j.scalar(100.0);

        INDArray result = arr.add(scalar);

        assertEquals(101.0, result.getDouble(0, 0), 1e-6);
        assertEquals(106.0, result.getDouble(1, 2), 1e-6);
    }

    @Test
    public void testBroadcastNegativeValues() {
        INDArray arr = Nd4j.create(new double[][] {{-1, -2, -3}, {-4, -5, -6}});
        INDArray vec = Nd4j.create(new double[] {10, 20, 30});

        INDArray result = arr.addRowVector(vec);

        assertEquals(9.0, result.getDouble(0, 0), 1e-6);   // -1 + 10
        assertEquals(24.0, result.getDouble(1, 2), 1e-6);  // -6 + 30
    }

    @Test
    public void testBroadcastZeroVector() {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        INDArray vec = Nd4j.zeros(3);

        INDArray result = arr.addRowVector(vec);

        assertEquals(1.0, result.getDouble(0, 0), 1e-6);
        assertEquals(6.0, result.getDouble(1, 2), 1e-6);
    }

    // ========== CHAINED OPERATION TESTS ==========

    @Test
    public void testChainedBroadcastOperations() {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        INDArray vec1 = Nd4j.create(new double[] {10, 20, 30});
        INDArray vec2 = Nd4j.create(new double[] {1, 2, 3}).reshape(2, 1);

        // (arr + row_vec) * col_vec
        INDArray result = arr.addRowVector(vec1).mulColumnVector(vec2);

        assertEquals(11.0, result.getDouble(0, 0), 1e-6);   // (1+10) * 1
        assertEquals(72.0, result.getDouble(1, 2), 1e-6);   // (6+30) * 2
    }

    @Test
    public void testBroadcastWithReduction() {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        INDArray vec = Nd4j.create(new double[] {10, 20, 30});

        // Add row vector then reduce
        INDArray added = arr.addRowVector(vec);
        INDArray sumResult = added.sum();

        // (1+10)+(2+20)+(3+30)+(4+10)+(5+20)+(6+30) = 11+22+33+14+25+36 = 141
        assertEquals(141.0, sumResult.getDouble(0), 1e-6);
    }

    /**
     * Regression test for broadcast_to producing all-zeros when the input is a VIEW
     * created by expand_dims on a non-contiguous tensor.
     *
     * SmolDocling GQA KV-head broadcast scenario:
     *   KV:  [1, 3, 8, 64]  →  expand_dims(dim=2)  →  VIEW [1, 3, 1, 8, 64]
     *        →  broadcast_to  →  [1, 3, 3, 8, 64]
     *
     * Green commit 5f3f34b2a4 produced correct output.
     * HEAD produced all-zeros in all 30 attention layers → garbage decode.
     */
    @Test
    public void testBroadcastToFromExpandDimsView() {
        // 1. Create a contiguous source tensor with distinct non-zero values
        int batch = 1, kvHeads = 3, seqLen = 8, headDim = 64;
        INDArray src = Nd4j.linspace(1.0, batch * kvHeads * seqLen * headDim,
                batch * kvHeads * seqLen * headDim, DataType.FLOAT)
                .reshape(batch, kvHeads, seqLen, headDim);

        // Verify source is not zero
        assertTrue(src.maxNumber().floatValue() > 0,
                "Source tensor must be non-zero");

        // 2. Create a VIEW via expand_dims at dim=2: [1,3,8,64] -> [1,3,1,8,64]
        INDArray viewInput = Nd4j.expandDims(src, 2);
        assertArrayEquals(new long[]{batch, kvHeads, 1, seqLen, headDim}, viewInput.shape(),
                "expand_dims shape mismatch");
        assertTrue(viewInput.maxNumber().floatValue() > 0,
                "VIEW input must be non-zero before broadcast");

        // 3. Run broadcast_to: [1,3,1,8,64] -> [1,3,3,8,64]  (GQA: 3 Q-heads per KV-head)
        int qHeads = kvHeads; // 3:1 ratio
        INDArray targetShapeArr = Nd4j.createFromArray(
                (long) batch, (long) kvHeads, (long) qHeads, (long) seqLen, (long) headDim);
        INDArray viewResult = Nd4j.create(DataType.FLOAT,
                batch, kvHeads, qHeads, seqLen, headDim);

        DynamicCustomOp broadcastOp = DynamicCustomOp.builder("broadcast_to")
                .addInputs(viewInput, targetShapeArr)
                .addOutputs(viewResult)
                .build();
        Nd4j.getExecutioner().exec(broadcastOp);
        Nd4j.getExecutioner().commit();

        // 4. Assert output is non-zero
        float maxViewResult = viewResult.maxNumber().floatValue();
        assertFalse(maxViewResult == 0.0f,
                "broadcast_to from VIEW input must produce non-zero output; got all-zeros (GQA regression)");

        // 5. Also run broadcast_to with a CONTIGUOUS (dup'd) input and compare
        INDArray contiguousInput = viewInput.dup();
        assertArrayEquals(viewInput.shape(), contiguousInput.shape());
        INDArray contiguousResult = Nd4j.create(DataType.FLOAT,
                batch, kvHeads, qHeads, seqLen, headDim);

        DynamicCustomOp broadcastOpContiguous = DynamicCustomOp.builder("broadcast_to")
                .addInputs(contiguousInput, targetShapeArr)
                .addOutputs(contiguousResult)
                .build();
        Nd4j.getExecutioner().exec(broadcastOpContiguous);
        Nd4j.getExecutioner().commit();

        float maxContiguousResult = contiguousResult.maxNumber().floatValue();
        assertFalse(maxContiguousResult == 0.0f,
                "broadcast_to from CONTIGUOUS input must produce non-zero output");

        // 6. Both results should match (view vs contiguous)
        assertEquals(maxContiguousResult, maxViewResult, 1e-4f,
                "broadcast_to VIEW result should match CONTIGUOUS result");

        // 7. Verify first element: src[0,0,0,0] = 1.0, so result[0,0,k,0,0] = 1.0 for all k
        float expectedFirst = src.getFloat(0, 0, 0, 0);
        for (int k = 0; k < qHeads; k++) {
            float actualView = viewResult.getFloat(0, 0, k, 0, 0);
            float actualContiguous = contiguousResult.getFloat(0, 0, k, 0, 0);
            assertEquals(expectedFirst, actualView, 1e-4f,
                    "VIEW broadcast_to mismatch at head=" + k);
            assertEquals(expectedFirst, actualContiguous, 1e-4f,
                    "CONTIGUOUS broadcast_to mismatch at head=" + k);
        }

        // 8. Verify the tiled values match expected: result[0, kv, q, s, d] == src[0, kv, s, d]
        for (int kv = 0; kv < kvHeads; kv++) {
            for (int q = 0; q < qHeads; q++) {
                for (int s = 0; s < seqLen; s++) {
                    float srcVal = src.getFloat(0, kv, s, 0);
                    float viewVal = viewResult.getFloat(0, kv, q, s, 0);
                    assertEquals(srcVal, viewVal, 1e-4f,
                            "Mismatch at kv=" + kv + " q=" + q + " s=" + s);
                }
            }
        }
    }
}
