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
package org.eclipse.deeplearning4j.nd4j.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Tile;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class BroadcastingOpsSmokeTests {

    @Test
    public void testRowVectorBroadcastOperations() {
        // Create a matrix and row vector
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });
        INDArray rowVector = Nd4j.create(new double[] {10, 20, 30});

        // Create corresponding Java arrays for validation
        double[][] matrixJava = new double[3][3];
        double[] rowVectorJava = new double[3];
        double[][] resultJava = new double[3][3];

        // Initialize Java arrays
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                matrixJava[i][j] = matrix.getDouble(i, j);
                rowVectorJava[j] = rowVector.getDouble(j);
            }
        }

        // Test addRowVector
        INDArray addResult = matrix.addRowVector(rowVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = matrixJava[i][j] + rowVectorJava[j];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], addResult.getDouble(i, j));
            }
        }

        // Test subRowVector
        INDArray subResult = matrix.subRowVector(rowVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = matrixJava[i][j] - rowVectorJava[j];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], subResult.getDouble(i, j));
            }
        }

        // Test mulRowVector
        INDArray mulResult = matrix.mulRowVector(rowVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = matrixJava[i][j] * rowVectorJava[j];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], mulResult.getDouble(i, j));
            }
        }

        // Test divRowVector
        INDArray divResult = matrix.divRowVector(rowVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = matrixJava[i][j] / rowVectorJava[j];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], divResult.getDouble(i, j), 1e-6);
            }
        }
    }

    @Test
    public void testColumnVectorBroadcastOperations() {
        // Create a matrix and column vector
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });
        INDArray columnVector = Nd4j.create(new double[] {10, 20, 30}).reshape(3, 1);

        // Create corresponding Java arrays for validation
        double[][] matrixJava = new double[3][3];
        double[] columnVectorJava = new double[3];
        double[][] resultJava = new double[3][3];

        // Initialize Java arrays
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                matrixJava[i][j] = matrix.getDouble(i, j);
            }
            columnVectorJava[i] = columnVector.getDouble(i, 0);
        }

        // Test addiColumnVector
        INDArray addResult = matrix.addColumnVector(columnVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = matrixJava[i][j] + columnVectorJava[i];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], addResult.getDouble(i, j));
            }
        }

        // Test subColumnVector
        INDArray subResult = matrix.subColumnVector(columnVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = matrixJava[i][j] - columnVectorJava[i];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], subResult.getDouble(i, j));
            }
        }

        // Test mulColumnVector
        INDArray mulResult = matrix.mulColumnVector(columnVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = matrixJava[i][j] * columnVectorJava[i];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], mulResult.getDouble(i, j));
            }
        }

        // Test divColumnVector
        INDArray divResult = matrix.divColumnVector(columnVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = matrixJava[i][j] / columnVectorJava[i];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], divResult.getDouble(i, j), 1e-6);
            }
        }
    }

    @Test
    public void testInPlaceBroadcastOperations() {
        // Create matrices for in-place operations
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });
        INDArray rowVector = Nd4j.create(new double[] {10, 20, 30});
        INDArray columnVector = Nd4j.create(new double[] {10, 20, 30}).reshape(3, 1);

        // Create corresponding Java arrays
        double[][] matrixJava = new double[3][3];
        double[] rowVectorJava = new double[3];
        double[] columnVectorJava = new double[3];

        // Initialize Java arrays
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                matrixJava[i][j] = matrix.getDouble(i, j);
            }
            rowVectorJava[i] = rowVector.getDouble(i);
            columnVectorJava[i] = columnVector.getDouble(i, 0);
        }

        // Test addiRowVector
        matrix.addiRowVector(rowVector);
        // Update Java array
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                matrixJava[i][j] += rowVectorJava[j];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(matrixJava[i][j], matrix.getDouble(i, j));
            }
        }

        // Reset matrix for next test
        matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });
        // Reset matrixJava
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                matrixJava[i][j] = matrix.getDouble(i, j);
            }
        }

        // Test addiColumnVector
        matrix.addiColumnVector(columnVector);
        // Update Java array
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                matrixJava[i][j] += columnVectorJava[i];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(matrixJava[i][j], matrix.getDouble(i, j));
            }
        }
    }

    @Test
    public void testBroadcastShape() {
        // Create source array with shape [4,5]
        INDArray input = Nd4j.ones(4, 5);

        // Reshape to [1,4,5] to match desired output rank
        input = input.reshape(1, 4, 5);

        // Create result array with target shape [3,4,5]
        INDArray result = Nd4j.createUninitialized(3, 4, 5);

        // Now we can tile with input rank 3
        Nd4j.getExecutioner().exec(new Tile(input, result, 3, 1, 1));

        // Verify shape
        assertArrayEquals(new long[]{3, 4, 5}, result.shape());

        // Verify the values were properly tiled/broadcasted
        for(int i = 0; i < result.size(0); i++) {
            assertEquals(input.slice(0), result.slice(i));
        }
    }
    @Test
    public void testBroadcastUnitDimensions() {
        // Create array with unit dimensions [1,3,1]
        INDArray input = Nd4j.ones(1, 3, 1);

        // Create result array with shape [2,3,4]
        INDArray result = Nd4j.createUninitialized(2, 3, 4);

        // Do the broadcast
        input.broadcast(result);

        // Verify shape
        assertArrayEquals(new long[]{2, 3, 4}, result.shape());

        // Verify all values are 1 since input was ones()
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 3; j++) {
                for(int k = 0; k < 4; k++) {
                    assertEquals(1.0, result.getDouble(i, j, k), 1e-5);
                }
            }
        }
    }

    @Test
    public void testBroadcastRowVector() {
        // Test row vector broadcasting which has special case handling
        INDArray rowVector = Nd4j.create(new double[]{1, 2, 3}).castTo(DataType.DOUBLE);
        INDArray result = Nd4j.createUninitialized(4, 3).castTo(DataType.DOUBLE);  // 4 rows, 3 cols

        // Broadcast row vector
        rowVector.broadcast(result);

        // Each row should be [1,2,3]
        for(int i = 0; i < 4; i++) {
            INDArray row = result.getRow(i);
            assertEquals(rowVector, row);
        }
    }

    @Test
    public void testBroadcastColumnVector() {
        // Test column vector broadcasting which has special case handling
        INDArray colVector = Nd4j.create(new double[]{1, 2, 3}).reshape(3,1).castTo(DataType.DOUBLE);
        INDArray result = Nd4j.createUninitialized(3, 4).castTo(DataType.DOUBLE); // 3 rows, 4 cols

        // Broadcast column vector
        colVector.broadcast(result);

        // Each column should equal original column
        for(int i = 0; i < 4; i++) {
            INDArray col = result.getColumn(i);
            assertEquals(colVector, col);
        }
    }

    @Test
    public void testInvalidBroadcast() {
        INDArray input = Nd4j.ones(2, 3);
        INDArray result = Nd4j.createUninitialized(3, 4); // Incompatible shape

        try {
            input.broadcast(result);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            // Expected
        }
    }
    @Test
    public void testTileOperations() {
        // Create a vector
        INDArray vector = Nd4j.create(new double[] {1, 2, 3});

        // Test tiling to create a matrix
        INDArray tiled = vector.repmat(3, 1);

        // Create expected result array
        double[][] expectedJava = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                expectedJava[i][j] = vector.getDouble(j);
            }
        }

        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedJava[i][j], tiled.getDouble(i, j));
            }
        }

        // Test tiling in multiple dimensions
        INDArray matrix = Nd4j.create(new double[][] {{1, 2}, {3, 4}});
        INDArray tiledMatrix = matrix.repmat(2, 3);

        // Verify shape
        assertArrayEquals(new long[]{4, 6}, tiledMatrix.shape());
    }

    @Test
    public void testBroadcastWithScalar() {
        // Test broadcasting scalar operations
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });

        // Create corresponding Java array
        double[][] matrixJava = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                matrixJava[i][j] = matrix.getDouble(i, j);
            }
        }

        double scalar = 10.0;

        // Test add scalar
        INDArray addResult = matrix.add(scalar);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(matrixJava[i][j] + scalar, addResult.getDouble(i, j));
            }
        }

        // Test multiply scalar
        INDArray mulResult = matrix.mul(scalar);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(matrixJava[i][j] * scalar, mulResult.getDouble(i, j));
            }
        }

        // Test in-place scalar operations
        matrix.addi(scalar);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(matrixJava[i][j] + scalar, matrix.getDouble(i, j));
            }
        }
    }

    @Test
    public void testBroadcastComparison() {
        // Create a matrix and vectors for comparison operations
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });
        INDArray rowVector = Nd4j.create(new double[] {2, 5, 8});

        // Create corresponding Java arrays
        double[][] matrixJava = new double[3][3];
        double[] rowVectorJava = new double[3];
        boolean[][] expectedGt = new boolean[3][3];
        boolean[][] expectedLt = new boolean[3][3];
        boolean[][] expectedEq = new boolean[3][3];

        // Initialize Java arrays
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                matrixJava[i][j] = matrix.getDouble(i, j);
                rowVectorJava[j] = rowVector.getDouble(j);
            }
        }

        // Compute expected comparison results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                expectedGt[i][j] = matrixJava[i][j] > rowVectorJava[j];
                expectedLt[i][j] = matrixJava[i][j] < rowVectorJava[j];
                expectedEq[i][j] = matrixJava[i][j] == rowVectorJava[j];
            }
        }

        // Test greater than
        INDArray gtResult = matrix.gt(rowVector);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedGt[i][j] ? 1 : 0, gtResult.getDouble(i, j));
            }
        }

        // Test less than
        INDArray ltResult = matrix.lt(rowVector);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedLt[i][j] ? 1 : 0, ltResult.getDouble(i, j));
            }
        }

        // Test equals
        INDArray eqResult = matrix.eq(rowVector);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedEq[i][j] ? 1 : 0, eqResult.getDouble(i, j));
            }
        }
    }

    @Test
    public void testBroadcast3DOperations() {
        // Create 3D array and vector for broadcasting
        INDArray array3D = Nd4j.create(new double[][][] {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}},
                {{9, 10}, {11, 12}}
        });
        INDArray vector = Nd4j.create(new double[] {10, 20});

        // Verify input shapes
        assertArrayEquals(new long[]{3, 2, 2}, array3D.shape());
        assertArrayEquals(new long[]{2}, vector.shape());

        // Create corresponding Java arrays
        double[][][] array3DJava = new double[3][2][2];
        double[] vectorJava = new double[2];
        double[][][] expectedAdd = new double[3][2][2];

        // Initialize Java arrays
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    array3DJava[i][j][k] = array3D.getDouble(i, j, k);
                }
            }
        }
        for (int i = 0; i < 2; i++) {
            vectorJava[i] = vector.getDouble(i);
        }

        // Broadcast vector along the last dimension
        INDArray broadcastResult = array3D.addRowVector(vector);

        // Verify output shape matches input
        assertArrayEquals(array3D.shape(), broadcastResult.shape());

        // Compute expected results with explicit position checks
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    expectedAdd[i][j][k] = array3DJava[i][j][k] + vectorJava[k];

                    // Get actual value at this position
                    double actual = broadcastResult.getDouble(i, j, k);

                    // Verify each position explicitly
                    String errorMsg = String.format("Failure at position [%d,%d,%d]: expected=%f, actual=%f, original=%f, vector=%f",
                            i, j, k, expectedAdd[i][j][k], actual, array3DJava[i][j][k], vectorJava[k]);

                    assertEquals(expectedAdd[i][j][k], actual, 1e-5,errorMsg);
                }
            }
        }

        // Additional test case: verify edge positions
        assertEquals(array3D.getDouble(0,0,0) + vector.getDouble(0),
                broadcastResult.getDouble(0,0,0), 1e-5);
        assertEquals(array3D.getDouble(2,1,1) + vector.getDouble(1),
                broadcastResult.getDouble(2,1,1), 1e-5);
    }
    @Test
    public void testBroadcastWithDifferentShapes() {
        // Test broadcasting between arrays of different but compatible shapes
        INDArray arr1 = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}}); // Shape: (2,3)
        INDArray arr2 = Nd4j.create(new double[] {10, 20, 30}); // Shape: (3)
        INDArray arr3 = Nd4j.create(new double[] {1, 2}).reshape(2, 1); // Shape: (2,1)

        // Test broadcasting arr2 to arr1's shape
        INDArray result1 = arr1.add(arr2);
        assertEquals(2, result1.rank());
        assertArrayEquals(new long[]{2, 3}, result1.shape());

        // Test broadcasting arr3 to arr1's shape
        INDArray result2 = arr1.addColumnVector(arr3);
        assertEquals(2, result2.rank());
        assertArrayEquals(new long[]{2, 3}, result2.shape());

        // Verify the results maintain the correct broadcasting pattern
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(arr1.getDouble(i, j) + arr2.getDouble(j), result1.getDouble(i, j));
                assertEquals(arr1.getDouble(i, j) + arr3.getDouble(i, 0), result2.getDouble(i, j));
            }
        }
    }

    @Test
    public void testReverseRowVectorBroadcastOperations() {
        // Create a matrix and row vector
        INDArray matrix = Nd4j.create(new double[][] {
                {2, 4, 8},
                {16, 32, 64},
                {128, 256, 512}
        });
        INDArray rowVector = Nd4j.create(new double[] {10, 20, 30});

        // Create corresponding Java arrays for validation
        double[][] matrixJava = new double[3][3];
        double[] rowVectorJava = new double[3];
        double[][] resultJava = new double[3][3];

        // Initialize Java arrays
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                matrixJava[i][j] = matrix.getDouble(i, j);
                rowVectorJava[j] = rowVector.getDouble(j);
            }
        }

        // Test rsubRowVector (rowVector - matrix)
        INDArray rsubResult = matrix.rsubRowVector(rowVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = rowVectorJava[j] - matrixJava[i][j];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], rsubResult.getDouble(i, j));
            }
        }

        // Test rdivRowVector (rowVector / matrix)
        INDArray rdivResult = matrix.rdivRowVector(rowVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = rowVectorJava[j] / matrixJava[i][j];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], rdivResult.getDouble(i, j), 1e-6);
            }
        }
    }

    @Test
    public void testReverseColumnVectorBroadcastOperations() {
        // Create a matrix and column vector
        INDArray matrix = Nd4j.create(new double[][] {
                {2, 4, 8},
                {16, 32, 64},
                {128, 256, 512}
        });
        INDArray columnVector = Nd4j.create(new double[] {10, 20, 30}).reshape(3, 1);

        // Create corresponding Java arrays for validation
        double[][] matrixJava = new double[3][3];
        double[] columnVectorJava = new double[3];
        double[][] resultJava = new double[3][3];

        // Initialize Java arrays
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                matrixJava[i][j] = matrix.getDouble(i, j);
            }
            columnVectorJava[i] = columnVector.getDouble(i, 0);
        }

        // Test rsubColumnVector (columnVector - matrix)
        INDArray rsubResult = matrix.rsubColumnVector(columnVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = columnVectorJava[i] - matrixJava[i][j];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], rsubResult.getDouble(i, j));
            }
        }

        // Test rdivColumnVector (columnVector / matrix)
        INDArray rdivResult = matrix.rdivColumnVector(columnVector);
        // Compute expected result
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                resultJava[i][j] = columnVectorJava[i] / matrixJava[i][j];
            }
        }
        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(resultJava[i][j], rdivResult.getDouble(i, j), 1e-6);
            }
        }
    }

    @Test
    public void testInPlaceReverseOperations() {
        // Test in-place reverse operations with scalar
        INDArray arr = Nd4j.create(new double[][] {{2, 4}, {8, 16}});
        double scalar = 10.0;

        // Create Java array for validation
        double[][] arrJava = new double[2][2];
        double[][] expectedJava = new double[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                arrJava[i][j] = arr.getDouble(i, j);
            }
        }

        // Test rsubi with scalar (scalar - array)
        INDArray rsubiResult = arr.rsubi(scalar);
        // Compute expected
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                expectedJava[i][j] = scalar - arrJava[i][j];
            }
        }
        // Verify results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expectedJava[i][j], rsubiResult.getDouble(i, j));
                assertEquals(expectedJava[i][j], arr.getDouble(i, j)); // Verify in-place modification
            }
        }

        // Reset array for next test
        arr = Nd4j.create(new double[][] {{2, 4}, {8, 16}});
        // Reset Java array
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                arrJava[i][j] = arr.getDouble(i, j);
            }
        }

        // Test rdivi with scalar (scalar / array)
        INDArray rdiviResult = arr.rdivi(scalar);
        // Compute expected
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                expectedJava[i][j] = scalar / arrJava[i][j];
            }
        }
        // Verify results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expectedJava[i][j], rdiviResult.getDouble(i, j), 1e-6);
                assertEquals(expectedJava[i][j], arr.getDouble(i, j), 1e-6); // Verify in-place modification
            }
        }
    }

    @Test
    public void testReverseBroadcastWithScalar() {
        INDArray arr = Nd4j.create(new double[][] {{2, 4}, {8, 16}});
        double scalar = 10.0;

        // Create Java array for validation
        double[][] arrJava = new double[2][2];
        double[][] expectedJava = new double[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                arrJava[i][j] = arr.getDouble(i, j);
            }
        }

        // Test rsub with scalar (scalar - array)
        INDArray rsubResult = arr.rsub(scalar);
        // Compute expected
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                expectedJava[i][j] = scalar - arrJava[i][j];
            }
        }
        // Verify results and check original array is unchanged
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expectedJava[i][j], rsubResult.getDouble(i, j));
                assertEquals(arrJava[i][j], arr.getDouble(i, j)); // Verify original unchanged
            }
        }

        // Test rdiv with scalar (scalar / array)
        INDArray rdivResult = arr.rdiv(scalar);
        // Compute expected
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                expectedJava[i][j] = scalar / arrJava[i][j];
            }
        }
        // Verify results and check original array is unchanged
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expectedJava[i][j], rdivResult.getDouble(i, j), 1e-6);
                assertEquals(arrJava[i][j], arr.getDouble(i, j)); // Verify original unchanged
            }
        }
    }

    @Test
    public void testReverseBroadcast3D() {
        // Create 3D array and vector for broadcasting
        INDArray array3D = Nd4j.create(new double[][][] {
                {{2, 4}, {8, 16}},
                {{32, 64}, {128, 256}},
                {{512, 1024}, {2048, 4096}}
        });
        INDArray vector = Nd4j.create(new double[] {1000, 2000});

        // Create corresponding Java arrays
        double[][][] array3DJava = new double[3][2][2];
        double[] vectorJava = new double[2];
        double[][][] expectedRDiv = new double[3][2][2];

        // Initialize Java arrays
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    array3DJava[i][j][k] = array3D.getDouble(i, j, k);
                }
            }
        }
        for (int i = 0; i < 2; i++) {
            vectorJava[i] = vector.getDouble(i);
        }

        // Broadcast vector along the last dimension for reverse division
        INDArray rdivResult = array3D.rdivRowVector(vector);

        // Compute expected results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    expectedRDiv[i][j][k] = vectorJava[k] / array3DJava[i][j][k];
                }
            }
        }

        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    assertEquals(expectedRDiv[i][j][k], rdivResult.getDouble(i, j, k), 1e-6);
                }
            }
        }
    }

    @Test
    public void testRowViewBroadcastOperations() {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });

        // Get row view
        INDArray rowView = matrix.getRow(1); // row [4, 5, 6]

        // Create a vector to broadcast
        INDArray vector = Nd4j.create(new double[] {10, 20, 30});

        // Test broadcasting operations on row view
        INDArray addResult = rowView.add(vector);
        INDArray mulResult = rowView.mul(vector);
        INDArray divResult = rowView.div(vector);
        INDArray subResult = rowView.sub(vector);

        // Verify results
        INDArray expectedAdd = Nd4j.create(new double[] {14, 25, 36});
        INDArray expectedMul = Nd4j.create(new double[] {40, 100, 180});
        INDArray expectedDiv = Nd4j.create(new double[] {0.4, 0.25, 0.2});
        INDArray expectedSub = Nd4j.create(new double[] {-6, -15, -24});

        assertEquals(expectedAdd, addResult);
        assertEquals(expectedMul, mulResult);
        assertEquals(expectedDiv, divResult);
        assertEquals(expectedSub, subResult);
    }

    @Test
    public void testColumnViewBroadcastOperations() {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        }).castTo(DataType.DOUBLE);

        // Get column view
        INDArray columnView = matrix.getColumn(1); // column [2, 5, 8]

        // Create a vector to broadcast
        INDArray vector = Nd4j.create(new double[] {10, 20, 30}).reshape(3, 1).castTo(DataType.DOUBLE);

        // Test broadcasting operations on column view
        INDArray addResult = columnView.add(vector);
        INDArray mulResult = columnView.mul(vector);
        INDArray divResult = columnView.div(vector);
        INDArray subResult = columnView.sub(vector);

        // Verify results
        INDArray expectedAdd = Nd4j.create(new double[] {12, 25, 38}).reshape(3,1).castTo(DataType.DOUBLE);
        INDArray expectedMul = Nd4j.create(new double[] {20, 100, 240}).reshape(3,1).castTo(DataType.DOUBLE);
        INDArray expectedDiv = Nd4j.create(new double[] {0.2, 0.25, 0.2666668}).reshape(3,1).castTo(DataType.DOUBLE);
        INDArray expectedSub = Nd4j.create(new double[] {-8, -15, -22}).reshape(3,1).castTo(DataType.DOUBLE);

        assertEquals(expectedAdd, addResult);
        assertEquals(expectedMul, mulResult);
        assertTrue(expectedDiv.equalsWithEps(divResult, 1e-1));
        assertEquals(expectedSub, subResult);
    }

    @Test
    public void testSubArrayViewBroadcastOperations() {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12}
        });

        // Get sub-array view using get(INDArrayIndex...)
        INDArray subView = matrix.get(
                NDArrayIndex.interval(0, 2),  // rows 0-1
                NDArrayIndex.interval(1, 3)   // columns 1-2
        );
        // subView is now:
        // [2, 3]
        // [6, 7]

        // Create vector for broadcasting
        INDArray rowVector = Nd4j.create(new double[] {10, 20});
        INDArray colVector = Nd4j.create(new double[] {10, 20}).reshape(2, 1);

        // Test row vector broadcasting
        INDArray rowAddResult = subView.addRowVector(rowVector);
        INDArray rowMulResult = subView.mulRowVector(rowVector);

        // Test column vector broadcasting
        INDArray colAddResult = subView.addColumnVector(colVector);
        INDArray colMulResult = subView.mulColumnVector(colVector);

        // Verify results
        INDArray expectedRowAdd = Nd4j.create(new double[][] {
                {12, 23},
                {16, 27}
        });
        INDArray expectedRowMul = Nd4j.create(new double[][] {
                {20, 60},
                {60, 140}
        });
        INDArray expectedColAdd = Nd4j.create(new double[][] {
                {12, 13},
                {26, 27}
        });
        INDArray expectedColMul = Nd4j.create(new double[][] {
                {20, 30},
                {120, 140}
        });

        assertEquals(expectedRowAdd, rowAddResult);
        assertEquals(expectedRowMul, rowMulResult);
        assertEquals(expectedColAdd, colAddResult);
        assertEquals(expectedColMul, colMulResult);
    }

    @Test
    public void testTensorAlongDimensionBroadcast() {
        // Create a 3D array
        INDArray array3D = Nd4j.create(new double[][][] {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}},
                {{9, 10}, {11, 12}}
        });

        // Get tensor along dimension 1 (middle dimension)
        INDArray tensor = array3D.tensorAlongDimension(0, 1, 2);
        // tensor is now the first slice: [[1,2], [3,4]]

        // Create vectors for broadcasting
        INDArray rowVector = Nd4j.create(new double[] {10, 20});
        INDArray colVector = Nd4j.create(new double[] {10, 20}).reshape(2, 1);

        // Test broadcasting operations on tensor view
        INDArray rowAddResult = tensor.addRowVector(rowVector);
        INDArray rowMulResult = tensor.mulRowVector(rowVector);
        INDArray colAddResult = tensor.addColumnVector(colVector);
        INDArray colMulResult = tensor.mulColumnVector(colVector);

        // Verify results
        INDArray expectedRowAdd = Nd4j.create(new double[][] {
                {11, 22},
                {13, 24}
        });
        INDArray expectedRowMul = Nd4j.create(new double[][] {
                {10, 40},
                {30, 80}
        });
        INDArray expectedColAdd = Nd4j.create(new double[][] {
                {11, 12},
                {23, 24}
        });
        INDArray expectedColMul = Nd4j.create(new double[][] {
                {10, 20},
                {60, 80}
        });

        assertEquals(expectedRowAdd, rowAddResult);
        assertEquals(expectedRowMul, rowMulResult);
        assertEquals(expectedColAdd, colAddResult);
        assertEquals(expectedColMul, colMulResult);
    }

    @Test
    public void testInPlaceViewBroadcastOperations() {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });

        // Get views
        INDArray rowView = matrix.getRow(1);
        INDArray colView = matrix.getColumn(1);

        // Create vectors for broadcasting
        INDArray rowVector = Nd4j.create(new double[] {10, 20, 30});
        INDArray colVector = Nd4j.create(new double[] {10, 20, 30}).reshape(3, 1);

        // Test in-place operations on row view
        rowView.addi(rowVector);
        assertEquals(Nd4j.create(new double[] {14, 25, 36}), rowView);

        // Test in-place operations on column view
        colView.addi(colVector);
        INDArray expected = Nd4j.create(new double[] {12, 45, 38}).reshape(3, 1);
        assertEquals(expected, colView);

        // Verify the original matrix was modified
        INDArray expectedMatrix = Nd4j.create(new double[][] {
                {1, 12, 3},
                {14, 45, 36},
                {7, 38, 9}
        });
        assertEquals(expectedMatrix, matrix);
    }

    @Test
    public void testViewReverseBroadcastOperations() {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {2, 4, 8},
                {16, 32, 64},
                {128, 256, 512}
        });

        // Get views
        INDArray rowView = matrix.getRow(1);  // [16, 32, 64]

        // Create vector for broadcasting
        INDArray vector = Nd4j.create(new double[] {100, 200, 400});

        // Test reverse operations (vector op view)
        INDArray rsubResult = rowView.rsub(vector);  // 100-16, 200-32, 400-64
        INDArray rdivResult = rowView.rdiv(vector);  // 100/16, 200/32, 400/64

        // Verify results
        INDArray expectedRsub = Nd4j.create(new double[] {84, 168, 336});
        INDArray expectedRdiv = Nd4j.create(new double[] {6.25, 6.25, 6.25});

        assertEquals(expectedRsub, rsubResult);
        assertEquals(expectedRdiv, rdivResult);
    }

    @Test
    public void testViewBroadcastWithScalar() {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });

        // Get views
        INDArray rowView = matrix.getRow(1);
        INDArray colView = matrix.getColumn(1);
        INDArray subView = matrix.get(
                NDArrayIndex.interval(0, 2),
                NDArrayIndex.interval(1, 3)
        );

        double scalar = 10.0;

        // Test scalar operations on views
        INDArray rowAddResult = rowView.add(scalar);
        INDArray colMulResult = colView.mul(scalar);
        INDArray subDivResult = subView.div(scalar);

        // Verify results
        INDArray expectedRowAdd = Nd4j.create(new double[] {14, 15, 16});
        INDArray expectedColMul = Nd4j.create(new double[] {20, 50, 80}).reshape(3,1);
        INDArray expectedSubDiv = Nd4j.create(new double[][] {
                {0.2, 0.3},
                {0.5, 0.6}
        });

        assertEquals(expectedRowAdd, rowAddResult);
        assertEquals(expectedColMul, colMulResult);
        assertEquals(expectedSubDiv, subDivResult);
    }

    @Test
    public void testViewBroadcastChaining() {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });

        // Get view
        INDArray rowView = matrix.getRow(1);

        // Create vector
        INDArray vector = Nd4j.create(new double[] {10, 20, 30});

        // Test chained operations
        INDArray result = rowView.add(vector).mul(2).sub(5);

        // Expected: ([4,5,6] + [10,20,30]) * 2 - 5
        INDArray expected = Nd4j.create(new double[] {23, 45, 67});
        assertEquals(expected, result);
    }

}