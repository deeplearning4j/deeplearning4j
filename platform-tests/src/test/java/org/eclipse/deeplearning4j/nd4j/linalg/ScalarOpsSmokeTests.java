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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class ScalarOpsSmokeTests {

    @Test
    public void testBasicScalarOperations() {
        // Create a 2D array
        INDArray arr = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });

        // Create a copy for Java operations
        double[][] javaArr = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                javaArr[i][j] = arr.getDouble(i, j);
            }
        }

        // Test scalar addition
        double scalar = 5.0;
        INDArray addResult = arr.add(scalar);
        
        // Compute expected results for addition
        double[][] expectedAdd = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                expectedAdd[i][j] = javaArr[i][j] + scalar;
            }
        }
        
        // Verify addition results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedAdd[i][j], addResult.getDouble(i, j), 1e-6);
                // Original array should be unchanged
                assertEquals(javaArr[i][j], arr.getDouble(i, j), 1e-6);
            }
        }

        // Test scalar subtraction
        INDArray subResult = arr.sub(scalar);
        
        // Compute expected results for subtraction
        double[][] expectedSub = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                expectedSub[i][j] = javaArr[i][j] - scalar;
            }
        }
        
        // Verify subtraction results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedSub[i][j], subResult.getDouble(i, j), 1e-6);
            }
        }

        // Test scalar multiplication
        INDArray mulResult = arr.mul(scalar);
        
        // Compute expected results for multiplication
        double[][] expectedMul = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                expectedMul[i][j] = javaArr[i][j] * scalar;
            }
        }
        
        // Verify multiplication results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedMul[i][j], mulResult.getDouble(i, j), 1e-6);
            }
        }

        // Test scalar division
        INDArray divResult = arr.div(scalar);
        
        // Compute expected results for division
        double[][] expectedDiv = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                expectedDiv[i][j] = javaArr[i][j] / scalar;
            }
        }
        
        // Verify division results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedDiv[i][j], divResult.getDouble(i, j), 1e-6);
            }
        }
    }

    @Test
    public void testInPlaceScalarOperations() {
        // Create a 2D array
        INDArray arr = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });

        // Create a copy for Java operations
        double[][] javaArr = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                javaArr[i][j] = arr.getDouble(i, j);
            }
        }

        // Test in-place scalar addition
        double scalar = 5.0;
        INDArray addiResult = arr.addi(scalar);

        // Update Java array for in-place addition
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                javaArr[i][j] += scalar;
            }
        }

        // Verify in-place addition results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(javaArr[i][j], addiResult.getDouble(i, j), 1e-6);
                assertEquals(javaArr[i][j], arr.getDouble(i, j), 1e-6);
                assertTrue(arr == addiResult); // Check it's the same object
            }
        }
        
        // Test in-place scalar subtraction
        scalar = 2.0;
        INDArray subiResult = arr.subi(scalar);
        
        // Update Java array for in-place subtraction
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                javaArr[i][j] -= scalar;
            }
        }
        
        // Verify in-place subtraction results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(javaArr[i][j], subiResult.getDouble(i, j), 1e-6);
                assertEquals(javaArr[i][j], arr.getDouble(i, j), 1e-6);
            }
        }
        
        // Test in-place scalar multiplication
        scalar = 3.0;
        INDArray muliResult = arr.muli(scalar);
        
        // Update Java array for in-place multiplication
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                javaArr[i][j] *= scalar;
            }
        }
        
        // Verify in-place multiplication results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(javaArr[i][j], muliResult.getDouble(i, j), 1e-6);
                assertEquals(javaArr[i][j], arr.getDouble(i, j), 1e-6);
            }
        }
        
        // Test in-place scalar division
        scalar = 4.0;
        INDArray diviResult = arr.divi(scalar);
        
        // Update Java array for in-place division
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                javaArr[i][j] /= scalar;
            }
        }
        
        // Verify in-place division results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(javaArr[i][j], diviResult.getDouble(i, j), 1e-6);
                assertEquals(javaArr[i][j], arr.getDouble(i, j), 1e-6);
            }
        }
    }

    @Test
    public void testReverseScalarOperations() {
        // Create a 2D array
        INDArray arr = Nd4j.create(new double[][] {
                {2, 4, 6},
                {8, 10, 12}
        });

        // Create a copy for Java operations
        double[][] javaArr = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                javaArr[i][j] = arr.getDouble(i, j);
            }
        }

        // Test scalar reverse subtraction (scalar - arr)
        double scalar = 10.0;
        INDArray rsubResult = arr.rsub(scalar);

        // Compute expected results for reverse subtraction
        double[][] expectedRsub = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                expectedRsub[i][j] = scalar - javaArr[i][j];
            }
        }

        // Verify reverse subtraction results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedRsub[i][j], rsubResult.getDouble(i, j), 1e-6);
                // Original array should be unchanged
                assertEquals(javaArr[i][j], arr.getDouble(i, j), 1e-6);
            }
        }

        // Test scalar reverse division (scalar / arr)
        INDArray rdivResult = arr.rdiv(scalar);

        // Compute expected results for reverse division
        double[][] expectedRdiv = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                expectedRdiv[i][j] = scalar / javaArr[i][j];
            }
        }

        // Verify reverse division results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedRdiv[i][j], rdivResult.getDouble(i, j), 1e-6);
                // Original array should be unchanged
                assertEquals(javaArr[i][j], arr.getDouble(i, j), 1e-6);
            }
        }

        // Test in-place reverse operations
        INDArray rsubiResult = arr.rsubi(20.0);

        // Update Java array for in-place reverse subtraction
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                javaArr[i][j] = 20.0 - javaArr[i][j];
            }
        }

        // Verify in-place reverse subtraction
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(javaArr[i][j], rsubiResult.getDouble(i, j), 1e-6);
                assertEquals(javaArr[i][j], arr.getDouble(i, j), 1e-6);
                assertTrue(arr == rsubiResult);
            }
        }
    }

    @Test
    public void testScalarOpsWithViews() {
        // Create a 3D array
        INDArray arr = Nd4j.create(new double[][][] {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}}
        });

        // Create a slice view
        INDArray view = arr.slice(1); // Second slice, shape [2,2]

        // Create Java array equivalent of the view
        double[][] javaView = new double[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                javaView[i][j] = view.getDouble(i, j);
            }
        }

        // Test scalar addition on view
        double scalar = 10.0;
        INDArray addResult = view.add(scalar);

        // Compute expected result
        double[][] expectedAdd = new double[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                expectedAdd[i][j] = javaView[i][j] + scalar;
            }
        }

        // Verify results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expectedAdd[i][j], addResult.getDouble(i, j), 1e-6);
                // Original view should be unchanged
                assertEquals(javaView[i][j], view.getDouble(i, j), 1e-6);
            }
        }

        // Test in-place scalar multiplication on view
        scalar = 2.0;
        INDArray muliResult = view.muli(scalar);

        // Update Java array for in-place operations
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                javaView[i][j] *= scalar;
            }
        }

        // Verify results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(javaView[i][j], muliResult.getDouble(i, j), 1e-6);
                // View should be modified
                assertEquals(javaView[i][j], view.getDouble(i, j), 1e-6);
                // Original array should reflect the changes
                assertEquals(javaView[i][j], arr.getDouble(1, i, j), 1e-6);
            }
        }
    }

    @Test
    public void testScalarOpsWithReshapedArray() {
        // Create a 1D array
        INDArray arr1D = Nd4j.linspace(1, 6, 6, DataType.DOUBLE);

        // Reshape to 2D
        INDArray arr2D = arr1D.reshape(2, 3);

        // Create Java array equivalent
        double[][] java2D = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                java2D[i][j] = arr2D.getDouble(i, j);
            }
        }

        // Test scalar addition
        double scalar = 5.0;
        INDArray addResult = arr2D.add(scalar);

        // Compute expected result
        double[][] expectedAdd = new double[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                expectedAdd[i][j] = java2D[i][j] + scalar;
            }
        }

        // Verify results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedAdd[i][j], addResult.getDouble(i, j), 1e-6);
            }
        }

        // Test in-place operation affects both original and reshaped arrays
        arr1D.addi(10);

        // Update Java array
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                java2D[i][j] += 10;
            }
        }

        // Verify both arrays are updated
        for (int i = 0; i < 6; i++) {
            assertEquals(i + 1 + 10, arr1D.getDouble(i), 1e-6);
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(java2D[i][j], arr2D.getDouble(i, j), 1e-6);
            }
        }
    }

    @Test
    public void testScalarOpsWithPermutedArray() {
        // Create a 2D array
        INDArray arr = Nd4j.create(new double[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        // Permute the array to swap dimensions
        INDArray permuted = arr.permute(1, 0); // Now shape [3,2]

        // Create Java array for the permuted array
        double[][] javaPermuted = new double[3][2];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                javaPermuted[i][j] = permuted.getDouble(i, j);
            }
        }

        // Test scalar subtraction on permuted array
        double scalar = 3.0;
        INDArray subResult = permuted.sub(scalar);

        // Compute expected result
        double[][] expectedSub = new double[3][2];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                expectedSub[i][j] = javaPermuted[i][j] - scalar;
            }
        }

        // Verify results
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expectedSub[i][j], subResult.getDouble(i, j), 1e-6);
            }
        }

        // Test in-place operations on the permuted array affect the original
        permuted.addi(10);

        // Update Java array
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                javaPermuted[i][j] += 10;
            }
        }

        // Verify permuted array is updated
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(javaPermuted[i][j], permuted.getDouble(i, j), 1e-6);
            }
        }

        // Verify original array is updated (note: indices are swapped)
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(permuted.getDouble(j, i), arr.getDouble(i, j), 1e-6);
            }
        }
    }

    @Test
    public void testScalarOpsWithSubViews() {
        // Create a 3D array
        INDArray arr = Nd4j.create(DataType.DOUBLE, 3, 4, 5);

        // Initialize with increasing values
        double value = 1.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 5; k++) {
                    arr.putScalar(i, j, k, value++);
                }
            }
        }

        // Get a subview of the array
        INDArray subView = arr.get(
                NDArrayIndex.interval(1, 3),
                NDArrayIndex.interval(1, 3),
                NDArrayIndex.interval(1, 4)
        );

        // Create Java array for validation
        double[][][] javaSubView = new double[2][2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    javaSubView[i][j][k] = subView.getDouble(i, j, k);
                }
            }
        }

        // Test scalar multiplication on subview
        double scalar = 2.0;
        INDArray mulResult = subView.mul(scalar);

        // Compute expected result
        double[][][] expectedMul = new double[2][2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    expectedMul[i][j][k] = javaSubView[i][j][k] * scalar;
                }
            }
        }

        // Verify results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(expectedMul[i][j][k], mulResult.getDouble(i, j, k), 1e-6);
                }
            }
        }

        // Test in-place scalar addition on subview
        scalar = 10.0;
        INDArray addiResult = subView.addi(scalar);

        // Update Java array for in-place operations
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    javaSubView[i][j][k] += scalar;
                }
            }
        }

        // Verify results
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    assertEquals(javaSubView[i][j][k], addiResult.getDouble(i, j, k), 1e-6);
                    // Subview should be modified
                    assertEquals(javaSubView[i][j][k], subView.getDouble(i, j, k), 1e-6);
                    // Original array should reflect the changes
                    assertEquals(javaSubView[i][j][k], arr.getDouble(i+1, j+1, k+1), 1e-6);
                }
            }
        }
    }

}
