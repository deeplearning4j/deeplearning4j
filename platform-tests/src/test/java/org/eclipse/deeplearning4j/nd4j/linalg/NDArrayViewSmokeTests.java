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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.*;

public class NDArrayViewSmokeTests {

    @Test
    public void testReshapeAssignmentsSimplified() {
        // Create an array of shape (4,4)
        INDArray arr = Nd4j.linspace(1, 16, 16).reshape('c', 4, 4);

        // Create a corresponding Java array
        double[][] arrJava = new double[4][4];

        // Initialize arrJava with values from arr
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                arrJava[i][j] = arr.getDouble(i, j);
            }
        }

        // Reshape the array to (2,8)
        INDArray reshaped = arr.reshape('c', 2, 8);

        // Create a Java array corresponding to reshaped
        double[][] reshapedJava = new double[2][8];

        // Initialize reshapedJava with values from reshaped
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 8; j++) {
                reshapedJava[i][j] = reshaped.getDouble(i, j);
            }
        }

        // Modify elements in reshaped array
        reshaped.putScalar(0, 0, 1000); // Corresponds to arr(0,0)
        reshaped.putScalar(1, 7, 2000); // Corresponds to arr(3,3)

        // Modify reshapedJava
        reshapedJava[0][0] = 1000;
        reshapedJava[1][7] = 2000;

        // Update arrJava accordingly
        arrJava[0][0] = 1000;
        arrJava[3][3] = 2000;

        // Print reshapedJava after modification
        System.out.println("reshapedJava after modifications:");
        for (int i = 0; i < reshapedJava.length; i++) {
            for (int j = 0; j < reshapedJava[i].length; j++) {
                System.out.print(reshapedJava[i][j] + " ");
            }
            System.out.println();
        }

        // Print arrJava after reshaped modifications
        System.out.println("arrJava after reshaped modifications:");
        for (int i = 0; i < arrJava.length; i++) {
            for (int j = 0; j < arrJava[i].length; j++) {
                System.out.print(arrJava[i][j] + " ");
            }
            System.out.println();
        }

        // Check that arrJava matches arr using getDouble
        assertEquals(arrJava[0][0], arr.getDouble(0, 0));
        assertEquals(arrJava[3][3], arr.getDouble(3, 3));

        // Check that arr matches arrJava using NDArrayIndex
        assertEquals(arrJava[0][0], arr.get(NDArrayIndex.point(0), NDArrayIndex.point(0)).getDouble());
        assertEquals(arrJava[3][3], arr.get(NDArrayIndex.point(3), NDArrayIndex.point(3)).getDouble());

        // Modify original array
        arr.putScalar(1, 1, 3000);

        // Update arrJava
        arrJava[1][1] = 3000;

        // Since arr(1,1) corresponds to linear index 5 in row-major order
        // In reshaped array (2,8), this linear index maps to (0,5)
        // Update reshapedJava
        reshapedJava[0][5] = 3000;

        // Print arrJava after arr modification
        System.out.println("arrJava after arr modification:");
        for (int i = 0; i < arrJava.length; i++) {
            for (int j = 0; j < arrJava[i].length; j++) {
                System.out.print(arrJava[i][j] + " ");
            }
            System.out.println();
        }

        // Print reshapedJava after arr modification
        System.out.println("reshapedJava after arr modification:");
        for (int i = 0; i < reshapedJava.length; i++) {
            for (int j = 0; j < reshapedJava[i].length; j++) {
                System.out.print(reshapedJava[i][j] + " ");
            }
            System.out.println();
        }

        // Check that reshapedJava matches reshaped using getDouble
        assertEquals(reshapedJava[0][5], reshaped.getDouble(0, 5));

        // Check that reshaped matches reshapedJava using NDArrayIndex
        assertEquals(reshapedJava[0][5], reshaped.get(NDArrayIndex.point(0), NDArrayIndex.point(5)).getDouble());

        // Check that reshaped view reflects the change
        assertEquals(3000, reshaped.getDouble(0, 5));
    }

    @Test
    public void testPermuteAssignmentsSimplified() {
        // Create an array of shape (3,3,3)
        INDArray arr = Nd4j.linspace(1, 27, 27).reshape('c', 3, 3, 3);

        // Create corresponding Java array
        double[][][] arrJava = new double[3][3][3];

        // Initialize arrJava
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    arrJava[i][j][k] = arr.getDouble(i, j, k);
                }
            }
        }

        // Permute the array to rearrange dimensions
        INDArray permuted = arr.permute(2, 1, 0);

        // Create Java array for permuted
        double[][][] permutedJava = new double[3][3][3];

        // Initialize permutedJava
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                for (int c = 0; c < 3; c++) {
                    permutedJava[a][b][c] = permuted.getDouble(a, b, c);
                }
            }
        }

        // Modify elements in permuted array
        permuted.putScalar(0, 0, 0, 1000); // Corresponds to arr(0,0,0)
        permuted.putScalar(2, 2, 2, 2000); // Corresponds to arr(2,2,2)

        // Modify permutedJava
        permutedJava[0][0][0] = 1000;
        permutedJava[2][2][2] = 2000;

        // Update arrJava accordingly
        arrJava[0][0][0] = 1000;
        arrJava[2][2][2] = 2000;

        // Print permutedJava after modification
        System.out.println("permutedJava after modifications:");
        for (int a = 0; a < permutedJava.length; a++) {
            for (int b = 0; b < permutedJava[a].length; b++) {
                for (int c = 0; c < permutedJava[a][b].length; c++) {
                    System.out.print(permutedJava[a][b][c] + " ");
                }
                System.out.print(" | ");
            }
            System.out.println();
        }

        // Print arrJava after permuted modifications
        System.out.println("arrJava after permuted modifications:");
        for (int i = 0; i < arrJava.length; i++) {
            for (int j = 0; j < arrJava[i].length; j++) {
                for (int k = 0; k < arrJava[i][j].length; k++) {
                    System.out.print(arrJava[i][j][k] + " ");
                }
                System.out.print(" | ");
            }
            System.out.println();
        }

        // Check that arrJava matches arr using getDouble
        assertEquals(arrJava[0][0][0], arr.getDouble(0, 0, 0));
        assertEquals(arrJava[2][2][2], arr.getDouble(2, 2, 2));

        // Check that arr matches arrJava using NDArrayIndex
        assertEquals(arrJava[0][0][0], arr.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0)).getDouble());
        assertEquals(arrJava[2][2][2], arr.get(NDArrayIndex.point(2), NDArrayIndex.point(2), NDArrayIndex.point(2)).getDouble());

        // Modify original array
        arr.putScalar(1, 1, 1, 3000);

        // Update arrJava
        arrJava[1][1][1] = 3000;

        // Update permutedJava
        permutedJava[1][1][1] = 3000;

        // Print arrJava after arr modification
        System.out.println("arrJava after arr modification:");
        for (int i = 0; i < arrJava.length; i++) {
            for (int j = 0; j < arrJava[i].length; j++) {
                for (int k = 0; k < arrJava[i][j].length; k++) {
                    System.out.print(arrJava[i][j][k] + " ");
                }
                System.out.print(" | ");
            }
            System.out.println();
        }

        // Print permutedJava after arr modification
        System.out.println("permutedJava after arr modification:");
        for (int a = 0; a < permutedJava.length; a++) {
            for (int b = 0; b < permutedJava[a].length; b++) {
                for (int c = 0; c < permutedJava[a][b].length; c++) {
                    System.out.print(permutedJava[a][b][c] + " ");
                }
                System.out.print(" | ");
            }
            System.out.println();
        }

        // Check that permutedJava matches permuted using getDouble
        assertEquals(permutedJava[1][1][1], permuted.getDouble(1, 1, 1));

        // Check that permuted matches permutedJava using NDArrayIndex
        assertEquals(permutedJava[1][1][1], permuted.get(NDArrayIndex.point(1), NDArrayIndex.point(1), NDArrayIndex.point(1)).getDouble());

        // Check that permuted is updated accordingly
        assertEquals(3000, permuted.getDouble(1, 1, 1));
    }

    @Test
    public void testViewModificationReflectsInOriginal() {
        // Create an array of shape (3,3)
        INDArray arr = Nd4j.linspace(1, 9, 9).reshape('c', 3, 3);

        // Create corresponding Java array
        double[][] arrJava = new double[3][3];

        // Initialize arrJava
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                arrJava[i][j] = arr.getDouble(i, j);
            }
        }

        // Create a view by selecting a subarray
        INDArray view = arr.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 3));

        // Create corresponding Java array for view
        double[][] viewJava = new double[2][2];

        // Initialize viewJava
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                viewJava[i][j] = view.getDouble(i, j);
            }
        }

        // Modify the view
        view.putScalar(0, 0, 100);
        view.putScalar(1, 1, 200);

        // Modify viewJava
        viewJava[0][0] = 100;
        viewJava[1][1] = 200;

        // Update arrJava accordingly
        arrJava[1][1] = 100;
        arrJava[2][2] = 200;

        // Print viewJava after modification
        System.out.println("viewJava after modifications:");
        for (int i = 0; i < viewJava.length; i++) {
            for (int j = 0; j < viewJava[i].length; j++) {
                System.out.print(viewJava[i][j] + " ");
            }
            System.out.println();
        }

        // Print arrJava after view modifications
        System.out.println("arrJava after view modifications:");
        for (int i = 0; i < arrJava.length; i++) {
            for (int j = 0; j < arrJava[i].length; j++) {
                System.out.print(arrJava[i][j] + " ");
            }
            System.out.println();
        }

        // Check that arrJava matches arr using getDouble
        assertEquals(arrJava[1][1], arr.getDouble(1, 1));
        assertEquals(arrJava[2][2], arr.getDouble(2, 2));

        // Check that arr matches arrJava using NDArrayIndex
        assertEquals(arrJava[1][1], arr.get(NDArrayIndex.point(1), NDArrayIndex.point(1)).getDouble());
        assertEquals(arrJava[2][2], arr.get(NDArrayIndex.point(2), NDArrayIndex.point(2)).getDouble());

        // Modify original array
        arr.putScalar(0, 0, 300);

        // Update arrJava
        arrJava[0][0] = 300;

        // Print arrJava after arr modification
        System.out.println("arrJava after arr modification:");
        for (int i = 0; i < arrJava.length; i++) {
            for (int j = 0; j < arrJava[i].length; j++) {
                System.out.print(arrJava[i][j] + " ");
            }
            System.out.println();
        }

        // Check that view is unaffected at this index
        // Since arr(0,0) is not part of the view
        assertEquals(300, arr.getDouble(0, 0));
        assertEquals(300, arr.get(NDArrayIndex.point(0), NDArrayIndex.point(0)).getDouble());

        // Modify an element in arr that is in the view
        arr.putScalar(1, 1, 400);

        // Update arrJava
        arrJava[1][1] = 400;

        // Update viewJava
        viewJava[0][0] = 400;

        // Print viewJava after arr modification
        System.out.println("viewJava after arr modification:");
        for (int i = 0; i < viewJava.length; i++) {
            for (int j = 0; j < viewJava[i].length; j++) {
                System.out.print(viewJava[i][j] + " ");
            }
            System.out.println();
        }

        // Check that view reflects the change using getDouble
        assertEquals(400, view.getDouble(0, 0));

        // Check that view matches viewJava using NDArrayIndex
        assertEquals(viewJava[0][0], view.get(NDArrayIndex.point(0), NDArrayIndex.point(0)).getDouble());
    }

    @Test
    public void testSlicingAssignmentsSimplified() {
        // Create an array of shape (3,3,3)
        INDArray arr = Nd4j.linspace(1, 27, 27).reshape('c', 3, 3, 3);

        // Create corresponding Java array
        double[][][] arrJava = new double[3][3][3];

        // Initialize arrJava
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    arrJava[i][j][k] = arr.getDouble(i, j, k);
                }
            }
        }

        // Take a slice along the first dimension
        INDArray slice = arr.slice(1); // Corresponds to arr(1,:,:)

        // Create corresponding Java array for slice
        double[][] sliceJava = new double[3][3];

        // Initialize sliceJava
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                sliceJava[j][k] = slice.getDouble(j, k);
            }
        }

        // Modify elements in the slice
        slice.putScalar(0, 0, 1000); // Corresponds to arr(1,0,0)
        slice.putScalar(2, 2, 2000); // Corresponds to arr(1,2,2)

        // Modify sliceJava
        sliceJava[0][0] = 1000;
        sliceJava[2][2] = 2000;

        // Update arrJava accordingly
        arrJava[1][0][0] = 1000;
        arrJava[1][2][2] = 2000;

        // Print sliceJava after modification
        System.out.println("sliceJava after modifications:");
        for (int j = 0; j < sliceJava.length; j++) {
            for (int k = 0; k < sliceJava[j].length; k++) {
                System.out.print(sliceJava[j][k] + " ");
            }
            System.out.println();
        }

        // Print arrJava after slice modifications
        System.out.println("arrJava after slice modifications:");
        for (int i = 0; i < arrJava.length; i++) {
            for (int j = 0; j < arrJava[i].length; j++) {
                for (int k = 0; k < arrJava[i][j].length; k++) {
                    System.out.print(arrJava[i][j][k] + " ");
                }
                System.out.print(" | ");
            }
            System.out.println();
        }

        // Check that arrJava matches arr using getDouble
        assertEquals(arrJava[1][0][0], arr.getDouble(1, 0, 0));
        assertEquals(arrJava[1][2][2], arr.getDouble(1, 2, 2));

        // Check that arr matches arrJava using NDArrayIndex
        assertEquals(arrJava[1][0][0], arr.get(NDArrayIndex.point(1), NDArrayIndex.point(0), NDArrayIndex.point(0)).getDouble());
        assertEquals(arrJava[1][2][2], arr.get(NDArrayIndex.point(1), NDArrayIndex.point(2), NDArrayIndex.point(2)).getDouble());

        // Modify original array
        arr.putScalar(1, 1, 1, 3000);

        // Update arrJava
        arrJava[1][1][1] = 3000;

        // Update sliceJava
        sliceJava[1][1] = 3000;

        // Print arrJava after arr modification
        System.out.println("arrJava after arr modification:");
        for (int i = 0; i < arrJava.length; i++) {
            for (int j = 0; j < arrJava[i].length; j++) {
                for (int k = 0; k < arrJava[i][j].length; k++) {
                    System.out.print(arrJava[i][j][k] + " ");
                }
                System.out.print(" | ");
            }
            System.out.println();
        }

        // Print sliceJava after arr modification
        System.out.println("sliceJava after arr modification:");
        for (int j = 0; j < sliceJava.length; j++) {
            for (int k = 0; k < sliceJava[j].length; k++) {
                System.out.print(sliceJava[j][k] + " ");
            }
            System.out.println();
        }

        // Check that sliceJava matches slice using getDouble
        assertEquals(sliceJava[1][1], slice.getDouble(1, 1));

        // Check that slice matches sliceJava using NDArrayIndex
        assertEquals(sliceJava[1][1], slice.get(NDArrayIndex.point(1), NDArrayIndex.point(1)).getDouble());

        // Check that slice is updated
        assertEquals(3000, slice.getDouble(1, 1));
    }

    @Test
    public void testAdvancedIndexingAssignmentsSimplified() {
        // Create an array of shape (5,5)
        INDArray arr = Nd4j.linspace(1, 25, 25).reshape('c', 5, 5);

        // Create corresponding Java array
        double[][] arrJava = new double[5][5];

        // Initialize arrJava
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                arrJava[i][j] = arr.getDouble(i, j);
            }
        }

        // Use advanced indexing to get a subarray
        INDArrayIndex[] indices = new INDArrayIndex[]{
                NDArrayIndex.interval(1, 4),
                NDArrayIndex.interval(1, 4)
        };
        INDArray subArr = arr.get(indices);

        // Create corresponding Java array for subArr
        double[][] subArrJava = new double[3][3];

        // Initialize subArrJava
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                subArrJava[i][j] = subArr.getDouble(i, j);
            }
        }

        // Modify elements in subArr
        subArr.putScalar(0, 0, 1000); // Corresponds to arr(1,1)
        subArr.putScalar(2, 2, 2000); // Corresponds to arr(3,3)

        // Modify subArrJava
        subArrJava[0][0] = 1000;
        subArrJava[2][2] = 2000;

        // Update arrJava accordingly
        arrJava[1][1] = 1000;
        arrJava[3][3] = 2000;

        // Print subArrJava after modification
        System.out.println("subArrJava after modifications:");
        for (int i = 0; i < subArrJava.length; i++) {
            for (int j = 0; j < subArrJava[i].length; j++) {
                System.out.print(subArrJava[i][j] + " ");
            }
            System.out.println();
        }

        // Print arrJava after subArr modifications
        System.out.println("arrJava after subArr modifications:");
        for (int i = 0; i < arrJava.length; i++) {
            for (int j = 0; j < arrJava[i].length; j++) {
                System.out.print(arrJava[i][j] + " ");
            }
            System.out.println();
        }

        // Check that arrJava matches arr using getDouble
        assertEquals(arrJava[1][1], arr.getDouble(1, 1));
        assertEquals(arrJava[3][3], arr.getDouble(3, 3));

        // Check that arr matches arrJava using NDArrayIndex
        assertEquals(arrJava[1][1], arr.get(NDArrayIndex.point(1), NDArrayIndex.point(1)).getDouble());
        assertEquals(arrJava[3][3], arr.get(NDArrayIndex.point(3), NDArrayIndex.point(3)).getDouble());

        // Modify original array
        arr.putScalar(2, 2, 3000);

        // Update arrJava
        arrJava[2][2] = 3000;

        // Update subArrJava
        subArrJava[1][1] = 3000;

        // Print arrJava after arr modification
        System.out.println("arrJava after arr modification:");
        for (int i = 0; i < arrJava.length; i++) {
            for (int j = 0; j < arrJava[i].length; j++) {
                System.out.print(arrJava[i][j] + " ");
            }
            System.out.println();
        }

        // Print subArrJava after arr modification
        System.out.println("subArrJava after arr modification:");
        for (int i = 0; i < subArrJava.length; i++) {
            for (int j = 0; j < subArrJava[i].length; j++) {
                System.out.print(subArrJava[i][j] + " ");
            }
            System.out.println();
        }

        // Check that subArrJava matches subArr using getDouble
        assertEquals(subArrJava[1][1], subArr.getDouble(1, 1));

        // Check that subArr matches subArrJava using NDArrayIndex
        assertEquals(subArrJava[1][1], subArr.get(NDArrayIndex.point(1), NDArrayIndex.point(1)).getDouble());

        // Check that subArr is updated
        assertEquals(3000, subArr.getDouble(1, 1));
    }

    @Test
    public void testAssignFromView() {
        // Create an array of shape (4,4)
        INDArray arr = Nd4j.linspace(1, 16, 16).reshape('c', 4, 4);

        // Create corresponding Java array
        double[][] arrJava = new double[4][4];

        // Initialize arrJava with values from arr
        for (int i = 0; i < arr.rows(); i++) {
            for (int j = 0; j < arr.columns(); j++) {
                arrJava[i][j] = arr.getDouble(i, j);
            }
        }

        // Create a view by selecting a subarray
        INDArray view = arr.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 3));

        // Create corresponding Java array for view
        double[][] viewJava = new double[view.rows()][view.columns()];

        // Initialize viewJava with values from view
        for (int i = 0; i < view.rows(); i++) {
            for (int j = 0; j < view.columns(); j++) {
                viewJava[i][j] = view.getDouble(i, j);
            }
        }

        // Create another array to assign from
        INDArray toAssign = Nd4j.ones(2, 2).mul(100);

        // Create corresponding Java array for toAssign
        double[][] toAssignJava = new double[2][2];
        for (int i = 0; i < toAssign.rows(); i++) {
            for (int j = 0; j < toAssign.columns(); j++) {
                toAssignJava[i][j] = toAssign.getDouble(i, j);
            }
        }

        // Update viewJava
        for (int i = 0; i < viewJava.length; i++) {
            for (int j = 0; j < viewJava[i].length; j++) {
                viewJava[i][j] = toAssignJava[i][j];
            }
        }

        // Since view is a view into arr, update arrJava accordingly
        for (int i = 0; i < viewJava.length; i++) {
            for (int j = 0; j < viewJava[i].length; j++) {
                arrJava[i + 1][j + 1] = viewJava[i][j];
            }
        }


        // Perform assignment
        view.assign(toAssign);


        // Check that the original array is updated
        assertEquals(100, arr.getDouble(1, 1));
        assertEquals(100, arr.getDouble(1, 2));
        assertEquals(100, arr.getDouble(2, 1));
        assertEquals(100, arr.getDouble(2, 2));

        // Check that arrJava matches arr using getDouble
        for (int i = 0; i < arr.rows(); i++) {
            for (int j = 0; j < arr.columns(); j++) {
                assertEquals(arrJava[i][j], arr.getDouble(i, j));
                assertEquals(arrJava[i][j], arr.get(NDArrayIndex.point(i), NDArrayIndex.point(j)).getDouble());
            }
        }

        // Print the updated arrJava
        System.out.println("arrJava after assign from view:");
        for (int i = 0; i < arrJava.length; i++) {
            for (int j = 0; j < arrJava[i].length; j++) {
                System.out.print(arrJava[i][j] + " ");
            }
            System.out.println();
        }

        // Print the updated viewJava
        System.out.println("viewJava after assign:");
        for (int i = 0; i < viewJava.length; i++) {
            for (int j = 0; j < viewJava[i].length; j++) {
                System.out.print(viewJava[i][j] + " ");
            }
            System.out.println();
        }
    }


    @Test
    public void testDupFromDifferentOrders() {
        // Create arrays in 'c' and 'f' orders
        INDArray arrC = Nd4j.linspace(1, 9, 9).reshape('c', 3, 3);
        INDArray arrF = Nd4j.linspace(1, 9, 9).reshape('f', 3, 3);

        // Create corresponding Java arrays
        double[][] arrCJava = new double[3][3];
        double[][] arrFJava = new double[3][3];

        // Initialize arrCJava and arrFJava with values from arrC and arrF
        for (int i = 0; i < arrC.rows(); i++) {
            for (int j = 0; j < arrC.columns(); j++) {
                arrCJava[i][j] = arrC.getDouble(i, j);
                arrFJava[i][j] = arrF.getDouble(i, j);
            }
        }

        // Duplicate arrays
        INDArray dupC = arrC.dup('c');
        INDArray dupF = arrF.dup('f');

        // Create corresponding Java arrays for duplicates
        double[][] dupCJava = new double[3][3];
        double[][] dupFJava = new double[3][3];

        // Initialize dupCJava and dupFJava with values from dupC and dupF
        for (int i = 0; i < dupC.rows(); i++) {
            for (int j = 0; j < dupC.columns(); j++) {
                dupCJava[i][j] = dupC.getDouble(i, j);
                dupFJava[i][j] = dupF.getDouble(i, j);
            }
        }

        // Modify duplicates
        dupC.putScalar(0, 0, 1000);
        dupF.putScalar(2, 2, 2000);

        // Update dupCJava and dupFJava
        dupCJava[0][0] = 1000;
        dupFJava[2][2] = 2000;

        // Check that originals are unaffected
        assertEquals(1, arrC.getDouble(0, 0));
        assertEquals(9, arrF.getDouble(2, 2));

        // Check that arrCJava and arrFJava match arrC and arrF
        for (int i = 0; i < arrC.rows(); i++) {
            for (int j = 0; j < arrC.columns(); j++) {
                assertEquals(arrCJava[i][j], arrC.getDouble(i, j));
                assertEquals(arrCJava[i][j], arrC.get(NDArrayIndex.point(i), NDArrayIndex.point(j)).getDouble());
                assertEquals(arrFJava[i][j], arrF.getDouble(i, j));
                assertEquals(arrFJava[i][j], arrF.get(NDArrayIndex.point(i), NDArrayIndex.point(j)).getDouble());
            }
        }

        // Print duplicates
        System.out.println("dupCJava after modification:");
        for (int i = 0; i < dupCJava.length; i++) {
            for (int j = 0; j < dupCJava[i].length; j++) {
                System.out.print(dupCJava[i][j] + " ");
            }
            System.out.println();
        }

        System.out.println("dupFJava after modification:");
        for (int i = 0; i < dupFJava.length; i++) {
            for (int j = 0; j < dupFJava[i].length; j++) {
                System.out.print(dupFJava[i][j] + " ");
            }
            System.out.println();
        }

        // Check that dupCJava and dupFJava match dupC and dupF
        for (int i = 0; i < dupC.rows(); i++) {
            for (int j = 0; j < dupC.columns(); j++) {
                assertEquals(dupCJava[i][j], dupC.getDouble(i, j));
                assertEquals(dupCJava[i][j], dupC.get(NDArrayIndex.point(i), NDArrayIndex.point(j)).getDouble());
                assertEquals(dupFJava[i][j], dupF.getDouble(i, j));
                assertEquals(dupFJava[i][j], dupF.get(NDArrayIndex.point(i), NDArrayIndex.point(j)).getDouble());
            }
        }
    }
}
