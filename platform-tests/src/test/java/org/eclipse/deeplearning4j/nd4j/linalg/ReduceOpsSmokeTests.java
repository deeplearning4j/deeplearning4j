
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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ReduceOpsSmokeTests {

    @Test
    public void test2DSumMeanNoDims() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });
        double[][] javaArr = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        double totalSum = 0.0;
        int count = 0;
        for (int i = 0; i < javaArr.length; i++) {
            for (int j = 0; j < javaArr[i].length; j++) {
                totalSum += javaArr[i][j];
                count++;
            }
        }
        double expectedMean = totalSum / count;

        INDArray sumResult = arr.sum();
        INDArray meanResult = arr.mean();

        assertTrue(sumResult.isScalar());
        assertTrue(meanResult.isScalar());
        assertEquals(totalSum, sumResult.getDouble(0), 1e-6);
        assertEquals(expectedMean, meanResult.getDouble(0), 1e-6);
    }

    @Test
    public void test2DSumMeanAlongDimensions() {
        INDArray arr = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });
        double[][] javaArr = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };

        INDArray sumDim0 = arr.sum(0);
        INDArray sumDim1 = arr.sum(1);
        INDArray meanDim0 = arr.mean(0);
        INDArray meanDim1 = arr.mean(1);

        double[] expectedSumDim0 = new double[3];
        double[] expectedSumDim1 = new double[2];
        double[] expectedMeanDim0 = new double[3];
        double[] expectedMeanDim1 = new double[2];

        for (int j = 0; j < 3; j++) {
            double colSum = 0;
            for (int i = 0; i < 2; i++) {
                colSum += javaArr[i][j];
            }
            expectedSumDim0[j] = colSum;
            expectedMeanDim0[j] = colSum / 2.0;
        }

        for (int i = 0; i < 2; i++) {
            double rowSum = 0;
            for (int j = 0; j < 3; j++) {
                rowSum += javaArr[i][j];
            }
            expectedSumDim1[i] = rowSum;
            expectedMeanDim1[i] = rowSum / 3.0;
        }

        for (int j = 0; j < 3; j++) {
            assertEquals(expectedSumDim0[j], sumDim0.getDouble(j), 1e-6);
            assertEquals(expectedMeanDim0[j], meanDim0.getDouble(j), 1e-6);
        }

        for (int i = 0; i < 2; i++) {
            assertEquals(expectedSumDim1[i], sumDim1.getDouble(i), 1e-6);
            assertEquals(expectedMeanDim1[i], meanDim1.getDouble(i), 1e-6);
        }
    }

    @Test
    public void test3DSumMeanNoDims() {
        INDArray arr = Nd4j.create(new double[][][] {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}}
        });
        double totalSum = 0.0;
        int count = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    totalSum += arr.getDouble(i,j,k);
                    count++;
                }
            }
        }
        double expectedMean = totalSum / count;

        INDArray sumResult = arr.sum();
        INDArray meanResult = arr.mean();
        assertTrue(sumResult.isScalar());
        assertTrue(meanResult.isScalar());
        assertEquals(totalSum, sumResult.getDouble(0), 1e-6);
        assertEquals(expectedMean, meanResult.getDouble(0), 1e-6);
    }

    @Test
    public void test3DSumMeanAlongDims() {
        INDArray arr = Nd4j.create(new double[][][] {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}},
                {{9, 10}, {11,12}}
        });

        INDArray sumDim0 = arr.sum(0);
        INDArray sumDim1 = arr.sum(1);
        INDArray sumDim2 = arr.sum(2);
        INDArray meanDim0 = arr.mean(0);
        INDArray meanDim1 = arr.mean(1);
        INDArray meanDim2 = arr.mean(2);

        double[][][] javaArr = new double[3][2][2];
        for (int i = 0; i < 3; i++) {
            for (int j=0; j<2; j++) {
                for (int k=0; k<2; k++) {
                    javaArr[i][j][k] = arr.getDouble(i,j,k);
                }
            }
        }

        double[][] expectedSumDim0 = new double[2][2];
        for (int j=0; j<2; j++) {
            for (int k=0; k<2; k++) {
                double s = 0;
                for (int i=0; i<3; i++) {
                    s += javaArr[i][j][k];
                }
                expectedSumDim0[j][k] = s;
            }
        }

        double[][] expectedSumDim1 = new double[3][2];
        for (int i=0; i<3; i++) {
            for (int k=0; k<2; k++) {
                double s=0;
                for (int j=0; j<2; j++) {
                    s += javaArr[i][j][k];
                }
                expectedSumDim1[i][k] = s;
            }
        }

        double[][] expectedSumDim2 = new double[3][2];
        for (int i=0; i<3; i++) {
            for (int j=0; j<2; j++) {
                double s=0;
                for (int k=0; k<2; k++) {
                    s += javaArr[i][j][k];
                }
                expectedSumDim2[i][j] = s;
            }
        }

        for (int j=0; j<2; j++) {
            for (int k=0; k<2; k++) {
                assertEquals(expectedSumDim0[j][k], sumDim0.getDouble(j,k), 1e-6);
                double meanVal = expectedSumDim0[j][k]/3.0;
                assertEquals(meanVal, meanDim0.getDouble(j,k), 1e-6);
            }
        }

        for (int i=0; i<3; i++) {
            for (int k=0; k<2; k++) {
                assertEquals(expectedSumDim1[i][k], sumDim1.getDouble(i,k), 1e-6);
                double meanVal = expectedSumDim1[i][k]/2.0;
                assertEquals(meanVal, meanDim1.getDouble(i,k),1e-6);
            }
        }

        for (int i=0; i<3; i++) {
            for (int j=0; j<2; j++) {
                assertEquals(expectedSumDim2[i][j], sumDim2.getDouble(i,j),1e-6);
                double meanVal = expectedSumDim2[i][j]/2.0;
                assertEquals(meanVal, meanDim2.getDouble(i,j),1e-6);
            }
        }
    }

    @Test
    public void test4DSumMeanNoDims() {
        INDArray arr = Nd4j.create(DataType.DOUBLE, 2,2,2,2);
        double val = 1.0;
        double totalSum = 0.0;
        int count = 0;
        for (int a=0; a<2; a++) {
            for (int b=0; b<2; b++) {
                for (int c=0; c<2; c++) {
                    for (int d=0; d<2; d++) {
                        arr.putScalar(a,b,c,d,val);
                        totalSum += val;
                        count++;
                        val++;
                    }
                }
            }
        }

        INDArray sumResult = arr.sum();
        INDArray meanResult = arr.mean();
        double expectedMean = totalSum / count;
        assertTrue(sumResult.isScalar());
        assertTrue(meanResult.isScalar());
        assertEquals(totalSum, sumResult.getDouble(0), 1e-6);
        assertEquals(expectedMean, meanResult.getDouble(0), 1e-6);
    }

    @Test
    public void test4DSumMeanAlongDims() {
        INDArray arr = Nd4j.create(DataType.DOUBLE, 2,2,2,2);
        double val = 1.0;
        double[][][][] javaArr = new double[2][2][2][2];
        for (int a=0; a<2; a++) {
            for (int b=0; b<2; b++) {
                for (int c=0; c<2; c++) {
                    for (int d=0; d<2; d++) {
                        arr.putScalar(a,b,c,d,val);
                        javaArr[a][b][c][d]=val;
                        val++;
                    }
                }
            }
        }

        INDArray sumDim0 = arr.sum(0);
        INDArray sumDim1 = arr.sum(1);
        INDArray sumDim2 = arr.sum(2);
        INDArray sumDim3 = arr.sum(3);

        INDArray meanDim0 = arr.mean(0);
        INDArray meanDim1 = arr.mean(1);
        INDArray meanDim2 = arr.mean(2);
        INDArray meanDim3 = arr.mean(3);

        double[][][] expectedSumDim0 = new double[2][2][2];
        for (int b=0; b<2; b++) {
            for (int c=0; c<2; c++) {
                for (int d=0; d<2; d++) {
                    double s=0;
                    for (int a=0; a<2; a++) {
                        s+=javaArr[a][b][c][d];
                    }
                    expectedSumDim0[b][c][d]=s;
                }
            }
        }

        double[][][] expectedSumDim1 = new double[2][2][2];
        for (int a=0; a<2; a++) {
            for (int c=0; c<2; c++) {
                for (int d=0; d<2; d++) {
                    double s=0;
                    for (int b=0; b<2; b++) {
                        s+=javaArr[a][b][c][d];
                    }
                    expectedSumDim1[a][c][d]=s;
                }
            }
        }

        double[][][] expectedSumDim2 = new double[2][2][2];
        for (int a=0; a<2; a++) {
            for (int b=0; b<2; b++) {
                for (int d=0; d<2; d++) {
                    double s=0;
                    for (int c=0; c<2; c++) {
                        s+=javaArr[a][b][c][d];
                    }
                    expectedSumDim2[a][b][d]=s;
                }
            }
        }

        double[][][] expectedSumDim3 = new double[2][2][2];
        for (int a=0; a<2; a++) {
            for (int b=0; b<2; b++) {
                for (int c=0; c<2; c++) {
                    double s=0;
                    for (int d=0; d<2; d++) {
                        s+=javaArr[a][b][c][d];
                    }
                    expectedSumDim3[a][b][c]=s;
                }
            }
        }

        for (int b=0; b<2; b++) {
            for (int c=0; c<2; c++) {
                for (int d=0; d<2; d++) {
                    assertEquals(expectedSumDim0[b][c][d], sumDim0.getDouble(b,c,d),1e-6);
                    assertEquals(expectedSumDim0[b][c][d]/2.0, meanDim0.getDouble(b,c,d),1e-6);
                }
            }
        }

        for (int a=0; a<2; a++) {
            for (int c=0; c<2; c++) {
                for (int d=0; d<2; d++) {
                    assertEquals(expectedSumDim1[a][c][d], sumDim1.getDouble(a,c,d),1e-6);
                    assertEquals(expectedSumDim1[a][c][d]/2.0, meanDim1.getDouble(a,c,d),1e-6);
                }
            }
        }

        for (int a=0; a<2; a++) {
            for (int b=0; b<2; b++) {
                for (int d=0; d<2; d++) {
                    assertEquals(expectedSumDim2[a][b][d], sumDim2.getDouble(a,b,d),1e-6);
                    assertEquals(expectedSumDim2[a][b][d]/2.0, meanDim2.getDouble(a,b,d),1e-6);
                }
            }
        }

        for (int a = 0 ; a < 2; a++) {
            for (int b=0; b < 2; b++) {
                for (int c = 0; c < 2; c++) {
                    assertEquals(expectedSumDim3[a][b][c], sumDim3.getDouble(a,b,c),1e-6);
                    assertEquals(expectedSumDim3[a][b][c]/2.0, meanDim3.getDouble(a,b,c),1e-6);
                }
            }
        }
    }

    @Test
    public void testDupWithDifferentOrders() {
        INDArray arrC = Nd4j.linspace(1,6,6, DataType.DOUBLE).reshape('c',2,3);
        INDArray arrF = arrC.dup('f');
        INDArray arrCsum = arrC.sum();
        INDArray arrFsum = arrF.sum();
        INDArray arrCmean = arrC.mean();
        INDArray arrFmean = arrF.mean();

        double total = 0;
        for (int i=0; i < 2; i++) {
            for (int j=0; j < 3; j++) {
                total+=arrC.getDouble(i,j);
            }
        }

        double mean = total/6.0;
        assertEquals(total, arrCsum.getDouble(0),1e-6);
        assertEquals(total, arrFsum.getDouble(0),1e-6);
        assertEquals(mean, arrCmean.getDouble(0),1e-6);
        assertEquals(mean, arrFmean.getDouble(0),1e-6);
    }

    @Test
    public void testViewReduce2D() {
        INDArray arr = Nd4j.linspace(1,12,12, DataType.DOUBLE).reshape('c',3,4);
        INDArray subView = arr.get(NDArrayIndex.interval(1,3), NDArrayIndex.interval(1,4));
        double[][] javaArr = new double[2][3];
        for (int i = 1; i < 3; i++) {
            for (int j = 1; j < 4; j++) {
                javaArr[i-1][j-1] = arr.getDouble(i,j);
            }
        }

        double sum = 0;
        int count = 0;
        for (int i = 0; i < 2; i++) {
            for (int j=0; j < 3; j++) {
                sum+=javaArr[i][j];
                count++;
            }
        }
        double mean = sum/count;

        INDArray sumResult = subView.sum();
        INDArray meanResult = subView.mean();
        assertEquals(sum, sumResult.getDouble(0),1e-6);
        assertEquals(mean, meanResult.getDouble(0),1e-6);
    }

    @Test
    public void testViewReduce3D() {
        INDArray arr = Nd4j.linspace(1,24,24, DataType.DOUBLE).reshape('c',2,3,4);
        INDArray subView = arr.get(NDArrayIndex.interval(0,2), NDArrayIndex.interval(1,3), NDArrayIndex.interval(2,4));
        double sum = 0;
        int count = 0;
        for (int a = 0; a < 2; a++) {
            for (int b=1; b < 3; b++) {
                for (int c = 2; c < 4; c++) {
                    sum += arr.getDouble(a,b,c);
                    count++;
                }
            }
        }
        double mean=sum/count;

        INDArray sumResult = subView.sum();
        INDArray meanResult=subView.mean();
        assertEquals(sum, sumResult.getDouble(0),1e-6);
        assertEquals(mean, meanResult.getDouble(0),1e-6);
    }

    @Test
    public void testScalarReduce() {
        INDArray scalar = Nd4j.scalar(5.0);
        assertEquals(5.0, scalar.sum().getDouble(0),1e-6);
        assertEquals(5.0, scalar.mean().getDouble(0),1e-6);
    }

    @Test
    public void testAllDimensions() {
        INDArray arr = Nd4j.create(new double[][] {
                {1,2,3},
                {4,5,6}
        });
        INDArray sumAll = arr.sum(0,1);
        INDArray meanAll = arr.mean(0,1);

        double total=0;
        int count=0;
        for (int i=0;i<2;i++) {
            for (int j=0;j<3;j++) {
                total+=arr.getDouble(i,j);
                count++;
            }
        }
        double expectedMean = total/count;
        assertEquals(total, sumAll.getDouble(0),1e-6);
        assertEquals(expectedMean, meanAll.getDouble(0),1e-6);
    }


    @Test
    public void testAllDimensions2DAnd3D() {
        INDArray arr2d = Nd4j.create(new double[][]{{1,2},{3,4}});
        INDArray arr3d = Nd4j.linspace(1,8,8,DataType.DOUBLE).reshape('c',2,2,2);

        INDArray sum2dAll = arr2d.sum(0,1);
        INDArray mean2dAll = arr2d.mean(0,1);
        double total2d=1 + 2 + 3 + 4;
        double mean2d = total2d/4.0;
        assertTrue(sum2dAll.isScalar());
        assertTrue(mean2dAll.isScalar());
        assertEquals(total2d,sum2dAll.getDouble(0),1e-6);
        assertEquals(mean2d,mean2dAll.getDouble(0),1e-6);

        double total3d = 0.0;
        int count3d = 0;
        for (int i = 0;i < 2;i++) {
            for (int j = 0;j < 2;j++) {
                for (int k = 0;k < 2;k++) {
                    total3d+=arr3d.getDouble(i,j,k);
                    count3d++;
                }
            }
        }
        double mean3d = total3d/count3d;
        INDArray sum3dAll = arr3d.sum(0,1,2);
        INDArray mean3dAll = arr3d.mean(0,1,2);
        assertTrue(sum3dAll.isScalar());
        assertTrue(mean3dAll.isScalar());
        assertEquals(total3d,sum3dAll.getDouble(0),1e-6);
        assertEquals(mean3d,mean3dAll.getDouble(0),1e-6);
    }

    @Test
    public void testDifferentOrderDups3D() {
        INDArray arr = Nd4j.create(new double[][][]{
                {{1,2},{3,4}},
                {{5,6},{7,8}}
        });
        INDArray arrF = arr.dup('f');
        INDArray arrC = arr.dup('c');

        INDArray sumF = arrF.sum();
        INDArray sumC = arrC.sum();
        INDArray meanF = arrF.mean();
        INDArray meanC = arrC.mean();

        double total = 0;
        int count=0;
        for (int i = 0;i < 2;i++) {
            for (int j=0;j < 2;j++) {
                for (int k=0;k < 2;k++) {
                    total+=arr.getDouble(i,j,k);
                    count++;
                }
            }
        }
        double m=total/count;

        assertEquals(total,sumF.getDouble(0),1e-6);
        assertEquals(total,sumC.getDouble(0),1e-6);
        assertEquals(m,meanF.getDouble(0),1e-6);
        assertEquals(m,meanC.getDouble(0),1e-6);
    }

    @Test
    public void testViewDupOperations() {
        INDArray arr = Nd4j.linspace(1,12,12,DataType.DOUBLE).reshape('c',3,4);
        INDArray view = arr.get(NDArrayIndex.interval(1,3),NDArrayIndex.interval(0,4));
        INDArray dupC = view.dup('c');
        INDArray dupF = view.dup('f');

        INDArray sumC = dupC.sum();
        INDArray sumF = dupF.sum();
        INDArray meanC = dupC.mean();
        INDArray meanF = dupF.mean();

        double total=0;
        int count=0;
        for (int i=1;i<3;i++){
            for (int j=0;j<4;j++){
                total+=arr.getDouble(i,j);
                count++;
            }
        }
        double m=total/count;

        assertEquals(total,sumC.getDouble(0),1e-6);
        assertEquals(total,sumF.getDouble(0),1e-6);
        assertEquals(m,meanC.getDouble(0),1e-6);
        assertEquals(m,meanF.getDouble(0),1e-6);
    }

    @Test
    public void testScalarFromVariousRanks() {
        INDArray arr1D = Nd4j.create(new double[]{10});
        INDArray arr2D = Nd4j.create(new double[][]{{42}});
        INDArray arr3D = Nd4j.create(new double[][][]{{{7}}});
        INDArray arr4D = Nd4j.create(DataType.DOUBLE,1,1,1,1);
        arr4D.putScalar(0,0,0,0,3.14);

        assertEquals(10.0, arr1D.sum().getDouble(0),1e-6);
        assertEquals(10.0, arr1D.mean().getDouble(0),1e-6);

        assertEquals(42.0, arr2D.sum().getDouble(0),1e-6);
        assertEquals(42.0, arr2D.mean().getDouble(0),1e-6);

        assertEquals(7.0, arr3D.sum().getDouble(0),1e-6);
        assertEquals(7.0, arr3D.mean().getDouble(0),1e-6);

        assertEquals(3.14, arr4D.sum().getDouble(0),1e-6);
        assertEquals(3.14, arr4D.mean().getDouble(0),1e-6);
    }

}
