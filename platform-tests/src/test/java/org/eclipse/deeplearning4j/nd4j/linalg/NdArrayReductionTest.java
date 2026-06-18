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

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.util.FastMath;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.common.util.MathUtils;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.bool.All;
import org.nd4j.linalg.api.ops.impl.reduce.custom.LogSumExp;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2;
import org.nd4j.linalg.api.ops.impl.reduce.same.Sum;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Focused test class for NDArray reduction operations extracted from Nd4jTestsC.
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
public class NdArrayReductionTest extends BaseNd4jTestWithBackends {

    @Override
    public long getTimeoutMilliseconds() {
        return 90000;
    }

    @BeforeEach
    public void before() throws Exception {
        Nd4j.getRandom().setSeed(123);
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLength(Nd4jBackend backend) {
        INDArray values = Nd4j.create(2, 2);
        INDArray values2 = Nd4j.create(2, 2);

        values.put(0, 0, 0);
        values2.put(0, 0, 2);
        values.put(1, 0, 0);
        values2.put(1, 0, 2);
        values.put(0, 1, 0);
        values2.put(0, 1, 0);
        values.put(1, 1, 2);
        values2.put(1, 1, 2);

        INDArray expected = Nd4j.repeat(Nd4j.scalar(DataType.DOUBLE, 2).reshape(1, 1), 2).reshape(2);

        val accum = new EuclideanDistance(values, values2);
        accum.setDimensions(1);

        INDArray results = Nd4j.getExecutioner().exec(accum);
        assertEquals(expected, results);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumAlongDim1sEdgeCases(Nd4jBackend backend) {
        val shapes = new long[][] {
                //Standard case:
                {2, 2, 3, 4},
                //Leading 1s:
                {1, 2, 3, 4}, {1, 1, 2, 3},
                //Trailing 1s:
                {4, 3, 2, 1}, {4, 3, 1, 1},
                //1s for non-leading/non-trailing dimensions
                {4, 1, 3, 2}, {4, 3, 1, 2}, {4, 1, 1, 2}};

        long[][] sumDims = {{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3},
                {0, 1, 2, 3}};

        for (val shape : shapes) {
            for (long[] dims : sumDims) {
                int length = ArrayUtil.prod(shape);
                INDArray inC = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape);
                INDArray inF = inC.dup('f');
                assertEquals(inC, inF);

                INDArray sumC = inC.sum(dims);
                INDArray sumF = inF.sum(dims);
                assertEquals(sumC, sumF);

                //Multiple runs: check for consistency between runs (threading issues, etc)
                for (int i = 0; i < 100; i++) {
                    assertEquals(sumC, inC.sum(dims));
                    assertEquals(sumF, inF.sum(dims));
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiSum(Nd4jBackend backend) {
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);

        /**
         * ([[[ 0.,  1.],
         [ 2.,  3.]],

         [[ 4.,  5.],
         [ 6.,  7.]]])

         [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]


         Rank: 3,Offset: 0
         Order: c shape: [2,2,2], stride: [4,2,1]
         */
        /* */
        INDArray arr = Nd4j.linspace(0, 7, 8, DataType.DOUBLE).reshape('c', 2, 2, 2);
        /* [0.0,4.0,2.0,6.0,1.0,5.0,3.0,7.0]
        *
        * Rank: 3,Offset: 0
            Order: f shape: [2,2,2], stride: [1,2,4]*/
        INDArray arrF = Nd4j.create(new long[] {2, 2, 2}, 'f').assign(arr);

        assertEquals(arr, arrF);
        //0,2,4,6 and 1,3,5,7
        assertEquals(Nd4j.create(new double[] {12, 16}), arr.sum(0, 1));
        //0,1,4,5 and 2,3,6,7
        assertEquals(Nd4j.create(new double[] {10, 18}), arr.sum(0, 2));
        //0,2,4,6 and 1,3,5,7
        assertEquals(Nd4j.create(new double[] {12, 16}), arrF.sum(0, 1));
        //0,1,4,5 and 2,3,6,7
        assertEquals(Nd4j.create(new double[] {10, 18}), arrF.sum(0, 2));

        //0,1,2,3 and 4,5,6,7
        assertEquals(Nd4j.create(new double[] {6, 22}), arr.sum(1, 2));
        //0,1,2,3 and 4,5,6,7
        assertEquals(Nd4j.create(new double[] {6, 22}), arrF.sum(1, 2));


        double[] data = new double[] {10, 26, 42};
        INDArray assertion = Nd4j.create(data);
        for (int i = 0; i < data.length; i++) {
            assertEquals(data[i], assertion.getDouble(i), 1e-1);
        }

        INDArray twoTwoByThree = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape('f', 2, 2, 3);
        INDArray multiSum = twoTwoByThree.sum(0, 1);
        assertEquals(assertion, multiSum);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum2dv2(Nd4jBackend backend) {
        INDArray in = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape('c', 2, 2, 2);

        val dims = new long[][] {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};
        double[][] exp = new double[][] {{16, 20}, {16, 20}, {14, 22}, {14, 22}, {10, 26}, {10, 26}};

        for (int i = 0; i < dims.length; i++) {
            val d = dims[i];
            double[] e = exp[i];

            INDArray out = in.sum(d);

            assertEquals(Nd4j.create(e, out.shape()), out);
        }
    }

    //Passes on 3.9:
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum3Of4_2222(Nd4jBackend backend) {
        int[] shape = {2, 2, 2, 2};
        int length = ArrayUtil.prod(shape);
        INDArray arrC = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape);
        INDArray arrF = Nd4j.create(arrC.shape()).assign(arrC);

        long[][] dimsToSum = new long[][] {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
        double[][] expD = new double[][] {{64, 72}, {60, 76}, {52, 84}, {36, 100}};

        for (int i = 0; i < dimsToSum.length; i++) {
            long[] d = dimsToSum[i];

            INDArray outC = arrC.sum(d);
            INDArray outF = arrF.sum(d);
            INDArray exp = Nd4j.create(expD[i], outC.shape()).castTo(DataType.DOUBLE);

            assertEquals(exp, outC);
            assertEquals(exp, outF);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum3Of4_3322(Nd4jBackend backend) {
        int[] shape = {3, 3, 2, 2};
        int length = ArrayUtil.prod(shape);
        INDArray arrC = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape('c', shape);
        INDArray arrF = Nd4j.create(arrC.shape()).assign(arrC);

        long[][] dimsToSum = new long[][] {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
        double[][] expD = new double[][] {{324, 342}, {315, 351}, {174, 222, 270}, {78, 222, 366}};

        for (int i = 0; i < dimsToSum.length; i++) {
            long[] d = dimsToSum[i];

            INDArray outC = arrC.sum(d);
            INDArray outF = arrF.sum(d);
            INDArray exp = Nd4j.create(expD[i], outC.shape());

            assertEquals(exp, outC);
            assertEquals(exp, outF);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNorm2Double(Nd4jBackend backend) {
        DataType initialType = Nd4j.dataType();

        INDArray n = Nd4j.create(new double[] {1, 2, 3, 4}).castTo(DataType.DOUBLE);
        double assertion = 5.47722557505;
        double norm3 = n.norm2Number().doubleValue();
        assertEquals(assertion, norm3, 1e-1, getFailureMessage(backend));

        INDArray row = Nd4j.create(new double[] {1, 2, 3, 4}, new long[] {2, 2}).castTo(DataType.DOUBLE);
        INDArray row1 = row.getRow(1);
        double norm2 = row1.norm2Number().doubleValue();
        double assertion2 = 5.0f;
        assertEquals(assertion2, norm2, 1e-1, getFailureMessage(backend));

        Nd4j.setDataType(initialType);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNorm2(Nd4jBackend backend) {
        INDArray n = Nd4j.create(new float[] {1, 2, 3, 4}).castTo(DataType.DOUBLE);
        float assertion = 5.47722557505f;
        float norm3 = n.norm2Number().floatValue();
        assertEquals(assertion, norm3, 1e-1, getFailureMessage(backend));

        INDArray row = Nd4j.create(new float[] {1, 2, 3, 4}, new long[] {2, 2}).castTo(DataType.DOUBLE);
        INDArray row1 = row.getRow(1);
        float norm2 = row1.norm2Number().floatValue();
        float assertion2 = 5.0f;
        assertEquals(assertion2, norm2, 1e-1, getFailureMessage(backend));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCosineSim(Nd4jBackend backend) {
        INDArray vec1 = Nd4j.create(new double[] {1, 2, 3, 4}).castTo(DataType.DOUBLE);
        INDArray vec2 = Nd4j.create(new double[] {1, 2, 3, 4}).castTo(DataType.DOUBLE);
        double sim = Transforms.cosineSim(vec1, vec2);
        assertEquals(1, sim, 1e-1, getFailureMessage(backend));

        INDArray vec3 = Nd4j.create(new float[] {0.2f, 0.3f, 0.4f, 0.5f});
        INDArray vec4 = Nd4j.create(new float[] {0.6f, 0.7f, 0.8f, 0.9f});
        sim = Transforms.cosineSim(vec3, vec4);
        assertEquals(0.98, sim, 1e-1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8, DataType.DOUBLE).data(), new long[] {2, 2, 2});
        INDArray test = Nd4j.create(new double[] {3, 7, 11, 15}, new long[] {2, 2});
        INDArray sum = n.sum(-1);
        assertEquals(test, sum);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum2(Nd4jBackend backend) {
        INDArray test = Nd4j.create(new float[] {1, 2, 3, 4}, new long[] {2, 2}).castTo(DataType.DOUBLE);
        INDArray sum = test.sum(1);
        INDArray assertion = Nd4j.create(new float[] {3, 7}).castTo(DataType.DOUBLE);
        assertEquals(assertion, sum);
        INDArray sum0 = Nd4j.create(new float[] {4, 6}).castTo(DataType.DOUBLE);
        assertEquals(sum0, test.sum(0));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeans(Nd4jBackend backend) {
        INDArray a = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray mean1 = a.mean(1);
        assertEquals(Nd4j.create(new double[] {1.5, 3.5}), mean1, getFailureMessage(backend));
        assertEquals(Nd4j.create(new double[] {2, 3}), a.mean(0), getFailureMessage(backend));
        assertEquals(2.5, Nd4j.linspace(1, 4, 4, DataType.DOUBLE).meanNumber().doubleValue(), 1e-1, getFailureMessage(backend));
        assertEquals(2.5, a.meanNumber().doubleValue(), 1e-1, getFailureMessage(backend));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSums(Nd4jBackend backend) {
        INDArray a = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        assertEquals(Nd4j.create(new double[] {3, 7}), a.sum(1), getFailureMessage(backend));
        assertEquals(Nd4j.create(new double[] {4, 6}), a.sum(0), getFailureMessage(backend));
        assertEquals(10, a.sumNumber().doubleValue(), 1e-1, getFailureMessage(backend));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSoftmaxStability(Nd4jBackend backend) {
        INDArray input = Nd4j.create(new double[] {-0.75, 0.58, 0.42, 1.03, -0.61, 0.19, -0.37, -0.40, -1.42, -0.04}).reshape(1, -1).transpose();
        INDArray output = Nd4j.create(10, 1);
        Nd4j.getExecutioner().exec(new SoftMax(input, output));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSquareMatrix(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8, DataType.DOUBLE).data(), new long[] {2, 2, 2});
        INDArray eightFirstTest = n.vectorAlongDimension(0, 2);
        INDArray eightFirstAssertion = Nd4j.create(new double[] {1, 2});
        assertEquals(eightFirstAssertion, eightFirstTest);

        INDArray eightFirstTestSecond = n.vectorAlongDimension(1, 2);
        INDArray eightFirstTestSecondAssertion = Nd4j.create(new double[] {3, 4});
        assertEquals(eightFirstTestSecondAssertion, eightFirstTestSecond);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNumVectorsAlongDimension(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 24, 24, DataType.DOUBLE).reshape(4, 3, 2);
        assertEquals(12, arr.vectorsAlongDimension(2));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumDifferentOrdersSquareMatrix(Nd4jBackend backend) {
        INDArray arrc = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray arrf = Nd4j.create(new long[] {2, 2}, 'f').assign(arrc);

        INDArray cSum = arrc.sum(0);
        INDArray fSum = arrf.sum(0);
        assertEquals(arrc, arrf);
        assertEquals(cSum, fSum); //Expect: 4,6. Getting [4, 4] for f order
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumDifferentOrders(Nd4jBackend backend) {
        INDArray arrc = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape('c', 3, 2);
        INDArray arrf = Nd4j.create(new double[6], new long[] {3, 2}, 'f').assign(arrc);

        assertEquals(arrc, arrf);
        INDArray cSum = arrc.sum(0);
        INDArray fSum = arrf.sum(0);
        assertEquals(cSum, fSum); //Expect: 0.51, 1.79; getting [0.51,1.71] for f order
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVarConst(Nd4jBackend backend) {
        INDArray x = Nd4j.linspace(1, 100, 100, DataType.DOUBLE).reshape(10, 10);
        assertFalse(Double.isNaN(x.var(0).sumNumber().doubleValue()));
        x.var(0);
        assertFalse(Double.isNaN(x.var(1).sumNumber().doubleValue()));
        x.var(1);

        // 2d array - all elements are the same
        INDArray a = Nd4j.ones(10, 10).mul(10);
        assertFalse(Double.isNaN(a.var(0).sumNumber().doubleValue()));
        a.var(0);
        assertFalse(Double.isNaN(a.var(1).sumNumber().doubleValue()));
        a.var(1);

        // 2d array - constant in one dimension
        INDArray nums = Nd4j.linspace(1, 10, 10, DataType.DOUBLE);
        INDArray b = Nd4j.ones(10, 10).mulRowVector(nums);
        assertFalse(Double.isNaN((Double) b.var(0).sumNumber()));
        b.var(0);
        assertFalse(Double.isNaN((Double) b.var(1).sumNumber()));
        b.var(1);

        assertFalse(Double.isNaN((Double) b.transpose().var(0).sumNumber()));
        b.transpose().var(0);
        assertFalse(Double.isNaN((Double) b.transpose().var(1).sumNumber()));
        b.transpose().var(1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDistance1and2(Nd4jBackend backend) {
        double[] d1 = new double[] {-1, 3, 2};
        double[] d2 = new double[] {0, 1.5, -3.5};
        INDArray arr1 = Nd4j.create(d1);
        INDArray arr2 = Nd4j.create(d2);

        double expD1 = 0.0;
        double expD2 = 0.0;
        for (int i = 0; i < d1.length; i++) {
            double diff = d1[i] - d2[i];
            expD1 += Math.abs(diff);
            expD2 += diff * diff;
        }
        expD2 = Math.sqrt(expD2);

        assertEquals(expD1, arr1.distance1(arr2), 1e-5);
        assertEquals(expD2, arr1.distance2(arr2), 1e-5);
        assertEquals(expD2 * expD2, arr1.squaredDistance(arr2), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReductionAgreement1(Nd4jBackend backend) {
        INDArray row = Nd4j.linspace(1, 3, 3, DataType.DOUBLE).reshape(1, 3);
        INDArray mean0 = row.mean(0);
        assertFalse(mean0 == row); //True: same object (should be a copy)

        INDArray col = Nd4j.linspace(1, 3, 3, DataType.DOUBLE).reshape(1, -1).transpose();
        INDArray mean1 = col.mean(1);
        assertFalse(mean1 == col);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarReduction1(Nd4jBackend backend) {
        val op = new Norm2(Nd4j.create(1).assign(1.0));
        double norm2 = Nd4j.getExecutioner().execAndReturn(op).getFinalResult().doubleValue();
        double norm1 = Nd4j.getExecutioner().execAndReturn(new Norm1(Nd4j.create(1).assign(1.0))).getFinalResult()
                .doubleValue();
        double sum = Nd4j.getExecutioner().execAndReturn(new Sum(Nd4j.create(1).assign(1.0))).getFinalResult()
                .doubleValue();

        assertEquals(1.0, norm2, 0.001);
        assertEquals(1.0, norm1, 0.001);
        assertEquals(1.0, sum, 0.001);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tesAbsReductions1(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {-1, -2, -3, -4}).castTo(DataType.DOUBLE);
        assertEquals(4, array.amaxNumber().intValue());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tesAbsReductions2(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {-1, -2, -3, -4}).castTo(DataType.DOUBLE);
        assertEquals(1, array.aminNumber().intValue());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tesAbsReductions3(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {-2, -2, 2, 2});
        assertEquals(2, array.ameanNumber().intValue());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tesAbsReductions4(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {-2, -2, 2, 3}).castTo(DataType.DOUBLE);
        assertEquals(1.0, array.sumNumber().doubleValue(), 1e-5);

        assertEquals(4, array.scan(org.nd4j.linalg.indexing.conditions.Conditions.absGreaterThanOrEqual(0.0)).intValue());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tesAbsReductions5(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {-2, 0.0, 2, 2});
        assertEquals(3, array.scan(org.nd4j.linalg.indexing.conditions.Conditions.absGreaterThan(0.0)).intValue());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_0(Nd4jBackend backend) {
        INDArray haystack = Nd4j.create(new double[] {-0.84443557262, -0.06822254508, 0.74266910552, 0.61765557527,
                        -0.77555125951, -0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130,
                        -1.25485503673, 0.62955373525, -0.31357592344, 1.03362500667, -0.59279078245, 1.1914824247})
                .reshape(3, 5).castTo(DataType.DOUBLE);
        INDArray needle = Nd4j.create(new double[] {-0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130,
                -1.25485503673}).castTo(DataType.DOUBLE);

        INDArray reduced = Nd4j.getExecutioner().exec(new CosineDistance(haystack, needle, 1));

        INDArray exp = Nd4j.create(new double[] {0.577452, 0.0, 1.80182}).castTo(DataType.DOUBLE);
        assertEquals(exp, reduced);

        for (int i = 0; i < haystack.rows(); i++) {
            val row = haystack.getRow(i).dup();
            double res = Nd4j.getExecutioner().execAndReturn(new CosineDistance(row, needle)).z().getDouble(0);
            assertEquals(reduced.getDouble(i), res, 1e-5, "Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduce3SignaturesEquality_1(Nd4jBackend backend) {
        val x = Nd4j.rand(DataType.DOUBLE, 3, 4, 5);
        val y = Nd4j.rand(DataType.DOUBLE, 3, 4, 5);

        val reduceOp = new ManhattanDistance(x, y, 0);
        val op = (org.nd4j.linalg.api.ops.Op) reduceOp;

        val z0 = Nd4j.getExecutioner().exec(reduceOp);
        val z1 = Nd4j.getExecutioner().exec(op);

        assertEquals(z0, z1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_1(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 10).castTo(DataType.DOUBLE);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(new double[] {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.getExecutioner().exec(new CosineSimilarity(initial, needle, 1));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            double res = Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(initial.getRow(i).dup(), needle))
                    .getFinalResult().doubleValue();
            assertEquals(reduced.getDouble(i), res, 0.001, "Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_2(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 10).castTo(DataType.DOUBLE);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.getExecutioner().exec(new ManhattanDistance(initial, needle, 1));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            double res = Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(initial.getRow(i).dup(), needle))
                    .getFinalResult().doubleValue();
            assertEquals(reduced.getDouble(i), res, 0.001, "Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_3(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 10).castTo(DataType.DOUBLE);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.getExecutioner().exec(new EuclideanDistance(initial, needle, 1));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            INDArray x = initial.getRow(i).dup();
            double res = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(x, needle)).getFinalResult()
                    .doubleValue();
            assertEquals(reduced.getDouble(i), res, 0.001, "Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_3_NEG(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 10).castTo(DataType.DOUBLE);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.getExecutioner().exec(new EuclideanDistance(initial, needle, -1));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            INDArray x = initial.getRow(i).dup();
            double res = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(x, needle)).getFinalResult()
                    .doubleValue();
            assertEquals(reduced.getDouble(i), res, 0.001, "Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_3_NEG_2(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 10).castTo(DataType.DOUBLE);
        for (int i = 0; i < initial.rows(); i++) {
            initial.getRow(i).assign(i + 1);
        }
        INDArray needle = Nd4j.create(10).assign(1.0).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.create(5).castTo(DataType.DOUBLE);
        Nd4j.getExecutioner().exec(new CosineSimilarity(initial, needle, reduced, -1));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < initial.rows(); i++) {
            INDArray x = initial.getRow(i).dup();
            double res = Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(x, needle)).getFinalResult()
                    .doubleValue();
            assertEquals(reduced.getDouble(i), res, 0.001, "Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadReduce3_4(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(5, 6, 7).castTo(DataType.DOUBLE);
        for (int i = 0; i < 5; i++) {
            initial.tensorAlongDimension(i, 1, 2).assign(i + 1);
        }
        INDArray needle = Nd4j.create(6, 7).assign(1.0).castTo(DataType.DOUBLE);
        INDArray reduced = Nd4j.getExecutioner().exec(new ManhattanDistance(initial, needle, 1, 2));

        log.warn("Reduced: {}", reduced);

        for (int i = 0; i < 5; i++) {
            double res = Nd4j.getExecutioner()
                    .execAndReturn(new ManhattanDistance(initial.tensorAlongDimension(i, 1, 2).dup(), needle))
                    .getFinalResult().doubleValue();
            assertEquals(reduced.getDouble(i), res, 0.001, "Failed at " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances1(Nd4jBackend backend) {
        INDArray initialX = Nd4j.create(5, 10);
        INDArray initialY = Nd4j.create(7, 10);
        for (int i = 0; i < initialX.rows(); i++) {
            initialX.getRow(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.rows(); i++) {
            initialY.getRow(i).assign(i + 101);
        }

        INDArray result = Transforms.allEuclideanDistances(initialX, initialY, 1);

        Nd4j.getExecutioner().commit();

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.rows(); x++) {
            INDArray rowX = initialX.getRow(x).dup();
            for (int y = 0; y < initialY.rows(); y++) {
                double res = result.getDouble(x, y);
                double exp = Transforms.euclideanDistance(rowX, initialY.getRow(y).dup());
                assertEquals(exp, res, 0.001, "Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances2(Nd4jBackend backend) {
        INDArray initialX = Nd4j.create(5, 10);
        INDArray initialY = Nd4j.create(7, 10);
        for (int i = 0; i < initialX.rows(); i++) {
            initialX.getRow(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.rows(); i++) {
            initialY.getRow(i).assign(i + 101);
        }

        INDArray result = Transforms.allManhattanDistances(initialX, initialY, 1);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.rows(); x++) {
            INDArray rowX = initialX.getRow(x).dup();
            for (int y = 0; y < initialY.rows(); y++) {
                double res = result.getDouble(x, y);
                double exp = Transforms.manhattanDistance(rowX, initialY.getRow(y).dup());
                assertEquals(exp, res, 0.001, "Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances2_Large(Nd4jBackend backend) {
        INDArray initialX = Nd4j.create(5, 2000);
        INDArray initialY = Nd4j.create(7, 2000);
        for (int i = 0; i < initialX.rows(); i++) {
            initialX.getRow(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.rows(); i++) {
            initialY.getRow(i).assign(i + 101);
        }

        INDArray result = Transforms.allManhattanDistances(initialX, initialY, 1);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.rows(); x++) {
            INDArray rowX = initialX.getRow(x).dup();
            for (int y = 0; y < initialY.rows(); y++) {
                double res = result.getDouble(x, y);
                double exp = Transforms.manhattanDistance(rowX, initialY.getRow(y).dup());
                assertEquals(exp, res, 0.001, "Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances3_Large(Nd4jBackend backend) {
        INDArray initialX = Nd4j.create(5, 2000);
        INDArray initialY = Nd4j.create(7, 2000);
        for (int i = 0; i < initialX.rows(); i++) {
            initialX.getRow(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.rows(); i++) {
            initialY.getRow(i).assign(i + 101);
        }

        INDArray result = Transforms.allEuclideanDistances(initialX, initialY, 1);

        Nd4j.getExecutioner().commit();

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.rows(); x++) {
            INDArray rowX = initialX.getRow(x).dup();
            for (int y = 0; y < initialY.rows(); y++) {
                double res = result.getDouble(x, y);
                double exp = Transforms.euclideanDistance(rowX, initialY.getRow(y).dup());
                assertEquals(exp, res, 0.001, "Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances3_Large_Columns(Nd4jBackend backend) {
        INDArray initialX = Nd4j.create(2000, 5);
        INDArray initialY = Nd4j.create(2000, 7);
        for (int i = 0; i < initialX.columns(); i++) {
            initialX.getColumn(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.columns(); i++) {
            initialY.getColumn(i).assign(i + 101);
        }

        INDArray result = Transforms.allEuclideanDistances(initialX, initialY, 0);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.columns(); x++) {
            INDArray colX = initialX.getColumn(x).dup();
            for (int y = 0; y < initialY.columns(); y++) {
                double res = result.getDouble(x, y);
                double exp = Transforms.euclideanDistance(colX, initialY.getColumn(y).dup());
                assertEquals(exp, res, 0.001, "Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances4_Large_Columns(Nd4jBackend backend) {
        INDArray initialX = Nd4j.create(2000, 5);
        INDArray initialY = Nd4j.create(2000, 7);
        for (int i = 0; i < initialX.columns(); i++) {
            initialX.getColumn(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.columns(); i++) {
            initialY.getColumn(i).assign(i + 101);
        }

        INDArray result = Transforms.allManhattanDistances(initialX, initialY, 0);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.columns(); x++) {
            INDArray colX = initialX.getColumn(x).dup();
            for (int y = 0; y < initialY.columns(); y++) {
                double res = result.getDouble(x, y);
                double exp = Transforms.manhattanDistance(colX, initialY.getColumn(y).dup());
                assertEquals(exp, res, 0.001, "Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances5_Large_Columns(Nd4jBackend backend) {
        INDArray initialX = Nd4j.create(2000, 5);
        INDArray initialY = Nd4j.create(2000, 7);
        for (int i = 0; i < initialX.columns(); i++) {
            initialX.getColumn(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.columns(); i++) {
            initialY.getColumn(i).assign(i + 101);
        }

        INDArray result = Transforms.allCosineDistances(initialX, initialY, 0);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.columns(); x++) {
            INDArray colX = initialX.getColumn(x).dup();
            for (int y = 0; y < initialY.columns(); y++) {
                double res = result.getDouble(x, y);
                double exp = Transforms.cosineDistance(colX, initialY.getColumn(y).dup());
                assertEquals(exp, res, 0.001, "Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances3_Small_Columns(Nd4jBackend backend) {
        INDArray initialX = Nd4j.create(200, 5);
        INDArray initialY = Nd4j.create(200, 7);
        for (int i = 0; i < initialX.columns(); i++) {
            initialX.getColumn(i).assign(i + 1);
        }

        for (int i = 0; i < initialY.columns(); i++) {
            initialY.getColumn(i).assign(i + 101);
        }

        INDArray result = Transforms.allManhattanDistances(initialX, initialY, 0);

        assertEquals(5 * 7, result.length());

        for (int x = 0; x < initialX.columns(); x++) {
            INDArray colX = initialX.getColumn(x).dup();
            for (int y = 0; y < initialY.columns(); y++) {
                double res = result.getDouble(x, y);
                double exp = Transforms.manhattanDistance(colX, initialY.getColumn(y).dup());
                assertEquals(exp, res, 0.001, "Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistances3(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(123);

        INDArray initialX = Nd4j.rand(5, 10).castTo(DataType.DOUBLE);
        INDArray initialY = initialX.mul(-1);

        INDArray result = Transforms.allCosineSimilarities(initialX, initialY, 1);

        assertEquals(5 * 5, result.length());

        for (int x = 0; x < initialX.rows(); x++) {
            INDArray rowX = initialX.getRow(x).dup();
            for (int y = 0; y < initialY.rows(); y++) {
                double res = result.getDouble(x, y);
                double exp = Transforms.cosineSim(rowX, initialY.getRow(y).dup());
                assertEquals(exp, res, 0.001, "Failed for [" + x + ", " + y + "]");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEntropy1(Nd4jBackend backend) {
        INDArray x = Nd4j.rand(1, 100).castTo(DataType.DOUBLE);

        double exp = MathUtils.entropy(x.data().asDouble());
        double res = x.entropyNumber().doubleValue();

        assertEquals(exp, res, 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEntropy2(Nd4jBackend backend) {
        INDArray x = Nd4j.rand(10, 100).castTo(DataType.DOUBLE);

        INDArray res = x.entropy(1);

        assertEquals(10, res.length());

        for (int t = 0; t < x.rows(); t++) {
            double exp = MathUtils.entropy(x.getRow(t).dup().data().asDouble());
            assertEquals(exp, res.getDouble(t), 1e-5);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEntropy3(Nd4jBackend backend) {
        INDArray x = Nd4j.rand(1, 100).castTo(DataType.DOUBLE);

        double exp = getShannonEntropy(x.data().asDouble());
        double res = x.shannonEntropyNumber().doubleValue();

        assertEquals(exp, res, 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEntropy4(Nd4jBackend backend) {
        INDArray x = Nd4j.rand(1, 100).castTo(DataType.DOUBLE);

        double exp = getLogEntropy(x.data().asDouble());
        double res = x.logEntropyNumber().doubleValue();

        assertEquals(exp, res, 1e-5);
    }

    protected double getShannonEntropy(double[] array) {
        double ret = 0;
        for (double x : array) {
            ret += x * FastMath.log(2., x);
        }
        return -ret;
    }

    protected double getLogEntropy(double[] array) {
        return Math.log(MathUtils.entropy(array));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPercentile1(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 10, 10, DataType.DOUBLE);
        Percentile percentile = new Percentile(50);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(50));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPercentile2(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 9, 9, DataType.DOUBLE);
        Percentile percentile = new Percentile(50);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(50));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPercentile3(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 9, 9, DataType.DOUBLE);
        Percentile percentile = new Percentile(75);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(75));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPercentile4(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 10, 10, DataType.DOUBLE);
        Percentile percentile = new Percentile(75);
        double exp = percentile.evaluate(array.data().asDouble());

        assertEquals(exp, array.percentileNumber(75));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPercentile5(Nd4jBackend backend) {
        val array = Nd4j.createFromArray(new int[]{1, 1982});
        val perc = array.percentileNumber(75);
        assertEquals(1982.f, perc.floatValue(), 1e-5f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadPercentile1(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 10, 10, DataType.DOUBLE);
        Transforms.reverse(array, false);
        Percentile percentile = new Percentile(75);
        double exp = percentile.evaluate(array.data().asDouble());

        INDArray matrix = Nd4j.create(10, 10);
        for (int i = 0; i < matrix.rows(); i++)
            matrix.getRow(i).assign(array);

        INDArray res = matrix.percentile(75, 1);

        for (int i = 0; i < matrix.rows(); i++)
            assertEquals(exp, res.getDouble(i), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @org.junit.jupiter.api.Disabled("Needs investigation")
    public void testLogExpSum1(Nd4jBackend backend) {
        INDArray matrix = Nd4j.create(3, 3);
        for (int r = 0; r < matrix.rows(); r++) {
            matrix.getRow(r).assign(Nd4j.create(new double[]{1, 2, 3}));
        }

        INDArray res = Nd4j.getExecutioner().exec(new LogSumExp(matrix, false, 1))[0];

        for (int e = 0; e < res.length(); e++) {
            assertEquals(3.407605, res.getDouble(e), 1e-5);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @org.junit.jupiter.api.Disabled("Needs investigation")
    public void testLogExpSum2(Nd4jBackend backend) {
        INDArray row = Nd4j.create(new double[]{1, 2, 3});

        double res = Nd4j.getExecutioner().exec(new LogSumExp(row))[0].getDouble(0);

        assertEquals(3.407605, res, 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test4DSumView(Nd4jBackend backend) {
        INDArray labels = Nd4j.linspace(1, 160, 160, DataType.DOUBLE).reshape(2, 5, 4, 4);

        val size1 = labels.size(1);
        INDArray classLabels = labels.get(NDArrayIndex.all(), NDArrayIndex.interval(4, size1), NDArrayIndex.all(), NDArrayIndex.all());

        assertEquals(classLabels, classLabels.dup());

        //Expect 0 or 1 for each entry (sum of all 0s, or 1-hot vector = 0 or 1)
        INDArray sum1 = classLabels.max(1);
        INDArray sum1_dup = classLabels.dup().max(1);

        assertEquals(sum1_dup, sum1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduction_Z1(Nd4jBackend backend) {
        val arrayX = Nd4j.create(10, 10, 10);

        val res = arrayX.max(1, 2);

        Nd4j.getExecutioner().commit();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduction_Z2(Nd4jBackend backend) {
        val arrayX = Nd4j.create(10, 10);

        val res = arrayX.max(0);

        Nd4j.getExecutioner().commit();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduction_Z3(Nd4jBackend backend) {
        val arrayX = Nd4j.create(200, 300);

        val res = arrayX.maxNumber().doubleValue();

        Nd4j.getExecutioner().commit();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSoftmaxZ1(Nd4jBackend backend) {
        val original = Nd4j.linspace(1, 100, 100, DataType.DOUBLE).reshape(10, 10);
        val reference = original.dup(original.ordering());
        val expected = original.dup(original.ordering());

        Nd4j.getExecutioner().execAndReturn((CustomOp) new SoftMax(expected, expected, -1));

        val result = Nd4j.getExecutioner().exec((CustomOp) new SoftMax(original, original.dup(original.ordering())))[0];

        assertEquals(reference, original);
        assertEquals(expected, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduce3AlexBug(Nd4jBackend backend) {
        val arr = Nd4j.linspace(1, 100, 100, DataType.DOUBLE).reshape('f', 10, 10).dup('c');
        val arr2 = Nd4j.linspace(1, 100, 100, DataType.DOUBLE).reshape('c', 10, 10);
        val out = Nd4j.getExecutioner().exec(new EuclideanDistance(arr, arr2, 1));
        val exp = Nd4j.create(new double[] {151.93748, 128.86038, 108.37435, 92.22256, 82.9759, 82.9759, 92.22256, 108.37435, 128.86038, 151.93748});

        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllDistancesEdgeCase1(Nd4jBackend backend) {
        val x = Nd4j.create(400, 20).assign(2.0).castTo(Nd4j.defaultFloatingPointType());
        val y = Nd4j.ones(1, 20).castTo(Nd4j.defaultFloatingPointType());
        val z = Transforms.allEuclideanDistances(x, y, 1);

        val exp = Nd4j.create(400, 1).assign(4.47214);

        assertEquals(exp, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAccumuationWithoutAxis_1(Nd4jBackend backend) {
        val array = Nd4j.create(3, 3).assign(1.0);

        val result = array.sum();

        assertEquals(1, result.length());
        assertEquals(9.0, result.getDouble(0), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSummaryStatsEquality_1(Nd4jBackend backend) {
        for (boolean biasCorrected : new boolean[]{false, true}) {
            INDArray indArray1 = Nd4j.rand(1, 4, 10).castTo(DataType.DOUBLE);
            double std = indArray1.stdNumber(biasCorrected).doubleValue();

            val standardDeviation = new org.apache.commons.math3.stat.descriptive.moment.StandardDeviation(biasCorrected);
            double std2 = standardDeviation.evaluate(indArray1.data().asDouble());

            assertEquals(std, std2, 1e-5);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeanEdgeCase_C() {
        INDArray arr = Nd4j.linspace(1, 30, 30, DataType.DOUBLE).reshape(new int[]{3, 10, 1}).dup('c');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));

        assertEquals(exp, arr2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeanEdgeCase_F() {
        INDArray arr = Nd4j.linspace(1, 30, 30, DataType.DOUBLE).reshape(new int[]{3, 10, 1}).dup('f');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));

        assertEquals(exp, arr2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeanEdgeCase2_C() {
        INDArray arr = Nd4j.linspace(1, 60, 60, DataType.DOUBLE).reshape(new int[]{3, 10, 2}).dup('c');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
        exp.addi(arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1)));
        exp.divi(2);

        assertEquals(exp, arr2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMeanEdgeCase2_F() {
        INDArray arr = Nd4j.linspace(1, 60, 60, DataType.DOUBLE).reshape(new int[]{3, 10, 2}).dup('f');
        INDArray arr2 = arr.mean(2);

        INDArray exp = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
        exp.addi(arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1)));
        exp.divi(2);

        assertEquals(exp, arr2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVariance_4D_1(Nd4jBackend backend) {
        val dtype = Nd4j.dataType();

        Nd4j.setDataType(DataType.FLOAT);

        val x = Nd4j.ones(10, 20, 30, 40);
        val result = x.var(false, 0, 2, 3);

        Nd4j.getExecutioner().commit();

        Nd4j.setDataType(dtype);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStatistics_1(Nd4jBackend backend) {
        val array = Nd4j.createFromArray(new float[] {-1.0f, 0.0f, 1.0f});
        val stats = Nd4j.getExecutioner().inspectArray(array);

        assertEquals(1, stats.getCountPositive());
        assertEquals(1, stats.getCountNegative());
        assertEquals(1, stats.getCountZero());
        assertEquals(0.0f, stats.getMeanValue(), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumEdgeCase() {
        INDArray row = Nd4j.create(1, 3);
        INDArray sum = row.sum(0);
        assertArrayEquals(new long[]{3}, sum.shape());

        INDArray twoD = Nd4j.create(2, 3);
        INDArray sum2 = twoD.sum(0);
        assertArrayEquals(new long[]{3}, sum2.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMedianEdgeCase() {
        INDArray rowVec = Nd4j.rand(DataType.FLOAT, 1, 10);
        INDArray median = rowVec.median(0);
        assertEquals(rowVec.reshape(10), median);

        INDArray colVec = Nd4j.rand(DataType.FLOAT, 10, 1);
        median = colVec.median(1);
        assertEquals(colVec.reshape(10), median);

        //Non-edge cases:
        rowVec.median(1);
        colVec.median(0);

        //full array case:
        rowVec.median();
        colVec.median();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduceAll_1(Nd4jBackend backend) {
        val x = Nd4j.empty(DataType.FLOAT);
        val e = Nd4j.scalar(true);
        val z = Nd4j.exec(new All(x));

        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduceAll_2(Nd4jBackend backend) {
        val x = Nd4j.ones(DataType.FLOAT, 0);
        val e = Nd4j.scalar(true);
        val z = Nd4j.exec(new All(x));

        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReduceAll_3(Nd4jBackend backend) {
        val x = Nd4j.create(DataType.FLOAT, 0);
        assertEquals(1, x.rank());

        val e = Nd4j.scalar(true);
        val z = Nd4j.exec(new All(x, 0));

        assertEquals(e, z);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMin2() {
        INDArray x = Nd4j.createFromArray(new double[][]{
                {-999, 0.2236, 0.7973, 0.0962},
                {0.7231, 0.3381, -0.7301, 0.9115},
                {-0.5094, 0.9749, -2.1340, 0.6023}});

        INDArray out = Nd4j.create(DataType.DOUBLE, 4);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("reduce_min")
                .addInputs(x)
                .addOutputs(out)
                .addIntegerArguments(0)
                .build());

        INDArray exp = Nd4j.createFromArray(-999, 0.2236, -2.1340, 0.0962);
        assertEquals(exp, out);

        INDArray out1 = Nd4j.create(DataType.DOUBLE, 3);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("reduce_min")
                .addInputs(x)
                .addOutputs(out1)
                .addIntegerArguments(1)
                .build());

        INDArray exp1 = Nd4j.createFromArray(-999, -0.7301, -2.1340);
        assertEquals(exp1, out1);
    }
}
