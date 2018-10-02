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

package org.nd4j.linalg.ops;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.exception.IllegalOpException;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.accum.floating.Mean;
import org.nd4j.linalg.api.ops.impl.accum.floating.Norm2;
import org.nd4j.linalg.api.ops.impl.accum.floating.NormMax;
import org.nd4j.linalg.api.ops.impl.accum.same.Max;
import org.nd4j.linalg.api.ops.impl.accum.same.Min;
import org.nd4j.linalg.api.ops.impl.accum.same.Sum;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMax;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarReverseSubtraction;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldMulOp;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Log;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * Created by agibsonccc on 2/22/15.
 */
@Slf4j
@RunWith(Parameterized.class)
public class OpExecutionerTestsC extends BaseNd4jTest {

    public OpExecutionerTestsC(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    DataType initialType;

    @After
    public void after() {
        Nd4j.setDataType(this.initialType);
    }


    @Test
    public void testSoftmaxReference() {
        INDArray input = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray dup = input.dup();
        Nd4j.getExecutioner().exec(new OldSoftMax(dup));
        INDArray result = Nd4j.zeros(2,2);
        Nd4j.getExecutioner().exec(new OldSoftMax(input,result));
        assertEquals(dup,result);


        dup = input.dup();
        Nd4j.getExecutioner().exec(new LogSoftMax(dup));

        result = Nd4j.zeros(2,2);
        Nd4j.getExecutioner().exec(new LogSoftMax(input,result));

        assertEquals(dup,result);



    }

    @Test
    public void testScalarReverseSub() {
        INDArray input = Nd4j.valueArrayOf(4,2.0);
        INDArray result= Nd4j.zeros(4);
        Nd4j.getExecutioner().exec(new ScalarReverseSubtraction(input,null,result,input.length(),1.0));
        INDArray assertion = Nd4j.valueArrayOf(4,-1.0);
        assertEquals(assertion,result);
    }


    @Test
    public void testBroadcastMultiDim() {
        INDArray data = Nd4j.linspace(1, 30, 30).reshape(2, 3, 5);
        System.out.println(data);
        INDArray mask = Nd4j.create(new double[][] {{1.00, 1.00, 1.00, 1.00, 1.00}, {1.00, 1.00, 1.00, 0.00, 0.00}});
        Nd4j.getExecutioner().exec(new BroadcastMulOp(data, mask, data, 0, 2));
        INDArray assertion = Nd4j.create(new double[] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 0.0, 0.0, 21.0, 22.0, 23.0, 0.0, 0.0, 26.0, 27.0, 28.0, 0.0,
                        0.0}).reshape(2, 3, 5);
        assertEquals(assertion, data);
    }


    @Test
    public void testCosineSimilarity() {
        INDArray vec1 = Nd4j.create(new float[] {1, 2, 3, 4, 5});
        INDArray vec2 = Nd4j.create(new float[] {1, 2, 3, 4, 5});
        double sim = Transforms.cosineSim(vec1, vec2);
        assertEquals(getFailureMessage(), 1, sim, 1e-1);
    }

    @Test
    public void testCosineDistance(){
        INDArray vec1 = Nd4j.create(new float[] {1, 2, 3});
        INDArray vec2 = Nd4j.create(new float[] {3, 5, 7});
        // 1-17*sqrt(2/581)
        double distance = Transforms.cosineDistance(vec1, vec2);
        assertEquals(getFailureMessage(), 0.0025851, distance, 1e-7);
    }

    @Test
    public void testLog() {
        INDArray log = Nd4j.linspace(1, 6, 6);
        INDArray transformed = Transforms.log(log);
        INDArray assertion = Nd4j.create(new double[] {0., 0.69314718, 1.09861229, 1.38629436, 1.60943791, 1.79175947});
        assertEquals(assertion, transformed);
    }

    @Test
    public void testNorm1AlongDimension() {
        INDArray arr = Nd4j.linspace(1, 8, 8).reshape(2, 4);
        INDArray arrNorm1 = arr.norm2(1);
        INDArray assertion = Nd4j.create(new double[] {5.47722558, 13.19090596});
        assertEquals(assertion, arrNorm1);
    }


    @Test
    public void testEuclideanDistance() {
        INDArray arr = Nd4j.create(new double[] {55, 55});
        INDArray arr2 = Nd4j.create(new double[] {60, 60});
        double result = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(arr, arr2)).getFinalResult()
                        .doubleValue();
        assertEquals(getFailureMessage(), 7.0710678118654755, result, 1e-1);
    }

    @Test
    public void testScalarMaxOp() {
        INDArray scalarMax = Nd4j.linspace(1, 6, 6).negi();
        INDArray postMax = Nd4j.ones(6);
        Nd4j.getExecutioner().exec(new ScalarMax(scalarMax, 1));
        assertEquals(getFailureMessage(), scalarMax, postMax);
    }

    @Test
    public void testSetRange() {
        INDArray linspace = Nd4j.linspace(1, 4, 4);
        Nd4j.getExecutioner().exec(new SetRange(linspace, 0, 1));
        for (int i = 0; i < linspace.length(); i++) {
            double val = linspace.getDouble(i);
            assertTrue(getFailureMessage(), val >= 0 && val <= 1);
        }

        INDArray linspace2 = Nd4j.linspace(1, 4, 4);
        Nd4j.getExecutioner().exec(new SetRange(linspace2, 2, 4));
        for (int i = 0; i < linspace2.length(); i++) {
            double val = linspace2.getDouble(i);
            assertTrue(getFailureMessage(), val >= 2 && val <= 4);
        }
    }


    @Test
    public void testNormMax() {
        INDArray arr = Nd4j.create(new float[] {1, 2, 3, 4});
        double normMax = Nd4j.getExecutioner().execAndReturn(new NormMax(arr)).getFinalResult().doubleValue();
        assertEquals(getFailureMessage(), 4, normMax, 1e-1);
    }


    @Test
    public void testNorm2() {
        INDArray arr = Nd4j.create(new float[] {1, 2, 3, 4});
        double norm2 = Nd4j.getExecutioner().execAndReturn(new Norm2(arr)).getFinalResult().doubleValue();
        assertEquals(getFailureMessage(), 5.4772255750516612, norm2, 1e-1);
    }

    @Test
    public void testAdd() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.ones(5);
        INDArray xDup = x.dup();
        INDArray solution = Nd4j.valueArrayOf(5, 2.0);
        opExecutioner.exec(new AddOp(new INDArray[]{x, xDup},new INDArray[]{x}));
        assertEquals(getFailureMessage(), solution, x);
    }

    @Test
    public void testMul() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.ones(5);
        INDArray xDup = x.dup();
        INDArray solution = Nd4j.valueArrayOf(5, 1.0);
        opExecutioner.exec(new OldMulOp(x, xDup, x));
        assertEquals(solution, x);
    }


    @Test
    public void testExecutioner() throws IllegalOpException {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.ones(5);
        INDArray xDup = x.dup();
        INDArray solution = Nd4j.valueArrayOf(5, 2.0);
        opExecutioner.exec(new AddOp(new INDArray[]{x, xDup},new INDArray[]{ x}));
        assertEquals(getFailureMessage(), solution, x);
        Sum acc = new Sum(x.dup());
        opExecutioner.exec(acc);
        assertEquals(getFailureMessage(), 10.0, acc.getFinalResult().doubleValue(), 1e-1);
        Prod prod = new Prod(x.dup());
        opExecutioner.exec(prod);
        assertEquals(getFailureMessage(), 32.0, prod.getFinalResult().doubleValue(), 1e-1);
    }


    @Test
    public void testMaxMin() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1, 5, 5);
        Max max = new Max(x);
        opExecutioner.exec(max);
        assertEquals(5, max.getFinalResult().doubleValue(), 1e-1);
        Min min = new Min(x);
        opExecutioner.exec(min);
        assertEquals(1, min.getFinalResult().doubleValue(), 1e-1);
    }

    @Test
    public void testProd() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        Prod prod = new Prod(linspace);
        double prod2 = Nd4j.getExecutioner().execAndReturn(prod).getFinalResult().doubleValue();
        assertEquals(720, prod2, 1e-1);
    }

    @Test
    public void testSum() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        Sum sum = new Sum(linspace);
        double sum2 = Nd4j.getExecutioner().execAndReturn(sum).getFinalResult().doubleValue();
        assertEquals(21, sum2, 1e-1);

        INDArray matrixSums = linspace.reshape(2, 3);
        INDArray rowSums = matrixSums.sum(1);
        assertEquals(Nd4j.create(new double[] {6, 15}), rowSums);
    }


    @Test
    public void testDescriptiveStatsDouble() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1, 5, 5);

        Mean mean = new Mean(x);
        opExecutioner.exec(mean);
        assertEquals(3.0, mean.getFinalResult().doubleValue(), 1e-1);

        Variance variance = new Variance(x.dup(), true);
        opExecutioner.exec(variance);
        assertEquals(getFailureMessage(), 2.5, variance.getFinalResult().doubleValue(), 1e-1);
    }


    @Test
    public void testDescriptiveStats() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1, 5, 5);

        Mean mean = new Mean(x);
        opExecutioner.exec(mean);
        assertEquals(getFailureMessage(), 3.0, mean.getFinalResult().doubleValue(), 1e-1);

        Variance variance = new Variance(x.dup(), true);
        opExecutioner.exec(variance);
        assertEquals(getFailureMessage(), 2.5, variance.getFinalResult().doubleValue(), 1e-1);
    }

    @Test
    public void testRowSoftmax() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6);
        OldSoftMax softMax = new OldSoftMax(arr);
        opExecutioner.exec(softMax);
        assertEquals(getFailureMessage(), 1.0, softMax.z().sumNumber().doubleValue(), 1e-1);
    }

    @Test
    public void testAddiRowVector() {
        INDArray arr = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray arr2 = Nd4j.linspace(1, 3, 3);
        INDArray assertion = Nd4j.create(new double[] {2, 4, 6, 5, 7, 9}).reshape(2, 3);
        INDArray test = arr.addRowVector(arr2);
        assertEquals(assertion, test);
    }

    @Test
    public void testTad() {
        INDArray arr = Nd4j.linspace(1, 12, 12).reshape(2, 3, 2);
        for (int i = 0; i < arr.tensorssAlongDimension(0); i++) {
            System.out.println(arr.tensorAlongDimension(i, 0));
        }
    }


    @Test
    public void testPow() {
        INDArray oneThroughSix = Nd4j.linspace(1, 6, 6);
        Pow pow = new Pow(oneThroughSix, 2);
        Nd4j.getExecutioner().exec(pow);
        INDArray answer = Nd4j.create(new float[] {1, 4, 9, 16, 25, 36});
        assertEquals(getFailureMessage(), answer, pow.z());
    }


    @Test
    public void testComparisonOps() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        INDArray ones = Nd4j.ones(6);
        INDArray zeros = Nd4j.zeros(6);
        assertEquals(ones, Nd4j.getExecutioner().execAndReturn(new ScalarGreaterThan(linspace, 0)));
        assertEquals(zeros, Nd4j.getExecutioner().execAndReturn(new ScalarGreaterThan(linspace, 7)));
        assertEquals(zeros, Nd4j.getExecutioner().execAndReturn(new ScalarLessThan(linspace, 0)));
        assertEquals(ones, Nd4j.getExecutioner().execAndReturn(new ScalarLessThan(linspace, 7)));
    }

    @Test
    public void testScalarArithmetic() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        INDArray plusOne = Nd4j.linspace(2, 7, 6);
        Nd4j.getExecutioner().exec(new ScalarAdd(linspace, 1));
        assertEquals(plusOne, linspace);
    }

    @Test
    public void testDimensionMax() {
        INDArray linspace = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        int axis = 0;
        INDArray row = linspace.slice(axis);
        Max max = new Max(row);
        double max2 = Nd4j.getExecutioner().execAndReturn(max).getFinalResult().doubleValue();
        assertEquals(3.0, max2, 1e-1);

        Min min = new Min(row);
        double min2 = Nd4j.getExecutioner().execAndReturn(min).getFinalResult().doubleValue();
        assertEquals(1.0, min2, 1e-1);
        Max matrixMax = new Max(linspace);
        INDArray exec2 = Nd4j.getExecutioner().exec(matrixMax, 1);
        assertEquals(Nd4j.create(new double[] {3, 6}), exec2);
    }


    @Test
    public void testStridedLog() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray slice = arr.slice(0);
        Log exp = new Log(slice);
        opExecutioner.exec(exp);
        INDArray assertion = Nd4j.create(Nd4j.createBuffer(new double[] {0.0, 0.6931471824645996, 1.0986123085021973}));
        assertEquals(getFailureMessage(), assertion, slice);
    }

    @Test
    public void testStridedExp() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray slice = arr.slice(0);
        // FIXME: int cast
        float[] expected = new float[(int) slice.length()];
        for (int i = 0; i < slice.length(); i++)
            expected[i] = (float) Math.exp(slice.getDouble(i));
        Exp exp = new Exp(slice);
        opExecutioner.exec(exp);
        assertEquals(getFailureMessage(), Nd4j.create(Nd4j.createBuffer(expected)), slice);
    }

    @Test
    public void testSoftMax() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6);
        OldSoftMax softMax = new OldSoftMax(arr);
        opExecutioner.exec(softMax);
        assertEquals(getFailureMessage(), 1.0, softMax.z().sumNumber().doubleValue(), 1e-1);

        INDArray linspace = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        OldSoftMax softmax = new OldSoftMax(linspace.dup());
        Nd4j.getExecutioner().exec(softmax);
        assertEquals(linspace.rows(), softmax.z().sumNumber().doubleValue(), 1e-1);
    }


    @Test
    public void testDimensionSoftMax() {
        INDArray linspace = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        OldSoftMax max = new OldSoftMax(linspace);
        Nd4j.getExecutioner().exec(max, 1);
        linspace.assign(max.z());
        assertEquals(getFailureMessage(), linspace.getRow(0).sumNumber().doubleValue(), 1.0, 1e-1);
    }

    @Test
    public void testColumnMean() {
        INDArray twoByThree = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray columnMean = twoByThree.mean(0);
        INDArray assertion = Nd4j.create(new float[] {2, 3});
        assertEquals(assertion, columnMean);
    }


    @Test
    public void testColumnVar() {
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnStd = twoByThree.var(0);
        INDArray assertion = Nd4j.create(new float[] {30200f, 30200f, 30200f, 30200f});
        assertEquals(assertion, columnStd);
    }

    @Test
    public void testColumnStd() {
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnStd = twoByThree.std(0);
        INDArray assertion = Nd4j.create(new float[] {173.78147196982766f, 173.78147196982766f, 173.78147196982766f,
                        173.78147196982766f});
        assertEquals(assertion, columnStd);
    }

    @Test
    public void testDim1() {
        INDArray sum = Nd4j.linspace(1, 2, 2).reshape(2, 1);
        INDArray same = sum.dup();
        assertEquals(same.sum(1), sum);
    }

    @Test
    public void testIMax() {
        INDArray arr = Nd4j.linspace(1, 10, 10);
        IMax imax = new IMax(arr);
        assertEquals(9, ((IndexAccumulation) Nd4j.getExecutioner().exec(imax)).getFinalResult());

        arr.muli(-1);
        imax = new IMax(arr);
        int maxIdx = ((IndexAccumulation) Nd4j.getExecutioner().exec(imax)).getFinalResult();
        assertEquals(0, maxIdx);
    }

    @Test
    public void testIMin() {
        INDArray arr = Nd4j.linspace(1, 10, 10);
        IMin imin = new IMin(arr);
        assertEquals(0, ((IndexAccumulation) Nd4j.getExecutioner().exec(imin)).getFinalResult());

        arr.muli(-1);
        imin = new IMin(arr);
        int minIdx = ((IndexAccumulation) Nd4j.getExecutioner().exec(imin)).getFinalResult();
        assertEquals(9, minIdx);
    }



    @Test
    public void testMeanSumSimple() {
        System.out.println("3d");
        INDArray arr = Nd4j.ones(1, 4, 4);
        assertEquals(Nd4j.ones(1), arr.mean(1, 2));
        assertEquals(Nd4j.ones(1).muli(16), arr.sum(1, 2));

        System.out.println("4d");
        INDArray arr4 = Nd4j.ones(1, 1, 4, 4);
        INDArray arr4m = arr4.mean(2, 3);
        INDArray arr4s = arr4.sum(2, 3);
        for (int i = 0; i < arr4m.length(); i++)
            assertEquals(arr4m.getDouble(i), 1, 1e-1);
        for (int i = 0; i < arr4s.length(); i++)
            assertEquals(arr4s.getDouble(i), 16, 1e-1);
        System.out.println("5d");
        INDArray arr5 = Nd4j.ones(1, 1, 4, 4, 4);
        INDArray arr5s = arr5.sum(2, 3);
        for (int i = 0; i < arr5s.length(); i++)
            assertEquals(arr5s.getDouble(i), 16, 1e-1);
        INDArray arr5m = arr5.mean(2, 3);
        for (int i = 0; i < arr5m.length(); i++)
            assertEquals(1, arr5m.getDouble(i), 1e-1);

        System.out.println("6d");
        INDArray arr6 = Nd4j.ones(1, 1, 4, 4, 4, 4);
        INDArray arr6m = arr6.mean(2, 3);
        for (int i = 0; i < arr6m.length(); i++)
            assertEquals(arr6m.getDouble(i), 1, 1e-1);

        INDArray arr6s = arr6.sum(2, 3);

        for (int i = 0; i < arr6s.length(); i++)
            assertEquals(arr6s.getDouble(i), 16, 1e-1);
    }

    @Test
    public void testSum6d() {
        INDArray arr6 = Nd4j.ones(1, 1, 4, 4, 4, 4);
        INDArray arr6s = arr6.sum(2, 3);
        for (int i = 0; i < arr6s.length(); i++)
            assertEquals(16, arr6s.getDouble(i), 1e-1);

    }

    @Test
    public void testMean() {
        int[] shape = new int[] {1, 2, 2, 2, 2, 2};
        int len = ArrayUtil.prod(shape);
        INDArray val = Nd4j.linspace(1, len, len).reshape('c', shape);
        /**
         * Failure comes from the lack of a jump
         * when doing tad offset in c++
         *
         * We need to jump from the last element rather than the
         * first for the next element.
         *
         * This happens when the index for a tad is >= the
         * stride[0]
         *
         * When the index is >= a stride[0] then you take
         * the offset at the end of the tad and use that +
         * (possibly the last stride?)
         * to get to the next offset.
         *
         * In order to get to the last element for a jump, just iterate
         * over the tad (coordinate wise) to get the coordinate pair +
         * offset at which to do compute.
         *
         * Another possible solution is to create an initialize pointer
         * method that will just set up the tad pointer directly.
         * Right now it is a simplistic base pointer + offset that
         * we could turn in to an init method instead.
         * This would allow use to use coordinate based techniques
         * on the pointer directly. The proposal here
         * would then be turning tad offset given an index
         * in to a pointer initialization method which
         * will auto insert the pointer at the right index.
         */
        INDArray sum = val.sum(2, 3);
        double[] assertionData = new double[] {28.0, 32.0, 36.0, 40.0, 92.0, 96.0, 100.0, 104.0};

        INDArray avgExpected = Nd4j.create(assertionData).reshape(1, 2, 2, 2);

        assertEquals(avgExpected, sum);
    }

    @Test
    public void testSum5d() throws Exception {
        System.out.println("5d");
        INDArray arr5 = Nd4j.ones(1, 1, 4, 4, 4);
        INDArray arr5s = arr5.sum(2, 3);
        Thread.sleep(1000);
        System.out.println("5d length: " + arr5s.length());
        for (int i = 0; i < arr5s.length(); i++)
            assertEquals(16, arr5s.getDouble(i), 1e-1);


        INDArray arrF = Nd4j.ones(1, 1, 4, 4, 4);
        System.out.println("A: " + arrF);
    }


    @Test
    public void testOneMinus() {
        INDArray in = Nd4j.linspace(1, 3, 3);
        INDArray out = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("timesoneminus", in));

        //Expect: 0, -2, -6 -> from 1*(1-1), 2*(1-2), 3*(1-3). Getting: [0,0,0]
        INDArray exp = Nd4j.create(new double[] {0, -2.0, -6.0});
        assertEquals(out, exp);
    }



    @Test
    public void testSubColumnVector() {
        INDArray vec = Nd4j.linspace(1, 18, 18);
        INDArray matrix = vec.dup().reshape(3, 6);
        INDArray vector = Nd4j.create(new double[] {6, 12, 18}).reshape(3, 1);
        INDArray assertion = Nd4j.create(new double[] {-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, -5.0, -4.0, -3.0, -2.0, -1.0,
                        0.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0}, new int[] {3, 6});
        INDArray test = matrix.subColumnVector(vector);
        assertEquals(assertion, test);
    }

    @Test
    public void testLogSoftmaxVector() {
        INDArray temp = Nd4j.create(new double[] {1.0, 2.0, 3.0, 4.0});
        INDArray logsoftmax = Nd4j.getExecutioner().execAndReturn(new LogSoftMax(temp.dup()));
        INDArray assertion = Nd4j.create(new double[] {-3.4401898, -2.4401898, -1.4401897, -0.44018975});
        assertEquals(assertion, logsoftmax);

    }


    @Test
    public void testSumDifferentOrder() {
        INDArray toAssign = Nd4j.linspace(0, 3, 4).reshape(2, 2);
        INDArray cOrder = Nd4j.create(new int[] {2, 2}, 'c').assign(toAssign);
        INDArray fOrder = Nd4j.create(new int[] {2, 2}, 'f').assign(toAssign);

        System.out.println(cOrder);
        System.out.println(cOrder.sum(0)); //[2,4] -> correct
        System.out.println(fOrder.sum(0)); //[2,3] -> incorrect

        assertEquals(cOrder, fOrder);
        assertEquals(cOrder.sum(0), fOrder.sum(0));
    }

    @Test
    public void testLogSoftmax() {
        INDArray test = Nd4j.create(new double[] {-0.115370326, -0.12137828, -0.120233774, -0.12121266, -0.11363905,
                        -0.101017155, -0.11571029, -0.116997495, -0.123033985, -0.1222254, -0.11120513, -0.11710341,
                        -0.12319958, -0.124424405, -0.105285235, -0.08768927, -0.10296882, -0.11346505, -0.10607526,
                        -0.10681274, -0.11604863, -0.1070115, -0.114202365, -0.11168295, -0.11615404, -0.120522454,
                        -0.11282451, -0.11514864, -0.11681116, -0.11987897, -0.12054029, -0.112625614, -0.10337835,
                        -0.098809384, -0.1222254, -0.11966098, -0.11500366, -0.1222254, -0.122691356, -0.1168594,
                        -0.11369472, -0.11666928, -0.12075868, -0.10658686, -0.10251844, -0.119958505, -0.10873747,
                        -0.12036781, -0.11125211, -0.118474, 0.07354958, 0.06268418, 0.08751996, 0.05259535, 0.07969022,
                        0.062334962, 0.07089297, -0.006484107, 0.0702586, 0.03601057, 0.03228142, 0.051330067,
                        0.048092633, 0.0753836, 0.0026741663, 0.060346458, 0.064265735, 0.03208362, 0.07322607,
                        0.034286126, 0.08459597, 0.040570714, 0.08494339, 0.06835921, 0.055334114, 0.06346921,
                        0.08284429, 0.09769646, 0.07128828, 0.0012985547, 0.033257447, 0.024084045, 0.03130147,
                        0.09381818, 0.062283173, 0.049273495, 0.0789609, 0.06648661, 0.030163772, 0.047266945,
                        0.05704684, 0.06862679, 0.04134995, 0.0029913357, 0.050757334, 0.031863946, 0.043180045,
                        0.053592253, -0.02633951, 0.04229047, 0.12401424, 0.1025523, 0.11914653, 0.10838079,
                        0.119204566, 0.120582364, 0.079642124, 0.1136303, 0.103594445, 0.12434465, 0.10481718,
                        0.10615024, 0.1161067, 0.101516, 0.11543929, 0.11498181, 0.1083647, 0.12498043, 0.117732316,
                        0.080594465, 0.12140614, 0.10168964, 0.11630502, 0.097365364, 0.11659742, 0.11525785,
                        0.095346555, 0.095523514, 0.1145297, 0.10820676, 0.113681756, 0.12088448, 0.11661095,
                        0.09196416, 0.09367608, 0.12396194, 0.11715822, 0.10781161, 0.09206241, 0.11529953, 0.12193694,
                        0.11471913, 0.1025523, 0.12246918, 0.12278436, 0.11647938, 0.09907566, 0.10939402, 0.11121245,
                        0.09931412, -0.2015398, -0.19392101, -0.19934568, -0.19083071, -0.20022182, -0.18812077,
                        -0.19819336, -0.19751601, -0.18787658, -0.1910854, -0.19982933, -0.19259657, -0.1910668,
                        -0.19623408, -0.20643783, -0.17979786, -0.20085241, -0.20226628, -0.1943775, -0.19513902,
                        -0.1944603, -0.19675966, -0.20814213, -0.19372807, -0.18230462, -0.18796724, -0.19594413,
                        -0.19937015, -0.20221426, -0.1900377, -0.18905015, -0.20246184, -0.18973471, -0.1917036,
                        -0.1910854, -0.2045007, -0.20772256, -0.1910854, -0.19349803, -0.19836159, -0.20438254,
                        -0.16650572, -0.19694945, -0.19511227, -0.18056169, -0.19521528, -0.19218414, -0.19556037,
                        -0.1989097, -0.19989866, 0.110895164, 0.09209204, 0.13636513, 0.09708423, 0.12663901,
                        0.11280878, 0.10437618, 0.008251642, 0.11656475, 0.062448665, 0.07663319, 0.076713376,
                        0.09773914, 0.1284772, 0.0019391886, 0.08873351, 0.10645666, 0.06874694, 0.12830636,
                        0.069761865, 0.12597786, 0.064558044, 0.14945637, 0.12600589, 0.08889626, 0.096229844,
                        0.13689923, 0.15111938, 0.11476847, 0.012906413, 0.06886689, 0.05653629, 0.056540295, 0.1647724,
                        0.1054803, 0.06795046, 0.12039944, 0.11954296, 0.052694272, 0.085520394, 0.110611565,
                        0.11398453, 0.07550961, 0.023511963, 0.090924345, 0.0600122, 0.07526812, 0.088270955,
                        -0.03518031, 0.073293336, 0.17944553, 0.16982275, 0.1886539, 0.18693338, 0.18788463, 0.2058602,
                        0.13861835, 0.20437749, 0.18895163, 0.16544276, 0.149991, 0.17463979, 0.17583887, 0.16696452,
                        0.16749835, 0.1592365, 0.17954215, 0.1818188, 0.21207899, 0.15266286, 0.17395115, 0.15906107,
                        0.21057771, 0.15467106, 0.17414747, 0.19151127, 0.14792846, 0.14762704, 0.1860418, 0.18808068,
                        0.19654934, 0.17514904, 0.18510495, 0.16045007, 0.18320344, 0.18669076, 0.16069236, 0.17718756,
                        0.14080223, 0.1681495, 0.17300002, 0.1528326, 0.16982275, 0.1817097, 0.16696694, 0.16177535,
                        0.1604718, 0.16464049, 0.15210003, 0.16091338, 0.19544502, 0.1334315, 0.16168839, 0.11322618,
                        0.19517533, 0.18929626, 0.17545204, 0.1665815, 0.09131178, 0.11004268, 0.20550796, 0.13831247,
                        0.10610545, 0.12289211, 0.27147663, 0.20504008, 0.2518754, 0.20981932, 0.20138234, 0.19962592,
                        0.15790789, 0.20949593, 0.23528637, 0.18096939, 0.08758456, 0.10911943, 0.18139273, 0.18525626,
                        0.19391479, 0.11438076, 0.1093913, 0.22006766, 0.18334126, 0.21811387, 0.11004268, 0.19371085,
                        0.23279056, 0.11004268, 0.11990581, 0.17242423, 0.21975593, 0.046734467, 0.1444371, 0.20759591,
                        0.13962208, 0.14867997, 0.17288592, 0.14028637, 0.19978605, 0.1737019, -0.038705423,
                        -0.03880039, -0.060744748, 0.005578369, -0.026154364, -0.09166601, -0.061155446, 0.008943805,
                        -0.04777039, -0.012912485, -0.010861377, -0.01913654, -0.0061141956, -0.09119834, 0.034481876,
                        -0.008210908, -0.09062711, -0.0464008, -0.0038113478, -0.006515413, -0.06737334, 0.022068182,
                        -0.078238964, -0.10467487, -0.012385059, -0.008899481, -0.0507185, -0.0612416, -0.05302817,
                        0.03657996, 0.0040081483, 0.0017336496, 0.00966107, -0.13457696, -0.106228024, -0.05810899,
                        -0.042826205, -0.004804179, -0.054947495, -0.0023088162, -0.083174944, -0.0812491, 0.0012216767,
                        0.017188948, -0.0416347, -0.0750825, -0.052436177, -0.028371494, 0.07799446, -0.02655019,
                        -0.04801802, -0.11302035, -0.114139326, -0.17401277, -0.11443192, -0.19375448, -0.08697115,
                        -0.22462566, -0.18594599, 0.029962104, -0.03072077, -0.10795037, -0.0687454, -0.08853653,
                        -0.02800453, -0.0044006817, -0.14119355, -0.057319514, -0.23839943, -0.09940908, -0.03132951,
                        -0.07696326, -0.23962279, -0.05578459, -0.073864885, -0.16175121, -0.046830498, -0.071334355,
                        -0.12525235, -0.1762308, -0.17853433, -0.05481769, -0.10788009, -0.12848935, -0.21946594,
                        -0.07054761, -0.0043790466, -0.1421547, -0.062456187, -0.038439218, -0.01970637, 0.04187341,
                        -0.11302035, -0.06571084, 0.012916437, 0.008474918, -0.058553338, -0.05822342, -0.0072570713,
                        -0.117029555}, new int[] {150, 3}, 'c');
        INDArray assertion = Nd4j.create(new double[] {-1.0949919, -1.1009998, -1.0998554, -1.1079034, -1.1003298,
                        -1.0877079, -1.0957471, -1.0970343, -1.1030709, -1.1040032, -1.0929829, -1.0988811, -1.1042137,
                        -1.1054386, -1.0862994, -1.0849832, -1.1002628, -1.110759, -1.0950522, -1.0957897, -1.1050256,
                        -1.0946627, -1.1018535, -1.0993341, -1.098271, -1.1026394, -1.0949415, -1.0964833, -1.0981458,
                        -1.1012137, -1.1069958, -1.0990812, -1.0898339, -1.0839114, -1.1073275, -1.104763, -1.0936487,
                        -1.1008704, -1.1013364, -1.0997316, -1.0965669, -1.0995414, -1.1094468, -1.0952749, -1.0912066,
                        -1.1022308, -1.0910097, -1.10264, -1.1618325, -1.1690543, -0.97703075, -1.1036359, -1.0788001,
                        -1.1137247, -1.0899199, -1.1072751, -1.0987172, -1.13885, -1.0621073, -1.0963553, -1.1102668,
                        -1.0912181, -1.0944556, -1.0698514, -1.1425608, -1.0848886, -1.0910273, -1.1232094, -1.0820669,
                        -1.1177288, -1.0674189, -1.1114442, -1.083288, -1.0998721, -1.1128973, -1.1165779, -1.0972028,
                        -1.0823506, -1.063015, -1.1330047, -1.1010458, -1.1247563, -1.1175389, -1.0550222, -1.0999088,
                        -1.1129185, -1.0832311, -1.0802083, -1.1165311, -1.0994279, -1.0973024, -1.0857224, -1.1129993,
                        -1.124351, -1.076585, -1.0954784, -1.0795343, -1.0691221, -1.1490538, -1.1465356, -1.0648118,
                        -1.0862738, -1.0950559, -1.1058216, -1.0949979, -1.0828075, -1.1237478, -1.0897596, -1.1059818,
                        -1.0852317, -1.1047591, -1.100405, -1.0904485, -1.1050392, -1.0961069, -1.0965644, -1.1031815,
                        -1.0815891, -1.0888373, -1.125975, -1.0903746, -1.1100911, -1.0954757, -1.1110255, -1.0917934,
                        -1.093133, -1.1051062, -1.1049292, -1.0859231, -1.1046766, -1.0992017, -1.0919989, -1.082815,
                        -1.1074618, -1.10575, -1.0909829, -1.0977867, -1.1071333, -1.116398, -1.0931609, -1.0865234,
                        -1.0971736, -1.1093404, -1.0894235, -1.0886579, -1.0949628, -1.1123666, -1.095872, -1.0940536,
                        -1.1059519, -1.1018884, -1.0942696, -1.0996943, -1.0963987, -1.1057898, -1.0936887, -1.102288,
                        -1.1016107, -1.0919713, -1.0952013, -1.1039451, -1.0967125, -1.0917866, -1.0969539, -1.1071577,
                        -1.0841576, -1.1052121, -1.106626, -1.098331, -1.0990925, -1.0984138, -1.095848, -1.1072304,
                        -1.0928164, -1.0921938, -1.0978565, -1.1058333, -1.1007886, -1.1036327, -1.0914562, -1.0939325,
                        -1.1073442, -1.0946171, -1.0945718, -1.0939536, -1.107369, -1.1089264, -1.0922892, -1.0947019,
                        -1.1073625, -1.1133835, -1.0755067, -1.1047142, -1.102877, -1.0883265, -1.0995088, -1.0964776,
                        -1.0998539, -1.2125868, -1.2135757, -0.9027819, -1.115231, -1.0709579, -1.1102388, -1.0866234,
                        -1.1004536, -1.1088862, -1.1537597, -1.0454466, -1.0995628, -1.1057239, -1.1056436, -1.0846179,
                        -1.0445701, -1.1711081, -1.0843138, -1.0936275, -1.1313372, -1.0717777, -1.1160054, -1.0597894,
                        -1.1212093, -1.0709189, -1.0943694, -1.131479, -1.1307347, -1.0900652, -1.0758451, -1.0502236,
                        -1.1520857, -1.0961251, -1.1360092, -1.1360053, -1.0277731, -1.091318, -1.1288478, -1.0763988,
                        -1.065361, -1.1322097, -1.0993836, -1.0881867, -1.0848137, -1.1232886, -1.133629, -1.0662166,
                        -1.0971287, -1.0676445, -1.0546416, -1.1780928, -1.1673087, -1.0611565, -1.0707793, -1.0977826,
                        -1.0995032, -1.0985519, -1.0761919, -1.1434338, -1.0776746, -1.0779177, -1.1014266, -1.1168783,
                        -1.0964613, -1.0952622, -1.1041365, -1.0999078, -1.1081696, -1.0878639, -1.0992746, -1.0690144,
                        -1.1284306, -1.1060928, -1.1209829, -1.0694662, -1.1174977, -1.0980213, -1.0806575, -1.1113796,
                        -1.111681, -1.0732663, -1.0971633, -1.0886947, -1.110095, -1.0898226, -1.1144775, -1.0917242,
                        -1.0868361, -1.1128345, -1.0963393, -1.1185608, -1.0912135, -1.086363, -1.1139716, -1.0969814,
                        -1.0850945, -1.0947206, -1.0999122, -1.1012157, -1.0932035, -1.105744, -1.0969306, -1.0670104,
                        -1.1290239, -1.100767, -1.1519758, -1.0700266, -1.0759057, -1.0683149, -1.0771854, -1.1524552,
                        -1.1406635, -1.0451982, -1.1123937, -1.1621376, -1.1453509, -0.99676645, -1.1160396, -1.0692043,
                        -1.1112604, -1.0837362, -1.0854926, -1.1272106, -1.0979462, -1.0721557, -1.1264727, -1.1378707,
                        -1.1163357, -1.0440625, -1.0785028, -1.0698442, -1.1493783, -1.1612072, -1.0505308, -1.0872571,
                        -1.0555155, -1.1635867, -1.0799185, -1.0216377, -1.1443856, -1.1345224, -1.0751246, -1.0277929,
                        -1.2008144, -1.1185431, -1.0553844, -1.1233582, -1.1039788, -1.0797728, -1.1123724, -1.0159799,
                        -1.0420641, -1.2544713, -1.1064723, -1.1284167, -1.0620935, -1.0654664, -1.1309781, -1.1004674,
                        -1.0726943, -1.1294085, -1.0945506, -1.0974507, -1.1057259, -1.0927036, -1.1695204, -1.0438402,
                        -1.086533, -1.1429209, -1.0986946, -1.0561051, -1.0885462, -1.149404, -1.0599625, -1.112509,
                        -1.1389449, -1.046655, -1.0674819, -1.1093009, -1.119824, -1.1481767, -1.0585686, -1.0911404,
                        -1.0579745, -1.050047, -1.194285, -1.136149, -1.08803, -1.0727472, -1.0830219, -1.1331651,
                        -1.0805265, -1.1281672, -1.1262413, -1.0437706, -1.0489775, -1.1078012, -1.141249, -1.1517346,
                        -1.1276698, -1.0213039, -1.0633042, -1.084772, -1.1497743, -1.0789506, -1.1388241, -1.0792432,
                        -1.125674, -1.0188907, -1.1565453, -1.2263924, -1.0104843, -1.0711672, -1.1182799, -1.079075,
                        -1.0988661, -1.0705098, -1.046906, -1.1836989, -1.0271709, -1.2082508, -1.0692605, -1.017894,
                        -1.0635278, -1.2261873, -1.0583237, -1.0764041, -1.1642903, -1.0648377, -1.0893415, -1.1432595,
                        -1.140007, -1.1423105, -1.0185939, -1.0557104, -1.0763197, -1.1672963, -1.09838, -1.0322114,
                        -1.1699871, -1.1210208, -1.0970039, -1.078271, -1.0132385, -1.1681323, -1.1208228, -1.0738388,
                        -1.0782803, -1.1453086, -1.0970035, -1.0460371, -1.1558095}, new int[] {150, 3}, 'c');
        Nd4j.getExecutioner().exec(new LogSoftMax(test));
        assertEquals(assertion, test);


    }

    @Test
    public void testSoftmax() {
        INDArray vec = Nd4j.linspace(1, 18, 18);
        INDArray matrix = vec.dup().reshape(3, 6);
        Nd4j.getExecutioner().exec(new OldSoftMax(matrix));
        INDArray assertion = Nd4j.create(
                        new double[] {0.0042697787, 0.011606461, 0.031549633, 0.085760795, 0.23312202, 0.6336913,
                                        0.0042697787, 0.011606461, 0.031549633, 0.085760795, 0.23312202, 0.6336913,
                                        0.0042697787, 0.011606461, 0.031549633, 0.085760795, 0.23312202, 0.6336913},
                        new int[] {3, 6}, 'c');
        assertEquals(assertion, matrix);
    }

    @Test
    public void testStdev() {
        INDArray arr = Nd4j.create(new float[] {0.9296161f, 0.31637555f, 0.1839188f}, new int[] {1, 3}, ordering());
        double stdev = arr.stdNumber().doubleValue();
        double stdev2 = arr.std(1).getDouble(0);
        assertEquals(stdev, stdev2, 1e-3);

        double exp = 0.39784279465675354;
        assertEquals(exp, stdev, 1e-7f);
    }

    @Test
    public void testVariance() {

        INDArray arr = Nd4j.create(new float[] {0.9296161f, 0.31637555f, 0.1839188f}, new int[] {1, 3}, ordering());
        double var = arr.varNumber().doubleValue();
        INDArray temp = arr.var(1);
        double var2 = arr.var(1).getDouble(0);
        assertEquals(var, var2, 1e-1);

        double exp = 0.15827888250350952;
        assertEquals(exp, var, 1e-7f);
    }

    @Test
    public void testEpsOps() {
        INDArray ones = Nd4j.ones(6);
        double tiny = 1.000000000000001;
        assertTrue(ones.eps(tiny).sumNumber().doubleValue() == 6);
        INDArray consec = Nd4j.linspace(1, 6, 6);
        assertTrue(consec.eps(5).sumNumber().doubleValue() == 1);
        assertTrue(consec.sub(1).eps(5).sumNumber().doubleValue() == 1);
        assertTrue(consec.sub(1).eps(5).getDouble(0, 5) == 1);
    }

    @Test
    public void testVarianceSingleVsMultipleDimensions() {
        // this test should always run in double
        DataType type = Nd4j.dataType();
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(12345);

        //Generate C order random numbers. Strides: [500,100,10,1]
        INDArray fourd = Nd4j.rand('c', new int[] {100, 5, 10, 10}).muli(10);
        INDArray twod = Shape.newShapeNoCopy(fourd, new int[] {100, 5 * 10 * 10}, false);

        //Population variance. These two should be identical
        INDArray var4 = fourd.var(false, 1, 2, 3);
        INDArray var2 = twod.var(false, 1);

        //Manual calculation of population variance, not bias corrected
        //https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Na.C3.AFve_algorithm
        double[] sums = new double[100];
        double[] sumSquares = new double[100];
        NdIndexIterator iter = new NdIndexIterator(fourd.shape());
        while (iter.hasNext()) {
            val next = iter.next();
            double d = fourd.getDouble(next);

            // FIXME: int cast
            sums[(int) next[0]] += d;
            sumSquares[(int) next[0]] += d * d;
        }

        double[] manualVariance = new double[100];
        val N = (fourd.length() / sums.length);
        for (int i = 0; i < sums.length; i++) {
            manualVariance[i] = (sumSquares[i] - (sums[i] * sums[i]) / N) / N;
        }

        INDArray var4bias = fourd.var(true, 1, 2, 3);
        INDArray var2bias = twod.var(true, 1);

        assertArrayEquals(var2.data().asDouble(), var4.data().asDouble(), 1e-5);
        assertArrayEquals(manualVariance, var2.data().asDouble(), 1e-5);
        assertArrayEquals(var2bias.data().asDouble(), var4bias.data().asDouble(), 1e-5);

        DataTypeUtil.setDTypeForContext(type);
    }


    @Test
    public void testHistogram1() throws Exception {
        INDArray x = Nd4j.linspace(1, 1000, 100000);
        INDArray z = Nd4j.zeros(20);

        INDArray xDup = x.dup();
        INDArray zDup = z.dup();

        INDArray zExp = Nd4j.create(20).assign(5000);

        Histogram histogram = new Histogram(x, z);

        Nd4j.getExecutioner().exec(histogram);

        assertEquals(xDup, x);
        assertNotEquals(zDup, z);

        log.info("bins: {}", z);

        assertEquals(zExp, z);
    }

    @Test
    public void testHistogram2() throws Exception {
        INDArray x = Nd4j.create(new float[] {0f, 0f, 0f, 5f, 5f, 5f, 10f, 10f, 10f});


        INDArray xDup = x.dup();

        INDArray zExp = Nd4j.zeros(10).putScalar(0, 3f).putScalar(5, 3f).putScalar(9, 3f);

        Histogram histogram = new Histogram(x, 10);

        Nd4j.getExecutioner().exec(histogram);

        INDArray z = histogram.z();

        assertEquals(xDup, x);

        log.info("bins: {}", z);

        assertEquals(zExp, z);
    }

    @Test
    public void testEuclideanManhattanDistanceAlongDimension_Rank4() {
        DataType initialType = Nd4j.dataType();
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(12345);
        INDArray firstOneExample = Nd4j.linspace(1, 8, 8).reshape('c', new int[] {1, 2, 2, 2});
        INDArray secondOneExample = firstOneExample.add(1);

        double[] d1 = firstOneExample.data().asDouble();
        double[] d2 = secondOneExample.data().asDouble();
        double sumSquaredDiff = 0.0;
        double expManhattanDistance = 0.0;
        for (int i = 0; i < d1.length; i++) {
            double diff = d1[i] - d2[i];
            sumSquaredDiff += diff * diff;
            expManhattanDistance += Math.abs(diff);
        }
        double expectedEuclidean = Math.sqrt(sumSquaredDiff);
        System.out.println("Expected, Euclidean: " + expectedEuclidean);
        System.out.println("Expected, Manhattan: " + expManhattanDistance);

        int mb = 2;
        INDArray firstOrig = Nd4j.create(mb, 2, 2, 2);
        INDArray secondOrig = Nd4j.create(mb, 2, 2, 2);
        for (int i = 0; i < mb; i++) {
            firstOrig.put(new INDArrayIndex[] {point(i), all(), all(), all()}, firstOneExample);
            secondOrig.put(new INDArrayIndex[] {point(i), all(), all(), all()}, secondOneExample);
        }

        for (char order : new char[] {'c', 'f'}) {
            INDArray first = firstOrig.dup(order);
            INDArray second = secondOrig.dup(order);

            assertEquals(firstOrig, first);
            assertEquals(secondOrig, second);


            INDArray out = Nd4j.getExecutioner().exec(new EuclideanDistance(first, second), 1, 2, 3);
            for (int i = 0; i < first.tensorssAlongDimension(1, 2, 3); i++) {
                val j = first.javaTensorAlongDimension(i, 1, 2, 3).shapeInfoDataBuffer().asLong();
                val t = first.tensorAlongDimension(i, 1, 2, 3).shapeInfoDataBuffer().asLong();

                log.info("J: {}", j);
                log.info("T: {}\n\n", t);
                assertArrayEquals(j, t);
            }
            Pair<DataBuffer, DataBuffer> firstTadInfo =
                            Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(first, 1, 2, 3);
            Pair<DataBuffer, DataBuffer> secondTadInfo =
                            Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(second, 1, 2, 3);
            for (int i = 0; i < first.tensorssAlongDimension(1, 2, 3); i++) {
                assertEquals(first.javaTensorAlongDimension(i, 1, 2, 3).offset(), firstTadInfo.getSecond().getLong(i));
                assertEquals(second.javaTensorAlongDimension(i, 1, 2, 3).offset(),
                                secondTadInfo.getSecond().getLong(i));
            }

            INDArray outManhattan = Nd4j.getExecutioner().exec(new ManhattanDistance(first, second), 1, 2, 3);

            System.out.println("\n\nOrder: " + order);
            System.out.println("Euclidean:");
            System.out.println(Arrays.toString(out.getRow(0).dup().data().asDouble()));
            System.out.println(Arrays.toString(out.getRow(1).dup().data().asDouble()));

            assertEquals(out.getRow(0), out.getRow(1));

            System.out.println("Manhattan:");
            System.out.println(Arrays.toString(outManhattan.getRow(0).dup().data().asDouble()));
            System.out.println(Arrays.toString(outManhattan.getRow(1).dup().data().asDouble()));

            assertEquals(expManhattanDistance, outManhattan.getRow(0).getDouble(0), 1e-5);
            assertEquals(expectedEuclidean, out.getRow(0).getDouble(0), 1e-5);
        }

        DataTypeUtil.setDTypeForContext(initialType);
    }

    @Test
    public void testPile1() throws Exception {
        List<INDArray> arrays = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            arrays.add(Nd4j.create(10, 10).assign(i));
        }

        INDArray pile = Nd4j.pile(arrays);

        assertEquals(3, pile.rank());
        for (int i = 0; i < 10; i++) {
            assertEquals((float) i, pile.tensorAlongDimension(i, 1, 2).getDouble(0), 0.01);
        }
    }

    @Test
    public void testPile2() throws Exception {
        List<INDArray> arrays = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            arrays.add(Nd4j.create(10, 10, 10).assign(i));
        }

        INDArray pile = Nd4j.pile(arrays);

        assertEquals(4, pile.rank());
        for (int i = 0; i < 10; i++) {
            assertEquals((float) i, pile.tensorAlongDimension(i, 1, 2, 3).getDouble(0), 0.01);
        }
    }

    @Test
    public void testMean1() throws Exception {
        INDArray array = Nd4j.create(32, 100, 100);
        for (int i = 0; i < 32; i++) {
            array.tensorAlongDimension(i, 1, 2).assign((float) 100 + i);
        }

        for (int i = 0; i < 32; i++) {
            INDArray tensor = array.tensorAlongDimension(i, 1, 2);
            assertEquals((float) (100 + i) * (100 * 100), tensor.sumNumber().floatValue(), 0.001f);
            assertEquals((float) 100 + i, tensor.meanNumber().floatValue(), 0.001f);
        }
    }

    @Test
    public void testMean2() throws Exception {
        INDArray array = Nd4j.create(32, 100, 100);
        for (int i = 0; i < 32; i++) {
            array.tensorAlongDimension(i, 1, 2).assign((float) 100 + i);
        }

        INDArray mean = array.mean(1, 2);
        for (int i = 0; i < 32; i++) {
            assertEquals((float) 100 + i, mean.getFloat(i), 0.001f);
        }
    }


    @Test
    public void testNorm2_1() throws Exception {
        INDArray array = Nd4j.rand(1769472, 9);

        INDArray max = array.max(1);
    }

    @Test
    public void testNorm2_2() throws Exception {
        INDArray array = Nd4j.rand(127, 164, 1, 100, Nd4j.getRandom());

        double norm2 = array.norm2Number().doubleValue();
    }

    /**
     * This test fails, but that's ok.
     * It's here only as reminder, that in some cases we can have EWS==1 for better performances.
     *
     * @throws Exception
     */
    @Test
    @Ignore
    public void testTadEws() throws Exception {
        INDArray array = Nd4j.create(32, 5, 10);
        assertEquals(1, array.tensorAlongDimension(0, 1, 2).elementWiseStride());
    }



    @Test
    public void testTear1() {
        List<INDArray> arrays = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            arrays.add(Nd4j.create(10, 10).assign(i));
        }

        INDArray pile = Nd4j.pile(arrays);

        INDArray[] tears = Nd4j.tear(pile, 1, 2);

        for (int i = 0; i < 10; i++) {
            assertEquals((float) i, tears[i].meanNumber().floatValue(), 0.01f);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
