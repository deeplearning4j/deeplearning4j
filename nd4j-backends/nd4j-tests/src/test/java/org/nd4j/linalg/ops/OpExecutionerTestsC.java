/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */


package org.nd4j.linalg.ops;

import static org.junit.Assert.*;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.exception.IllegalOpException;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 2/22/15.
 */
@RunWith(Parameterized.class)
public  class OpExecutionerTestsC extends BaseNd4jTest {

    public OpExecutionerTestsC(Nd4jBackend backend) {
        super(backend);
    }




    @Test
    public void testCosineSimilarity() {
        INDArray vec1 = Nd4j.create(new float[]{1, 2, 3, 4, 5});
        INDArray vec2 = Nd4j.create(new float[]{1, 2, 3, 4, 5});
        double sim = Transforms.cosineSim(vec1, vec2);
        assertEquals(getFailureMessage(), 1, sim, 1e-1);
    }

    @Test
    public void testLog() {
        INDArray log = Nd4j.linspace(1, 6, 6);
        INDArray transformed = Transforms.log(log);
        INDArray assertion = Nd4j.create(new double[]{0., 0.69314718, 1.09861229, 1.38629436, 1.60943791,
                1.79175947});
        assertEquals(assertion, transformed);
    }

    @Test
    public void testNorm1AlongDimension() {
        INDArray arr = Nd4j.linspace(1,8,8).reshape(2,4);
        INDArray arrNorm1 = arr.norm2(1);
        INDArray assertion = Nd4j.create(new double[]{5.47722558,  13.19090596});
        assertEquals(assertion,arrNorm1);
    }




    @Test
    public void testEuclideanDistance() {
        INDArray arr = Nd4j.create(new double[]{55, 55});
        INDArray arr2 = Nd4j.create(new double[]{60, 60});
        double result = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(arr,arr2)).currentResult().doubleValue();
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
            assertTrue(getFailureMessage(),val >= 0 && val <= 1);
        }

        INDArray linspace2 = Nd4j.linspace(1, 4, 4);
        Nd4j.getExecutioner().exec(new SetRange(linspace2, 2, 4));
        for (int i = 0; i < linspace2.length(); i++) {
            double val = linspace2.getDouble(i);
            assertTrue(getFailureMessage(),val >= 2 && val <= 4);
        }
    }

    @Test
    public void testAlongDimension() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2, 2);
        Sigmoid sigmoid = new Sigmoid(arr.dup());
        Nd4j.getExecutioner().exec(sigmoid,1);
        INDArray sigmoidREsult = Transforms.sigmoid(arr);
        assertEquals(sigmoidREsult, sigmoid.z());
    }


    @Test
    public void testNormMax() {
        INDArray arr = Nd4j.create(new float[]{1, 2, 3, 4});
        double normMax = Nd4j.getExecutioner().execAndReturn(new NormMax(arr)).currentResult().doubleValue();
        assertEquals(getFailureMessage(), 4, normMax, 1e-1);
    }




    @Test
    public void testNorm2() {
        INDArray arr = Nd4j.create(new float[]{1, 2, 3, 4});
        double norm2 = Nd4j.getExecutioner().execAndReturn(new Norm2(arr)).currentResult().doubleValue();
        assertEquals(getFailureMessage(),5.4772255750516612, norm2, 1e-1);
    }

    @Test
    public void testAdd() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.ones(5);
        INDArray xDup = x.dup();
        INDArray solution = Nd4j.valueArrayOf(5, 2.0);
        opExecutioner.exec(new AddOp(x, xDup, x));
        assertEquals(getFailureMessage(),solution, x);
    }

    @Test
    public void testMul() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.ones(5);
        INDArray xDup = x.dup();
        INDArray solution = Nd4j.valueArrayOf(5, 1.0);
        opExecutioner.exec(new MulOp(x, xDup, x));
        assertEquals(solution, x);
    }


    @Test
    public void testExecutioner() throws IllegalOpException {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.ones(5);
        INDArray xDup = x.dup();
        INDArray solution = Nd4j.valueArrayOf(5, 2.0);
        opExecutioner.exec(new AddOp(x, xDup, x));
        assertEquals(getFailureMessage(),solution, x);
        Sum acc = new Sum(x.dup());
        opExecutioner.exec(acc);
        assertEquals(getFailureMessage(), 10.0, acc.currentResult().doubleValue(), 1e-1);
        Prod prod = new Prod(x.dup());
        opExecutioner.exec(prod);
        assertEquals(getFailureMessage(),32.0, prod.currentResult().doubleValue(), 1e-1);
    }


    @Test
    public void testMaxMin() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1, 5, 5);
        Max max = new Max(x);
        opExecutioner.exec(max);
        assertEquals(5, max.currentResult().doubleValue(), 1e-1);
        Min min = new Min(x);
        opExecutioner.exec(min);
        assertEquals(1, min.currentResult().doubleValue(), 1e-1);
    }

    @Test
    public void testProd() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        Prod prod = new Prod(linspace);
        double prod2 = Nd4j.getExecutioner().execAndReturn(prod).currentResult().doubleValue();
        assertEquals(720, prod2, 1e-1);
    }

    @Test
    public void testSum() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        Sum sum = new Sum(linspace);
        double sum2 = Nd4j.getExecutioner().execAndReturn(sum).currentResult().doubleValue();
        assertEquals(21, sum2, 1e-1);

        INDArray matrixSums = linspace.reshape(2, 3);
        INDArray rowSums = matrixSums.sum(1);
        assertEquals(Nd4j.create(new double[]{6, 15}),rowSums);
    }


    @Test
    public void testDescriptiveStatsDouble() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1, 5, 5);

        Mean mean = new Mean(x);
        opExecutioner.exec(mean);
        assertEquals(3.0, mean.currentResult().doubleValue(), 1e-1);

        Variance variance = new Variance(x.dup(), true);
        opExecutioner.exec(variance);
        assertEquals(getFailureMessage(),2.5, variance.currentResult().doubleValue(), 1e-1);
    }

    @Test
    public void testBias() {
        INDArray bias = Nd4j.linspace(1, 4, 4);
        Bias biaOp = new Bias(bias);
        Nd4j.getExecutioner().exec(biaOp);
        assertEquals(0.0,biaOp.currentResult().doubleValue(),1e-1);
    }




    @Test
    public void testDescriptiveStats() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1, 5, 5);

        Mean mean = new Mean(x);
        opExecutioner.exec(mean);
        assertEquals(getFailureMessage(),3.0, mean.currentResult().doubleValue(), 1e-1);

        Variance variance = new Variance(x.dup(), true);
        opExecutioner.exec(variance);
        assertEquals(getFailureMessage(),2.5, variance.currentResult().doubleValue(), 1e-1);
    }

    @Test
    public void testRowSoftmax() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6);
        SoftMax softMax = new SoftMax(arr);
        opExecutioner.exec(softMax);
        assertEquals(getFailureMessage(),1.0, softMax.z().sumNumber().doubleValue(), 1e-1);
    }

    @Test
    public void testRowLogSoftMax(){
        //For moderate input values, LogSoftMax op should be identical to log(softmax)
        // through is numerically more stable for
        int[][] shapes = new int[][]{{5,3},{5,100},{1,5},{1,100}};

        double eps = 1e-3;

        for( int[] shape : shapes ){
            INDArray orig = Nd4j.rand(shape);

            INDArray orig1 = orig.dup();
            INDArray orig2 = orig.dup();

            //First: standard log(softmax)
            Nd4j.getExecutioner().exec(new SoftMax(orig1), 1);
            Nd4j.getExecutioner().exec(new Log(orig1));

            //Second: LogSoftMax op
            Nd4j.getExecutioner().exec(new LogSoftMax(orig2),1);

            for( int i=0; i<shape[0]; i++ ){
                for( int j=0; j<shape[1]; j++ ){
                    double o1 = orig1.getDouble(i);
                    double o2 = orig2.getDouble(i);
                    if(Math.abs(o1-o2)>eps){
                        System.out.println();
                    }
                    assertEquals(o1,o2,eps);
                }
            }
        }
    }

    @Test
    public void testPow() {
        INDArray oneThroughSix = Nd4j.linspace(1, 6, 6);
        Pow pow = new Pow(oneThroughSix, 2);
        Nd4j.getExecutioner().exec(pow);
        INDArray answer = Nd4j.create(new float[]{1, 4, 9, 16, 25, 36});
        assertEquals(getFailureMessage(),answer, pow.z());
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
        double max2 = Nd4j.getExecutioner().execAndReturn(max).currentResult().doubleValue();
        assertEquals(3.0, max2, 1e-1);

        Min min = new Min(row);
        double min2 = Nd4j.getExecutioner().execAndReturn(min).currentResult().doubleValue();
        assertEquals(1.0, min2, 1e-1);
        Max matrixMax = new Max(linspace);
        INDArray exec2 = Nd4j.getExecutioner().exec(matrixMax, 1);
        assertEquals(Nd4j.create(new double[]{3, 6}),exec2);
    }


    @Test
    public void testStridedLog() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray slice = arr.slice(0);
        Log exp = new Log(slice);
        opExecutioner.exec(exp);
        INDArray assertion = Nd4j.create(Nd4j.createBuffer(new double[]{0.0, 0.6931471824645996, 1.0986123085021973}));
        assertEquals(getFailureMessage(),assertion, slice);
    }

    @Test
    public void testStridedExp() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray slice = arr.slice(0);
        float[] expected = new float[slice.length()];
        for( int i=0; i<slice.length(); i++) expected[i] = (float)Math.exp(slice.getDouble(i));
        Exp exp = new Exp(slice);
        opExecutioner.exec(exp);
        assertEquals(getFailureMessage(),Nd4j.create(Nd4j.createBuffer(expected)), slice);
    }

    @Test
    public void testSoftMax() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6);
        SoftMax softMax = new SoftMax(arr);
        opExecutioner.exec(softMax);
        assertEquals(getFailureMessage(), 1.0, softMax.z().sumNumber().doubleValue(), 1e-1);

        INDArray linspace = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        SoftMax softmax = new SoftMax(linspace.dup());
        Nd4j.getExecutioner().exec(softmax);
        assertEquals(linspace.rows(), softmax.z().sumNumber().doubleValue(), 1e-1);
    }



    @Test
    public void testDimensionSoftMax() {
        INDArray linspace = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        SoftMax max = new SoftMax(linspace);
        Nd4j.getExecutioner().exec(max, 1);
        linspace.assign(max.z());
        assertEquals(getFailureMessage(), linspace.getRow(0).sumNumber().doubleValue(), 1.0, 1e-1);
    }

    @Test
    public void testColumnMean() {
        INDArray twoByThree = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray columnMean = twoByThree.mean(0);
        INDArray assertion = Nd4j.create(new float[]{2, 3});
        assertEquals(assertion, columnMean);
    }




    @Test
    public void testColumnVar() {
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnStd = twoByThree.var(0);
        INDArray assertion = Nd4j.create(new float[]{30200f, 30200f, 30200f, 30200f});
        assertEquals(assertion, columnStd);
    }

    @Test
    public void testColumnStd() {
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnStd = twoByThree.std(0);
        INDArray assertion = Nd4j.create(new float[]{173.78147196982766f, 173.78147196982766f, 173.78147196982766f, 173.78147196982766f});
        assertEquals(assertion, columnStd);
    }

    @Test
    public void testDim1() {
        INDArray sum = Nd4j.linspace(1,2, 2).reshape(2, 1);
        INDArray same = sum.dup();
        assertEquals(same.sum(1), sum);
    }

    @Test
    public void testIMax(){
        INDArray arr = Nd4j.linspace(1, 10, 10);
        IMax imax = new IMax(arr);
        assertEquals(9, ((IndexAccumulation) Nd4j.getExecutioner().exec(imax)).getFinalResult());

        arr.muli(-1);
        imax = new IMax(arr);
        int maxIdx = ((IndexAccumulation) Nd4j.getExecutioner().exec(imax)).getFinalResult();
        assertEquals(0,maxIdx);
    }

    @Test
    public void testIMin(){
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
        INDArray arr = Nd4j.ones(1,4,4);
        assertEquals(Nd4j.ones(1),arr.mean(1, 2));
        assertEquals(Nd4j.ones(1).muli(16), arr.sum(1,2));

        System.out.println("4d");
        INDArray arr4 = Nd4j.ones(1, 1, 4, 4);
        INDArray arr4m = arr4.mean(2, 3);
        INDArray arr4s = arr4.sum(2, 3);
        for(int i = 0; i < arr4m.length(); i++)
            assertEquals(arr4m.getDouble(i),1,0.0);
        for(int i = 0; i < arr4s.length(); i++)
            assertEquals(arr4s.getDouble(i),16,0.0);

        System.out.println("5d");
        INDArray arr5 = Nd4j.ones(1,1,4,4,4);
        INDArray arr5m = arr5.mean(2, 3);
        INDArray arr5s = arr5.sum(2,3);
        for( int i=0; i< arr5m.length(); i++)
            assertEquals(arr5m.getDouble(i),1,0.0);
        for( int i=0; i<arr5s.length(); i++ )
            assertEquals(arr5s.getDouble(i),16,0.0);

        System.out.println("6d");
        INDArray arr6 = Nd4j.ones(1,1,4,4,4,4);
        INDArray arr6m = arr6.mean(2, 3);
        INDArray arr6s = arr6.sum(2,3);
        for( int i = 0; i<arr6m.length(); i++ )
            assertEquals(arr6m.getDouble(i),1,0.0);
        for( int i = 0; i<arr6s.length(); i++ )
            assertEquals(arr6s.getDouble(i),16,0.0);
    }


    @Test
    public void testReductionIndex() {
        INDArray x = Nd4j.linspace(1,24,24).reshape(2,2,3,2);
        Map<Integer,Integer> assertionMap = new HashMap<>();
        assertionMap.put(0,0);
        assertionMap.put(1,0);
        assertionMap.put(2,0);
        assertionMap.put(3,1);
        assertionMap.put(4,1);
        assertionMap.put(5,1);
        assertionMap.put(6,2);
        assertionMap.put(7,2);
        assertionMap.put(8,2);
        assertionMap.put(9,3);
        assertionMap.put(10,3);
        assertionMap.put(11,3);
        assertionMap.put(12,3);

        int[] smallerDimension ={3};
        int[] biggerDImensions = new int[] {2,3};
        assertEquals(3,TadCollapseAccumulation.tadsPerReduceIndex(4,12));
        for(int i = 0; i < 12; i++) {
            int val = assertionMap.get(i);
            assertEquals(val, TadCollapseAccumulation.reductionIndexForTad(i, 4, 12));
        }


    }

    @Test
    public void testStdev() {

        INDArray arr = Nd4j.create(new float[]{0.9296161f, 0.31637555f, 0.1839188f}, new int[]{1, 3}, ordering());
        double stdev = arr.stdNumber().doubleValue();
        double stdev2 = arr.std(1).getDouble(0);
        assertEquals(stdev,stdev2,1e-3);

        double exp = 0.37003588676452637;
        assertEquals(exp,stdev,1e-7f);
    }

    @Test
    public void testVariance(){

        INDArray arr = Nd4j.create(new float[]{0.9296161f, 0.31637555f, 0.1839188f},new int[]{1,3},ordering());
        double var = arr.varNumber().doubleValue();
        INDArray temp = arr.var(1);
        double var2 = arr.var(1).getDouble(0);
        assertEquals(var,var2,1e-1);

        double exp = 0.1369265615940094;
        assertEquals(exp,var,1e-7f);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
