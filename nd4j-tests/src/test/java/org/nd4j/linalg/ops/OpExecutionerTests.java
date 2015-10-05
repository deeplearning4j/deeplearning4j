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


import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.exception.IllegalOpException;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEquals;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarSetValue;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.*;
import org.nd4j.linalg.api.parallel.bufferops.AccumulationViaTensorDataBufferTask;
import org.nd4j.linalg.api.parallel.bufferops.IndexAccumulationViaTensorDataBufferTask;
import org.nd4j.linalg.api.parallel.bufferops.ScalarViaTensorDataBufferAction;
import org.nd4j.linalg.api.parallel.bufferops.TransformViaTensorDataBufferTask;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 2/22/15.
 */
public  class OpExecutionerTests extends BaseNd4jTest {
    public OpExecutionerTests() {
    }

    public OpExecutionerTests(Nd4jBackend backend) {
        super(backend);
    }

    public OpExecutionerTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public OpExecutionerTests(String name) {
        super(name);
    }



    @Test
    public void testCosineSimilarity() {
        INDArray vec1 = Nd4j.create(new float[]{1, 2, 3, 4,5});
        INDArray vec2 = Nd4j.create(new float[]{1, 2, 3, 4,5});
        double sim = Transforms.cosineSim(vec1, vec2);
        assertEquals(getFailureMessage(), 1, sim, 1e-1);

    }

    @Test
    public void testEuclideanDistance() {
        INDArray arr = Nd4j.create(new double[]{55,55});
        INDArray arr2 = Nd4j.create(new double[]{60, 60});
        double result = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(arr,arr2)).currentResult().doubleValue();
        assertEquals(getFailureMessage(),7.0710678118654755,result,1e-1);
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
    public void testNormMax() {
        INDArray arr = Nd4j.create(new float[]{1, 2, 3, 4});
        double normMax = Nd4j.getExecutioner().execAndReturn(new NormMax(arr)).currentResult().doubleValue();
        assertEquals(getFailureMessage(), 4, normMax, 1e-1);
    }

    @Test
    public void testLog() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2, 2);
        INDArray assertion = Nd4j.create(new double[][]{
                {0., 1.09861229},
                {0.69314718, 1.38629436}
        });

        INDArray logTest = Transforms.log(arr);
        assertEquals(assertion,logTest);
        arr = Nd4j.linspace(1,6,6).reshape(2,3);
        assertion = Nd4j.create(new double[][]{
                {0., 1.09861229, 1.60943791},
                {0.69314718,  1.38629436,  1.79175947}
        });

        logTest = Transforms.log(arr);
        assertEquals(assertion,logTest);
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
        assertEquals(getFailureMessage(),10.0, acc.currentResult().doubleValue(), 1e-1);
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
    public void testIamax() {
        INDArray linspace = Nd4j.linspace(1,4,4);
        assertEquals(getFailureMessage(),3,Nd4j.getBlasWrapper().iamax(linspace));
    }

    @Test
    public void testIamax2() {
        INDArray linspace = Nd4j.linspace(1, 4, 4);
        assertEquals(getFailureMessage(), 3, Nd4j.getBlasWrapper().iamax(linspace));
        int iamax = Nd4j.getExecutioner().execAndReturn(new IAMax(linspace)).getFinalResult();
        assertEquals(3,iamax);
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
        assertEquals(5.0, max2, 1e-1);

        Min min = new Min(row);
        double min2 = Nd4j.getExecutioner().execAndReturn(min).currentResult().doubleValue();
        assertEquals(1.0, min2, 1e-1);
    }


    @Test
    public void testStridedLog() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray slice = arr.slice(0);
        Log log = new Log(slice);
        opExecutioner.exec(log);
        INDArray assertion = Nd4j.create(Nd4j.createBuffer(new float[]{0.f, 1.09861229f, 1.60943791f}));
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
        assertEquals(getFailureMessage(),1.0, softMax.z().sumNumber().doubleValue(), 1e-1);
    }


    @Test
    public void testOpExecutionerTransformOps() throws Exception {
        //Basic idea: results should be identical, whether executed in serial vs. parallel, heap vs. direct,
        // or direct execution vs. split via tensors, x == z vs. x != z
        final DataBuffer.AllocationMode origAlloc = Nd4j.alloc;
        DefaultOpExecutioner opExec = (DefaultOpExecutioner)Nd4j.getExecutioner();

        List<Class<? extends TransformOp>> testClasses = new ArrayList<>();
        testClasses.add(AddOp.class);
        testClasses.add(CopyOp.class);
        testClasses.add(MulOp.class);
        testClasses.add(DivOp.class);
        testClasses.add(RDivOp.class);
        testClasses.add(RSubOp.class);
        testClasses.add(SubOp.class);
        testClasses.add(Tanh.class);
        testClasses.add(Sigmoid.class);
        testClasses.add(RectifedLinear.class);
        testClasses.add(SoftMax.class);

        int[] shape = {30,50};

        for(DataBuffer.Type dtype : DataBuffer.Type.values() ) {

            Nd4j.getRandom().setSeed(12345);
            INDArray origFirst = Nd4j.rand(shape);
            INDArray origSecond = Nd4j.rand(shape);

            for (Class<? extends TransformOp> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype="+dtype;
                Constructor<? extends TransformOp> xyzConstructor = opClass.getConstructor(INDArray.class, INDArray.class, INDArray.class);

                // --- First: serial, heap, x =/= z and x == z ---
                DefaultOpExecutioner.setParallelThreshold(Integer.MAX_VALUE);

                INDArray x1 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1 = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z1 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);

                TransformOp op = xyzConstructor.newInstance(x1, y1, z1);
                opExec.exec(op);

                assertEquals(x1, origFirst);
                assertEquals(y1, origSecond);

                INDArray x2 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y2 = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x2, y2, x2);
                opExec.exec(op);
                assertEquals(y2, origSecond);
                assertEquals(x2, z1);

                //Same thing, but split via tensors first:
                INDArray x1a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1a = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z1a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyzConstructor.newInstance(x1a, y1a, z1a);
                new TransformViaTensorDataBufferTask(op, Integer.MAX_VALUE, x1a, y1a, z1a).invoke();
                assertEquals(msg, x1a, origFirst);
                assertEquals(msg, y1a, origSecond);
                assertEquals(msg, z1a, z1);

                INDArray x2a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y2a = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x2a, y2a, x2a);
                new TransformViaTensorDataBufferTask(op, Integer.MAX_VALUE, x2a, y2a, x2a).invoke();
                assertEquals(msg, y2a, origSecond);
                assertEquals(msg, x2a, z1);


                // --- Second: parallel, heap ---
                DefaultOpExecutioner.setParallelThreshold(5);
                INDArray x3 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3 = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z3 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x3, y3, z3);
                opExec.exec(op);

                assertEquals(msg, x3, origFirst);
                assertEquals(msg, y3, origSecond);
                assertEquals(msg, z3, z1);

                INDArray x4 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y4 = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x4, y4, x4);
                opExec.exec(op);
                assertEquals(msg, y4, origSecond);
                assertEquals(msg, x4, z1);

                //Same thing, but split via tensors first:
                INDArray x3a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3a = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z3a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyzConstructor.newInstance(x3a, y3a, z3a);
                new TransformViaTensorDataBufferTask(op, 5, x3a, y3a, z3a).invoke();
                assertEquals(msg, x3a, origFirst);
                assertEquals(msg, y3a, origSecond);
                assertEquals(msg, z3a, z1);

                INDArray x4a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y4a = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x4a, y4a, x4a);
                new TransformViaTensorDataBufferTask(op, 5, x4a, y4a, x4a).invoke();
                assertEquals(msg, y4a, origSecond);
                assertEquals(msg, x4a, z1);


                // --- Third: serial, direct ---
                DefaultOpExecutioner.setParallelThreshold(Integer.MAX_VALUE);

                INDArray x5 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5 = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z5 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x5, y5, z5);
                opExec.exec(op);

                assertEquals(msg, x5, origFirst);
                assertEquals(msg, y5, origSecond);
                assertEquals(msg, z5, z1);

                INDArray x6 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y6 = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x6, y6, x6);
                opExec.exec(op);
                assertEquals(msg, y6, origSecond);
                assertEquals(msg, x6, z1);

                //Same thing, but split via tensors first:
                INDArray x5a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5a = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z5a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x5a, y5a, z5a);
                new TransformViaTensorDataBufferTask(op, Integer.MAX_VALUE, x5a, y5a, z5a).invoke();
                assertEquals(msg, x5a, origFirst);
                assertEquals(msg, y5a, origSecond);
                assertEquals(msg, z5a, z5);

                INDArray x6a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y6a = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x6a, y6a, x6a);
                new TransformViaTensorDataBufferTask(op, Integer.MAX_VALUE, x6a, y6a, x6a).invoke();
                assertEquals(msg, y6a, origSecond);
                assertEquals(msg, x6a, z1);


                // --- Fourth: parallel, direct ---
                DefaultOpExecutioner.setParallelThreshold(5);

                INDArray x7 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7 = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z7 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x7, y7, z7);
                opExec.exec(op);

                assertEquals(msg, x7, origFirst);
                assertEquals(msg, y7, origSecond);
                assertEquals(msg, z7, z1);


                INDArray x8 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y8 = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x8, y8, x8);
                opExec.exec(op);
                assertEquals(msg, y8, origSecond);
                assertEquals(msg, x8, z1);

                //Same thing, but split via tensors first:
                INDArray x7a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7a = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z7a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x7a, y7a, z7a);
                new TransformViaTensorDataBufferTask(op, 5, x7a, y7a, z7a).invoke();
                assertEquals(msg, x7a, origFirst);
                assertEquals(msg, y7a, origSecond);
                assertEquals(msg, z7a, z1);

                INDArray x8a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y8a = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x8a, y8a, x8a);
                new TransformViaTensorDataBufferTask(op, 5, x8a, y8a, x8a).invoke();
                assertEquals(msg, y8a, origSecond);
                assertEquals(msg, x8a, z1);
            }
        }

        Nd4j.alloc = origAlloc;
    }

    @Test
    public void testOpExecutionerScalarOps() throws Exception {
        //Basic idea: results should be identical, whether executed in serial vs. parallel, heap vs. direct,
        // or direct execution vs. split via tensors, x == z vs. x != z
        final DataBuffer.AllocationMode origAlloc = Nd4j.alloc;
        DefaultOpExecutioner opExec = (DefaultOpExecutioner)Nd4j.getExecutioner();

        List<Class<? extends ScalarOp>> testClasses = new ArrayList<>();
        testClasses.add(ScalarAdd.class);
        testClasses.add(ScalarDivision.class);
        testClasses.add(ScalarMax.class);
        testClasses.add(ScalarMultiplication.class);
        testClasses.add(ScalarReverseDivision.class);
        testClasses.add(ScalarReverseSubtraction.class);
        testClasses.add(ScalarSet.class);
        testClasses.add(ScalarSubtraction.class);
        testClasses.add(ScalarEquals.class);
        testClasses.add(ScalarGreaterThan.class);
        testClasses.add(ScalarSetValue.class);

        int[] shape = {30,50};

        for(DataBuffer.Type dtype : DataBuffer.Type.values() ) {

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape);
            INDArray origZ = Nd4j.zeros(shape);

            for (Class<? extends ScalarOp> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype="+dtype;
                Constructor<? extends ScalarOp> xyzConstructor = opClass.getConstructor(INDArray.class, INDArray.class,
                        INDArray.class, int.class, Number.class);

                // --- First: serial, heap, x =/= z and x == z ---
                DefaultOpExecutioner.setParallelThreshold(Integer.MAX_VALUE);

                INDArray x1 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z1 = getCopyOf(origZ, DataBuffer.AllocationMode.HEAP, dtype);

                ScalarOp op = xyzConstructor.newInstance(x1, null, z1, x1.length(), 0.4);
                opExec.exec(op);

                assertEquals(x1, origX);

                INDArray x2 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x2, null, x2, x2.length(), 0.4);
                opExec.exec(op);
                assertEquals(x2, z1);

                //Same thing, but split via tensors first:
                INDArray x1a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z1a = getCopyOf(origZ, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyzConstructor.newInstance(x1a, null, z1a, x1a.length(), 0.4);
                new ScalarViaTensorDataBufferAction(op, Integer.MAX_VALUE, x1a, z1a).invoke();
                assertEquals(msg, x1a, origX);
                assertEquals(msg, z1a, z1);

                INDArray x2a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x2a, null, x2a, x2a.length(), 0.4);
                new ScalarViaTensorDataBufferAction(op, Integer.MAX_VALUE, x2a, x2a).invoke();
                assertEquals(msg, x2a, z1);


                // --- Second: parallel, heap ---
                DefaultOpExecutioner.setParallelThreshold(5);
                INDArray x3 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z3 = getCopyOf(origZ, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x3, null, z3, x3.length(), 0.4);
                opExec.exec(op);

                assertEquals(msg, x3, origX);
                assertEquals(msg, z3, z1);

                INDArray x4 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x4, null, x4, x4.length(), 0.4);
                opExec.exec(op);
                assertEquals(msg, x4, z1);

                //Same thing, but split via tensors first:
                INDArray x3a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z3a = getCopyOf(origZ, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyzConstructor.newInstance(x3a, null, z3a, x3a.length(), 0.4);
                new ScalarViaTensorDataBufferAction(op, 5, x3a, z3a).invoke();
                assertEquals(msg, x3a, origX);
                assertEquals(msg, z3a, z1);

                INDArray x4a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x4a, null, x4a, x4a.length(), 0.4);
                new ScalarViaTensorDataBufferAction(op, 5, x4a, x4a).invoke();
                assertEquals(msg, x4a, z1);


                // --- Third: serial, direct ---
                DefaultOpExecutioner.setParallelThreshold(Integer.MAX_VALUE);

                INDArray x5 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z5 = getCopyOf(origZ, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x5, null, z5, x5.length(), 0.4);
                opExec.exec(op);

                assertEquals(msg, x5, origX);
                assertEquals(msg, z5, z1);

                INDArray x6 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x6, null, x6, x6.length(), 0.4);
                opExec.exec(op);
                assertEquals(msg, x6, z1);

                //Same thing, but split via tensors first:
                INDArray x5a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z5a = getCopyOf(origZ, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x5a, null, z5a, x5a.length(), 0.4);
                new ScalarViaTensorDataBufferAction(op, Integer.MAX_VALUE, x5a, z5a).invoke();
                assertEquals(msg, x5a, origX);
                assertEquals(msg, z5a, z5);

                INDArray x6a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x6a, null, x6a, x6a.length(), 0.4);
                new ScalarViaTensorDataBufferAction(op, Integer.MAX_VALUE, x6a, x6a).invoke();
                assertEquals(msg, x6a, z1);


                // --- Fourth: parallel, direct ---
                DefaultOpExecutioner.setParallelThreshold(5);

                INDArray x7 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z7 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x7, null, z7, x7.length(), 0.4);
                opExec.exec(op);
                assertEquals(msg, x7, origX);
                assertEquals(msg, z7, z1);


                INDArray x8 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x8, null, x8, x8.length(), 0.4);
                opExec.exec(op);
                assertEquals(msg, x8, z1);

                //Same thing, but split via tensors first:
                INDArray x7a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z7a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x7a, null, z7a, x7a.length(), 0.4);
                new ScalarViaTensorDataBufferAction(op, 5, x7a, z7a).invoke();
                assertEquals(msg, x7a, origX);
                assertEquals(msg, z7a, z1);

                INDArray x8a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x8a, null, x8a, x8a.length(), 0.4);
                new ScalarViaTensorDataBufferAction(op, 5, x8a, x8a).invoke();
                assertEquals(msg, x8a, z1);
            }
        }

        Nd4j.alloc = origAlloc;
    }

    @Test
    public void testOpExecutionerAccumulationOps() throws Exception {
        //Basic idea: results should be identical, whether executed in serial vs. parallel, heap vs. direct,
        // or direct execution vs. split via tensors
        final DataBuffer.AllocationMode origAlloc = Nd4j.alloc;
        DefaultOpExecutioner opExec = (DefaultOpExecutioner)Nd4j.getExecutioner();

        List<Class<? extends Accumulation>> testClasses = new ArrayList<>();
        testClasses.add(Bias.class);
        testClasses.add(Dot.class);
        testClasses.add(Max.class);
        testClasses.add(Mean.class);
        testClasses.add(Min.class);
        testClasses.add(Norm1.class);
        testClasses.add(Norm2.class);
        testClasses.add(NormMax.class);
        testClasses.add(Prod.class);
        testClasses.add(StandardDeviation.class);
        testClasses.add(Sum.class);
        testClasses.add(Variance.class);

        int[] shape = {30,50};

        for(DataBuffer.Type dtype : DataBuffer.Type.values() ) {

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape);
            INDArray origY = Nd4j.rand(shape);

            for (Class<? extends Accumulation> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype="+dtype;
                Constructor<? extends Accumulation> xyConstructor = opClass.getConstructor(INDArray.class, INDArray.class);

                // --- First: serial, heap ---
                DefaultOpExecutioner.setParallelThreshold(Integer.MAX_VALUE);

                INDArray x1 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                Accumulation op = xyConstructor.newInstance(x1, y1);
                opExec.exec(op);

                assertEquals(x1, origX);
                double result1 = op.getFinalResult().doubleValue();

                //Same thing, but split via tensors first:
                INDArray x1a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1a = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyConstructor.newInstance(x1a, y1a);
                new AccumulationViaTensorDataBufferTask(op, Integer.MAX_VALUE, x1a, y1a).invoke();
                assertEquals(msg, x1a, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                // --- Second: parallel, heap ---
                DefaultOpExecutioner.setParallelThreshold(5);
                INDArray x3 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyConstructor.newInstance(x3, y3);
                opExec.exec(op);

                assertEquals(msg, x3, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                //Same thing, but split via tensors first:
                INDArray x3a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3a = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyConstructor.newInstance(x3a, y3a);
                new AccumulationViaTensorDataBufferTask(op, 5, x3a, y3a).invoke();
                assertEquals(msg, x3a, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                // --- Third: serial, direct ---
                DefaultOpExecutioner.setParallelThreshold(Integer.MAX_VALUE);

                INDArray x5 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x5, y5);
                opExec.exec(op);

                assertEquals(msg, x5, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                //Same thing, but split via tensors first:
                INDArray x5a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5a = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x5a, y5a);
                new AccumulationViaTensorDataBufferTask(op, Integer.MAX_VALUE, x5a, y5a).invoke();
                assertEquals(msg, x5a, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                // --- Fourth: parallel, direct ---
                DefaultOpExecutioner.setParallelThreshold(5);

                INDArray x7 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x7, y7);
                opExec.exec(op);
                assertEquals(msg, x7, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                //Same thing, but split via tensors first:
                INDArray x7a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7a = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x7a,y7a);
                new AccumulationViaTensorDataBufferTask(op, 5, x7a, y7a).invoke();
                assertEquals(msg, x7a, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);
            }
        }

        Nd4j.alloc = origAlloc;
    }


    @Test
    public void testOpExecutionerIndexAccumulationOps() throws Exception {
        //Basic idea: results should be identical, whether executed in serial vs. parallel, heap vs. direct,
        // or direct execution vs. split via tensors
        final DataBuffer.AllocationMode origAlloc = Nd4j.alloc;
        DefaultOpExecutioner opExec = (DefaultOpExecutioner)Nd4j.getExecutioner();

        List<Class<? extends IndexAccumulation>> testClasses = new ArrayList<>();
        testClasses.add(IAMax.class);
        testClasses.add(IMax.class);
        testClasses.add(IMin.class);

        int[] shape = {30,50};

        for(DataBuffer.Type dtype : DataBuffer.Type.values() ) {

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape);
            INDArray origY = Nd4j.rand(shape);

            for (Class<? extends IndexAccumulation> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype="+dtype;
                Constructor<? extends IndexAccumulation> xyConstructor = opClass.getConstructor(INDArray.class, INDArray.class);

                // --- First: serial, heap ---
                DefaultOpExecutioner.setParallelThreshold(Integer.MAX_VALUE);

                INDArray x1 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                IndexAccumulation op = xyConstructor.newInstance(x1, y1);
                opExec.exec(op);

                assertEquals(x1, origX);
                int result1 = op.getFinalResult();

                //Same thing, but split via tensors first:
                INDArray x1a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1a = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyConstructor.newInstance(x1a, y1a);
                new IndexAccumulationViaTensorDataBufferTask(op, Integer.MAX_VALUE, x1a, y1a).invoke();
                assertEquals(msg, x1a, origX);
                assertEquals(msg, result1, op.getFinalResult());

                // --- Second: parallel, heap ---
                DefaultOpExecutioner.setParallelThreshold(5);
                INDArray x3 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyConstructor.newInstance(x3, y3);
                opExec.exec(op);

                assertEquals(msg, x3, origX);
                assertEquals(msg, result1, op.getFinalResult());

                //Same thing, but split via tensors first:
                INDArray x3a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3a = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyConstructor.newInstance(x3a, y3a);
                new IndexAccumulationViaTensorDataBufferTask(op, 5, x3a, y3a).invoke();
                assertEquals(msg, x3a, origX);
                assertEquals(msg, result1, op.getFinalResult());

                // --- Third: serial, direct ---
                DefaultOpExecutioner.setParallelThreshold(Integer.MAX_VALUE);

                INDArray x5 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x5, y5);
                opExec.exec(op);

                assertEquals(msg, x5, origX);
                assertEquals(msg, result1, op.getFinalResult());

                //Same thing, but split via tensors first:
                INDArray x5a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5a = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x5a, y5a);
                new IndexAccumulationViaTensorDataBufferTask(op, Integer.MAX_VALUE, x5a, y5a).invoke();
                assertEquals(msg, x5a, origX);
                assertEquals(msg, result1, op.getFinalResult());

                // --- Fourth: parallel, direct ---
                DefaultOpExecutioner.setParallelThreshold(5);

                INDArray x7 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x7, y7);
                opExec.exec(op);
                assertEquals(msg, x7, origX);
                assertEquals(msg, result1, op.getFinalResult());

                //Same thing, but split via tensors first:
                INDArray x7a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7a = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x7a,y7a);
                new IndexAccumulationViaTensorDataBufferTask(op, 5, x7a, y7a).invoke();
                assertEquals(msg, x7a, origX);
                assertEquals(msg, result1, op.getFinalResult());
            }
        }

        Nd4j.alloc = origAlloc;
    }


    private static INDArray getCopyOf(INDArray original, DataBuffer.AllocationMode mode, DataBuffer.Type type ){
        DataBuffer.AllocationMode origMode = Nd4j.alloc;
        DataBuffer.Type origType = Nd4j.dataType();
        Nd4j.alloc = mode;
        Nd4j.dtype = type;
        Nd4j.factory().setDType(type);
        INDArray out = Nd4j.create(original.shape());
        if(type == DataBuffer.Type.FLOAT){
            for (int i = 0; i < original.length(); i++) out.putScalar(i, original.getFloat(i));
        } else {
            for (int i = 0; i < original.length(); i++) out.putScalar(i, original.getDouble(i));
        }
        Nd4j.alloc = origMode;
        Nd4j.dtype = origType;
        Nd4j.factory().setDType(origType);
        assertEquals(mode, out.data().allocationMode());
        assertEquals( original, out );
        return out;
    }



    @Override
    public char ordering() {
        return 'f';
    }
}
