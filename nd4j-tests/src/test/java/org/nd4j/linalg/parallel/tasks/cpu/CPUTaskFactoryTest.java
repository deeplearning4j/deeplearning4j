package org.nd4j.linalg.parallel.tasks.cpu;

import org.apache.commons.math3.util.Pair;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEquals;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarSetValue;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.*;
import org.nd4j.linalg.api.ops.impl.vector.*;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.cpu.CPUTaskFactory;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;

public class CPUTaskFactoryTest extends BaseNd4jTest {


    @Test
    public void testOpExecutionerTransformOps() throws Exception {
        //Basic idea: results should be identical, whether executed in serial vs. parallel, heap vs. direct,
        // or direct execution vs. split via tensors, x == z vs. x != z
        final DataBuffer.AllocationMode origAlloc = Nd4j.alloc;

        CPUTaskFactory taskFactory = new CPUTaskFactory();

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

        int[] shape = {30, 50};

        for (DataBuffer.Type dtype : DataBuffer.Type.values()) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origFirst = Nd4j.rand(shape);
            INDArray origSecond = Nd4j.rand(shape);

            for (Class<? extends TransformOp> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends TransformOp> xyzConstructor = opClass.getConstructor(INDArray.class, INDArray.class, INDArray.class);

                // --- First: serial, heap, x =/= z and x == z ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x1 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1 = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z1 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);

                TransformOp op = xyzConstructor.newInstance(x1, y1, z1);
                Task<Void> task = taskFactory.getTransformAction(op);
                task.invokeBlocking();

                assertEquals(msg, x1, origFirst);
                assertEquals(msg, y1, origSecond);

                INDArray x2 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y2 = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x2, y2, x2);
                task = taskFactory.getTransformAction(op);
                task.invokeBlocking();
                assertEquals(msg, y2, origSecond);
                assertEquals(msg, x2, z1);

                //Same thing, but split via tensors first:
                if (!op.isPassThrough()) {
                    //have to execute passthrough ops via OpExecutioner
                    INDArray x1a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                    INDArray y1a = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                    INDArray z1a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);

                    op = xyzConstructor.newInstance(x1a, y1a, z1a);
                    task = taskFactory.getTransformAction(op);
                    task.invokeBlocking();
                    assertEquals(msg, x1a, origFirst);
                    assertEquals(msg, y1a, origSecond);
                    assertEquals(msg, z1a, z1);

                    INDArray x2a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                    INDArray y2a = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                    op = xyzConstructor.newInstance(x2a, y2a, x2a);
                    task = taskFactory.getTransformAction(op);
                    task.invokeBlocking();
                    assertEquals(msg, y2a, origSecond);
                    assertEquals(msg, x2a, z1);
                }


                // --- Second: parallel, heap ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x3 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3 = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z3 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x3, y3, z3);
                task = taskFactory.getTransformAction(op);
                task.invokeBlocking();

                assertEquals(msg, x3, origFirst);
                assertEquals(msg, y3, origSecond);
                assertEquals(msg, z3, z1);

                INDArray x4 = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y4 = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x4, y4, x4);
                task = taskFactory.getTransformAction(op);
                task.invokeBlocking();
                assertEquals(msg, y4, origSecond);
                assertEquals(msg, x4, z1);

                //Same thing, but split via tensors first:
                if (!op.isPassThrough()) {
                    //have to execute passthrough ops via OpExecutioner
                    INDArray x3a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                    INDArray y3a = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                    INDArray z3a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);

                    op = xyzConstructor.newInstance(x3a, y3a, z3a);
                    task = taskFactory.getTransformAction(op);
                    task.invokeBlocking();
                    assertEquals(msg, x3a, origFirst);
                    assertEquals(msg, y3a, origSecond);
                    assertEquals(msg, z3a, z1);

                    INDArray x4a = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                    INDArray y4a = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                    op = xyzConstructor.newInstance(x4a, y4a, x4a);
                    task = taskFactory.getTransformAction(op);
                    task.invokeBlocking();
                    assertEquals(msg, y4a, origSecond);
                    assertEquals(msg, x4a, z1);
                }


                // --- Third: serial, direct ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x5 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5 = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z5 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x5, y5, z5);
                task = taskFactory.getTransformAction(op);
                task.invokeBlocking();

                assertEquals(msg, x5, origFirst);
                assertEquals(msg, y5, origSecond);
                assertEquals(msg, z5, z1);

                INDArray x6 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y6 = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x6, y6, x6);
                task = taskFactory.getTransformAction(op);
                task.invokeBlocking();
                assertEquals(msg, y6, origSecond);
                assertEquals(msg, x6, z1);

                //Same thing, but split via tensors first:
                if (!op.isPassThrough()) {
                    //have to execute passthrough ops via OpExecutioner
                    INDArray x5a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                    INDArray y5a = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                    INDArray z5a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);

                    op = xyzConstructor.newInstance(x5a, y5a, z5a);
                    task = taskFactory.getTransformAction(op);
                    task.invokeBlocking();
                    assertEquals(msg, x5a, origFirst);
                    assertEquals(msg, y5a, origSecond);
                    assertEquals(msg, z5a, z5);

                    INDArray x6a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                    INDArray y6a = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                    op = xyzConstructor.newInstance(x6a, y6a, x6a);
                    task = taskFactory.getTransformAction(op);
                    task.invokeBlocking();
                    assertEquals(msg, y6a, origSecond);
                    assertEquals(msg, x6a, z1);
                }


                // --- Fourth: parallel, direct ---
                taskFactory.setParallelThreshold(5);
                task = taskFactory.getTransformAction(op);
                task.invokeBlocking();
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x7 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7 = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z7 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x7, y7, z7);
                task = taskFactory.getTransformAction(op);
                task.invokeBlocking();

                assertEquals(msg, x7, origFirst);
                assertEquals(msg, y7, origSecond);
                assertEquals(msg, z7, z1);


                INDArray x8 = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y8 = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x8, y8, x8);
                task = taskFactory.getTransformAction(op);
                task.invokeBlocking();
                assertEquals(msg, y8, origSecond);
                assertEquals(msg, x8, z1);

                //Same thing, but split via tensors first:
                if (!op.isPassThrough()) {
                    //have to execute passthrough ops via OpExecutioner
                    INDArray x7a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                    INDArray y7a = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                    INDArray z7a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);

                    op = xyzConstructor.newInstance(x7a, y7a, z7a);
                    task = taskFactory.getTransformAction(op);
                    task.invokeBlocking();
                    assertEquals(msg, x7a, origFirst);
                    assertEquals(msg, y7a, origSecond);
                    assertEquals(msg, z7a, z1);

                    INDArray x8a = getCopyOf(origFirst, DataBuffer.AllocationMode.DIRECT, dtype);
                    INDArray y8a = getCopyOf(origSecond, DataBuffer.AllocationMode.DIRECT, dtype);
                    op = xyzConstructor.newInstance(x8a, y8a, x8a);
                    task = taskFactory.getTransformAction(op);
                    task.invokeBlocking();
                    assertEquals(msg, y8a, origSecond);
                    assertEquals(msg, x8a, z1);
                }
            }
        }

        Nd4j.alloc = origAlloc;
    }

    @Test
    public void testOpExecutionerScalarOps() throws Exception {
        //Basic idea: results should be identical, whether executed in serial vs. parallel, heap vs. direct,
        // or direct execution vs. split via tensors, x == z vs. x != z
        final DataBuffer.AllocationMode origAlloc = Nd4j.alloc;

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

        CPUTaskFactory taskFactory = new CPUTaskFactory();

        int[] shape = {30, 50};

        for (DataBuffer.Type dtype : DataBuffer.Type.values()) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape);
            INDArray origZ = Nd4j.zeros(shape);

            for (Class<? extends ScalarOp> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends ScalarOp> xyzConstructor = opClass.getConstructor(INDArray.class, INDArray.class,
                        INDArray.class, int.class, Number.class);

                // --- First: serial, heap, x =/= z and x == z ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x1 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z1 = getCopyOf(origZ, DataBuffer.AllocationMode.HEAP, dtype);

                ScalarOp op = xyzConstructor.newInstance(x1, null, z1, x1.length(), 0.4);
                Task<Void> task = taskFactory.getScalarAction(op);
                task.invokeBlocking();

                assertEquals(msg, x1, origX);

                INDArray x2 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x2, null, x2, x2.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x2, z1);

                //Same thing, but split via tensors first:
                INDArray x1a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z1a = getCopyOf(origZ, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyzConstructor.newInstance(x1a, null, z1a, x1a.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x1a, origX);
                assertEquals(msg, z1a, z1);

                INDArray x2a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x2a, null, x2a, x2a.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x2a, z1);


                // --- Second: parallel, heap ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x3 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z3 = getCopyOf(origZ, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x3, null, z3, x3.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();

                assertEquals(msg, x3, origX);
                assertEquals(msg, z3, z1);

                INDArray x4 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x4, null, x4, x4.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x4, z1);

                //Same thing, but split via tensors first:
                INDArray x3a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray z3a = getCopyOf(origZ, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyzConstructor.newInstance(x3a, null, z3a, x3a.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x3a, origX);
                assertEquals(msg, z3a, z1);

                INDArray x4a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyzConstructor.newInstance(x4a, null, x4a, x4a.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x4a, z1);


                // --- Third: serial, direct ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x5 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z5 = getCopyOf(origZ, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x5, null, z5, x5.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();

                assertEquals(msg, x5, origX);
                assertEquals(msg, z5, z1);

                INDArray x6 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x6, null, x6, x6.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x6, z1);

                //Same thing, but split via tensors first:
                INDArray x5a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z5a = getCopyOf(origZ, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x5a, null, z5a, x5a.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x5a, origX);
                assertEquals(msg, z5a, z5);

                INDArray x6a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x6a, null, x6a, x6a.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x6a, z1);


                // --- Fourth: parallel, direct ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x7 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z7 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x7, null, z7, x7.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x7, origX);
                assertEquals(msg, z7, z1);


                INDArray x8 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x8, null, x8, x8.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x8, z1);

                //Same thing, but split via tensors first:
                INDArray x7a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray z7a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyzConstructor.newInstance(x7a, null, z7a, x7a.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
                assertEquals(msg, x7a, origX);
                assertEquals(msg, z7a, z1);

                INDArray x8a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                op = xyzConstructor.newInstance(x8a, null, x8a, x8a.length(), 0.4);
                task = taskFactory.getScalarAction(op);
                task.invokeBlocking();
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

        CPUTaskFactory taskFactory = new CPUTaskFactory();

        int[] shape = {30, 50};

        for (DataBuffer.Type dtype : DataBuffer.Type.values()) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape);
            INDArray origY = Nd4j.rand(shape);

            for (Class<? extends Accumulation> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends Accumulation> xyConstructor = opClass.getConstructor(INDArray.class, INDArray.class);

                // --- First: serial, heap ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x1 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                Accumulation op = xyConstructor.newInstance(x1, y1);
                Task<Double> task = taskFactory.getAccumulationTask(op);
                double out = task.invokeBlocking();

                assertEquals(msg, x1, origX);
                double result1 = op.getFinalResult().doubleValue();
                assertEquals(msg, result1, out, 0.0);

                //Same thing, but split via tensors first:
                INDArray x1a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1a = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyConstructor.newInstance(x1a, y1a);
                task = taskFactory.getAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, out, 0.0);
                assertEquals(msg, x1a, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                // --- Second: parallel, heap ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x3 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyConstructor.newInstance(x3, y3);
                task = taskFactory.getAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, out, 0.0);

                assertEquals(msg, x3, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                //Same thing, but split via tensors first:
                INDArray x3a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3a = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyConstructor.newInstance(x3a, y3a);
                task = taskFactory.getAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, out, 0.0);
                assertEquals(msg, x3a, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                // --- Third: serial, direct ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x5 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x5, y5);
                task = taskFactory.getAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, out, 0.0);

                assertEquals(msg, x5, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                //Same thing, but split via tensors first:
                INDArray x5a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5a = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x5a, y5a);
                task = taskFactory.getAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, out, 0.0);
                assertEquals(msg, x5a, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                // --- Fourth: parallel, direct ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x7 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x7, y7);
                task = taskFactory.getAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, out, 0.0);
                assertEquals(msg, x7, origX);
                assertEquals(msg, result1, op.getFinalResult().doubleValue(), 1e-2);

                //Same thing, but split via tensors first:
                INDArray x7a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7a = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x7a, y7a);
                task = taskFactory.getAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, out, 0.0);
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

        List<Class<? extends IndexAccumulation>> testClasses = new ArrayList<>();
        testClasses.add(IAMax.class);
        testClasses.add(IMax.class);
        testClasses.add(IMin.class);

        CPUTaskFactory taskFactory = new CPUTaskFactory();

        int[] shape = {30, 50};

        for (DataBuffer.Type dtype : DataBuffer.Type.values()) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape);
            INDArray origY = Nd4j.rand(shape);

            for (Class<? extends IndexAccumulation> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends IndexAccumulation> xyConstructor = opClass.getConstructor(INDArray.class, INDArray.class);

                // --- First: serial, heap ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x1 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                IndexAccumulation op = xyConstructor.newInstance(x1, y1);
                Task<Pair<Double, Integer>> task = taskFactory.getIndexAccumulationTask(op);
                Pair<Double, Integer> out = task.invokeBlocking();

                assertEquals(msg, x1, origX);
                int result1 = op.getFinalResult();
                double result1d = op.z().getDouble(result1);
                assertEquals(msg, result1, (int) out.getSecond());
                assertEquals(msg, result1d, out.getFirst(), 0.0);

                //Same thing, but split via tensors first:
                INDArray x1a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1a = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyConstructor.newInstance(x1a, y1a);
                task = taskFactory.getIndexAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, (int) out.getSecond());
                assertEquals(msg, result1d, out.getFirst(), 0.0);
                assertEquals(msg, op.z().getDouble(result1), out.getFirst(), 0.0);

                assertEquals(msg, x1a, origX);
                assertEquals(msg, result1, op.getFinalResult());

                // --- Second: parallel, heap ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x3 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyConstructor.newInstance(x3, y3);
                task = taskFactory.getIndexAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, (int) out.getSecond());
                assertEquals(msg, result1d, out.getFirst(), 0.0);
                assertEquals(msg, op.z().getDouble(result1), out.getFirst(), 0.0);

                assertEquals(msg, x3, origX);
                assertEquals(msg, result1, op.getFinalResult());

                //Same thing, but split via tensors first:
                INDArray x3a = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3a = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                op = xyConstructor.newInstance(x3a, y3a);
                task = taskFactory.getIndexAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, (int) out.getSecond());
                assertEquals(msg, result1d, out.getFirst(), 0.0);
                assertEquals(msg, op.z().getDouble(result1), out.getFirst(), 0.0);
                assertEquals(msg, x3a, origX);
                assertEquals(msg, result1, op.getFinalResult());

                // --- Third: serial, direct ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x5 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x5, y5);
                task = taskFactory.getIndexAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, (int) out.getSecond());
                assertEquals(msg, result1d, out.getFirst(), 0.0);
                assertEquals(msg, op.z().getDouble(result1), out.getFirst(), 0.0);

                assertEquals(msg, x5, origX);
                assertEquals(msg, result1, op.getFinalResult());

                //Same thing, but split via tensors first:
                INDArray x5a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5a = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x5a, y5a);
                task = taskFactory.getIndexAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, (int) out.getSecond());
                assertEquals(msg, result1d, out.getFirst(), 0.0);
                assertEquals(msg, op.z().getDouble(result1), out.getFirst(), 0.0);
                assertEquals(msg, x5a, origX);
                assertEquals(msg, result1, op.getFinalResult());

                // --- Fourth: parallel, direct ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x7 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x7, y7);
                task = taskFactory.getIndexAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, (int) out.getSecond());
                assertEquals(msg, result1d, out.getFirst(), 0.0);
                assertEquals(msg, op.z().getDouble(result1), out.getFirst(), 0.0);
                assertEquals(msg, x7, origX);
                assertEquals(msg, result1, op.getFinalResult());

                //Same thing, but split via tensors first:
                INDArray x7a = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7a = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                op = xyConstructor.newInstance(x7a, y7a);
                task = taskFactory.getIndexAccumulationTask(op);
                out = task.invokeBlocking();
                assertEquals(msg, result1, (int) out.getSecond());
                assertEquals(msg, result1d, out.getFirst(), 0.0);
                assertEquals(msg, op.z().getDouble(result1), out.getFirst(), 0.0);
                assertEquals(msg, x7a, origX);
                assertEquals(msg, result1, op.getFinalResult());
            }
        }

        Nd4j.alloc = origAlloc;
    }


    @Test
    public void testOpExecutionerAccumulationOpsAlongDimensions() throws Exception {
        //Test accumulation ops along dimensions
        //Basic idea: results should be identical, whether executed in serial vs. parallel, heap vs. direct,
        // or direct execution vs. split via tensors
        final DataBuffer.AllocationMode origAlloc = Nd4j.alloc;

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

        CPUTaskFactory taskFactory = new CPUTaskFactory();

        int[] shape = {30, 50};
        int[] shape0 = {1, shape[1]};
        int[] shape1 = {1, shape[0]};

        for (DataBuffer.Type dtype : DataBuffer.Type.values()) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape);
            INDArray origY = Nd4j.rand(shape);

            for (Class<? extends Accumulation> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends Accumulation> xyConstructor = opClass.getConstructor(INDArray.class, INDArray.class);

                // --- First: serial, heap ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x1 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                //Along d0
                Accumulation op = xyConstructor.newInstance(x1, y1);
                Task<INDArray> task = taskFactory.getAccumulationTask(op, 0);
                INDArray out0 = task.invokeBlocking();
                assertArrayEquals(msg, shape0, out0.shape());
                assertEquals(msg, x1, origX);

                //Along d1
                op = xyConstructor.newInstance(x1, y1);
                task = taskFactory.getAccumulationTask(op, 1);
                INDArray out1 = task.invokeBlocking();
                assertArrayEquals(msg, shape1, out1.shape());
                assertEquals(msg, x1, origX);

                // --- Second: parallel, heap ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x3 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                //Along d0 then d1
                op = xyConstructor.newInstance(x3, y3);
                task = taskFactory.getAccumulationTask(op, 0);
                INDArray out = task.invokeBlocking();
                assertEquals(msg, out0, out);
                task = taskFactory.getAccumulationTask(op, 1);
                out = task.invokeBlocking();
                assertEquals(msg, out1, out);

                // --- Third: serial, direct ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x5 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                //Along d0 then d1
                op = xyConstructor.newInstance(x5, y5);
                task = taskFactory.getAccumulationTask(op, 0);
                out = task.invokeBlocking();
                assertEquals(msg, out0, out);
                task = taskFactory.getAccumulationTask(op, 1);
                out = task.invokeBlocking();
                assertEquals(msg, out1, out);


                // --- Fourth: parallel, direct ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x7 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                //Along d0 then d1
                op = xyConstructor.newInstance(x7, y7);
                task = taskFactory.getAccumulationTask(op, 0);
                out = task.invokeBlocking();
                assertEquals(msg, out0, out);
                task = taskFactory.getAccumulationTask(op, 1);
                out = task.invokeBlocking();
                assertEquals(msg, out1, out);
            }
        }

        Nd4j.alloc = origAlloc;
    }

    @Test
    public void testOpExecutionerIndexAccumulationOpsAlongDimensions() throws Exception {
        //Test index accumulation ops along dimensions
        //Basic idea: results should be identical, whether executed in serial vs. parallel, heap vs. direct,
        // or direct execution vs. split via tensors
        final DataBuffer.AllocationMode origAlloc = Nd4j.alloc;

        List<Class<? extends IndexAccumulation>> testClasses = new ArrayList<>();
        testClasses.add(IAMax.class);
        testClasses.add(IMax.class);
        testClasses.add(IMin.class);

        CPUTaskFactory taskFactory = new CPUTaskFactory();

        int[] shape = {30, 50};
        int[] shape0 = {1, shape[1]};
        int[] shape1 = {1, shape[0]};

        for (DataBuffer.Type dtype : DataBuffer.Type.values()) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape);
            INDArray origY = Nd4j.rand(shape);

            for (Class<? extends IndexAccumulation> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends IndexAccumulation> xyConstructor = opClass.getConstructor(INDArray.class, INDArray.class);

                // --- First: serial, heap ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x1 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                //Along d0
                IndexAccumulation op = xyConstructor.newInstance(x1, y1);
                Task<INDArray> task = taskFactory.getIndexAccumulationTask(op, 0);
                INDArray out0 = task.invokeBlocking();
                assertArrayEquals(msg, shape0, out0.shape());
                assertEquals(msg, x1, origX);

                //Along d1
                op = xyConstructor.newInstance(x1, y1);
                task = taskFactory.getIndexAccumulationTask(op, 1);
                INDArray out1 = task.invokeBlocking();
                assertArrayEquals(msg, shape1, out1.shape());
                assertEquals(msg, x1, origX);

                // --- Second: parallel, heap ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x3 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3 = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);

                //Along d0 then d1
                op = xyConstructor.newInstance(x3, y3);
                task = taskFactory.getIndexAccumulationTask(op, 0);
                INDArray out = task.invokeBlocking();
                assertEquals(msg, out0, out);
                task = taskFactory.getIndexAccumulationTask(op, 0);
                out = task.invokeBlocking();
                assertEquals(msg, out1, out);


                // --- Third: serial, direct ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x5 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                //Along d0 then d1
                op = xyConstructor.newInstance(x5, y5);
                task = taskFactory.getIndexAccumulationTask(op, 0);
                out = task.invokeBlocking();
                assertEquals(msg, out0, out);
                task = taskFactory.getIndexAccumulationTask(op, 0);
                out = task.invokeBlocking();
                assertEquals(msg, out1, out);

                // --- Fourth: parallel, direct ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x7 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7 = getCopyOf(origY, DataBuffer.AllocationMode.DIRECT, dtype);

                //Along d0 then d1
                op = xyConstructor.newInstance(x7, y7);
                task = taskFactory.getIndexAccumulationTask(op, 0);
                out = task.invokeBlocking();
                assertEquals(msg, out0, out);
                task = taskFactory.getIndexAccumulationTask(op, 0);
                out = task.invokeBlocking();
                assertEquals(msg, out1, out);
            }
        }

        Nd4j.alloc = origAlloc;
    }


    private static INDArray getCopyOf(INDArray original, DataBuffer.AllocationMode mode, DataBuffer.Type type) {
        DataBuffer.AllocationMode origMode = Nd4j.alloc;
        DataBuffer.Type origType = Nd4j.dataType();
        Nd4j.alloc = mode;
        Nd4j.dtype = type;
        Nd4j.factory().setDType(type);
        INDArray out = Nd4j.create(original.shape());
        if (type == DataBuffer.Type.FLOAT) {
            for (int i = 0; i < original.length(); i++) out.putScalar(i, original.getFloat(i));
        } else {
            for (int i = 0; i < original.length(); i++) out.putScalar(i, original.getDouble(i));
        }
        Nd4j.alloc = origMode;
        Nd4j.dtype = origType;
        Nd4j.factory().setDType(origType);
        assertEquals(mode, out.data().allocationMode());
        assertEquals(original, out);
        return out;
    }

    @Test
    public void testOpExecutionerVectorOp() throws Exception {
        //Test accumulation ops along dimensions
        //Basic idea: results should be identical, whether executed in serial vs. parallel, heap vs. direct,
        // or direct execution vs. split via tensors
        final DataBuffer.AllocationMode origAlloc = Nd4j.alloc;

        List<Class<? extends VectorOp>> testClasses = new ArrayList<>();
        testClasses.add(VectorAddOp.class);
        testClasses.add(VectorCopyOp.class);
        testClasses.add(VectorDivOp.class);
        testClasses.add(VectorMulOp.class);
        testClasses.add(VectorRDivOp.class);
        testClasses.add(VectorRSubOp.class);
        testClasses.add(VectorSubOp.class);

        CPUTaskFactory taskFactory = new CPUTaskFactory();

        int[] shape = {30, 50};
        int[] rowShape = {1,shape[1]};
        int[] colShape = {shape[0],1};

        for (DataBuffer.Type dtype : DataBuffer.Type.values()) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape);
            INDArray origY0 = Nd4j.rand(rowShape);
            INDArray origY1 = Nd4j.rand(colShape);

            for (Class<? extends VectorOp> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends VectorOp> xyConstructor = opClass.getConstructor(INDArray.class, INDArray.class, INDArray.class, int.class);

                // --- First: serial, heap ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x1 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1_0 = getCopyOf(origY0, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y1_1 = getCopyOf(origY1, DataBuffer.AllocationMode.HEAP, dtype);

                //Along d0
                VectorOp op = xyConstructor.newInstance(x1, y1_0, x1, 0);
                Task<Void> task = taskFactory.getVectorOpAction(op);
                task.invokeBlocking();
                INDArray zOut_0 = op.z();
                assertEquals(msg, y1_0, origY0);

                //Along d1
                x1 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyConstructor.newInstance(x1, y1_1, x1, 1);
                task = taskFactory.getVectorOpAction(op);
                task.invokeBlocking();
                INDArray zOut_1 = op.z();
                assertEquals(msg, y1_1, origY1);

                // --- Second: parallel, heap ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;

                INDArray x3 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3_0 = getCopyOf(origY0, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray y3_1 = getCopyOf(origY1, DataBuffer.AllocationMode.HEAP, dtype);

                //Along d0 then d1
                op = xyConstructor.newInstance(x3, y3_0, x3, 0);
                task = taskFactory.getVectorOpAction(op);
                task.invokeBlocking();
                assertEquals(msg, zOut_0, op.z());
                assertEquals(msg, origY0, y3_0);
                x3 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyConstructor.newInstance(x3, y3_1, x3, 1);
                task = taskFactory.getVectorOpAction(op);
                task.invokeBlocking();
                assertEquals(msg, zOut_1, op.z());
                assertEquals(msg, origY1, y3_1);

                // --- Third: serial, direct ---
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x5 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5_0 = getCopyOf(origY0, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y5_1 = getCopyOf(origY1, DataBuffer.AllocationMode.DIRECT, dtype);

                //Along d0 then d1
                op = xyConstructor.newInstance(x5, y5_0, x5, 0);
                task = taskFactory.getVectorOpAction(op);
                task.invokeBlocking();
                assertEquals(msg, zOut_0, op.z());
                assertEquals(msg, origY0, y5_0);
                x5 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyConstructor.newInstance(x5, y5_1, x5, 1);
                task = taskFactory.getVectorOpAction(op);
                task.invokeBlocking();
                assertEquals(msg, zOut_1, op.z());
                assertEquals(msg, origY1, y5_1);


                // --- Fourth: parallel, direct ---
                taskFactory.setParallelThreshold(5);
                Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;

                INDArray x7 = getCopyOf(origX, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7_0 = getCopyOf(origY0, DataBuffer.AllocationMode.DIRECT, dtype);
                INDArray y7_1 = getCopyOf(origY1, DataBuffer.AllocationMode.DIRECT, dtype);

                //Along d0 then d1
                op = xyConstructor.newInstance(x7, y7_0, x7, 0);
                task = taskFactory.getVectorOpAction(op);
                task.invokeBlocking();
                assertEquals(msg, zOut_0, op.z());
                assertEquals(msg, origY0, y7_0);
                x7 = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyConstructor.newInstance(x7, y7_1, x7, 1);
                task = taskFactory.getVectorOpAction(op);
                task.invokeBlocking();
                assertEquals(msg, zOut_1, op.z());
                assertEquals(msg, origY1, y7_1);
            }
        }

        Nd4j.alloc = origAlloc;
    }

}
