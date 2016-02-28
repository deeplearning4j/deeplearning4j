package org.nd4j.linalg.parallel.tasks.cpu;

import org.apache.commons.math3.util.Pair;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEquals;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarSetValue;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.*;
import org.nd4j.linalg.api.ops.impl.broadcast.*;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.cpu.CPUTaskFactory;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.lang.reflect.Constructor;
import java.util.*;

import static org.junit.Assert.*;
@RunWith(Parameterized.class)
public class CPUTaskFactoryTest extends BaseNd4jTest {
    public CPUTaskFactoryTest(Nd4jBackend backend) {
        super(backend);
    }

    @Before
    public void before()  {
        Nd4j nd4j = new Nd4j();
        Nd4jBackend backend = null;
        try {
            backend = (Nd4jBackend)Class.forName("org.nd4j.linalg.cpu.CpuBackend").newInstance();
            nd4j.initWithBackend(backend);
            Nd4j.factory().setOrder('c');
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

    }

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
        testClasses.add(LogSoftMax.class);

        int[] shape = {30, 50};

        for (DataBuffer.Type dtype : new DataBuffer.Type[]{DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT}) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origFirst = Nd4j.rand(shape).muli(2).subi(1);      //-1 to +1
            INDArray origSecond = Nd4j.rand(shape).muli(2).subi(1);     //-1 to +1

            for (Class<? extends TransformOp> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends TransformOp> xyzConstructor = opClass.getConstructor(INDArray.class, INDArray.class, INDArray.class);

                //Get expected result
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                INDArray origFirstDup = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray origSecondDup = getCopyOf(origSecond, DataBuffer.AllocationMode.HEAP, dtype);
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                Nd4j.alloc = DataBuffer.AllocationMode.HEAP;
                INDArray expectedZ = getCopyOf(origFirst, DataBuffer.AllocationMode.HEAP, dtype);
                TransformOp op = xyzConstructor.newInstance(origFirstDup, origSecondDup, expectedZ);
                Task<Void> task = taskFactory.getTransformAction(op);
                task.invokeBlocking();

                // For each combination of: serial/parallel, heap/direct
                // do ops with: x =/= z and x == z
                // And compare z with expectedZ

                int[] thresholds = {Integer.MAX_VALUE, 5, Integer.MAX_VALUE, 5};
                DataBuffer.AllocationMode[] allocModes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                        DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

                for (int t = 0; t < 4; t++) {
                    int threshold = thresholds[t];
                    DataBuffer.AllocationMode mode = allocModes[t];
                    taskFactory.setParallelThreshold(threshold);
                    Nd4j.alloc = mode;

                    //Test combinations of different types of NDArrays (with different combinations of offsets, strides, etc)
                    List<Pair<INDArray, String>> list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    List<Pair<INDArray, String>> list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    List<Pair<INDArray, String>> list3 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    Random r = new Random(12345);
                    Collections.shuffle(list2, r);
                    Collections.shuffle(list3, r);

                    for (int i = 0; i < list1.size(); i++) {
                        String msg2 = msg + ", i=" + i + ", threshold=" + threshold + ", mode=" + mode;

                        INDArray x1 = list1.get(i).getFirst().assign(origFirst);
                        INDArray y1 = list2.get(i).getFirst().assign(origSecond);
                        INDArray z1 = list3.get(i).getFirst().assign(0);
                        assertEquals(x1.data().allocationMode(), mode);
                        assertEquals(y1.data().allocationMode(), mode);
                        assertEquals(z1.data().allocationMode(), mode);
                        assertEquals(x1.data().dataType(), dtype);
                        assertEquals(y1.data().dataType(), dtype);
                        assertEquals(z1.data().dataType(), dtype);

                        op = xyzConstructor.newInstance(x1, y1, z1);
                        if (op.isPassThrough()) continue;    //Have to execute passthrough via op executioner
                        task = taskFactory.getTransformAction(op);
                        task.invokeBlocking();
                        assertEquals(msg2, x1, origFirst);
                        assertEquals(msg2, y1, origSecond);
                        assertEquals(msg2, z1, expectedZ);

                        INDArray x2 = list1.get(i).getFirst().assign(origFirst);
                        INDArray y2 = list2.get(i).getFirst().assign(origSecond);
                        op = xyzConstructor.newInstance(x2, y2, x2);
                        task = taskFactory.getTransformAction(op);
                        task.invokeBlocking();
                        assertEquals(msg2, y2, origSecond);
                        assertEquals(msg2, x2, expectedZ);
                    }
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

//        int[] shape = {30, 50};
        int[] shape = {3, 5};

        for (DataBuffer.Type dtype : new DataBuffer.Type[]{DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT}) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape).muli(2).subi(1);      //-1 to +1
            INDArray origZ = Nd4j.rand(shape).muli(2).subi(1);      //-1 to +1

            for (Class<? extends ScalarOp> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends ScalarOp> xyzConstructor = opClass.getConstructor(INDArray.class, INDArray.class,
                        INDArray.class, int.class, Number.class);

                //Get expected result:
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                INDArray origXDup = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray expectedZ = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                ScalarOp op = xyzConstructor.newInstance(origXDup, null, expectedZ, origXDup.length(), 0.4);
                Task<Void> task = taskFactory.getScalarAction(op);
                task.invokeBlocking();


                // For each combination of: serial/parallel, heap/direct
                // do ops with: x =/= z and x == z
                // And compare z with expectedZ

                int[] thresholds = {Integer.MAX_VALUE, 5, Integer.MAX_VALUE, 5};
                DataBuffer.AllocationMode[] allocModes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                        DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

                for (int t = 0; t < 4; t++) {
                    int threshold = thresholds[t];
                    DataBuffer.AllocationMode mode = allocModes[t];
                    taskFactory.setParallelThreshold(threshold);
                    Nd4j.alloc = mode;

                    //Test combinations of different types of NDArrays (with different combinations of offsets, strides, etc)
                    List<Pair<INDArray, String>> list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    List<Pair<INDArray, String>> list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    Random r = new Random(12345);
                    Collections.shuffle(list2, r);

                    for (int i = 0; i < list1.size(); i++) {
                        String msg2 = msg + ", i=" + i + ", threshold=" + threshold + ", mode=" + mode;

                        INDArray x1 = list1.get(i).getFirst().assign(origX);
                        INDArray z1 = list2.get(i).getFirst().assign(origZ);
                        assertEquals(x1.data().allocationMode(), mode);
                        assertEquals(z1.data().allocationMode(), mode);
                        assertEquals(x1.data().dataType(), dtype);
                        assertEquals(z1.data().dataType(), dtype);

                        op = xyzConstructor.newInstance(x1, null, z1, x1.length(), 0.4);
                        task = taskFactory.getScalarAction(op);
                        task.invokeBlocking();
                        assertEquals(msg2, x1, origX);
                        assertEquals(msg2, z1, expectedZ);

                        INDArray x2 = list1.get(i).getFirst().assign(origX);
                        op = xyzConstructor.newInstance(x2, null, x2, x2.length(), 0.4);
                        task = taskFactory.getScalarAction(op);
                        task.invokeBlocking();
                        assertEquals(msg2, x2, expectedZ);
                    }
                }
            }
        }

        Nd4j.alloc = origAlloc;
    }

    @Test
    public void testOpExecutionerAccumulationOps() throws Exception {
        //Basic idea: results should be identical, whether executed in serial vs. parallel, heap vs. direct,
        // or direct execution vs. split via tensors
        final DataBuffer.AllocationMode origAlloc = Nd4j.alloc;

        double eps;

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
        testClasses.add(CosineSimilarity.class);
        testClasses.add(EuclideanDistance.class);
        testClasses.add(ManhattanDistance.class);


        CPUTaskFactory taskFactory = new CPUTaskFactory();

        int[] shape = {30, 50};

        for (DataBuffer.Type dtype : new DataBuffer.Type[]{DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT}) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape).muli(2).subi(1);      //-1 to +1
            INDArray origY = Nd4j.rand(shape).muli(2).subi(1);      //-1 to +1

            if (dtype == DataBuffer.Type.DOUBLE) eps = 1e-10;
            else eps = 1e-3;

            for (Class<? extends Accumulation> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends Accumulation> xyConstructor = opClass.getConstructor(INDArray.class, INDArray.class);

                //Get expected result:
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                INDArray origXDup = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray origYDup = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);
                Accumulation op = xyConstructor.newInstance(origXDup, origYDup);
                if (op.isPassThrough()) continue;    //Have to execute passthrough via op executioner, not via task
                Task<Double> task = taskFactory.getAccumulationTask(op);
                double expected = task.invokeBlocking();
                assertEquals(msg, expected, op.getFinalResult().doubleValue(), eps);

                // For each combination of: serial/parallel, heap/direct
                // do ops with: x =/= z and x == z
                // And compare z with expectedZ

                int[] thresholds = {Integer.MAX_VALUE, 5, Integer.MAX_VALUE, 5};
                DataBuffer.AllocationMode[] allocModes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                        DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

                for (int t = 0; t < 4; t++) {
                    int threshold = thresholds[t];
                    DataBuffer.AllocationMode mode = allocModes[t];
                    taskFactory.setParallelThreshold(threshold);
                    Nd4j.alloc = mode;

                    //Test combinations of different types of NDArrays (with different combinations of offsets, strides, etc)
                    List<Pair<INDArray, String>> list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    List<Pair<INDArray, String>> list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    Random r = new Random(12345);
                    Collections.shuffle(list2, r);

                    for (int i = 0; i < list1.size(); i++) {
                        String msg2 = msg + ", i=" + i + ", threshold=" + threshold + ", mode=" + mode;

                        INDArray x1 = list1.get(i).getFirst().assign(origX);
                        INDArray y1 = list2.get(i).getFirst().assign(origY);
                        assertEquals(x1.data().allocationMode(), mode);
                        assertEquals(y1.data().allocationMode(), mode);
                        assertEquals(x1.data().dataType(), dtype);
                        assertEquals(y1.data().dataType(), dtype);

                        op = xyConstructor.newInstance(x1, y1);
                        task = taskFactory.getAccumulationTask(op);
                        double out1 = task.invokeBlocking();
                        assertEquals(msg2, x1, origX);
                        assertEquals(msg2, y1, origY);
                        assertEquals(msg2,expected, out1, eps);

                        INDArray x2 = list1.get(i).getFirst().assign(origX);
                        INDArray y2 = list2.get(i).getFirst().assign(origY);
                        op = xyConstructor.newInstance(x2, y2);
                        task = taskFactory.getAccumulationTask(op);
                        double out2 = task.invokeBlocking();
                        assertEquals(msg2, y2, origY);
                        assertEquals(expected, out2, eps);
                    }
                }
            }
        }

        Nd4j.alloc = origAlloc;
    }


    @Test
    public void testOpExecutionerIndexAccumulationOps() throws Exception {
        //Basic idea: results should be identical, whether executed in serial vs. parallel, heap vs. direct,
        // or direct execution vs. split via tensors
        final DataBuffer.AllocationMode origAlloc = Nd4j.alloc;

        double eps;

        List<Class<? extends IndexAccumulation>> testClasses = new ArrayList<>();
        testClasses.add(IAMax.class);
        testClasses.add(IMax.class);
        testClasses.add(IMin.class);

        CPUTaskFactory taskFactory = new CPUTaskFactory();

        int[] shape = {30, 50};

        for (DataBuffer.Type dtype : new DataBuffer.Type[]{DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT}) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape).muli(2).subi(1);      //-1 to +1
            INDArray origY = Nd4j.rand(shape).muli(2).subi(1);      //-1 to +1

            if (dtype == DataBuffer.Type.DOUBLE) eps = 1e-10;
            else eps = 1e-5;

            for (Class<? extends IndexAccumulation> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends IndexAccumulation> xyConstructor = opClass.getConstructor(INDArray.class, INDArray.class);

                //Get expected result:
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                INDArray origXDup = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray origYDup = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);
                IndexAccumulation op = xyConstructor.newInstance(origXDup, origYDup);
                Task<Pair<Double,Integer>> task = taskFactory.getIndexAccumulationTask(op);
                Pair<Double,Integer> expectedPair = task.invokeBlocking();
                double expectedD = expectedPair.getFirst();
                int expectedI = expectedPair.getSecond();
                assertEquals(msg, expectedI, op.getFinalResult());
                assertEquals(msg, expectedD, op.op(origXDup.getDouble(expectedI), origYDup.getDouble(expectedI)),eps);

                // For each combination of: serial/parallel, heap/direct
                // do ops with: x =/= z and x == z
                // And compare z with expectedZ

                int[] thresholds = {Integer.MAX_VALUE, 5, Integer.MAX_VALUE, 5};
                DataBuffer.AllocationMode[] allocModes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                        DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

                for (int t = 0; t < 4; t++) {
                    int threshold = thresholds[t];
                    DataBuffer.AllocationMode mode = allocModes[t];
                    taskFactory.setParallelThreshold(threshold);
                    Nd4j.alloc = mode;

                    //Test combinations of different types of NDArrays (with different combinations of offsets, strides, etc)
                    List<Pair<INDArray, String>> list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    List<Pair<INDArray, String>> list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    Random r = new Random(12345);
                    Collections.shuffle(list2, r);

                    for (int i = 0; i < list1.size(); i++) {
                        String msg2 = msg + ", i=" + i + ", threshold=" + threshold + ", mode=" + mode;

                        INDArray x1 = list1.get(i).getFirst().assign(origX);
                        INDArray y1 = list2.get(i).getFirst().assign(origY);
                        assertEquals(x1.data().allocationMode(), mode);
                        assertEquals(y1.data().allocationMode(), mode);
                        assertEquals(x1.data().dataType(), dtype);
                        assertEquals(y1.data().dataType(), dtype);

                        op = xyConstructor.newInstance(x1, y1);
                        task = taskFactory.getIndexAccumulationTask(op);
                        Pair<Double,Integer> out1 = task.invokeBlocking();
                        assertEquals(msg2, x1, origX);
                        assertEquals(msg2, y1, origY);
                        assertEquals(msg2, expectedI, (int)out1.getSecond());
                        assertEquals(msg2, expectedD, out1.getFirst(), eps);

                        INDArray x2 = list1.get(i).getFirst().assign(origX);
                        INDArray y2 = list2.get(i).getFirst().assign(origY);
                        op = xyConstructor.newInstance(x2, y2);
                        task = taskFactory.getIndexAccumulationTask(op);
                        Pair<Double,Integer> out2 = task.invokeBlocking();
                        assertEquals(msg2, x2, origX);
                        assertEquals(msg2, y2, origY);
                        assertEquals(msg2, expectedI, (int)out2.getSecond());
                        assertEquals(msg2, expectedD, out2.getFirst(), eps);
                    }
                }
            }
        }

        Nd4j.alloc = origAlloc;
    }

    @Test
    public void testDimensionZero() {
        INDArray arr = Nd4j.linspace(1,8,8).reshape(2,4);
        INDArray norm1 = arr.norm2(0);
        INDArray assertion = Nd4j.create(new double[]{5.09901951,  6.32455532,  7.61577311,  8.94427191});
        assertEquals(assertion,norm1);

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
        testClasses.add(CosineSimilarity.class);
        testClasses.add(EuclideanDistance.class);
        testClasses.add(ManhattanDistance.class);

        CPUTaskFactory taskFactory = new CPUTaskFactory();

        int[] shape = {30, 50};

        for (DataBuffer.Type dtype : new DataBuffer.Type[]{DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT}) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape).muli(2).subi(1);      //-1 to +1
            INDArray origY = Nd4j.rand(shape).muli(2).subi(1);      //-1 to +1

            for (Class<? extends Accumulation> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends Accumulation> xyConstructor = opClass.getConstructor(INDArray.class, INDArray.class);

                //Get expected result:
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                INDArray origXDup = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray origYDup = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);
                Accumulation op = xyConstructor.newInstance(origXDup, origYDup);
                if (op.isPassThrough()) continue;    //Have to execute passthrough via op executioner, not via task
                Task<INDArray> task = taskFactory.getAccumulationTask(op, 0);
                INDArray expected0 = task.invokeBlocking();
                assertTrue(expected0==op.z());

                origXDup = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                origYDup = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyConstructor.newInstance(origXDup, origYDup);
                task = taskFactory.getAccumulationTask(op, 1);
                INDArray expected1 = task.invokeBlocking();
                assertEquals(expected1,op.z());

                // For each combination of: serial/parallel, heap/direct
                // And compare output with expected

                int[] thresholds = {Integer.MAX_VALUE, 5, Integer.MAX_VALUE, 5};
                DataBuffer.AllocationMode[] allocModes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                        DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

                for (int t = 0; t < thresholds.length; t++) {
                    int threshold = thresholds[t];
                    DataBuffer.AllocationMode mode = allocModes[t];
                    taskFactory.setParallelThreshold(threshold);
                    Nd4j.alloc = mode;

                    //Test combinations of different types of NDArrays (with different combinations of offsets, strides, etc)
                    List<Pair<INDArray, String>> list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    List<Pair<INDArray, String>> list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    Random r = new Random(12345);
                    Collections.shuffle(list2, r);

                    for (int i = 0; i < list1.size(); i++) {
                        String msg2 = msg + ", i=" + i + ", threshold=" + threshold + ", mode=" + mode;

                        INDArray x1 = list1.get(i).getFirst().assign(origX);
                        INDArray y1 = list2.get(i).getFirst().assign(origY);
                        assertEquals(x1.data().allocationMode(), mode);
                        assertEquals(y1.data().allocationMode(), mode);
                        assertEquals(x1.data().dataType(), dtype);
                        assertEquals(y1.data().dataType(), dtype);

                        op = xyConstructor.newInstance(x1, y1);
                        task = taskFactory.getAccumulationTask(op, 0);
                        INDArray out0_xz = task.invokeBlocking();
                        assertEquals(msg2, x1, origX);
                        assertEquals(msg2, y1, origY);
                        assertEquals(msg2, expected0, out0_xz);
                        assertTrue(out0_xz == op.z());

                        op = xyConstructor.newInstance(x1, y1);
                        task = taskFactory.getAccumulationTask(op, 1);
                        INDArray out1_xz = task.invokeBlocking();
                        assertEquals(msg2, x1, origX);
                        assertEquals(msg2, y1, origY);
                        assertEquals(msg2, expected1, out1_xz);
                        assertEquals(out1_xz , op.z());
                    }
                }
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

        for (DataBuffer.Type dtype : new DataBuffer.Type[]{DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT}) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape).muli(2).subi(1);
            INDArray origY = Nd4j.rand(shape).muli(2).subi(1);

            for (Class<? extends IndexAccumulation> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends IndexAccumulation> xyConstructor = opClass.getConstructor(INDArray.class, INDArray.class);


                //Get expected result:
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                INDArray origXDup = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray origYDup = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);
                IndexAccumulation op = xyConstructor.newInstance(origXDup, origYDup);
                Task<INDArray> task = taskFactory.getIndexAccumulationTask(op, 0);
                INDArray expected0 = task.invokeBlocking();
                assertTrue(expected0==op.z());

                origXDup = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                origYDup = getCopyOf(origY, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyConstructor.newInstance(origXDup, origYDup);
                task = taskFactory.getIndexAccumulationTask(op, 1);
                INDArray expected1 = task.invokeBlocking();
                assertTrue(expected1==op.z());

                // For each combination of: serial/parallel, heap/direct
                // And compare output with expected

                int[] thresholds = {Integer.MAX_VALUE, 5, Integer.MAX_VALUE, 5};
                DataBuffer.AllocationMode[] allocModes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                        DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

                for (int t = 0; t < 4; t++) {
                    int threshold = thresholds[t];
                    DataBuffer.AllocationMode mode = allocModes[t];
                    taskFactory.setParallelThreshold(threshold);
                    Nd4j.alloc = mode;

                    //Test combinations of different types of NDArrays (with different combinations of offsets, strides, etc)
                    List<Pair<INDArray, String>> list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    List<Pair<INDArray, String>> list2 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    Random r = new Random(12345);
                    Collections.shuffle(list2, r);

                    for (int i = 0; i < list1.size(); i++) {
                        String msg2 = msg + ", i=" + i + ", threshold=" + threshold + ", mode=" + mode;

                        INDArray x1 = list1.get(i).getFirst().assign(origX);
                        INDArray y1 = list2.get(i).getFirst().assign(origY);
                        assertEquals(x1.data().allocationMode(), mode);
                        assertEquals(y1.data().allocationMode(), mode);
                        assertEquals(x1.data().dataType(), dtype);
                        assertEquals(y1.data().dataType(), dtype);

                        op = xyConstructor.newInstance(x1, y1);
                        task = taskFactory.getIndexAccumulationTask(op, 0);
                        INDArray out0_xz = task.invokeBlocking();
                        assertEquals(msg2, x1, origX);
                        assertEquals(msg2, y1, origY);
                        assertEquals(msg2, expected0, out0_xz);
                        assertTrue(out0_xz == op.z());

                        op = xyConstructor.newInstance(x1, y1);
                        task = taskFactory.getIndexAccumulationTask(op, 1);
                        INDArray out1_xz = task.invokeBlocking();
                        assertEquals(msg2, x1, origX);
                        assertEquals(msg2, y1, origY);
                        assertEquals(msg2, expected1, out1_xz);
                        assertTrue(out1_xz == op.z());
                    }
                }
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

        List<Class<? extends BroadcastOp>> testClasses = new ArrayList<>();
        testClasses.add(BroadcastAddOp.class);
        testClasses.add(BroadcastCopyOp.class);
        testClasses.add(BroadcastDivOp.class);
        testClasses.add(BroadcastMulOp.class);
        testClasses.add(BroadcastRDivOp.class);
        testClasses.add(BroadcastRSubOp.class);
        testClasses.add(BroadcastSubOp.class);

        CPUTaskFactory taskFactory = new CPUTaskFactory();

        int[] shape = {30, 50};
        int[] rowShape = {1, shape[1]};
        int[] colShape = {shape[0], 1};

        for (DataBuffer.Type dtype : new DataBuffer.Type[]{DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT} ) {

            Nd4j.dtype = dtype;
            Nd4j.factory().setDType(dtype);

            Nd4j.getRandom().setSeed(12345);
            INDArray origX = Nd4j.rand(shape).muli(2).subi(1);
            INDArray origY0 = Nd4j.rand(colShape).muli(2).subi(1);  //Along d0 = column
            INDArray origY1 = Nd4j.rand(rowShape).muli(2).subi(1);  //Along d1 = row
            INDArray origZ = Nd4j.rand(shape).muli(2).subi(1);

            for (Class<? extends BroadcastOp> opClass : testClasses) {
                String msg = "class: " + opClass.getName() + ", dtype=" + dtype;
                Constructor<? extends BroadcastOp> xyznConstructor = opClass.getConstructor(INDArray.class, INDArray.class, INDArray.class, int.class);

                //Get expected result:
                taskFactory.setParallelThreshold(Integer.MAX_VALUE);
                INDArray origXDup = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray origY0Dup = getCopyOf(origY0, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray expectedZ0 = getCopyOf(origZ, DataBuffer.AllocationMode.HEAP, dtype);
                BroadcastOp op = xyznConstructor.newInstance(origXDup, origY0Dup, expectedZ0, 0);
                Task<Void> task = taskFactory.getBroadcastOpAction(op);
                task.invokeBlocking();


                origXDup = getCopyOf(origX, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray origY1Dup = getCopyOf(origY1, DataBuffer.AllocationMode.HEAP, dtype);
                INDArray expectedZ1 = getCopyOf(origZ, DataBuffer.AllocationMode.HEAP, dtype);
                op = xyznConstructor.newInstance(origXDup, origY1Dup, expectedZ1, 1);
                task = taskFactory.getBroadcastOpAction(op);
                task.invokeBlocking();

                // For each combination of: serial/parallel, heap/direct
                // And compare output with expected

                int[] thresholds = {Integer.MAX_VALUE, 5, Integer.MAX_VALUE, 5};
                DataBuffer.AllocationMode[] allocModes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                        DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

                for (int t = 0; t < 4; t++) {
                    int threshold = thresholds[t];
                    DataBuffer.AllocationMode mode = allocModes[t];
                    taskFactory.setParallelThreshold(threshold);
                    Nd4j.alloc = mode;

                    //Test combinations of different types of NDArrays (with different combinations of offsets, strides, etc)
                    List<Pair<INDArray, String>> list1 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    List<Pair<INDArray, String>> list2_0 = NDArrayCreationUtil.getAllTestMatricesWithShape(colShape[0], colShape[1], 123);
                    List<Pair<INDArray, String>> list2_1 = NDArrayCreationUtil.getAllTestMatricesWithShape(rowShape[0], rowShape[1], 123);
                    List<Pair<INDArray, String>> list3 = NDArrayCreationUtil.getAllTestMatricesWithShape(shape[0], shape[1], 123);
                    Random r = new Random(12345);
                    Collections.shuffle(list2_0, r);
                    Collections.shuffle(list2_1, r);
                    Collections.shuffle(list3, r);

                    for (int i = 0; i < list1.size(); i++) {
                        String msg2 = msg + ", i=" + i + ", threshold=" + threshold + ", mode=" + mode;

                        //z=/=x, then z=x
                        //Along d0:
                        INDArray x1 = list1.get(i).getFirst().assign(origX);
                        INDArray y1_0 = list2_0.get(i).getFirst().assign(origY0);
                        INDArray z1 = list3.get(i).getFirst().assign(origZ);
                        assertEquals(x1.data().allocationMode(), mode);
                        assertEquals(y1_0.data().allocationMode(), mode);
                        assertEquals(z1.data().allocationMode(), mode);
                        assertEquals(x1.data().dataType(), dtype);
                        assertEquals(y1_0.data().dataType(), dtype);
                        assertEquals(z1.data().dataType(), dtype);

                        op = xyznConstructor.newInstance(x1, y1_0, z1, 0);
                        task = taskFactory.getBroadcastOpAction(op);
                        task.invokeBlocking();
                        assertEquals(msg2, x1, origX);
                        assertEquals(msg2, y1_0, origY0);
                        assertEquals(msg2, expectedZ0, z1);

                        op = xyznConstructor.newInstance(x1, y1_0, x1, 0);
                        task = taskFactory.getBroadcastOpAction(op);
                        task.invokeBlocking();
                        assertEquals(msg2, y1_0, origY0);
                        assertEquals(msg2, expectedZ0, x1);

                        //Along d1
                        x1 = list1.get(i).getFirst().assign(origX);
                        INDArray y1_1 = list2_1.get(i).getFirst().assign(origY1);
                        z1 = list3.get(i).getFirst().assign(origZ);
                        assertEquals(y1_1.data().allocationMode(), mode);
                        assertEquals(y1_1.data().dataType(), dtype);

                        op = xyznConstructor.newInstance(x1, y1_1, z1, 1);
                        task = taskFactory.getBroadcastOpAction(op);
                        task.invokeBlocking();
                        assertEquals(msg2, x1, origX);
                        assertEquals(msg2, y1_1, origY1);
                        assertEquals(msg2, expectedZ1, z1);

                        op = xyznConstructor.newInstance(x1, y1_1, x1, 1);
                        task = taskFactory.getBroadcastOpAction(op);
                        task.invokeBlocking();
                        assertEquals(msg2, y1_1, origY1);
                        assertEquals(msg2, expectedZ1, x1);
                    }
                }
            }
        }

        Nd4j.alloc = origAlloc;
    }

}
