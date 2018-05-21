package org.nd4j.linalg.jcublas.ops.executioner;

import org.bytedeco.javacpp.Pointer;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.impl.accum.EqualsWithEps;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.accum.Sum;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMultiplication;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarSet;
import org.nd4j.linalg.api.ops.impl.transforms.Abs;
import org.nd4j.linalg.api.ops.impl.transforms.Set;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class GridExecutionerTest {
    @Before
    public void setUp() throws Exception {
        CudaEnvironment.getInstance().getConfiguration()
                .enableDebug(true);
    }
///////////////////////////////////////////////////////////////////////////
/*/////////////////////////////////////////////////////////////////////////

    MatchMeta tests are checking, how ops are matching for MetaOp requirements

*//////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

    @Test
    public void isMatchingMetaOp1() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray array = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        ScalarAdd opB = new ScalarAdd(array, 10f);

        executioner.exec(opA);
        assertEquals(CudaGridExecutioner.MetaType.NOT_APPLICABLE, executioner.getMetaOpType(opB));
    }

    @Test
    public void isMatchingMetaOp2() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray array = Nd4j.create(10);
        INDArray array2 = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        ScalarAdd opB = new ScalarAdd(array2, 10f);

        executioner.exec(opA);
        assertEquals(executioner.getMetaOpType(opB), CudaGridExecutioner.MetaType.NOT_APPLICABLE);
    }

    @Test
    public void isMatchingMetaOp3() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray array = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        Max opB = new Max(array);

        executioner.exec(opA);
        assertEquals(CudaGridExecutioner.MetaType.NOT_APPLICABLE, executioner.getMetaOpType(opB));
    }

    @Test
    public void isMatchingMetaOp4() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(10);
        INDArray arrayY = Nd4j.create(10);

        Set opA = new Set(arrayX, arrayY, arrayX, arrayX.length());

        ScalarAdd opB = new ScalarAdd(arrayX, 10f);

        executioner.exec(opA);

        assertEquals(CudaGridExecutioner.MetaType.INVERTED_PREDICATE, executioner.getMetaOpType(opB));
    }

///////////////////////////////////////////////////////////////////////////
/*/////////////////////////////////////////////////////////////////////////

    GridFlow tests are checking how ops are getting queued upon exec() calls

*//////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

    @Test
    public void testGridFlow1() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        assertEquals(0, executioner.getQueueLength());

        INDArray array = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        executioner.exec(opA);

        long time1 = System.nanoTime();

        Max opB = new Max(array);

        executioner.exec(opB);

        assertEquals(0, executioner.getQueueLength());

        long time2 = System.nanoTime();

        opB = new Max(array);

        executioner.exec(opB);

        long time3 = System.nanoTime();

        assertEquals(0, executioner.getQueueLength());



        long firstExec = time2 - time1;
        long secondExec = time3 - time2;

        System.out.println("First exec time: " + firstExec);
        System.out.println("Second exec time: " + secondExec);

        System.out.println("Array: " + array);

    }


    @Test
    public void testGridFlow2() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(10);
        INDArray arrayY = Nd4j.create(new float[] {1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f});
        INDArray exp = Nd4j.create(new float[] {3f, 3f, 3f, 3f, 3f, 3f, 3f, 3f, 3f, 3f});

        Set opA = new Set(arrayX, arrayY, arrayX, arrayX.length());

        executioner.exec(opA);

        assertEquals(1, executioner.getQueueLength());

        long time1 = System.nanoTime();

        ScalarAdd opB = new ScalarAdd(arrayX, 2f);

        executioner.exec(opB);

        assertEquals(0, executioner.getQueueLength());

        long time2 = System.nanoTime();


        long time3 = System.nanoTime();

        assertEquals(0, executioner.getQueueLength());



        long firstExec = time2 - time1;
        long secondExec = time3 - time2;

        System.out.println("First exec time: " + firstExec);
        System.out.println("Second exec time: " + secondExec);

        assertEquals(exp, arrayX);
        System.out.println("ArrayX: " + arrayX);
        System.out.println("ArrayExp: " + exp);

    }

    @Test
    public void testGridFlow3() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(10);
        INDArray arrayY = Nd4j.create(new float[] {1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f});

        EqualsWithEps op = new EqualsWithEps(arrayX, arrayY, Nd4j.EPS_THRESHOLD);

        executioner.exec(op);

        assertEquals(0, executioner.getQueueLength());
        assertNotEquals(null, op.getFinalResult());
        assertEquals(10, op.getFinalResult().intValue());
    }

    @Test
    public void testGridFlow4() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(new float[] {1f, 1f, 1f, 1f, 1f});

        Sum op = new Sum(arrayX);

        executioner.exec(op);

        assertEquals(0, executioner.getQueueLength());
        assertNotEquals(null, op.getFinalResult());
        assertEquals(5, op.getFinalResult().intValue());
    }

    @Test
    public void testGridFlow5() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(5, 5);

        Sum op = new Sum(arrayX);

        executioner.exec(op, 1);

        assertEquals(0, executioner.getQueueLength());
        assertEquals(0, op.z().getFloat(0), 0.1f);
    }

    @Test
    public void testGridFlow6() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(new float[] {-1f, -1f, 1f});
        INDArray exp = Nd4j.create(new float[] {1f, 1f, 1f});

        Abs op = new Abs(arrayX);

        executioner.exec(op);

        op = new Abs(arrayX);
        executioner.exec(op);

        assertEquals(0, executioner.getQueueLength());
        assertEquals(exp, arrayX);
    }


    @Test
    public void testGridFlow7() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(new float[] {0f, 0f, 0f});
        INDArray arrayY1 = Nd4j.create(new float[] {-1f, -1f, 1f});
        INDArray arrayY2 = Nd4j.create(new float[] {1f, 1f, 1f});
        INDArray exp = Nd4j.create(new float[] {1f, 1f, 1f});

        Set opA = new Set(arrayX, arrayY1, arrayX, arrayY1.length());

        executioner.exec(opA);

        assertEquals(1, executioner.getQueueLength());

        Set opB = new Set(arrayX, arrayY2, arrayX, arrayY1.length());
        executioner.exec(opB);

        assertEquals(1, executioner.getQueueLength());

        assertEquals(1, executioner.getExecutionCounter());
        //System.out.println("---------------------------");

        executioner.flushQueueBlocking();
        arrayX.getFloat(0);


        // it should be 0, because getFloat() should trigger flushQueue
        assertEquals(2, executioner.getExecutionCounter());
        assertEquals(0, executioner.getQueueLength());

        assertEquals(1f, arrayX.getFloat(0), 0.1f);
//        assertEquals(exp, arrayX);
    }

    @Test
    public void testGridFlow8() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(new float[] {0f, 0f, 0f});
        INDArray arrayY1 = Nd4j.create(new float[] {-1f, -1f, 1f});
        INDArray arrayY2 = Nd4j.create(new float[] {1f, 1f, 1f});
        INDArray exp = Nd4j.create(new float[] {1f, 1f, 1f});

        Set opA = new Set(arrayX, arrayY1, arrayX, arrayY1.length());

        executioner.exec(opA);

        assertEquals(1, executioner.getQueueLength());

        ScalarSet opB = new ScalarSet(arrayX, 1f);
        executioner.exec(opB);

        assertEquals(0, executioner.getQueueLength());
        assertEquals(1f, arrayX.getFloat(0), 0.1f);
        assertEquals(1f, arrayX.getFloat(1), 0.1f);
        //assertEquals(exp, arrayX);
    }
/*
    @Test
    public void testGridFlow9() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(new float[] {0f, 0f, 0f});
        INDArray arrayY1 = Nd4j.create(new float[] {-1f, -1f, 1f});
        INDArray arrayY2 = Nd4j.create(new float[] {1f, 1f, 1f});
        INDArray exp = Nd4j.create(new float[] {1f, 1f, 1f});

        Set opA = new Set(arrayX, arrayY1, arrayX, arrayY1.length());

        executioner.exec(opA);

        assertEquals(1, executioner.getQueueLength());

        ScalarSet opB = new ScalarSet(arrayX, 1f);
        executioner.exec(opB);

        assertEquals(0, executioner.getQueueLength());
        assertEquals(1f, arrayX.getFloat(0), 0.1f);
        assertEquals(1f, arrayX.getFloat(1), 0.1f);
        //assertEquals(exp, arrayX);
    }
*/
    @Test
    public void testGridFlowFlush1() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(10);
        INDArray arrayY = Nd4j.create(new float[] {1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f});
        INDArray exp = Nd4j.create(new float[] {3f, 3f, 3f, 3f, 3f, 3f, 3f, 3f, 3f, 3f});

        Set opA = new Set(arrayX, arrayY, arrayX, arrayX.length());

        executioner.exec(opA);

        executioner.flushQueue();

        assertEquals(arrayY, arrayX);
    }


    @Test
    public void testGridFlowFlush2() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(10);
        INDArray arrayX2 = Nd4j.create(10);
        INDArray arrayY = Nd4j.create(new float[] {1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f});
        INDArray exp = Nd4j.create(new float[] {3f, 3f, 3f, 3f, 3f, 3f, 3f, 3f, 3f, 3f});
        INDArray exp2 = Nd4j.create(new float[] {10f, 10f, 10f, 10f, 10f, 10f, 10f, 10f, 10f, 10f});

        Set opA = new Set(arrayX, arrayY, arrayX, arrayX.length());

        executioner.exec(opA);

        assertEquals(1, executioner.getQueueLength());

        ScalarAdd opB = new ScalarAdd(arrayX2, 10f);

        executioner.exec(opB);

        assertEquals(0, executioner.getQueueLength());

        assertEquals(arrayY, arrayX);
        assertEquals(exp2, arrayX2);
    }




    /////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/*
    Performance test for combined op
*/
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
    @Test
    public void testGridPerformance1() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray arrayX = Nd4j.create(1024);
        INDArray arrayY = Nd4j.create(1024);

        Set opA = new Set(arrayX, arrayY, arrayX, arrayX.length());
        ScalarAdd opB = new ScalarAdd(arrayX, 2f);

        long time1 = System.nanoTime();
        for (int x = 0; x < 1000000; x++) {
            executioner.exec(opA);
            executioner.exec(opB);
        }
        long time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 1000000));
    }



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/*
    Pointerize tests are checking how Ops are converted into GridPointers
*/
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

    @Test
    public void testOpPointerizeScalar1() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray array = Nd4j.create(10);
        ScalarMultiplication opA = new ScalarMultiplication(array, 10f);

        GridPointers pointers = executioner.pointerizeOp(opA, null);

        assertEquals(opA.opNum(), pointers.getOpNum());
        assertEquals(Op.Type.SCALAR, pointers.getType());

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        Pointer x = AtomicAllocator.getInstance().getPointer(array, context);
        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(array.shapeInfoDataBuffer(), context);

        assertEquals(x, pointers.getX());
        assertEquals(null, pointers.getY());
        assertEquals(x, pointers.getZ());

        assertEquals(1, pointers.getXStride());
        assertEquals(-1, pointers.getYStride());
        assertEquals(1, pointers.getZStride());

        assertEquals(xShapeInfo, pointers.getXShapeInfo());
        assertEquals(null, pointers.getYShapeInfo());
        assertEquals(xShapeInfo, pointers.getZShapeInfo());

        assertEquals(null, pointers.getDimensions());
        assertEquals(0, pointers.getDimensionsLength());

        assertEquals(null, pointers.getTadShape());
        assertEquals(null, pointers.getTadOffsets());

        assertEquals(null, pointers.getExtraArgs());
    }

    /**
     * Reduce along dimensions
     *
     * @throws Exception
     */
    @Test
    public void testOpPointerizeReduce1() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray array = Nd4j.create(10, 10);

        Sum opA = new Sum(array);

        // we need exec here, to init Op.Z for specific dimension
        executioner.exec(opA, 1);

        GridPointers pointers = executioner.pointerizeOp(opA, 1);

        assertEquals(opA.opNum(), pointers.getOpNum());
        assertEquals(Op.Type.REDUCE, pointers.getType());

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        Pointer x = AtomicAllocator.getInstance().getPointer(array, context);
        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(array.shapeInfoDataBuffer(), context);

        Pointer z = AtomicAllocator.getInstance().getPointer(opA.z(), context);
        Pointer zShapeInfo = AtomicAllocator.getInstance().getPointer(opA.z().shapeInfoDataBuffer(), context);

        DataBuffer dimBuff = Nd4j.getConstantHandler().getConstantBuffer(new int[] {1});

        Pointer ptrBuff = AtomicAllocator.getInstance().getPointer(dimBuff, context);

        assertEquals(x, pointers.getX());
        assertEquals(null, pointers.getY());
        assertNotEquals(null, pointers.getZ());
        assertEquals(z, pointers.getZ());

        assertEquals(10, opA.z().length());
        assertEquals(10, pointers.getZLength());

/*      // We dont really care about EWS here, since we're testing TAD-based operation

        assertEquals(1, pointers.getXStride());
        assertEquals(-1, pointers.getYStride());
        assertEquals(1, pointers.getZStride());
*/
        assertEquals(xShapeInfo, pointers.getXShapeInfo());
        assertEquals(null, pointers.getYShapeInfo());
        assertEquals(zShapeInfo, pointers.getZShapeInfo());

        assertEquals(ptrBuff, pointers.getDimensions());
        assertEquals(1, pointers.getDimensionsLength());

        assertNotEquals(null, pointers.getTadShape());
        assertNotEquals(null, pointers.getTadOffsets());

        assertEquals(null, pointers.getExtraArgs());
    }

    /**
     * Reduce along all dimensions
     *
     * @throws Exception
     */
    @Test
    public void testOpPointerizeReduce2() throws Exception {
        CudaGridExecutioner executioner = new CudaGridExecutioner();

        INDArray array = Nd4j.create(10, 10);

        Sum opA = new Sum(array);

        // we need exec here, to init Op.Z for specific dimension
        executioner.exec(opA);

        GridPointers pointers = executioner.pointerizeOp(opA, null);

        assertEquals(opA.opNum(), pointers.getOpNum());
        assertEquals(Op.Type.REDUCE, pointers.getType());

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        Pointer x = AtomicAllocator.getInstance().getPointer(array, context);
        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(array.shapeInfoDataBuffer(), context);

        Pointer z = AtomicAllocator.getInstance().getPointer(opA.z(), context);
        Pointer zShapeInfo = AtomicAllocator.getInstance().getPointer(opA.z().shapeInfoDataBuffer(), context);

        DataBuffer dimBuff = Nd4j.getConstantHandler().getConstantBuffer(new int[] {1});

        Pointer ptrBuff = AtomicAllocator.getInstance().getPointer(dimBuff, context);

        assertEquals(x, pointers.getX());
        assertEquals(null, pointers.getY());
        assertNotEquals(null, pointers.getZ());
        assertEquals(z, pointers.getZ());

        assertEquals(1, opA.z().length());
        assertEquals(1, pointers.getZLength());


/*      // We dont really care about EWS here, since we're testing TAD-based operation

        assertEquals(1, pointers.getXStride());
        assertEquals(-1, pointers.getYStride());
        assertEquals(1, pointers.getZStride());
*/
        assertEquals(xShapeInfo, pointers.getXShapeInfo());
        assertEquals(null, pointers.getYShapeInfo());
        assertEquals(zShapeInfo, pointers.getZShapeInfo());

        assertEquals(null, pointers.getDimensions());
        assertEquals(0, pointers.getDimensionsLength());

        assertEquals(null, pointers.getTadShape());
        assertEquals(null, pointers.getTadOffsets());

        assertEquals(null, pointers.getExtraArgs());
    }

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/*
    MetaOp concatenation tests
*/
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

    /**
     * This test checks
     * @throws Exception
     */
    @Test
    public void testMetaOpScalarTransform1() throws Exception {

    }

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/*
    Reverse flow tests
*/
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

    @Test
    public void testReverseFlow1() throws Exception {
        CudaGridExecutioner executioner = ((CudaGridExecutioner) Nd4j.getExecutioner());

        INDArray put = Nd4j.create(new double[]{5, 6});

        INDArray row1 = Nd4j.linspace(1, 4, 4);

        AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(row1);

        assertEquals(0, executioner.getQueueLength());
        assertEquals(true, point.isActualOnHostSide());
        assertEquals(false, point.isActualOnDeviceSide());

        System.out.println("A: --------------------------");

        row1 = row1.reshape(2, 2);

        assertEquals(true, point.isActualOnHostSide());
        assertEquals(false, point.isActualOnDeviceSide());

        System.out.println("B: --------------------------");

//        ((CudaGridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

        row1.putRow(1, put);

        assertEquals(true, point.isActualOnHostSide());
        assertEquals(false, point.isActualOnDeviceSide());

        System.out.println("C: --------------------------");

        assertEquals(1, executioner.getQueueLength());

        executioner.flushQueueBlocking();

        assertEquals(0, executioner.getQueueLength());

        assertEquals(false, point.isActualOnHostSide());
        assertEquals(true, point.isActualOnDeviceSide());

        System.out.println("D: --------------------------");


    //    ((CudaGridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

        //System.out.println(row1);
        assertArrayEquals(new float[]{1, 2, 5, 6}, row1.data().asFloat(), 0.1f);

    }

    @Test
    public void testReverseFlow2() {
        CudaGridExecutioner executioner = ((CudaGridExecutioner) Nd4j.getExecutioner());

        INDArray n1 = Nd4j.scalar(1);
        INDArray n2 = Nd4j.scalar(2);
        INDArray n3 = Nd4j.scalar(3);
        INDArray n4 = Nd4j.scalar(4);

        System.out.println("0: ------------------------");

        INDArray nClone = n1.add(n2);

        assertEquals(Nd4j.scalar(3), nClone);
        INDArray n1PlusN2 = n1.add(n2);
        assertFalse(n1PlusN2.equals(n1));

        System.out.println("2: ------------------------");

        System.out.println(n4);

        INDArray subbed = n4.sub(n3);
        INDArray mulled = n4.mul(n3);
        INDArray div = n4.div(n3);

        System.out.println("Subbed: " + subbed);
        System.out.println("Mulled: " + mulled);
        System.out.println("Div: " + div);
        System.out.println("4: ------------------------");

        assertFalse(subbed.equals(n4));
        assertFalse(mulled.equals(n4));

        assertEquals(0, executioner.getQueueLength());

        assertEquals(Nd4j.scalar(1), subbed);
        assertEquals(Nd4j.scalar(12), mulled);
        assertEquals(Nd4j.scalar(1.333333333333333333333), div);
    }

    @Test
    public void testReverseFlow3() throws Exception {
        INDArray toSort = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray ascending = Nd4j.sort(toSort.dup(), 1, true);
        assertEquals(toSort, ascending);

        INDArray columnSorted = Nd4j.create(new float[]{2, 1, 4, 3}, new int[]{2, 2});

        // in this particular code point, toSort.dup() isn't executed, but queued
        INDArray dupd = toSort.dup();

        // if we execute muli - dup() op will be flushed as metaOp
        //dupd.muli(1.0);

        INDArray sorted = Nd4j.sort(dupd, 1, false);
        assertEquals(columnSorted, sorted);
    }

    @Test
    public void testDupLocality1() throws Exception {
        INDArray array1 = Nd4j.create(new double[]{1,2,3,4,5});

        AllocationPoint point1 = AtomicAllocator.getInstance().getAllocationPoint(array1);
        assertEquals(true, point1.isActualOnDeviceSide());
        assertEquals(false, point1.isActualOnHostSide());

        INDArray array2 = array1.dup();

        //((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

        AllocationPoint point2 = AtomicAllocator.getInstance().getAllocationPoint(array2);

        assertEquals(true, point2.isActualOnDeviceSide());
        assertEquals(true, point2.isActualOnHostSide());
    }

    @Test
    public void testDupLocality2() throws Exception {
        INDArray array2 = Nd4j.createUninitialized(new int[]{10, 10}, 'c');

        //((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

        AllocationPoint point2 = AtomicAllocator.getInstance().getAllocationPoint(array2);

        assertEquals(true, point2.isActualOnDeviceSide());
        assertEquals(true, point2.isActualOnHostSide());
    }

    @Test
    public void testDupLocality3() throws Exception {
        INDArray array1 = Nd4j.create(new float[]{1f, 1f, 1f, 1f, 1f});
        INDArray exp1 = Nd4j.create(new float[]{0f, 1f, 1f, 1f, 1f});
        INDArray exp2 = Nd4j.create(new float[]{1f, 1f, 1f, 1f, 1f});
        INDArray array2 = array1.dup();


        AllocationPoint point1 = AtomicAllocator.getInstance().getAllocationPoint(array1);
        AllocationPoint point2 = AtomicAllocator.getInstance().getAllocationPoint(array2);
        assertTrue(point1.isActualOnDeviceSide());
        assertTrue(point2.isActualOnDeviceSide());

        assertTrue(point1.isEnqueued());
        assertTrue(point2.isEnqueued());

        array1.putScalar(0, 0f);

        assertEquals(0, ((GridExecutioner)Nd4j.getExecutioner()).getQueueLength());

        assertFalse(point1.isActualOnDeviceSide());

        assertEquals(exp1, array1);
        assertEquals(exp2, array2);
    }

    @Test
    public void testDupLocality4() throws Exception {
        int nIn = 8;
        int layerSize = 10;
        int nOut = 4;

        INDArray in = Nd4j.ones(1,10).dup('c');
        AllocationPoint point1 = AtomicAllocator.getInstance().getAllocationPoint(in);
        assertEquals(true, point1.isEnqueued());

        //assertEquals(1, ((GridExecutioner) Nd4j.getExecutioner()).getQueueLength());

        INDArray out = Nd4j.zeros(1, 10).dup('c');
        AllocationPoint point1A = AtomicAllocator.getInstance().getAllocationPoint(in);
        AllocationPoint point2 = AtomicAllocator.getInstance().getAllocationPoint(out);
        assertEquals(1, ((GridExecutioner) Nd4j.getExecutioner()).getQueueLength());
        assertTrue(point1 == point1A);
        assertEquals(true, point2.isEnqueued());
        assertEquals(false, point1.isEnqueued());


        assertEquals(Nd4j.ones(1, 10), in);
        assertEquals(Nd4j.zeros(1, 10), out);

        INDArray inCopy = in.dup('c');
        AllocationPoint point3 = AtomicAllocator.getInstance().getAllocationPoint(inCopy);

        assertEquals(false, point2.isEnqueued());
        assertEquals(true, point3.isEnqueued());
        assertEquals(true, point1.isEnqueued());
    }
}
