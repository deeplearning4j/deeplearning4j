package org.nd4j.linalg.jcublas.ops.executioner;

import org.bytedeco.javacpp.Pointer;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.accum.Sum;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMultiplication;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class GridExecutionerTest {
    @Before
    public void setUp() throws Exception {

    }
///////////////////////////////////////////////////////////////////////////
/*/////////////////////////////////////////////////////////////////////////

    MatchMeta tests are checking, how ops are matching for MetaOp requirements

*//////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

    @Test
    public void isMatchingMetaOp1() throws Exception {
        GridExecutioner executioner = new GridExecutioner();

        INDArray array = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        ScalarAdd opB = new ScalarAdd(array, 10f);

        executioner.exec(opA);
        assertTrue(executioner.isMatchingMetaOp(opB));
    }

    @Test
    public void isMatchingMetaOp2() throws Exception {
        GridExecutioner executioner = new GridExecutioner();

        INDArray array = Nd4j.create(10);
        INDArray array2 = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        ScalarAdd opB = new ScalarAdd(array2, 10f);

        executioner.exec(opA);
        assertFalse(executioner.isMatchingMetaOp(opB));
    }

    @Test
    public void isMatchingMetaOp3() throws Exception {
        GridExecutioner executioner = new GridExecutioner();

        INDArray array = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        Max opB = new Max(array);

        executioner.exec(opA);
        assertFalse(executioner.isMatchingMetaOp(opB));
    }

///////////////////////////////////////////////////////////////////////////
/*/////////////////////////////////////////////////////////////////////////

    GridFlow tests are checking how ops are getting queued upon exec() calls

*//////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

    @Test
    public void testGridFlow1() throws Exception {
        GridExecutioner executioner = new GridExecutioner();

        assertEquals(0, executioner.getQueueLength());

        INDArray array = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        executioner.exec(opA);

        long time1 = System.nanoTime();

        Max opB = new Max(array);

        executioner.exec(opB);

        assertEquals(1, executioner.getQueueLength());

        long time2 = System.nanoTime();

        opB = new Max(array);

        executioner.exec(opB);

        long time3 = System.nanoTime();

        assertEquals(2, executioner.getQueueLength());



        long firstExec = time2 - time1;
        long secondExec = time3 - time2;

        System.out.println("First exec time: " + firstExec);
        System.out.println("Second exec time: " + secondExec);

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
        GridExecutioner executioner = new GridExecutioner();

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
        GridExecutioner executioner = new GridExecutioner();

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
        GridExecutioner executioner = new GridExecutioner();

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
}