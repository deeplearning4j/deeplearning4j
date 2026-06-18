/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reproducer tests for shape cache issues in DynamicCustomOp.
 *
 * The shape cache in DynamicCustomOp stores DataBuffer objects returned by
 * calculateOutputShapes2 (native). These DataBuffers wrap native pointers
 * from a temporary ShapeList that gets freed after the call. On cache hit,
 * the stale pointer is read, returning garbage (e.g. expectedShape=[] for
 * a multiply of [1,1024,3072] inputs).
 */
public class ShapeCacheReproducerTest {

    /**
     * Core reproducer: call calculateOutputShape twice with the same inputs.
     * Second call hits cache - verify the cached shape is still valid.
     */
    @Test
    public void testShapeCacheReturnsSameShapeOnSecondCall() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 1, 1024, 3072);
        INDArray y = Nd4j.rand(DataType.FLOAT, 1, 1024, 3072);

        MulOp op = new MulOp();

        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArrays(x, y);

        // First call - computes shape natively
        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        assertNotNull(shapes1);
        assertEquals(1, shapes1.size());
        long[] shape1 = Shape.shape(shapes1.get(0).asLong());
        assertArrayEquals(new long[]{1, 1024, 3072}, shape1,
                "First calculateOutputShape should return [1,1024,3072]");

        // Second call with same inputs - should hit cache
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertNotNull(shapes2);
        assertEquals(1, shapes2.size());
        long[] shape2 = Shape.shape(shapes2.get(0).asLong());
        assertArrayEquals(new long[]{1, 1024, 3072}, shape2,
                "Cached calculateOutputShape should still return [1,1024,3072], not []");
    }

    /**
     * Test that shape cache invalidates when input shapes change.
     */
    @Test
    public void testShapeCacheInvalidatesOnShapeChange() {
        MulOp op = new MulOp();

        // First call with [2, 3]
        INDArray x1 = Nd4j.rand(DataType.FLOAT, 2, 3);
        INDArray y1 = Nd4j.rand(DataType.FLOAT, 2, 3);
        OpContext oc1 = Nd4j.getExecutioner().buildContext();
        oc1.setInputArrays(x1, y1);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc1);
        long[] shape1 = Shape.shape(shapes1.get(0).asLong());
        assertArrayEquals(new long[]{2, 3}, shape1);

        // Second call with [4, 5] - different shape, cache must invalidate
        INDArray x2 = Nd4j.rand(DataType.FLOAT, 4, 5);
        INDArray y2 = Nd4j.rand(DataType.FLOAT, 4, 5);
        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArrays(x2, y2);

        List<DataBuffer> shapes2 = op.calculateOutputShape(oc2);
        long[] shape2 = Shape.shape(shapes2.get(0).asLong());
        assertArrayEquals(new long[]{4, 5}, shape2,
                "Shape cache must invalidate when input shapes change");
    }

    /**
     * Test shape cache with broadcast: inputs [3,1] * [1,4] -> [3,4]
     */
    @Test
    public void testShapeCacheBroadcast() {
        MulOp op = new MulOp();

        INDArray x = Nd4j.rand(DataType.FLOAT, 3, 1);
        INDArray y = Nd4j.rand(DataType.FLOAT, 1, 4);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArrays(x, y);

        // First call
        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        long[] shape1 = Shape.shape(shapes1.get(0).asLong());
        assertArrayEquals(new long[]{3, 4}, shape1);

        // Second call - cache hit
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        long[] shape2 = Shape.shape(shapes2.get(0).asLong());
        assertArrayEquals(new long[]{3, 4}, shape2,
                "Cached broadcast shape should still be [3,4]");
    }

    /**
     * Stress test: repeated calculateOutputShape calls with GC pressure
     * to expose use-after-free of cached native pointers.
     */
    @Test
    public void testShapeCacheUnderGcPressure() {
        MulOp op = new MulOp();

        INDArray x = Nd4j.rand(DataType.FLOAT, 1, 1024, 3072);
        INDArray y = Nd4j.rand(DataType.FLOAT, 1, 1024, 3072);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArrays(x, y);

        // First call to populate cache
        List<DataBuffer> shapes = op.calculateOutputShape(oc);
        long[] expected = Shape.shape(shapes.get(0).asLong());
        assertArrayEquals(new long[]{1, 1024, 3072}, expected);

        // Allocate and discard lots of native memory to trigger reuse of freed addresses
        for (int i = 0; i < 50; i++) {
            INDArray temp = Nd4j.rand(DataType.FLOAT, 1, 1024, 3072);
            // Do a multiply to trigger native calculateOutputShapes2 calls
            // which allocate and free ShapeLists, potentially reusing the
            // address that our cached DataBuffer points to
            MulOp tempOp = new MulOp();
            OpContext tempOc = Nd4j.getExecutioner().buildContext();
            tempOc.setInputArrays(temp, temp);
            tempOp.calculateOutputShape(tempOc);
            temp.close();
        }

        // Force GC
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException e) {}

        // Now read the cached shape again - if it was a stale native pointer,
        // this will return garbage or crash
        List<DataBuffer> shapesAfterGc = op.calculateOutputShape(oc);
        long[] actual = Shape.shape(shapesAfterGc.get(0).asLong());
        assertArrayEquals(new long[]{1, 1024, 3072}, actual,
                "Shape cache should survive GC pressure - " +
                "cached DataBuffer must not point to freed native memory");
    }

    /**
     * Test that executing the multiply op end-to-end works correctly
     * when shape cache is involved (the actual VLM failure scenario).
     */
    @Test
    public void testMultiplyExecutionWithShapeCache() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 1, 1024, 3072);

        // First multiply: x * x (same variable used twice, like GELU activation)
        INDArray result1 = Nd4j.exec(new MulOp(x, x))[0];
        assertArrayEquals(new long[]{1, 1024, 3072}, result1.shape());

        // Second multiply with same shapes - exercises cache
        INDArray x2 = Nd4j.rand(DataType.FLOAT, 1, 1024, 3072);
        INDArray result2 = Nd4j.exec(new MulOp(x2, x2))[0];
        assertArrayEquals(new long[]{1, 1024, 3072}, result2.shape());

        // Verify values are correct (x*x should be non-negative)
        assertTrue(result2.minNumber().floatValue() >= 0.0f,
                "x*x should be non-negative");
    }
}
