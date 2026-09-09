package org.eclipse.deeplearning4j.nd4j.linalg;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.LongPointer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Shape inference must synchronize shape values, never the reshape data payload. */
class CudaReshapeShapeSyncTest {
    private static final float STALE_HOST_SENTINEL = -1234.0f;

    @Test
    void integerArgumentReshapeDoesNotReadDataValues() {
        assumeTrue(Nd4j.getBackend().getClass().getSimpleName().contains("JCublas"));
        NativeOps ops = Nd4j.getNativeOps();
        // Include small/scalar inputs: the metadata contract overrides the old size heuristic.
        for (int length : new int[]{1, 8192}) {
            try (INDArray data = Nd4j.ones(DataType.FLOAT, 1, length)) {
                ops.dbForceSyncToPrimary(data.data().opaqueBuffer());
                ops.dbForceSyncToSpecial(data.data().opaqueBuffer());
                Nd4j.getExecutioner().commit();
                FloatPointer host = new FloatPointer(ops.dbPrimaryBuffer(data.data().opaqueBuffer()));
                // Deliberately stale host bytes without marking a host write: device data is authoritative.
                // Read this borrowed pointer directly, so the assertion itself cannot synchronize.
                host.put(0, STALE_HOST_SENTINEL);
                try {
                    ReshapeNoCopy op = new ReshapeNoCopy(data, new long[]{length}, null);
                    var shapes = Nd4j.getExecutioner().calculateOutputShape(op);
                    assertArrayEquals(new long[]{length}, Shape.shape(shapes.get(0).asLong()));
                    assertEquals(STALE_HOST_SENTINEL, host.get(0),
                            "reshape shape inference copied data input to host, length=" + length);
                } finally {
                    ops.dbForceSyncToPrimary(data.data().opaqueBuffer());
                }
                assertEquals(1.0f, host.get(0));
            }
        }
    }

    @Test
    void tensorShapeRefreshesDeviceWrittenDimensionsButNotData() {
        assumeTrue(Nd4j.getBackend().getClass().getSimpleName().contains("JCublas"));
        NativeOps ops = Nd4j.getNativeOps();
        try (INDArray data = Nd4j.ones(DataType.FLOAT, 64);
             INDArray dimensions = Nd4j.createFromArray(2L, 8L)) {
            ops.dbForceSyncToPrimary(data.data().opaqueBuffer());
            ops.dbForceSyncToSpecial(data.data().opaqueBuffer());
            dimensions.muli(2); // Device-written target [4,16].
            Nd4j.getExecutioner().commit();
            ops.dbAllocatePrimaryBuffer(dimensions.data().opaqueBuffer());
            FloatPointer dataHost = new FloatPointer(ops.dbPrimaryBuffer(data.data().opaqueBuffer()));
            LongPointer shapeHost = new LongPointer(ops.dbPrimaryBuffer(dimensions.data().opaqueBuffer()));
            dataHost.put(0, STALE_HOST_SENTINEL);
            shapeHost.put(0, -7L).put(1, -7L);
            try {
                DynamicCustomOp op = DynamicCustomOp.builder("reshape_no_copy")
                        .addInputs(data, dimensions)
                        .addIntegerArguments(ReshapeNoCopy.RESHAPE_NO_COPY_C_ORDER_MARKER).build();
                var shapes = Nd4j.getExecutioner().calculateOutputShape(op);
                assertArrayEquals(new long[]{4, 16}, Shape.shape(shapes.get(0).asLong()));
                assertEquals(4L, shapeHost.get(0));
                assertEquals(16L, shapeHost.get(1));
                assertEquals(STALE_HOST_SENTINEL, dataHost.get(0),
                        "only the shape tensor may be synchronized for shape inference");
            } finally {
                ops.dbForceSyncToPrimary(data.data().opaqueBuffer());
                ops.dbForceSyncToPrimary(dimensions.data().opaqueBuffer());
            }
        }
    }
}
