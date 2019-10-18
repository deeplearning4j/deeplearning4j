package org.nd4j.autodiff.samediff.internal.memory;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A simple "no-op" memory manager that relies on JVM garbage collector for memory management.
 * Assuming other references have been cleared (they should have been) the arrays will be cleaned up by the
 * garbage collector at some point.
 *
 * This memory management strategy is not recommended for performance or memory reasons, and should only be used
 * for testing and debugging purposes
 *
 * @author Alex Black
 */
public class NoOpMemoryMgr implements SessionMemMgr {

    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        return Nd4j.createUninitialized(dataType, shape);
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        return Nd4j.create(descriptor, false);
    }

    @Override
    public void release(@NonNull INDArray array) {
        //No-op, rely on GC to clear arrays
    }

    @Override
    public void close() {
        //No-op
    }

}