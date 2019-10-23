package org.nd4j.autodiff.samediff.internal.memory;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A simple memory management strategy that deallocates memory as soon as it is no longer needed.<br>
 * This should result in a minimal amount of memory, but will have some overhead - notably, the cost of deallocating
 * and reallocating memory all the time.
 *
 * @author Alex Black
 */
@Slf4j
public class ArrayCloseMemoryMgr extends AbstractMemoryMgr implements SessionMemMgr {

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
        if (!array.wasClosed() && array.closeable()) {
            array.close();
            log.trace("Closed array (deallocated) - id={}", array.getId());
        }
    }

    @Override
    public void close() {
        //No-op
    }
}
