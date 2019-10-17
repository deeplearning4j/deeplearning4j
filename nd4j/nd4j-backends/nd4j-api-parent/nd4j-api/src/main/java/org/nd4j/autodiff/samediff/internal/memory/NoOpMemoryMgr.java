package org.nd4j.autodiff.samediff.internal.memory;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.internal.SessionMemMrg;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

public class NoOpMemoryMgr implements SessionMemMrg {

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