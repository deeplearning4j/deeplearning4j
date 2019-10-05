package org.nd4j.autodiff.samediff.internal;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.io.Closeable;

public interface SessionMemMrg extends Closeable {

    INDArray allocate(boolean detached, DataType dataType, long... shape);

    INDArray allocate(boolean detached, LongShapeDescriptor descriptor);

    void release(INDArray array);

    void close();

}
