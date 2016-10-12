package org.nd4j.linalg.api.ops.aggregates;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseAggregate implements Aggregate {
    protected List<INDArray> arguments;

    public List<INDArray> getArguments() {
        return arguments;
    }
}
