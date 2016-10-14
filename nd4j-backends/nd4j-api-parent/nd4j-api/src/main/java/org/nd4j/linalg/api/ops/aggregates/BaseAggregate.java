package org.nd4j.linalg.api.ops.aggregates;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseAggregate implements Aggregate {
    protected List<INDArray> arguments = new ArrayList<>();
    protected List<DataBuffer> shapes = new ArrayList<>();
    protected List<Integer> indexingArguments = new ArrayList<>();
    protected List<Double> realArguments = new ArrayList<>();


    protected Number finalResult = 0.0;

    public List<INDArray> getArguments() {
        return arguments;
    }

    @Override
    public Number getFinalResult() {
        return finalResult;
    }

    @Override
    public void setFinalResult(Number result) {
        this.finalResult = result;
    }

    @Override
    public List<DataBuffer> getShapes() {
        return shapes;
    }

    @Override
    public List<Integer> getIndexingArguments() {
        return indexingArguments;
    }

    @Override
    public List<Double> getRealArguments() {
        return realArguments;
    }
}
