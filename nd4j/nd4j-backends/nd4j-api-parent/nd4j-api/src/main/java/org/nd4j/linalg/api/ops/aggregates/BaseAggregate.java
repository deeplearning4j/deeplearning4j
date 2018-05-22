package org.nd4j.linalg.api.ops.aggregates;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public abstract class   BaseAggregate implements Aggregate {
    protected List<INDArray> arguments = new ArrayList<>();
    protected List<DataBuffer> shapes = new ArrayList<>();
    protected List<int[]> intArrayArguments = new ArrayList<>();
    protected List<Integer> indexingArguments = new ArrayList<>();
    protected List<Number> realArguments = new ArrayList<>();

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
    public List<Number> getRealArguments() {
        return realArguments;
    }

    @Override
    public List<int[]> getIntArrayArguments() {
        return intArrayArguments;
    }

    @Override
    public long getRequiredBatchMemorySize() {
        long result = maxIntArrays() * maxIntArraySize() * 4;
        result += maxArguments() * 8; // pointers
        result += maxShapes() * 8; // pointers
        result += maxIndexArguments() * 4;
        result += maxRealArguments() * (Nd4j.dataType() == DataBuffer.Type.DOUBLE ? 8
                        : Nd4j.dataType() == DataBuffer.Type.FLOAT ? 4 : 2);
        result += 5 * 4; // numArgs

        return result * Batch.getBatchLimit();
    }
}
