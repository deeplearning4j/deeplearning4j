package org.nd4j.linalg.api.ops.aggregates.impl;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This op describes Axpy call
 * that'll happen soon(TM) in batch mode
 *
 * @author raver119
 */
public class AggregateAxpy extends BaseAggregate {
    private int vectorLength;

    public AggregateAxpy(@NonNull INDArray x, @NonNull INDArray y, double alpha) {
        this.arguments.add(x);
        this.arguments.add(y);

        this.indexingArguments.add((int) x.length());

        this.realArguments.add(alpha);
        this.vectorLength = (int) x.length();
    }

    /**
     * This method returns amount of shared memory required for this specific Aggregate.
     * PLEASE NOTE: this method is especially important for CUDA backend. On
     * CPU backend it might be ignored, depending on Aggregate.
     *
     * @return
     */
    @Override
    public int getSharedMemorySize() {
        return (getThreadsPerInstance() * Nd4j.sizeOfDataType()) + 256;
    }

    /**
     * This method returns desired number of threads per Aggregate instance
     * PLEASE NOTE: this method is especially important for
     * CUDA backend. On CPU backend it might be ignored,
     * depending on Aggregate.
     *
     * @return
     */
    @Override
    public int getThreadsPerInstance() {
        return Math.min(768, vectorLength);
    }

    @Override
    public String name() {
        return "aggregate_axpy";
    }

    @Override
    public int opNum() {
        return 2;
    }


    @Override
    public int maxArguments() {
        return 2;
    }

    @Override
    public int maxShapes() {
        return 0;
    }

    @Override
    public int maxIntArrays() {
        return 0;
    }

    @Override
    public int maxIntArraySize() {
        return 0;
    }

    @Override
    public int maxIndexArguments() {
        return 2;
    }

    @Override
    public int maxRealArguments() {
        return 2;
    }
}
