package org.nd4j.linalg.api.ops.aggregates.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;

/**
 * This op describes Dot call that'll happen soon(TM) in batch mode
 *
 * @author raver119@gmail.com
 */
public class AggregateDot extends BaseAggregate {

    public AggregateDot(INDArray x, INDArray y) {
        this.arguments.add(x);
        this.arguments.add(y);

        this.indexingArguments.add(x.length());
    }

    @Override
    public String name() {
        return "aggregate_dot";
    }

    @Override
    public int opNum() {
        return 1;
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
        return 1;
    }

    @Override
    public int maxRealArguments() {
        return 0;
    }
}
