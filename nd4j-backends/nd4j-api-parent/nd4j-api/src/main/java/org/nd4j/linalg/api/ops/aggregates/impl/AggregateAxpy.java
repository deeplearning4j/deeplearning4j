package org.nd4j.linalg.api.ops.aggregates.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;

/**
 * This op describes Axpy call that'll happen soon(TM) in batch mode
 *
 * @author raver119
 */
public class AggregateAxpy extends BaseAggregate {

    public AggregateAxpy(INDArray x, INDArray y, double alpha) {
        this.arguments.add(x);
        this.arguments.add(y);

        this.realArguments.add(alpha);
    }

    @Override
    public String name() {
        return "aggregate_axpy";
    }

    @Override
    public int opNum() {
        return 2;
    }
}
