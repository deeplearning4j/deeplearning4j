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
    }

    @Override
    public String name() {
        return "aggregate_dot";
    }

    @Override
    public int opNum() {
        return 1;
    }
}
