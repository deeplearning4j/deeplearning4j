package org.nd4j.linalg.api.ops.aggregates.impl;

import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;

/**
 * @author raver119@gmail.com
 */
@Deprecated
public abstract class NegativeSampling extends BaseAggregate {

    @Override
    public String name() {
        return "negative_sampling";
    }

    @Override
    public int opNum() {
        return -1;
    }


}
