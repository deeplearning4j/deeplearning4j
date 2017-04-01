package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;

/**
 * BernoulliDistribution implementation
 *
 * @author raver119@gmail.com
 */
public class BernoulliDistribution extends BaseRandomOp {
    private double prob;

    public BernoulliDistribution() {
        super();
    }

    /**
     * This op fills Z with random values within from...to boundaries
     * @param z
    
     */
    public BernoulliDistribution(@NonNull INDArray z, double prob) {
        init(null, null, z, z.lengthLong());
        this.prob = prob;
        this.extraArgs = new Object[] {this.prob};
    }


    @Override
    public int opNum() {
        return 7;
    }

    @Override
    public String name() {
        return "distribution_bernoulli";
    }
}
