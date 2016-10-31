package org.nd4j.linalg.api.ops.random.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;

/**
 * This Op generates binomial distribution
 *
 * @author raver119@gmail.com
 */
public class BinomialDistribution extends BaseRandomOp {
    private int trials;
    private double probability;

    public BinomialDistribution(){
        super();
    }

    /**
     * This op fills Z with random values within stddev..mean..stddev boundaries
     * @param z
     * @param trials
     * @param probability
     */
    public BinomialDistribution(INDArray z, int trials, double probability) {
        init(z, z, z, z.length());
        this.trials = trials;
        this.probability = probability;
        this.extraArgs = new Object[]{(double) this.trials, this.probability};
    }


    @Override
    public int opNum() {
        return 8;
    }

    @Override
    public String name() {
        return "distribution_binomial";
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }
}
