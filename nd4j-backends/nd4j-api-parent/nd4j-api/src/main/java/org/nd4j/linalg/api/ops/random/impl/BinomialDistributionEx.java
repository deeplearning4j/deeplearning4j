package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;

/**
 * This Op generates binomial distribution
 *
 * @author raver119@gmail.com
 */
public class BinomialDistributionEx extends BaseRandomOp {
    private int trials;
    private double probability;

    public BinomialDistributionEx() {
        super();
    }

    /**
     * This op fills Z with binomial distribution over given trials with single given probability for all trials
     * @param z
     * @param trials
     * @param probability
     */
    public BinomialDistributionEx(@NonNull INDArray z, int trials, double probability) {
        init(z, z, z, z.lengthLong());
        this.trials = trials;
        this.probability = probability;
        this.extraArgs = new Object[] {(double) this.trials, this.probability};
    }

    /**
     * This op fills Z with binomial distribution over given trials with probability for each trial given as probabilities INDArray
     * @param z
     * @param trials
     * @param probabilities array with probability value for each trial
     */
    public BinomialDistributionEx(@NonNull INDArray z, int trials, @NonNull INDArray probabilities) {
        if (z.lengthLong() != probabilities.lengthLong())
            throw new IllegalStateException("Length of probabilities array should match length of target array");

        if (probabilities.elementWiseStride() < 1)
            throw new IllegalStateException("Probabilities array shouldn't have negative elementWiseStride");

        init(z, probabilities, z, z.lengthLong());

        this.trials = trials;
        this.probability = 0.0;
        this.extraArgs = new Object[] {(double) this.trials, this.probability};
    }


    /**
     * This op fills Z with binomial distribution over given trials with probability for each trial given as probabilities INDArray
     *
     * @param z
     * @param probabilities
     */
    public BinomialDistributionEx(@NonNull INDArray z, @NonNull INDArray probabilities) {
        this(z, probabilities.length(), probabilities);
    }

    @Override
    public int opNum() {
        return 9;
    }

    @Override
    public String name() {
        return "distribution_binomial_ex";
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }
}
