package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
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

    public BinomialDistribution() {
        super();
    }

    /**
     * This op fills Z with binomial distribution over given trials with single given probability for all trials
     * @param z
     * @param trials
     * @param probability
     */
    public BinomialDistribution(@NonNull INDArray z, int trials, double probability) {
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
    public BinomialDistribution(@NonNull INDArray z, int trials, @NonNull INDArray probabilities) {
        if (trials > probabilities.lengthLong())
            throw new IllegalStateException("Number of trials is > then amount of probabilities provided");

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
    public BinomialDistribution(@NonNull INDArray z, @NonNull INDArray probabilities) {
        this(z, probabilities.length(), probabilities);
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
