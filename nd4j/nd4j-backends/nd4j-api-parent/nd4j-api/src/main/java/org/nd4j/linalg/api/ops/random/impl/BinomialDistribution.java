package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * This Op generates binomial distribution
 *
 * @author raver119@gmail.com
 */
public class BinomialDistribution extends BaseRandomOp {
    private int trials;
    private double probability;

    public BinomialDistribution(SameDiff sd, int trials, double probability, long[] shape){
        super(sd, shape);
        this.trials = trials;
        this.probability = probability;
        this.extraArgs = new Object[] {(double) this.trials, this.probability};
    }

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


    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("trials",trials);
        ret.put("probability",probability);
        return ret;
    }


    /**
     * This op fills Z with binomial distribution over given trials with probability for each trial given as probabilities INDArray
     *
     * @param z
     * @param probabilities
     */
    public BinomialDistribution(@NonNull INDArray z, @NonNull INDArray probabilities) {
        this(z, (int) probabilities.length(), probabilities);
    }

    @Override
    public int opNum() {
        return 8;
    }

    @Override
    public String opName() {
        return "distribution_binomial";
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.emptyList();
    }
}
