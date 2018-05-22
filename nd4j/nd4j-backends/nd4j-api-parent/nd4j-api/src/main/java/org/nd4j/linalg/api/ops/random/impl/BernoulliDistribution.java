package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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
     * This op fills Z with bernoulli trial results, so 0, or 1, depending by common probability
     * @param z
    
     */
    public BernoulliDistribution(@NonNull INDArray z, double prob) {
        init(null, null, z, z.lengthLong());
        this.prob = prob;
        this.extraArgs = new Object[] {this.prob};
    }

    /**
     * This op fills Z with bernoulli trial results, so 0, or 1, each element will have it's own success probability defined in prob array
     * @param prob array with probabilities
     * @param z
    
     */
    public BernoulliDistribution(@NonNull INDArray z, @NonNull INDArray prob) {
        if (prob.elementWiseStride() != 1)
            throw new ND4JIllegalStateException("Probabilities should have ElementWiseStride of 1");

        if (prob.lengthLong() != z.lengthLong())
            throw new ND4JIllegalStateException("Length of probabilities array [" + prob.lengthLong()
                            + "] doesn't match length of output array [" + z.lengthLong() + "]");

        init(prob, null, z, z.lengthLong());
        this.prob = 0.0;
        this.extraArgs = new Object[] {this.prob};
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("prob",prob);
        return ret;
    }



    @Override
    public int opNum() {
        return 7;
    }

    @Override
    public String opName() {
        return "distribution_bernoulli";
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
        return null;
    }
}
