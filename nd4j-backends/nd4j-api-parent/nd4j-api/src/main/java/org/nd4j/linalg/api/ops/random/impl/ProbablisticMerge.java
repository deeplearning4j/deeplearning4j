package org.nd4j.linalg.api.ops.random.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;

/**
 * @author raver119@gmail.com
 */
public class ProbablisticMerge extends BaseRandomOp {
    private double probability;

    public ProbablisticMerge() {
        super();
    }

    public ProbablisticMerge(INDArray x, INDArray y, INDArray z, double probability) {
        init(x, y, z, x.length());
        this.probability = probability;
        this.extraArgs = new Object[]{probability};
    }

    public ProbablisticMerge(INDArray x, INDArray y, double probability) {
        this(x, y, x, probability);
    }

    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public String name() {
        return "probablistic_merge";
    }
}
