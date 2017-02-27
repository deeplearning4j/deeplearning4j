package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
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

    public ProbablisticMerge(@NonNull INDArray x, @NonNull INDArray y, @NonNull INDArray z, double probability) {
        init(x, y, z, x.lengthLong());
        this.probability = probability;
        this.extraArgs = new Object[] {probability};
    }

    public ProbablisticMerge(@NonNull INDArray x, @NonNull INDArray y, double probability) {
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
