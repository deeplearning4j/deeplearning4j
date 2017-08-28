package org.deeplearning4j.nn.conf.constraint;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

public class MaxNormConstraint extends BaseConstraint {

    private double maxNorm;

    public MaxNormConstraint(double maxNorm, int... dimensions) {
        this(maxNorm, true, false, dimensions);
    }

    public MaxNormConstraint(double maxNorm, boolean applyToWeights, boolean applyToBiases, int... dimensions){
        super(applyToWeights, applyToBiases, DEFAULT_EPSILON, dimensions);
        this.maxNorm = maxNorm;
    }


    @Override
    public void apply(INDArray param, boolean isBias){
        INDArray norm = param.norm2(dimensions);
        INDArray clipped = norm.unsafeDuplication();
        BooleanIndexing.replaceWhere(clipped, maxNorm, Conditions.greaterThan(maxNorm));
        norm.addi(epsilon);

        clipped.divi(norm);

        Broadcast.mul(param, clipped, param, dimensions );
    }
}
