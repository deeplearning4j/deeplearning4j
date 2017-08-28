package org.deeplearning4j.nn.conf.constraint;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

public class MinMaxNormConstraint extends BaseConstraint {

    private double min;
    private double max;

    public MinMaxNormConstraint(double min, double max, int... dimensions){
        this(min, max, true, false, dimensions);
    }

    public MinMaxNormConstraint(double min, double max, boolean applyToWeights, boolean applyToBiases, int... dimensions){
        super(applyToWeights, applyToBiases, dimensions);
        this.min = min;
        this.max = max;
    }

    @Override
    public void apply(INDArray param, boolean isBias) {
        BooleanIndexing.replaceWhere(param, min, Conditions.lessThan(min));
        BooleanIndexing.replaceWhere(param, max, Conditions.lessThan(max));
    }
}
