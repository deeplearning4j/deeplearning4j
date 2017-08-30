package org.deeplearning4j.nn.conf.constraint;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

@Data
@EqualsAndHashCode(callSuper = true)
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

    @Override
    public MinMaxNormConstraint clone() {
        return new MinMaxNormConstraint(min, max, applyToWeights, applyToBiases, dimensions);
    }
}
