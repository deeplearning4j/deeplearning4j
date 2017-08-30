package org.deeplearning4j.nn.conf.constraint;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

@Data
@EqualsAndHashCode(callSuper = true)
public class MinMaxNormConstraint extends BaseConstraint {
    public static final double DEFAULT_RATE = 1.0;

    private double min;
    private double max;
    private double rate;

    public MinMaxNormConstraint(double min, double max, int... dimensions){
        this(min, max, DEFAULT_RATE, true, false, dimensions);
    }

    public MinMaxNormConstraint(double min, double max, double rate, boolean applyToWeights, boolean applyToBiases, int... dimensions){
        super(applyToWeights, applyToBiases, dimensions);
        if(rate <= 0 || rate > 1.0){
            throw new IllegalStateException("Invalid rate: must be in interval (0,1]: got " + rate);
        }
        this.min = min;
        this.max = max;
        this.rate = rate;
    }

    @Override
    public void apply(INDArray param, boolean isBias) {
        INDArray norm = param.norm2(dimensions);
        INDArray clipped = norm.unsafeDuplication();
        BooleanIndexing.replaceWhere(clipped, max, Conditions.greaterThan(max));
        BooleanIndexing.replaceWhere(clipped, min, Conditions.lessThan(min));

        norm.addi(epsilon);
        clipped.divi(norm);

        if(rate != 1.0){
            clipped.muli(rate).addi(norm.muli(1.0-rate));
        }

        Broadcast.mul(param, clipped, param, getBroadcastDims(dimensions, param.rank()) );
    }

    @Override
    public MinMaxNormConstraint clone() {
        return new MinMaxNormConstraint(min, max, rate, applyToWeights, applyToBiases, dimensions);
    }
}
