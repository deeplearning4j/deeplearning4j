package org.deeplearning4j.nn.conf.constraint;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;

@Data
@EqualsAndHashCode(callSuper = true)
public class UnitNormConstraint extends BaseConstraint{

    public UnitNormConstraint(int... dimensions){
        this(true, false, dimensions);
    }

    public UnitNormConstraint(boolean applyToWeights, boolean applyToBiases, int... dimensions){
        super(applyToWeights, applyToBiases, dimensions);
    }

    @Override
    public void apply(INDArray param, boolean isBias) {
        INDArray norm2 = param.norm2(dimensions);
        Broadcast.div(param, norm2, param, dimensions );
    }

    @Override
    public UnitNormConstraint clone() {
        return new UnitNormConstraint(applyToWeights, applyToBiases, dimensions);
    }
}
