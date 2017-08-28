package org.deeplearning4j.nn.conf.constraint;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;


public class UnitNormConstraint extends BaseConstraint{

    public UnitNormConstraint(int... dimensions){
        super(true, false, dimensions);
    }

    @Override
    public void apply(INDArray param, boolean isBias) {
        INDArray norm2 = param.norm2(dimensions);
        Broadcast.div(param, norm2, param, dimensions );
    }
}
