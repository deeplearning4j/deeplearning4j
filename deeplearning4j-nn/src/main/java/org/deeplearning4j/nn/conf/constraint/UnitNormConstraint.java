package org.deeplearning4j.nn.conf.constraint;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;

import java.util.Collections;
import java.util.Set;

/**
 * Constrain the L2 norm of the incoming weights for each unit to be 1.0
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class UnitNormConstraint extends BaseConstraint {

    private UnitNormConstraint(){
        //No arg for json ser/de
    }

    /**
     * Apply to weights but not biases by default
     *
     * @param dimensions     Dimensions to apply to. For DenseLayer, OutputLayer, RnnOutputLayer, LSTM, etc: this should
     *                       be dimension 1. For CNNs, this should be dimensions [1,2,3] corresponding to last 3 of
     *                       parameters which have order [depthOut, depthIn, kH, kW]
     */
    public UnitNormConstraint(int... dimensions){
        this(Collections.<String>emptySet(), dimensions);
    }


    /**
     * @param dimensions     Dimensions to apply to. For DenseLayer, OutputLayer, RnnOutputLayer, LSTM, etc: this should
     *                       be dimension 1. For CNNs, this should be dimensions [1,2,3] corresponding to last 3 of
     *                       parameters which have order [depthOut, depthIn, kH, kW]
     */
    public UnitNormConstraint(Set<String> paramNames, int... dimensions){
        super(paramNames, dimensions);
    }

    @Override
    public void apply(INDArray param) {
        INDArray norm2 = param.norm2(dimensions);
        Broadcast.div(param, norm2, param, getBroadcastDims(dimensions, param.rank()) );
    }

    @Override
    public UnitNormConstraint clone() {
        return new UnitNormConstraint( params, dimensions);
    }
}
