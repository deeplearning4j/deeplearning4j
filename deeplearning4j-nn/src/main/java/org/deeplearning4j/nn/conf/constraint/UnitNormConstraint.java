package org.deeplearning4j.nn.conf.constraint;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;

/**
 * Constrain the L2 norm of the incoming weights for each unit to be 1.0
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class UnitNormConstraint extends BaseConstraint{

    private UnitNormConstraint(){
        //No arg for json ser/de
    }

    /**
     * Apply to weights but not biases by default
     *
     * @param dimensions     Dimensions to apply to. For DenseLayer, OutputLayer, RnnOutputLayer, LSTM, etc: this should
     *                       be dimension 1. For CNNs, this should be dimensions [1,2,3] correspending to last 3 of
     *                       parameters which have order [depthOut, depthIn, kH, kW]
     */
    public UnitNormConstraint(int... dimensions){
        this(true, false, dimensions);
    }

    /**
     * Apply to weights but not biases by default
     *
     * @param applyToWeights If constraint should be applied to weights
     * @param applyToBiases  If constraint should be applied to biases (usually false)
     * @param dimensions     Dimensions to apply to. For DenseLayer, OutputLayer, RnnOutputLayer, LSTM, etc: this should
     *                       be dimension 1. For CNNs, this should be dimensions [1,2,3] correspending to last 3 of
     *                       parameters which have order [depthOut, depthIn, kH, kW]
     */
    public UnitNormConstraint(boolean applyToWeights, boolean applyToBiases, int... dimensions){
        super(applyToWeights, applyToBiases, dimensions);
    }

    @Override
    public void apply(INDArray param, boolean isBias) {
        INDArray norm2 = param.norm2(dimensions);
        Broadcast.div(param, norm2, param, getBroadcastDims(dimensions, param.rank()) );
    }

    @Override
    public UnitNormConstraint clone() {
        return new UnitNormConstraint(applyToWeights, applyToBiases, dimensions);
    }
}
