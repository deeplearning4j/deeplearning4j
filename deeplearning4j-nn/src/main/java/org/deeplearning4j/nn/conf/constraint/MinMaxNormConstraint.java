package org.deeplearning4j.nn.conf.constraint;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * Constrain the minimum AND maximum L2 norm of the incoming weights for each unit to be between the specified values.
 * If the L2 norm exceeds the specified max value, the weights will be scaled down to satisfy the constraint; if the
 * L2 norm is less than the specified min value, the weights will be scaled up<br>
 * Note that this constraint supports a rate parameter (default: 1.0, which is equivalent to a strict constraint).
 * If rate < 1.0, the applied norm2 constraint will be (1-rate)*norm2 + rate*clippedNorm2, where clippedNorm2 is the
 * norm2 value after applying clipping to min/max values.
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class MinMaxNormConstraint extends BaseConstraint {
    public static final double DEFAULT_RATE = 1.0;

    private double min;
    private double max;
    private double rate;

    /**
     * Apply to weights but not biases by default
     *
     * @param max            Maximum L2 value
     * @param min            Minimum L2 value
     * @param dimensions     Dimensions to apply to. For DenseLayer, OutputLayer, RnnOutputLayer, LSTM, etc: this should
     *                       be dimension 1. For CNNs, this should be dimensions [1,2,3] correspending to last 3 of
     *                       parameters which have order [depthOut, depthIn, kH, kW]
     */
    public MinMaxNormConstraint(double min, double max, int... dimensions){
        this(min, max, DEFAULT_RATE, true, false, dimensions);
    }

    /**
     *
     * @param max            Maximum L2 value
     * @param min            Minimum L2 value
     * @param applyToWeights If constraint should be applied to weights
     * @param applyToBiases  If constraint should be applied to biases
     * @param dimensions     Dimensions to apply to. For DenseLayer, OutputLayer, RnnOutputLayer, LSTM, etc: this should
     *                       be dimension 1. For CNNs, this should be dimensions [1,2,3] correspending to last 3 of
     *                       parameters which have order [depthOut, depthIn, kH, kW]
     */
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
