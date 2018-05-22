package org.deeplearning4j.nn.conf.constraint;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.Collections;
import java.util.Set;

/**
 * Constrain the maximum L2 norm of the incoming weights for each unit to be less than or equal to the specified value.
 * If the L2 norm exceeds the specified value, the weights will be scaled down to satisfy the constraint.
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class MaxNormConstraint extends BaseConstraint {

    private double maxNorm;

    private MaxNormConstraint(){
        //No arg for json ser/de
    }

    /**
     * @param maxNorm        Maximum L2 value
     * @param paramNames     Which parameter names to apply constraint to
     * @param dimensions     Dimensions to apply to. For DenseLayer, OutputLayer, RnnOutputLayer, LSTM, etc: this should
     *                       be dimension 1. For CNNs, this should be dimensions [1,2,3] corresponding to last 3 of
     *                       parameters which have order [depthOut, depthIn, kH, kW]
     */
    public MaxNormConstraint(double maxNorm, Set<String> paramNames, int... dimensions){
        super(paramNames, DEFAULT_EPSILON, dimensions);
        this.maxNorm = maxNorm;
    }

    /**
     * Apply to weights but not biases by default
     *
     * @param maxNorm        Maximum L2 value
     * @param dimensions     Dimensions to apply to. For DenseLayer, OutputLayer, RnnOutputLayer, LSTM, etc: this should
     *                       be dimension 1. For CNNs, this should be dimensions [1,2,3] corresponding to last 3 of
     *                       parameters which have order [depthOut, depthIn, kH, kW]
     */
    public MaxNormConstraint(double maxNorm, int... dimensions) {

        this(maxNorm, Collections.<String>emptySet(), dimensions);
    }

    @Override
    public void apply(INDArray param){
        INDArray norm = param.norm2(dimensions);
        INDArray clipped = norm.unsafeDuplication();
        BooleanIndexing.replaceWhere(clipped, maxNorm, Conditions.greaterThan(maxNorm));
        norm.addi(epsilon);
        clipped.divi(norm);

        Broadcast.mul(param, clipped, param, getBroadcastDims(dimensions, param.rank()) );
    }

    @Override
    public MaxNormConstraint clone() {
        return new MaxNormConstraint(maxNorm,  params, dimensions);
    }
}
