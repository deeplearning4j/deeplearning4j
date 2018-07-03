package org.nd4j.linalg.api.ops.impl.accum.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * Backprop op for squared norm (sum_i x_i^2) reduction operation
 *
 * @author Alex Black
 */

public class SquaredNormBp extends BaseReductionBp {

    public SquaredNormBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean keepDims, int... dimensions) {
        super(sameDiff, origInput, gradAtOutput, keepDims, dimensions);
    }

    public SquaredNormBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean keepDims, int... dimensions){
        super(origInput, gradAtOutput, output, keepDims, dimensions);
    }

    public SquaredNormBp(){}

    @Override
    public String opName() {
        return "reduce_sqnorm_bp";
    }
}
