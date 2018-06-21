package org.nd4j.linalg.api.ops.impl.accum.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * Backprop op for Min reduction operation
 *
 * @author Alex Black
 */

public class MinBp extends BaseReductionBp {

    public MinBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean keepDims, int... dimensions) {
        super(sameDiff, origInput, gradAtOutput, keepDims, dimensions);
    }

    public MinBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean keepDims, int... dimensions){
        super(origInput, gradAtOutput, output, keepDims, dimensions);
    }

    public MinBp(){}

    @Override
    public String opName() {
        return "reduce_min_bp";
    }
}
