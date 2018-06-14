package org.nd4j.linalg.api.ops.impl.accum.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * Backprop op for Dot pairwise reduction operation
 *
 * @author Alex Black
 */

public class DotBp extends BaseReductionBp {

    public DotBp(SameDiff sameDiff, SDVariable origInput1, SDVariable origInput2, SDVariable gradAtOutput, boolean keepDims, int... dimensions) {
        super(sameDiff, origInput1, origInput2, gradAtOutput, keepDims, dimensions);
    }

    public DotBp(INDArray origInput1, INDArray origInput2, INDArray gradAtOutput, INDArray output, boolean keepDims, int... dimensions){
        super(origInput1, origInput2, gradAtOutput, output, keepDims, dimensions);
    }

    public DotBp(){}

    @Override
    public String opName() {
        return "reduce_dot_bp";
    }
}
