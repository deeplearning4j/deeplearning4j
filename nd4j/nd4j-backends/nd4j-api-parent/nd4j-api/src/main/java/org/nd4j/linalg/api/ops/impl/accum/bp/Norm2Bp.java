package org.nd4j.linalg.api.ops.impl.accum.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * Backprop op for Norm2 reduction operation
 *
 * @author Alex Black
 */

public class Norm2Bp extends BaseReductionBp {

    public Norm2Bp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean keepDims, int... dimensions) {
        super(sameDiff, origInput, gradAtOutput, keepDims, dimensions);
    }

    public Norm2Bp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean keepDims, int... dimensions){
        super(origInput, gradAtOutput, output, keepDims, dimensions);
    }

    public Norm2Bp(){}

    @Override
    public String opName() {
        return "reduce_norm2_bp";
    }
}
