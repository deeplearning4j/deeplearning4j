package org.nd4j.linalg.api.ops.impl.accum.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * Backprop op for variance reduction operation
 *
 * @author Alex Black
 */

public class VarianceBp extends BaseReductionBp {

    private boolean biasCorrected;

    public VarianceBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean biasCorrected, boolean keepDims, int... dimensions) {
        super(sameDiff, origInput, gradAtOutput, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        addTArgument(biasCorrected ? 1.0 : 0.0);
    }

    public VarianceBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean biasCorrected, boolean keepDims, int... dimensions){
        super(origInput, gradAtOutput, output, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        addTArgument(biasCorrected ? 1.0 : 0.0);
    }

    public VarianceBp(){}

    @Override
    public String opName() {
        return "reduce_variance_bp";
    }
}
