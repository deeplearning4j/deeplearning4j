package org.nd4j.linalg.api.ops.impl.accum.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;


/**
 * Backprop op for cumulative sum operation
 *
 * @author Alex Black
 */

public class CumSumBp extends BaseReductionBp {

    private boolean exclusive;
    private boolean reverse;

    public CumSumBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean exclusive, boolean reverse, int... dimensions) {
        super(sameDiff, origInput, gradAtOutput, false, dimensions);
        this.exclusive = exclusive;
        this.reverse = reverse;
    }

    public CumSumBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean exclusive, boolean reverse, int... dimensions){
        super(origInput, gradAtOutput, output, false, dimensions);
        this.exclusive = exclusive;
        this.reverse = reverse;
    }

    public CumSumBp(){}

    @Override
    protected void addArgs(){
        addTArgument(exclusive ? 1.0 : 0.0);
        addTArgument(reverse ? 1.0 : 0.0);
        if(dimensions != null && dimensions.length > 0){
            addIArgument(dimensions);
        }
    }

    @Override
    public String opName() {
        return "cumsum_bp";
    }
}
