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

    public CumSumBp(SameDiff sameDiff, SDVariable origInput, SDVariable axis, SDVariable gradAtOutput, boolean exclusive, boolean reverse) {
        super(sameDiff, origInput, axis, gradAtOutput, false);
        this.exclusive = exclusive;
        this.reverse = reverse;
        addArgs();
    }

    public CumSumBp(INDArray origInput, INDArray axis, INDArray gradAtOutput, INDArray output, boolean exclusive, boolean reverse){
        super(origInput, gradAtOutput, output, false);
        this.exclusive = exclusive;
        this.reverse = reverse;
        addArgs();
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
