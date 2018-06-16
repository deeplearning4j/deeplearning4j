package org.nd4j.linalg.api.ops.impl.accum.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * Backprop op for cumulative product operation
 *
 * @author Alex Black
 */

public class CumProdBp extends BaseReductionBp {

    private boolean exclusive;
    private boolean reverse;

    public CumProdBp(SameDiff sameDiff, SDVariable origInput, SDVariable axis, SDVariable gradAtOutput, boolean exclusive, boolean reverse) {
        super(sameDiff, origInput, gradAtOutput, false);
        this.exclusive = exclusive;
        this.reverse = reverse;
        addArgs();
    }

    public CumProdBp(INDArray origInput, INDArray axis, INDArray gradAtOutput, INDArray output, boolean exclusive, boolean reverse){
        super(origInput, gradAtOutput, output, false);
        this.exclusive = exclusive;
        this.reverse = reverse;
        addArgs();
    }

    public CumProdBp(){}

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
        return "cumprod_bp";
    }
}
