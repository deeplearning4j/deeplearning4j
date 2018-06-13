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

    public CumProdBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean exclusive, boolean reverse, int... dimensions) {
        super(sameDiff, origInput, gradAtOutput, false, dimensions);
        this.exclusive = exclusive;
        this.reverse = reverse;
    }

    public CumProdBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean exclusive, boolean reverse, int... dimensions){
        super(origInput, gradAtOutput, output, false, dimensions);
        this.exclusive = exclusive;
        this.reverse = reverse;
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
        return "reduce_cum_prod_bp";
    }
}
