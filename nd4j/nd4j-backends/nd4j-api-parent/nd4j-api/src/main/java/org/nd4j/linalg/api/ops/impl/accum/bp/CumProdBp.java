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

    public CumProdBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean exclusive, boolean reverse, int... axis) {
        super(sameDiff, origInput, gradAtOutput, false, axis);
        this.exclusive = exclusive;
        this.reverse = reverse;

        iArguments.clear();
        tArguments.clear();
        addArgs();
    }

    public CumProdBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean exclusive, boolean reverse, int... axis){
        super(origInput, gradAtOutput, output, false, axis);
        this.exclusive = exclusive;
        this.reverse = reverse;

        iArguments.clear();
        tArguments.clear();
        addArgs();
    }

    public CumProdBp(){}

    @Override
    public int getNumOutputs() {
        if (args().length == 2)
            return 1;
        else
            return 2;
    }


    @Override
    protected void addArgs(){
        addIArgument(exclusive ? 1 : 0);
        addIArgument(reverse ? 1 : 0);
        if(dimensions != null && dimensions.length > 0){
            addIArgument(dimensions);
        }
    }

    @Override
    public String opName() {
        return "cumprod_bp";
    }
}
