package org.nd4j.linalg.api.ops.impl.accum.bp;

import lombok.val;
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

    public CumSumBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean exclusive, boolean reverse, int... axis) {
        super(sameDiff, origInput, gradAtOutput, false, axis);
        this.exclusive = exclusive;
        this.reverse = reverse;

        iArguments.clear();
        tArguments.clear();
        addArgs();
    }

    public CumSumBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean exclusive, boolean reverse, int... axis){
        super(origInput, gradAtOutput, output, false, axis);
        this.exclusive = exclusive;
        this.reverse = reverse;

        iArguments.clear();
        tArguments.clear();
        addArgs();
    }

    @Override
    public int getNumOutputs() {
        if (args().length == 2)
            return 1;
        else
            return 2;
    }

    public CumSumBp(){}

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
        return "cumsum_bp";
    }
}
