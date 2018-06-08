package org.nd4j.linalg.api.ops.impl.accum.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;


/**
 * @author Alex Black
 */

public abstract class BaseReductionBp extends DynamicCustomOp {

    private boolean keepDims;
    private int[] dimensions;

    public BaseReductionBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean keepDims, int... dimensions) {
        super(null, sameDiff, new SDVariable[]{origInput, gradAtOutput}, false);
        addArgs();
    }

    public BaseReductionBp(){}

    protected void addArgs(){
        addIArgument(keepDims ? 1 : 0);
        if(dimensions != null && dimensions.length > 0){
            addIArgument(dimensions);
        }
    }

}
