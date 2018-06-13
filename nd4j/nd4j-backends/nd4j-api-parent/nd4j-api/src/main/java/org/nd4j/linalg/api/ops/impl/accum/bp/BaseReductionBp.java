package org.nd4j.linalg.api.ops.impl.accum.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;


/**
 * @author Alex Black
 */

public abstract class BaseReductionBp extends DynamicCustomOp {

    protected boolean keepDims;
    protected int[] dimensions;

    /**
     *
     * @param origInput    Pre-reduced input
     * @param gradAtOutput Gradient at the output
     * @param keepDims     If true: reduction dimensions were kept
     * @param dimensions   Dimensions to reduce. May be null
     */
    public BaseReductionBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean keepDims, int... dimensions) {
        super(null, sameDiff, new SDVariable[]{origInput, gradAtOutput}, false);
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        addArgs();
    }

    /**
     *
     * @param origInput    Pre-reduced input
     * @param gradAtOutput Gradient at the output
     * @param output       Output array - i.e., gradient at the input to the reduction function
     * @param keepDims     If true: reduction dimensions were kept
     * @param dimensions   Dimensions to reduce. May be null
     */
    public BaseReductionBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean keepDims, int... dimensions){
        super(null, new INDArray[]{origInput, gradAtOutput}, (output == null ? null : new INDArray[]{output}));
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        addArgs();
    }

    public BaseReductionBp(){}

    protected void addArgs(){
        addTArgument(keepDims ? 1 : 0);
        if(dimensions != null && dimensions.length > 0){
            if(dimensions.length != 1 || dimensions[0] != Integer.MAX_VALUE ){
                //Integer.MAX_VALUE means "full array" but here no dimension args == full array
                addIArgument(dimensions);
            }
        }
    }

    public abstract String opName();

}
