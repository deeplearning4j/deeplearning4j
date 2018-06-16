package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.bp.BaseReductionBp;

import java.util.Collections;
import java.util.List;


/**
 * Mean reduction operation
 *
 * @author Alex Black
 */

public class ReduceMean extends BaseReduction {
    public ReduceMean(SameDiff sameDiff, SDVariable input, boolean keepDims, int... dimensions) {
        super(sameDiff, input, keepDims, dimensions);
    }

    public ReduceMean(INDArray input, INDArray output, boolean keepDims, int... dimensions){
        super(input, output, keepDims, dimensions);
    }

    public ReduceMean(){}

    @Override
    public String opName() {
        return "reduce_mean";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        return Collections.singletonList(f().meanBp(arg(), grad.get(0), keepDims, dimensions));
    }
}
