package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;


/**
 * Norm 2 reduction operation
 *
 * @author Alex Black
 */

public class ReduceNorm2 extends BaseReduction {
    public ReduceNorm2(SameDiff sameDiff, SDVariable input, boolean keepDims, int... dimensions) {
        super(sameDiff, input, keepDims, dimensions);
    }

    public ReduceNorm2(INDArray input, INDArray output, boolean keepDims, int... dimensions){
        super(input, output, keepDims, dimensions);
    }

    public ReduceNorm2(){}

    @Override
    public String opName() {
        return "reduce_norm2";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        return Collections.singletonList(f().norm2Bp(arg(), grad.get(0), keepDims, dimensions));
    }
}
