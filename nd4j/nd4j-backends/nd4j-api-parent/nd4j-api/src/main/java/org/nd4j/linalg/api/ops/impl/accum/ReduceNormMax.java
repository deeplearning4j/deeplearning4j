package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;


/**
 * Norm Max reduction operation
 *
 * @author Alex Black
 */

public class ReduceNormMax extends BaseReduction {
    public ReduceNormMax(SameDiff sameDiff, SDVariable input, boolean keepDims, int... dimensions) {
        super(sameDiff, input, keepDims, dimensions);
    }

    public ReduceNormMax(INDArray input, INDArray output, boolean keepDims, int... dimensions){
        super(input, output, keepDims, dimensions);
    }

    public ReduceNormMax(){}

    @Override
    public String opName() {
        return "reduce_norm_max";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        return Collections.singletonList(f().normmaxBp(arg(), grad.get(0), keepDims, dimensions));
    }
}
