package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;


/**
 * Squared norm (sum_i x_i^2) reduction operation
 *
 * @author Alex Black
 */

public class ReduceSquaredNorm extends BaseReduction {
    public ReduceSquaredNorm(SameDiff sameDiff, SDVariable input, boolean keepDims, int... dimensions) {
        super(sameDiff, input, keepDims, dimensions);
    }

    public ReduceSquaredNorm(INDArray input, INDArray output, boolean keepDims, int... dimensions){
        super(input, output, keepDims, dimensions);
    }

    public ReduceSquaredNorm(){}

    @Override
    public String opName() {
        return "reduce_sqnorm";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        return Collections.singletonList(f().squaredNormBp(arg(), grad.get(0), keepDims, dimensions));
    }
}
