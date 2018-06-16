package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;


/**
 * Sum reduction operation
 *
 * @author Alex Black
 */

public class ReduceStandardDeviation extends BaseReduction {

    private boolean biasCorrected;

    public ReduceStandardDeviation(SameDiff sameDiff, SDVariable input, boolean biasCorrected, boolean keepDims, int... dimensions) {
        super(sameDiff, input, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        addTArgument(biasCorrected ? 1.0 : 0.0);
    }

    public ReduceStandardDeviation(INDArray input, INDArray output, boolean biasCorrected, boolean keepDims, int... dimensions){
        super(input, output, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        addTArgument(biasCorrected ? 1.0 : 0.0);
    }

    public ReduceStandardDeviation(){}

    @Override
    public String opName() {
        return "reduce_stdev";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        return Collections.singletonList(f().stdBp(arg(), grad.get(0), biasCorrected, keepDims, dimensions));
    }
}
