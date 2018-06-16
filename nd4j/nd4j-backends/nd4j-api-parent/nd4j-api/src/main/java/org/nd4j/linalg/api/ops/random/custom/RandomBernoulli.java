package org.nd4j.linalg.api.ops.random.custom;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Random bernoulli distribution: p(x=1) = p, p(x=0) = 1-p
 * i.e., output is 0 or 1 with probability p.
 *
 * @author Alex Black
 */
@Slf4j
public class RandomBernoulli extends DynamicCustomOp {
    private double p = 0.0;

    public RandomBernoulli() {
        //
    }

    public RandomBernoulli(SameDiff sd, SDVariable shape, double p){
        super(null, sd, new SDVariable[]{shape});
        Preconditions.checkState(p >= 0 && p <= 1.0, "Probability must be between 0 and 1 - got %s", p);
        this.p = p;
        addTArgument(p);
    }

    public RandomBernoulli(INDArray shape, INDArray out, double p){
        super(null, new INDArray[]{shape}, new INDArray[]{out}, Collections.singletonList(p), (List<Integer>)null);
        Preconditions.checkState(p >= 0 && p <= 1.0, "Probability must be between 0 and 1 - got %s", p);
    }

    @Override
    public String opName() {
        return "random_bernoulli";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }
}
