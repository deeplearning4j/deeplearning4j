package org.nd4j.linalg.api.ops.random.custom;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Uniform distribution wrapper
 *
 * @author raver119@gmail.com
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

    @Override
    public String opName() {
        return "random_bernoulli";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }
}
