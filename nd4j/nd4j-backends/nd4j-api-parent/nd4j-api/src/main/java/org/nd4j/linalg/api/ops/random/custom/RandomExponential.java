package org.nd4j.linalg.api.ops.random.custom;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Random exponential distribution: p(x) = lambda * exp(-lambda * x)
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class RandomExponential extends DynamicCustomOp {
    private double lambda = 0.0;

    public RandomExponential() {
        //
    }

    public RandomExponential(SameDiff sd, SDVariable shape, double lambda){
        super(null, sd, new SDVariable[]{shape});
        Preconditions.checkState(lambda >= 0, "Lambda parameter must be > 0 - got %s", lambda);
        this.lambda = lambda;
        addTArgument(lambda);
    }

    public RandomExponential(INDArray shape,INDArray out, double lambda){
        super(null, new INDArray[]{shape}, new INDArray[]{out}, Collections.singletonList(lambda), (List<Integer>)null);
        this.lambda = lambda;
    }

    @Override
    public String opName() {
        return "random_exponential";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }
}
