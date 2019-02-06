package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;

import java.util.Arrays;
import java.util.List;

/**
 * Huber loss
 *
 * @author Alex Black
 */
public class HuberLoss extends BaseLoss {

    private double delta;

    public HuberLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, double delta){
        super(sameDiff, lossReduce, predictions, weights, labels);
        Preconditions.checkState(delta >= 0.0, "Delta must be >= 0.0. Got: %s", delta);
        this.delta = delta;
        tArguments.add(delta);
    }

    public HuberLoss(){ }

    @Override
    public String opName() {
        return "huber_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient
        //Args are: predictions, weights, label
        SDVariable[] grads = f().lossHuberBp(arg(2), arg(0), arg(1), lossReduce, delta);
        return Arrays.asList(grads);
    }


}
