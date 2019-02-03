package org.nd4j.linalg.api.ops.impl.loss.bp;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;

/**
 * Hinge loss
 *
 * @author Alex Black
 */
public class HuberLossBp extends BaseLossBp {
    private double delta;

    public HuberLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, double delta){
        super(sameDiff, lossReduce, predictions, weights, labels);
        Preconditions.checkState(delta >= 0.0, "Delta must be >= 0.0. Got: %s", delta);
        this.delta = delta;
        tArguments.add(delta);
    }

    public HuberLossBp(){ }

    @Override
    public String opName() {
        return "huber_loss_grad";
    }


}
