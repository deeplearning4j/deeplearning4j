package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Huber loss
 *
 * @author Alex Black
 */
public class HuberLoss extends BaseLoss {


    public HuberLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }

    public HuberLoss(){ }

    @Override
    public String opName() {
        return "huber_loss";
    }


}
