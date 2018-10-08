package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Hinge loss
 *
 * @author Alex Black
 */
public class HingeLoss extends BaseLoss {


    public HingeLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }

    public HingeLoss(){ }

    @Override
    public String opName() {
        return "hinge_loss";
    }


}
