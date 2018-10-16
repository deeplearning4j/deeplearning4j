package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Absolute difference loss
 *
 * @author Alex Black
 */
public class AbsoluteDifferenceLoss extends BaseLoss {


    public AbsoluteDifferenceLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }

    public AbsoluteDifferenceLoss(){ }

    @Override
    public String opName() {
        return "absolute_difference_loss";
    }


}
