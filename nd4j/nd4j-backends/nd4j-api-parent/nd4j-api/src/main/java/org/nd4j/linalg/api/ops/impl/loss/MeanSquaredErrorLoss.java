package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Mean squared error loss
 *
 * @author Alex Black
 */
public class MeanSquaredErrorLoss extends BaseLoss {


    public MeanSquaredErrorLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }

    public MeanSquaredErrorLoss(){ }

    @Override
    public String opName() {
        return "mean_sqerr_loss";
    }


}
