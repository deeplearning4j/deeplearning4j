package org.nd4j.linalg.api.ops.impl.loss.bp;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.loss.BaseLoss;

/**
 * Mean squared error loss
 *
 * @author Alex Black
 */
public class MeanSquaredErrorLossBp extends BaseLossBp {


    public MeanSquaredErrorLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }

    public MeanSquaredErrorLossBp(){ }

    @Override
    public String opName() {
        return "mean_sqerr_loss_grad";
    }


}
