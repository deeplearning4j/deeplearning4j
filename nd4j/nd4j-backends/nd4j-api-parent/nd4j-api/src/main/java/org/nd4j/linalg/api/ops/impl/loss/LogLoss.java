package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Absolute difference loss
 *
 * @author Alex Black
 */
public class LogLoss extends BaseLoss {


    public LogLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }

    public LogLoss(){ }

    @Override
    public String opName() {
        return "log_loss";
    }


}
