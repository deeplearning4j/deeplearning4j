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
    public static final double DEFAULT_EPSILON = 1e-7;

    private double epsilon;

    public LogLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, double epsilon){
        super(sameDiff, lossReduce, predictions, weights, labels);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public LogLoss(){ }

    @Override
    public String opName() {
        return "log_loss";
    }


}
