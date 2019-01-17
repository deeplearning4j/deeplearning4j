package org.nd4j.linalg.api.ops.impl.loss.bp;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.loss.BaseLoss;

/**
 * Binary log loss, or cross entropy loss:
 * {@code -1/numExamples * sum_i (labels[i] * log(predictions[i] + epsilon) + (1-labels[i]) * log(1-predictions[i] + epsilon))}
 *
 * @author Alex Black
 */
public class LogLossBp extends BaseLossBp {

    private double epsilon;

    public LogLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, double epsilon){
        super(sameDiff, lossReduce, predictions, weights, labels);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public LogLossBp(){ }

    @Override
    public String opName() {
        return "log_loss_grad";
    }


}
