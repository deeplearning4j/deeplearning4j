package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Softmax cross entropy with logits loss
 *
 * @author Alex Black
 */
public class SoftmaxCrossEntropyWithLogitsLoss extends BaseLoss {


    public SoftmaxCrossEntropyWithLogitsLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable labels, SDVariable weights ){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }

    @Override
    public String opName() {
        return "softmax_cross_entropy_loss";
    }


}
