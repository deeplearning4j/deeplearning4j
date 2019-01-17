package org.nd4j.linalg.api.ops.impl.loss.bp;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.loss.BaseLoss;

/**
 * Hinge loss
 *
 * @author Alex Black
 */
public class HingeLossBp extends BaseLossBp {


    public HingeLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }

    public HingeLossBp(){ }

    @Override
    public String opName() {
        return "hinge_loss_grad";
    }


}
