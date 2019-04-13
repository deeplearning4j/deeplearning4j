package org.nd4j.linalg.api.ops.impl.loss.bp;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.loss.BaseLoss;

import java.util.List;

/**
 * Absolute difference loss backprop
 *
 * @author Alex Black
 */
public class AbsoluteDifferenceLossBp extends BaseLossBp {


    public AbsoluteDifferenceLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }

    public AbsoluteDifferenceLossBp(){ }

    @Override
    public String opName() {
        return "absolute_difference_loss_grad";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }

}
