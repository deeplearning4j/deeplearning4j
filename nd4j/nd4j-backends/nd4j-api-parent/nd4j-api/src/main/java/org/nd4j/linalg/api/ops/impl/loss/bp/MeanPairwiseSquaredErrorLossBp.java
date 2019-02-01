package org.nd4j.linalg.api.ops.impl.loss.bp;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

/**
 * Mean Pairwise Squared Error Loss Backprop
 *
 * @author Paul Dubs
 */
public class MeanPairwiseSquaredErrorLossBp extends BaseLossBp {


    public MeanPairwiseSquaredErrorLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce,  predictions, weights, labels);
    }

    public MeanPairwiseSquaredErrorLossBp(){ }

    @Override
    public String opName() {
        return "mean_pairwssqerr_loss_grad";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }
}
