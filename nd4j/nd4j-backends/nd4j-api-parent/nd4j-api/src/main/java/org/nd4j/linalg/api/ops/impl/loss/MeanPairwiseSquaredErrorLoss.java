package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

/**
 * Absolute difference loss
 *
 * @author Alex Black
 */
public class MeanPairwiseSquaredErrorLoss extends DynamicCustomOp {


    public MeanPairwiseSquaredErrorLoss(SameDiff sameDiff, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(null, sameDiff, new SDVariable[]{predictions, weights, labels});
    }

    public MeanPairwiseSquaredErrorLoss(){ }

    @Override
    public String opName() {
        return "mean_pairwssqerr_loss";
    }


}
