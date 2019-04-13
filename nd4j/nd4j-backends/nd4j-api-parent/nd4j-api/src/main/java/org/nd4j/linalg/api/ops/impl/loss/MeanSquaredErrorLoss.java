package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Arrays;
import java.util.List;

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

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient
        //Args are: predictions, weights, label
        SDVariable[] grads = f().lossMeanSquaredErrorBp(arg(2), arg(0), arg(1), lossReduce);
        return Arrays.asList(grads);
    }

}
