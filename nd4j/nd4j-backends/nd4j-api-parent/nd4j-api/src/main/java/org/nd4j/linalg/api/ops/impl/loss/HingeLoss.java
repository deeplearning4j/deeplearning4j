package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Arrays;
import java.util.List;

/**
 * Hinge loss
 *
 * @author Alex Black
 */
public class HingeLoss extends BaseLoss {


    public HingeLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }

    public HingeLoss(){ }

    @Override
    public String opName() {
        return "hinge_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient
        //Args are: predictions, weights, label
        SDVariable[] grads = f().lossHingeBp(arg(2), arg(0), arg(1), lossReduce);
        return Arrays.asList(grads);
    }

}
