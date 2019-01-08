package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.Collections;
import java.util.List;

/**
 * Absolute difference loss
 *
 * @author Alex Black
 */
public class AbsoluteDifferenceLoss extends BaseLoss {


    public AbsoluteDifferenceLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }

    public AbsoluteDifferenceLoss(){ }

    @Override
    public String opName() {
        return "absolute_difference_loss";
    }

}
