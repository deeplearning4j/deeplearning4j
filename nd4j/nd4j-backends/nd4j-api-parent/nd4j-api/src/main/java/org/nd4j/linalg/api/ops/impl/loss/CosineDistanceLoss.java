package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Arrays;
import java.util.List;

/**
 * Cosine distance loss
 *
 * @author Alex Black
 */
public class CosineDistanceLoss extends BaseLoss {

    private int dimension;

    public CosineDistanceLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, int dimension){
        super(sameDiff, lossReduce, predictions, weights, labels);
        this.dimension = dimension;
        this.addIArgument(dimension);
    }

    public CosineDistanceLoss(){ }

    @Override
    public String opName() {
        return "cosine_distance_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient.
        //Args are: predictions, weights, label
        SDVariable[] grads = f().lossCosineDistanceBp(arg(2), arg(0), arg(1), lossReduce, dimension);
        return Arrays.asList(grads);
    }

}
