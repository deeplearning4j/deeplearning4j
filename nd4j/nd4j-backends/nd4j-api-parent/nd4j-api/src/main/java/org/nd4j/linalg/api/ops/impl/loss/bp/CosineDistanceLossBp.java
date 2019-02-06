package org.nd4j.linalg.api.ops.impl.loss.bp;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.loss.BaseLoss;

import java.util.List;

/**
 * Cosine distance loss
 *
 * @author Alex Black
 */
public class CosineDistanceLossBp extends BaseLossBp {

    private int dimension;

    public CosineDistanceLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, int dimension){
        super(sameDiff, lossReduce, predictions, weights, labels);
        this.dimension = dimension;
        this.addIArgument(dimension);
    }

    public CosineDistanceLossBp(){ }

    @Override
    public String opName() {
        return "cosine_distance_loss_grad";
    }


}
