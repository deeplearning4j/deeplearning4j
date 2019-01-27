package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Mean Pairwise Squared Error Loss
 *
 * @author Paul Dubs
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

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3, "Expected exactly 3 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));    //Same as predictions
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient
        //Args are: predictions, weights, label
        SDVariable[] grads = f().lossMeanPairwiseSquaredErrorBp(arg(2), arg(0), arg(1));
        return Arrays.asList(grads);
    }
}
