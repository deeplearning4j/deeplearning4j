package org.nd4j.linalg.api.ops.impl.loss.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Mean Pairwise Squared Error Loss Backprop
 *
 * @author Paul Dubs
 */
public class MeanPairwiseSquaredErrorLossBp extends DynamicCustomOp {


    public MeanPairwiseSquaredErrorLossBp(SameDiff sameDiff, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(null, sameDiff, new SDVariable[]{predictions, weights, labels});
    }

    public MeanPairwiseSquaredErrorLossBp(){ }

    @Override
    public String opName() {
        return "mean_pairwssqerr_loss_grad";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes.get(0).isFPType(), "Input 0 (predictions) must be a floating point type; inputs datatypes are %s for %s",
                inputDataTypes, getClass());
        DataType dt0 = inputDataTypes.get(0);
        DataType dt1 = arg(1).dataType();
        DataType dt2 = arg(2).dataType();
        if(!dt1.isFPType())
            dt1 = dt0;
        if(!dt2.isFPType())
            dt2 = dt0;
        return Arrays.asList(dt0, dt1, dt2);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }
}
