package org.nd4j.linalg.api.ops.impl.loss.bp;

import lombok.NonNull;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public abstract class BaseLossBp extends DynamicCustomOp {

    protected LossReduce lossReduce;

    public BaseLossBp(@NonNull SameDiff sameDiff, @NonNull LossReduce lossReduce, @NonNull SDVariable predictions, @NonNull SDVariable weights,
                      @NonNull SDVariable labels){
        super(null, sameDiff, new SDVariable[]{predictions, weights, labels});
        this.lossReduce = lossReduce;
        addArgs();
    }

    protected BaseLossBp(){ }

    protected void addArgs(){
        iArguments.clear();
        tArguments.clear();
        addIArgument(lossReduce.ordinal()); //Ops: 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
    }

    public abstract String opName();

    @Override
    public int getNumOutputs(){
        return 3;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){

        return Arrays.asList(arg(0).dataType(), arg(1).dataType(), arg(2).dataType());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }
}
