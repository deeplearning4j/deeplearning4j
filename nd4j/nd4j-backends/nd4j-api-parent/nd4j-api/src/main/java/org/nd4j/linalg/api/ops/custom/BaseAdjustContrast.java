package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public abstract class BaseAdjustContrast extends DynamicCustomOp {
    public BaseAdjustContrast() {
    }

    public BaseAdjustContrast(INDArray in, double factor, INDArray out) {
        Preconditions.checkArgument(in.rank() >= 3,
                String.format("AdjustContrast: op expects rank of input array to be >= 3, but got %d instead", in.rank()));
        inputArguments.add(in);
        outputArguments.add(out);

        addTArgument(factor);
    }

    public BaseAdjustContrast(SameDiff sameDiff, SDVariable[] vars) {
        super("", sameDiff, vars);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}