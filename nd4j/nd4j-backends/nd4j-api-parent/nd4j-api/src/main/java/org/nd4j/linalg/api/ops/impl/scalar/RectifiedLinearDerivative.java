package org.nd4j.linalg.api.ops.impl.scalar;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.shade.guava.base.Preconditions;

import java.util.Collections;
import java.util.List;

public class RectifiedLinearDerivative extends DynamicCustomOp {

    public RectifiedLinearDerivative(){ }

    public RectifiedLinearDerivative(SameDiff sd, SDVariable input, SDVariable gradient){
        super(sd, new SDVariable[]{input, gradient});
    }

    public RectifiedLinearDerivative(@NonNull INDArray input, @NonNull INDArray gradient, INDArray output){
        super(new INDArray[]{input, gradient}, wrapOrNull(output));
    }

    @Override
    public String opName(){
        return "relu_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkArgument(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes, got %s", dataTypes);
        Preconditions.checkArgument(dataTypes.get(0).isFPType() && dataTypes.get(1).isFPType(), "Input datatypes must be floating point, got %s", dataTypes);

        return Collections.singletonList(dataTypes.get(0));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Not supported");
    }
}
