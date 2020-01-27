package org.nd4j.linalg.api.ops.impl.reduce.bp;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.BaseArithmeticBackpropOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@NoArgsConstructor
public class PowBp extends BaseDynamicTransformOp {

    public PowBp(SameDiff sameDiff, SDVariable x, SDVariable y, SDVariable dLdz) {
        super(sameDiff,new SDVariable[]{x,y,dLdz}, false);
    }

    public PowBp(INDArray x, INDArray y, INDArray dLdz,
                 INDArray dLdx, INDArray dLdy) {
        super(new INDArray[]{x,y, dLdz}, new INDArray[]{dLdx, dLdy});
    }

    @Override
    public String opName() {
        return "Pow_bp";
    }

    @Override
    public boolean isInplaceCall() {
        return false;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 3, "Expected exactly 3 input datatypes for %s, got input %s", getClass(), dataTypes);
        //Gradient types: same as input
        return Arrays.asList(arg(0).dataType(), arg(1).dataType());
    }
}
