package org.nd4j.linalg.api.ops.impl.transforms.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Broadcastable element-wise power operation: x[i]^y[i]
 *
 * @author Alex Black
 */
public class Pow extends DynamicCustomOp {

    public Pow(SameDiff sameDiff, SDVariable x, SDVariable y){
        super(sameDiff, new SDVariable[]{x, y});
    }

    public Pow(){ }

    @Override
    public String opName(){
        return "Pow";
    }

    @Override
    public String tensorflowName(){
        return "Pow";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Not yet implemneted");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected exactly 2 input datatypes for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }
}
