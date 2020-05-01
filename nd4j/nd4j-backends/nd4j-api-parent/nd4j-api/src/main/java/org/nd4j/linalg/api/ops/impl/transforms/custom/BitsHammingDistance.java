package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public class BitsHammingDistance extends DynamicCustomOp {

    public BitsHammingDistance(){ }

    public BitsHammingDistance(@NonNull SameDiff sd, @NonNull SDVariable x, @NonNull SDVariable y){
        super(sd, new SDVariable[]{x, y});
    }

    public BitsHammingDistance(@NonNull INDArray x, @NonNull INDArray y){
        super(new INDArray[]{x, y}, null);
    }

    @Override
    public String opName() {
        return "bits_hamming_distance";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 2, "Expected 2 input datatypes, got %s", dataTypes);
        Preconditions.checkState(dataTypes.get(0).isIntType() && dataTypes.get(1).isIntType(), "Input datatypes must be integer type, got %s", dataTypes);
        return Collections.singletonList(DataType.LONG);
    }
}
