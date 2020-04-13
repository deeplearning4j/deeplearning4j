package org.nd4j.linalg.api.ops.impl.shape.bp;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.List;


@NoArgsConstructor
public class MergeMaxBp extends DynamicCustomOp {

    public MergeMaxBp(SameDiff sameDiff, @NonNull SDVariable[] inputs, @NonNull SDVariable gradO) {
        super("mergemax_bp", sameDiff, ArrayUtils.add(inputs, gradO));
    }

    @Override
    public String opName() {
        return "mergemax_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        ArrayList<DataType> list = new ArrayList<DataType>();
        for (int i=0; i< args().length-1;i++){list.add(inputDataTypes.get(0));}
        return list;

    }

    @Override
    public int getNumOutputs(){
        return args().length-1;
    }
}
