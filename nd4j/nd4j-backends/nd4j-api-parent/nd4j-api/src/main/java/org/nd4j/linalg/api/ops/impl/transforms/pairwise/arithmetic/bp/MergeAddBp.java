package org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@NoArgsConstructor
public class MergeAddBp extends DynamicCustomOp {

    public MergeAddBp(SameDiff sameDiff, @NonNull SDVariable[] inputs, @NonNull SDVariable gradO) {
        super("mergeadd_bp", sameDiff, ArrayUtils.add(inputs, gradO));
    }

    @Override
    public String opName() {
        return "mergeadd_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Arrays.asList(inputDataTypes.get(0));

    }
}