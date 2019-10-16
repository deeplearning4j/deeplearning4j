package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

public class BitCast extends DynamicCustomOp {
    public BitCast() {}

    public BitCast(INDArray in, int dataType, INDArray out) {
        inputArguments.add(in);
        outputArguments.add(out);
        iArguments.add(Long.valueOf(dataType));
    }

    public BitCast(SameDiff sameDiff, SDVariable in, SDVariable dataType) {
        super("", sameDiff, new SDVariable[]{in, dataType});
    }

    @Override
    public String opName() {
        return "bitcast";
    }
}