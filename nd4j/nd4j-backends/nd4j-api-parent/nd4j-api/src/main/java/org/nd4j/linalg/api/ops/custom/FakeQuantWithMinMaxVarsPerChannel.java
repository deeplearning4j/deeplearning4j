package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class FakeQuantWithMinMaxVarsPerChannel extends DynamicCustomOp {
    public FakeQuantWithMinMaxVarsPerChannel() {}

    public FakeQuantWithMinMaxVarsPerChannel(INDArray x, INDArray min, INDArray max,
                                             INDArray output) {
        Preconditions.checkArgument(min.isVector() && max.isVector() &&
                        min.length() == max.length(),
                "FakeQuantWithMinMaxVarsPerChannel: min and max should be 1D tensors with the same length");
        inputArguments.add(x);
        inputArguments.add(min);
        inputArguments.add(max);
        outputArguments.add(output);
    }

    public FakeQuantWithMinMaxVarsPerChannel(SameDiff sameDiff, SDVariable x, SDVariable min, SDVariable max) {
        super("", sameDiff, new SDVariable[]{x, min, max});
    }

    @Override
    public String opName() {
        return "fake_quant_with_min_max_vars_per_channel";
    }

    @Override
    public String tensorflowName() {
        return "FakeQuantWithMinMaxVarsPerChannel";
    }
}