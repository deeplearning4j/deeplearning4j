package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

public class Reverse extends DynamicCustomOp {

    public Reverse(SameDiff sameDiff, SDVariable i_v, int... dimensions) {
        super(null, sameDiff, new SDVariable[]{i_v}, false);
        this.dimensions = dimensions;
        addIArgument(dimensions);
    }

    public Reverse() {
    }

    @Override
    public String opName() {
        return "reverse";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Reverse";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable ret = f().reverse(f1.get(0), dimensions);
        return Arrays.asList(ret);
    }

}
