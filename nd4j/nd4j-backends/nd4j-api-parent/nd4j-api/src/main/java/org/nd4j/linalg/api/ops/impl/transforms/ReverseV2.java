package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * This is compatibility op for ReverseV2
 */
public class ReverseV2 extends DynamicCustomOp {
    protected final boolean isLegacy = true;

    public ReverseV2() {
        iArguments.add(isLegacy ? 1L : 0L);
    }

    @Override
    public String opName() {
        return "reverse_v2";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "ReverseV2";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("BP mode isn't supported for this op");
    }

}
