package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * SConv2DDerivative operation
 */
@Slf4j
public class SConv2DDerivative extends SConv2D {

    @Builder(builderMethodName = "sameDiffDerivativeBuilder")
    public SConv2DDerivative(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(sameDiff, inputs, inPlace, kh, kw, sy, sx, ph, pw, dh, dw, isSameMode);
    }

    @Builder(builderMethodName = "execDerivativeBuilder")
    public SConv2DDerivative(INDArray[] inputs, INDArray[] outputs, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(inputs,outputs, kh, kw, sy, sx, ph, pw, dh, dw, isSameMode);
    }

    public SConv2DDerivative() {}

    @Override
    public String opName() {
        return "sconv2d_bp";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
