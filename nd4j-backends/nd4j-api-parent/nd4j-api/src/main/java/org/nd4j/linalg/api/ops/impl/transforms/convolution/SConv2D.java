package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Pooling2DDerivative operation
 */
@Slf4j
public class SConv2D extends Conv2D {

    @Builder(builderMethodName = "sameDiffSBuilder")
    public SConv2D(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(sameDiff, inputs, inPlace, kh, kw, sy, sx, ph, pw, dh, dw, isSameMode);
    }

    @Builder(builderMethodName = "execSBuilder")
    public SConv2D(INDArray[] inputs, INDArray[] outputs, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(inputs,outputs, kh, kw, sy, sx, ph, pw, dh, dw, isSameMode);
    }

    public SConv2D() {}

    @Override
    public String opName() {
        return "sconv2d";
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        SConv2DDerivative conv2DDerivative = SConv2DDerivative.sameDiffDerivativeBuilder()
                .dh(dh)
                .dw(dw)
                .isSameMode(isSameMode)
                .kh(kh)
                .kw(kw)
                .ph(ph)
                .pw(pw)
                .sx(sx)
                .sy(sy)
                .inputs(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .build();
        ret.addAll(Arrays.asList(conv2DDerivative.getOutputFunctions()));
        return ret;
    }

}
