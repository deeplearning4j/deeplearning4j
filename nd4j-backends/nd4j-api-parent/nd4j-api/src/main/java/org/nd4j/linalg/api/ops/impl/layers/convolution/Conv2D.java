package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Conv2D operation
 */
@Slf4j
@Getter
public class Conv2D extends DynamicCustomOp {


    protected int kh, kw, sy, sx, ph, pw, dh, dw;
    protected boolean isSameMode;

    @Builder(builderMethodName = "sameDiffBuilder")
    public Conv2D(SameDiff sameDiff, DifferentialFunction[] inputFunctions, boolean inPlace, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(null,sameDiff, inputFunctions, inPlace);
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
        addArgs();

    }

    @Builder(builderMethodName = "execBuilder")
    public Conv2D(INDArray[] inputArrays, INDArray[] outputs, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(null,inputArrays,outputs);
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
        addArgs();
    }

    public Conv2D() {}

    protected void addArgs() {
        getIArguments().add(kh);
        getIArguments().add(kw);
        getIArguments().add(sy);
        getIArguments().add(sx);
        getIArguments().add(ph);
        getIArguments().add(pw);
        getIArguments().add(dh);
        getIArguments().add(dw);
        getIArguments().add(fromBoolean(isSameMode));

    }


    @Override
    public String opName() {
        return "conv2d";
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Conv2DDerivative conv2DDerivative = Conv2DDerivative.sameDiffDerivativeBuilder()
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
