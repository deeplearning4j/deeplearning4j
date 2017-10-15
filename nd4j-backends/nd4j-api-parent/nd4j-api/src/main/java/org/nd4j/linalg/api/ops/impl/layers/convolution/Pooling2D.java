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
 * Pooling2D operation
 */
@Slf4j
@Getter
public class Pooling2D extends DynamicCustomOp {

    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }



    private int kh, kw, sy, sx, ph, pw, dh, dw,virtualHeight,virtualWidth;
    private double extra;
    private Pooling2DType type;
    private boolean isSameMode;

    public Pooling2D() {}

    @Builder(builderMethodName = "sameDiffBuilder")
    @SuppressWarnings("Used in lombok")
    public Pooling2D(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, int virtualHeight,int virtualWidth,double extra,Pooling2DType type, boolean isSameMode) {
        super(null,sameDiff, inputs, inPlace);
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.virtualHeight = virtualHeight;
        this.virtualWidth = virtualWidth;
        this.type = type;
        this.extra = extra;
        this.isSameMode = isSameMode;
        addArgs();
    }

    @Builder(builderMethodName = "execBuilder")
    @SuppressWarnings("Used in lombok")
    public Pooling2D(INDArray[] arrayInputs, INDArray[] arrayOutputs,int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw,  int virtualHeight,int virtualWidth, double extra,Pooling2DType type, boolean isSameMode) {
        super(null,arrayInputs,arrayOutputs);
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.virtualWidth = virtualWidth;
        this.virtualHeight = virtualHeight;
        this.extra = extra;
        this.type = type;
        this.isSameMode = isSameMode;
        addArgs();
    }


    private void addArgs() {
        getIArguments().add(kh);
        getIArguments().add(kw);
        getIArguments().add(sy);
        getIArguments().add(sx);
        getIArguments().add(ph);
        getIArguments().add(pw);
        getIArguments().add(dh);
        getIArguments().add(virtualHeight);
        getIArguments().add(virtualWidth);
        getIArguments().add(fromBoolean(isSameMode));

        getTArguments().add(extra);

    }

    @Override
    public String opName() {
        return getPoolingPrefix() + "pool2d";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Pooling2DDerivative pooling2DDerivative = Pooling2DDerivative.sameDiffDerivativeBuilder()
                .dh(dh)
                .dw(dw)
                .extra(extra)
                .isSameMode(isSameMode)
                .kh(kh)
                .kw(kw)
                .ph(ph)
                .pw(pw)
                .type(type)
                .sx(sx)
                .sy(sy)
                .virtualHeight(virtualHeight)
                .virtualWidth(virtualWidth)
                .inputs(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .build();
        ret.addAll(Arrays.asList(pooling2DDerivative.getOutputFunctions()));
        return ret;
    }

    public String getPoolingPrefix() {
        switch(type) {
            case AVG:return "avg";
            case MAX: return "max";
            case PNORM: return "pnorm";
            default: throw new IllegalStateException("No pooling type found.");
        }
    }

}
