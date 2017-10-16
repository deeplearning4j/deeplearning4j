package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;


/**
 * Pooling3D operation
 */
@Slf4j
public class Pooling3D extends DynamicCustomOp {

    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }


    private int kT,kW,kH,dT,dW,dH,pT,pW,pH,dilationT,dilationW,dilationH;
    private Pooling2DType type;
    private boolean ceilingMode;

    public Pooling3D() {}

    @Builder(builderMethodName = "sameDiffBuilder")
    public Pooling3D(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, Pooling2DType type, boolean ceilingMode) {
        super(null,sameDiff, inputs, inPlace);
        this.kT = kT;
        this.kW = kW;
        this.kH = kH;
        this.dT = dT;
        this.dW = dW;
        this.dH = dH;
        this.pT = pT;
        this.pW = pW;
        this.pH = pH;
        this.dilationT = dilationT;
        this.dilationW = dilationW;
        this.dilationH = dilationH;
        this.type = type;
        this.ceilingMode = ceilingMode;
        addArgs();
    }

    @Builder(builderMethodName = "execBuilder")
    public Pooling3D(INDArray[] inputs, INDArray[] outputs, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, Pooling2DType type, boolean ceilingMode) {
        super(null,inputs,outputs);
        this.kT = kT;
        this.kW = kW;
        this.kH = kH;
        this.dT = dT;
        this.dW = dW;
        this.dH = dH;
        this.pT = pT;
        this.pW = pW;
        this.pH = pH;
        this.dilationT = dilationT;
        this.dilationW = dilationW;
        this.dilationH = dilationH;
        this.type = type;
        this.ceilingMode = ceilingMode;
        addArgs();
    }


    private void addArgs() {
        getIArguments().add(kT);
        getIArguments().add(kW);
        getIArguments().add(kH);
        getIArguments().add(dT);
        getIArguments().add(dW);
        getIArguments().add(dH);
        getIArguments().add(pT);
        getIArguments().add(pW);
        getIArguments().add(pH);
        getIArguments().add(dilationT);
        getIArguments().add(dilationW);
        getIArguments().add(dilationH);

    }

    @Override
    public String opName() {
        return getPoolingPrefix() + "pool3d";
    }

  @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
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
