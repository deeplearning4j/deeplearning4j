package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * Pooling3DDerivative operation
 */
@Slf4j
public class Pooling3DDerivative extends Pooling3D {

    @Builder(builderMethodName = "sameDiffDerivativeBuilder")
    public Pooling3DDerivative(SameDiff sameDiff, DifferentialFunction[] inputs, boolean inPlace, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, Pooling2DType type, boolean ceilingMode) {
        super(sameDiff, inputs, inPlace, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, type, ceilingMode);
    }

    @Builder(builderMethodName = "execDerivativeBuilder")
    public Pooling3DDerivative(INDArray[] inputs, INDArray[] outputs, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, Pooling2DType type, boolean ceilingMode) {
        super(inputs, outputs, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, type, ceilingMode);
    }

    public Pooling3DDerivative() {}



    @Override
    public String opName() {
        return "pooling3d_bp";
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
