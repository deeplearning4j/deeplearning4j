package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * FullConv3DDerivative operation
 */
@Slf4j
public class FullConv3DDerivative extends FullConv3D {

    public FullConv3DDerivative() {}

    @Builder(builderMethodName = "sameDiffDerivativeBuilder")
    public FullConv3DDerivative(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, boolean biasUsed) {
        super(sameDiff, inputs, inPlace, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH, biasUsed);
    }

    @Builder(builderMethodName = "execDerivativeBuilder")
    public FullConv3DDerivative(INDArray[] inputs, INDArray[] outputs, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, boolean biasUsed) {
        super(inputs,outputs, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH, biasUsed);
    }


    @Override
    public String opName() {
        return "fullconv3d_bp";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
