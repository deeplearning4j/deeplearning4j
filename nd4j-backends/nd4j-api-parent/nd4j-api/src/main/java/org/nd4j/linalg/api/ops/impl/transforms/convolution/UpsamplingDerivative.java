package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * UpsamplingDerivative operation
 */
@Slf4j
public class UpsamplingDerivative extends Upsampling {

    public UpsamplingDerivative() {}

    @Builder(builderMethodName = "sameDiffDerivativeBuilder")
    public UpsamplingDerivative(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int scaleFactor) {
        super(sameDiff, inputs, inPlace, scaleFactor);
    }

    @Builder(builderMethodName = "execDerivativeBuilder")
    public UpsamplingDerivative(INDArray[] inputs, INDArray[] outputs, int scaleFactor) {
        super(inputs,outputs, scaleFactor);
    }


    @Override
    public String opName() {
        return "upsampling_bp";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
