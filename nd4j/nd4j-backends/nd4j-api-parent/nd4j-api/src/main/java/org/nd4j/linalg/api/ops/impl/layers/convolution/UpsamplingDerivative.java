package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * UpsamplingDerivative operation
 */
@Slf4j
public class UpsamplingDerivative extends Upsampling {

    public UpsamplingDerivative() {}

    @Builder(builderMethodName = "derivativeBuilder")
    public UpsamplingDerivative(SameDiff sameDiff, SDVariable[] inputs, INDArray[] inputArrays, INDArray[] outputs, boolean inPlace, int scaleFactor) {
        super(sameDiff, inputs, inputArrays, outputs, inPlace, scaleFactor);
    }

    @Override
    public String opName() {
        return "upsampling_bp";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
