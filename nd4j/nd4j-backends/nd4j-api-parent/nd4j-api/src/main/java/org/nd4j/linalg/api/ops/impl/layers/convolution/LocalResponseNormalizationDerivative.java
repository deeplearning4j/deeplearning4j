package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;

import java.util.List;


/**
 * LocalResponseNormalizationDerivative operation
 */
@Slf4j
public class LocalResponseNormalizationDerivative extends LocalResponseNormalization {
    @Builder(builderMethodName = "derivativeBuilder")
    public LocalResponseNormalizationDerivative(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputs, INDArray[] outputs, boolean inPlace, LocalResponseNormalizationConfig config) {
        super(sameDiff, inputFunctions, inputs, outputs, inPlace, config);
    }

    public LocalResponseNormalizationDerivative() {}

    @Override
    public String opName() {
        return "lrn_bp";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
