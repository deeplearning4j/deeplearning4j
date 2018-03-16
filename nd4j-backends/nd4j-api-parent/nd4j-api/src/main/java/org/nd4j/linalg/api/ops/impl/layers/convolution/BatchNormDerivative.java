package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * BatchNormDerivative operation
 */
@Slf4j
public class BatchNormDerivative extends BatchNorm {

    @Builder(builderMethodName = "derivativeBuilder")
    public BatchNormDerivative(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputArrays,
                               INDArray[] outputArrays, boolean inPlace, boolean applyGamma,
                               boolean applyBeta, double epsilon) {
        super(sameDiff, inputFunctions, inputArrays, outputArrays, inPlace, applyGamma, applyBeta, epsilon);
    }

    public BatchNormDerivative() {}


    @Override
    public String opName() {
        return "batchnorm_bp";
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
