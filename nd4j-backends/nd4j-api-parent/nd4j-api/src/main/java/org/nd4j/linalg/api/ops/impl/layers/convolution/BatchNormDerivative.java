package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * BatchNormDerivative operation
 */
@Slf4j
public class BatchNormDerivative extends BatchNorm {

    @Builder(builderMethodName = "derivativeBuilder")
    public BatchNormDerivative(SameDiff sameDiff, DifferentialFunction[] inputFunctions, INDArray[] inputArrays, INDArray[] outputArrays, boolean inPlace, boolean training, boolean isLockGammaBeta, boolean isMiniBatch) {
        super(sameDiff, inputFunctions, inputArrays, outputArrays, inPlace, training, isLockGammaBeta, isMiniBatch);
    }

    public BatchNormDerivative() {}


    @Override
    public String opName() {
        return "batchnorm_bp";
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
