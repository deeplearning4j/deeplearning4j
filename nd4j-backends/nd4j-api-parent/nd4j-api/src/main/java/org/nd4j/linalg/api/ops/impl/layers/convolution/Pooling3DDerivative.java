package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;

import java.util.List;


/**
 * Pooling3DDerivative operation
 */
@Slf4j
public class Pooling3DDerivative extends Pooling3D {

    @Builder(builderMethodName = "derivativeBuilder")
    public Pooling3DDerivative(SameDiff sameDiff, DifferentialFunction[] inputs, INDArray[] inputArrays, INDArray[] outputs, boolean inPlace, Pooling3DConfig pooling3DConfig) {
        super(sameDiff, inputs, inputArrays, outputs, inPlace, pooling3DConfig);
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
