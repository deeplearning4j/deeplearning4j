package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;

import java.util.List;


/**
 * Conv2DDerivative operation
 */
@Slf4j
public class Conv2DDerivative extends Conv2D {

    @Builder(builderMethodName = "derivativeBuilder")
    public Conv2DDerivative(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputArrays, INDArray[] outputs, Conv2DConfig config) {
        super(sameDiff, inputFunctions, inputArrays, outputs, config);
    }

    public Conv2DDerivative() {}



    @Override
    public String onnxName() {
       throw new NoOpNameFoundException("No op name found for backwards.");
    }

    @Override
    public String tensorflowName() {
       throw new NoOpNameFoundException("No op name found for backwards");
    }

    @Override
    public String[] tensorflowNames() {
        throw new NoOpNameFoundException("No op name found for backwards");
    }

    @Override
    public String opName() {
        return "conv2d_bp";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
