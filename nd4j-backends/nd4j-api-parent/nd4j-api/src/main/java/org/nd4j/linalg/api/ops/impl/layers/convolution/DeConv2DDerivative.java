package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;

import java.util.List;


/**
 * DeConv2DDerivative operation
 */
@Slf4j
public class DeConv2DDerivative extends DeConv2D {

    public DeConv2DDerivative() {}

    @Builder(builderMethodName = "derivativeBuilder")
    public DeConv2DDerivative(SameDiff sameDiff, SDVariable[] inputs, INDArray[] inputArrays, INDArray[] outputs, DeConv2DConfig config) {
        super(sameDiff, inputs, inputArrays, outputs, config);
    }

    @Override
    public String opName() {
        return "deconv2d_bp";
    }



    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No op name found for backwards.");
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No op name found for backwards");
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");

    }

}
