package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;

import java.util.List;


/**
 * Conv3DDerivative operation
 */
@Slf4j
public class Conv3DDerivative extends Conv3D {

    public Conv3DDerivative() {}

    @Builder(builderMethodName = "derivativeBuilder")
    public Conv3DDerivative(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputs, INDArray[] outputs, Conv3DConfig conv3DConfig) {
        super(sameDiff, inputFunctions, inputs, outputs, conv3DConfig);
    }

    @Override
    public String opName() {
        return "conv3d_bp";
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op name found for conv3d derivative");
    }

    @Override
    public String[] tensorflowNames() {
        throw new NoOpNameFoundException("No tensorflow op name found for conv3d derivative");
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op name found for conv3d derivative");
    }

    @Override
    public String[] onnxNames() {
        throw new NoOpNameFoundException("No onnx op name found for conv3d derivative");
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to differentiate from a derivative op");
    }

}
