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
 * SConv2DDerivative operation
 */
@Slf4j
public class SConv2DDerivative extends SConv2D {

    @Builder(builderMethodName = "sDerviativeBuilder")
    public SConv2DDerivative(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputArrays, INDArray[] outputs, Conv2DConfig conv2DConfig) {
        super(sameDiff, inputFunctions, inputArrays, outputs, conv2DConfig);
    }

    public SConv2DDerivative() {}

    @Override
    public String opName() {
        return "sconv2d_bp";
    }

    @Override
    public String[] tensorflowNames() {
        throw new NoOpNameFoundException("No op name found for backwards");
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

    @Override
    public int getNumOutputs(){
        //Inputs: in, gradAtOutput, weightsDepth, optional weightsPoint, optional weightsBias       3 req, 2 optional
        //Outputs: gradAtInput, gradWD, optional gradWP, optional gradB                             2 req, 2 optional
        SDVariable[] args = args();
        return args.length - 1;
    }

}
