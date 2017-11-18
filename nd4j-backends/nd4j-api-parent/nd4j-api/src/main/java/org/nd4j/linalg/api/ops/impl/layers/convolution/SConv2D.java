package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Pooling2DDerivative operation
 */
@Slf4j
public class SConv2D extends Conv2D {

    @Builder(builderMethodName = "sBuilder")
    public SConv2D(SameDiff sameDiff, DifferentialFunction[] inputFunctions, INDArray[] inputArrays, INDArray[] outputs, Conv2DConfig conv2DConfig) {
        super(sameDiff, inputFunctions, inputArrays, outputs, conv2DConfig);
    }

    public SConv2D() {}

    @Override
    public String opName() {
        return "sconv2d";
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        SConv2DDerivative conv2DDerivative = SConv2DDerivative.sDerviativeBuilder()
                .conv2DConfig(conv2DConfig)
                .inputFunctions(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .build();
        ret.addAll(Arrays.asList(conv2DDerivative.getOutputFunctions()));
        return ret;
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for op " + opName());
    }

    @Override
    public String tensorflowName() {
        return "separable_conv2d";
    }

}
