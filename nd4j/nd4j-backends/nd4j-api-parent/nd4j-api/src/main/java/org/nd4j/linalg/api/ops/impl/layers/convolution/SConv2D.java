package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


/**
 * Separable convolution 2D operation
 */
@Slf4j
public class SConv2D extends Conv2D {

    @Builder(builderMethodName = "sBuilder")
    public SConv2D(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputArrays, INDArray[] outputs, Conv2DConfig conv2DConfig) {
        super(sameDiff, inputFunctions, inputArrays, outputs, conv2DConfig);
    }

    public SConv2D() {}

    @Override
    public String opName() {
        return "sconv2d";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //Args at libnd4j level: in, gradAtOut, wD, wP, bias
        //Args for SConv2d libnd4j: input, wD, wP, bias
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(arg(0));
        inputs.add(f1.get(0));
        SDVariable[] args = args();
        for( int i=1; i<args.length; i++ ){ //Skip input, already added
            inputs.add(args[i]);
        }
        SConv2DDerivative conv2DDerivative = SConv2DDerivative.sDerviativeBuilder()
                .conv2DConfig(config)
                .inputFunctions(inputs.toArray(new SDVariable[inputs.size()]))
                .sameDiff(sameDiff)
                .build();
        List<SDVariable> ret = Arrays.asList(conv2DDerivative.outputVariables());
        return ret;
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    @Override
    public boolean isConfigProperties() {
        return true;
    }

    @Override
    public String configFieldName() {
        return "config";
    }

    @Override
    public void setValueFor(Field target, Object value) {
        config.setValueFor(target,value);
    }


    @Override
    public String[] tensorflowNames() {
        throw new NoOpNameFoundException("No op name found for backwards");
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
