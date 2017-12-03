package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * DeConv2D operation
 */
@Slf4j
@Getter
public class DeConv2D extends DynamicCustomOp {


    protected DeConv2DConfig config;

    public DeConv2D() {}


    @Builder(builderMethodName = "builder")
    public DeConv2D(SameDiff sameDiff, DifferentialFunction[] inputs,INDArray[] inputArrays, INDArray[] outputs,boolean inPlace, DeConv2DConfig config) {
        super(null,sameDiff, inputs, inPlace);
        this.config = config;
        if(inputArrays != null) {
            addInputArgument(inputArrays);
        }

        if(outputs != null) {
            addOutputArgument(outputs);
        }


        addArgs();
    }



    private void addArgs() {
       addIArgument(config.getKY());
       addIArgument(config.getKX());
       addIArgument(config.getSY());
       addIArgument(config.getSX());
       addIArgument(config.getPY());
       addIArgument(config.getPX());
       addIArgument(config.getDY());
       addIArgument(config.getDX());
       addIArgument(fromBoolean(config.isSameMode()));

    }


    @Override
    public String opName() {
        return "deconv2d";
    }

    @Override
    public String onnxName() {
        return "ConvTranspose";
    }

    @Override
    public String tensorflowName() {
        return "conv2d_transpose";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.addAll(f1);
        DeConv2DDerivative deConv2DDerivative = DeConv2DDerivative.derivativeBuilder()
                .sameDiff(sameDiff)
                .config(config)
                .inputs(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .build();
        ret.addAll(Arrays.asList(deConv2DDerivative.outputFunctions()));
        return ret;
    }

}
