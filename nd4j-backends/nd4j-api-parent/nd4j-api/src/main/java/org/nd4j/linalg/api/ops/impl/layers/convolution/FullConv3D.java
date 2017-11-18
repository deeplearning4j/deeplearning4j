package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.FullConv3DConfig;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * FullConv3D operation
 */
@Slf4j
public class FullConv3D extends DynamicCustomOp {
    protected FullConv3DConfig conv3DConfig;

    @Builder(builderMethodName = "builder")
    public FullConv3D(SameDiff sameDiff, DifferentialFunction[] inputFunctions, INDArray[] inputs, INDArray[] outputs, FullConv3DConfig conv3DConfig) {
        super(null,sameDiff, inputFunctions, false);
        this.conv3DConfig = conv3DConfig;
        if(inputs != null) {
            getInputArguments().addAll(Arrays.asList(inputs));
        }

        if(outputs != null) {
            getOutputArguments().addAll(Arrays.asList(outputs));
        }

        addArgs();
    }


    public FullConv3D() {}



    private void addArgs() {
        getIArguments().add(conv3DConfig.getDT());
        getIArguments().add(conv3DConfig.getDW());
        getIArguments().add(conv3DConfig.getDH());
        getIArguments().add(conv3DConfig.getPT());
        getIArguments().add(conv3DConfig.getPW());
        getIArguments().add(conv3DConfig.getPH());
        getIArguments().add(conv3DConfig.getDilationT());
        getIArguments().add(conv3DConfig.getDilationW());
        getIArguments().add(conv3DConfig.getDilationH());
        getIArguments().add(conv3DConfig.getAT());
        getIArguments().add(conv3DConfig.getAW());
        getIArguments().add(conv3DConfig.getAH());
        getIArguments().add(fromBoolean(conv3DConfig.isBiasUsed()));


    }

    @Override
    public String opName() {
        return "fullconv3d";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.addAll(f1);
        List<DifferentialFunction> ret = new ArrayList<>();
        FullConv3DDerivative fullConv3DDerivative = FullConv3DDerivative.derivativeBuilder()
                .conv3DConfig(conv3DConfig)
                .sameDiff(sameDiff)
                .inputFunctions(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .build();
        ret.addAll(Arrays.asList(fullConv3DDerivative.getOutputFunctions()));
        return ret;
    }

}
