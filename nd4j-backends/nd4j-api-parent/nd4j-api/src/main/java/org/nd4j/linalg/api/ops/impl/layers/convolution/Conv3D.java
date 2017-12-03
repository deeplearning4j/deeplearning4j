package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Conv3D operation
 */
@Slf4j
@Getter
public class Conv3D extends DynamicCustomOp {

    protected Conv3DConfig config;

    public Conv3D() {}

    @Builder(builderMethodName = "builder")
    public Conv3D(SameDiff sameDiff, DifferentialFunction[] inputFunctions,INDArray[] inputs, INDArray[] outputs,Conv3DConfig conv3DConfig) {
        super(null,sameDiff, inputFunctions, false);
        setSameDiff(sameDiff);
        if(inputFunctions != null)
            sameDiff.associateFunctionsAsArgs(inputFunctions,this);
        if(inputs != null)
            addInputArgument(inputs);
        if(outputs != null)
            addOutputArgument(outputs);
        this.config = conv3DConfig;
        addArgs();

    }


    private void addArgs() {
        addIArgument(new int[]{getConfig().getDT(),
        getConfig().getDW(),
        getConfig().getDH(),
        getConfig().getPT(),
        getConfig().getPW(),
        getConfig().getPH(),
        getConfig().getDilationT(),
        getConfig().getDilationW(),
        getConfig().getDilationH(),
        getConfig().getAT(),
        getConfig().getAW(),
        getConfig().getAH(),
        fromBoolean(getConfig().isBiasUsed())});

    }

    @Override
    public String opName() {
        return "conv3d";
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Conv3DDerivative conv3DDerivative = Conv3DDerivative.derivativeBuilder()
               .conv3DConfig(config)
                .inputFunctions(args())
                .outputs(outputArguments())
                .inputFunctions(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .sameDiff(sameDiff)
                .build();
        ret.addAll(Arrays.asList(conv3DDerivative.outputFunctions()));
        return ret;
    }



    @Override
    public String onnxName() {
        return "Conv";
    }

    @Override
    public String tensorflowName() {
        return "Conv3D";
    }
}
