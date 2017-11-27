package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;

import java.util.List;


/**
 * Im2col operation
 */
public class Im2col extends DynamicCustomOp {

    protected Conv2DConfig conv2DConfig;

    @Builder(builderMethodName = "builder")
    public Im2col(SameDiff sameDiff, DifferentialFunction[] inputFunctions, INDArray[] inputArrays, INDArray[] outputs, Conv2DConfig conv2DConfig) {
        super(null,inputArrays,outputs);
        this.sameDiff = sameDiff;
        sameDiff.associateFunctionsAsArgs(inputFunctions,this);
        this.conv2DConfig = conv2DConfig;

        addArgs();
    }

    public Im2col() {}

    protected void addArgs() {
        getIArguments().add(conv2DConfig.getKh());
        getIArguments().add(conv2DConfig.getKw());
        getIArguments().add(conv2DConfig.getSy());
        getIArguments().add(conv2DConfig.getSx());
        getIArguments().add(conv2DConfig.getPh());
        getIArguments().add(conv2DConfig.getPw());
        getIArguments().add(conv2DConfig.getDh());
        getIArguments().add(conv2DConfig.getDw());
        getIArguments().add(fromBoolean(conv2DConfig.isSameMode()));

    }



    @Override
    public String opName() {
        return "im2col";
    }




    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new UnsupportedOperationException("Unable to run derivative on im2col op");
    }
}
