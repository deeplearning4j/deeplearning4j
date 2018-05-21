package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Upsampling operation
 */
@Slf4j
@Getter
public class Upsampling extends DynamicCustomOp {


    protected int scaleFactor;

    @Builder(builderMethodName = "sameDiffBuilder")
    public Upsampling(SameDiff sameDiff, SDVariable[] inputs,INDArray[] inputArrays, INDArray[] outputs,boolean inPlace, int scaleFactor) {
        super(null,sameDiff, inputs, inPlace);
        this.scaleFactor = scaleFactor;
        if(inputArrays != null) {
            addInputArgument(inputArrays);
        }

        if(outputs != null) {
            addOutputArgument(outputs);
        }

        addIArgument(scaleFactor);
    }


    public Upsampling() {}


    @Override
    public String opName() {
        return "upsampling2d";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<SDVariable> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        UpsamplingDerivative conv2DDerivative = UpsamplingDerivative.derivativeBuilder()
                .scaleFactor(scaleFactor)
                .inputs(inputs.toArray(new SDVariable[inputs.size()]))
                .build();
        ret.addAll(Arrays.asList(conv2DDerivative.outputVariables()));
        return ret;
    }

}
