package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Pooling2D operation
 */
@Slf4j
@Getter
public class Pooling2D extends DynamicCustomOp {

    protected Pooling2DConfig config;

    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }

    public Pooling2D() {}

    @Builder(builderMethodName = "builder")
    @SuppressWarnings("Used in lombok")
    public Pooling2D(SameDiff sameDiff, DifferentialFunction[] inputs,INDArray[] arrayInputs, INDArray[] arrayOutputs,Pooling2DConfig config) {
        super(null,sameDiff, inputs, false);
       if(arrayInputs != null) {
           getInputArguments().addAll(Arrays.asList(arrayInputs));
       }

       if(arrayOutputs != null) {
           getOutputArguments().addAll(Arrays.asList(arrayOutputs));
       }

       this.config = config;


        addArgs();
    }


    private void addArgs() {
        getIArguments().add(config.getKh());
        getIArguments().add(config.getKw());
        getIArguments().add(config.getSy());
        getIArguments().add(config.getSx());
        getIArguments().add(config.getPh());
        getIArguments().add(config.getPw());
        getIArguments().add(config.getDh());
        getIArguments().add(config.getDw());
        getIArguments().add(fromBoolean(config.isSameMode()));
        getIArguments().add((int) config.getExtra());

    }

    @Override
    public String opName() {
        return getPoolingPrefix() + "pool2d";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Pooling2DDerivative pooling2DDerivative = Pooling2DDerivative.derivativeBuilder()
                .inputs(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .sameDiff(sameDiff)
                .config(config)
                .build();
        ret.addAll(Arrays.asList(pooling2DDerivative.getOutputFunctions()));
        return ret;
    }

    public String getPoolingPrefix() {
        switch(config.getType()) {
            case AVG:return "avg";
            case MAX: return "max";
            case PNORM: return "pnorm";
            default: throw new IllegalStateException("No pooling type found.");
        }
    }

}
