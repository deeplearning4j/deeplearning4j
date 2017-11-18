package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Pooling3D operation
 */
@Slf4j
public class Pooling3D extends DynamicCustomOp {
    protected Pooling3DConfig config;

    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }


    public Pooling3D() {}

    @Builder(builderMethodName = "builder")
    public Pooling3D(SameDiff sameDiff, DifferentialFunction[] inputs,INDArray[] inputArrays, INDArray[] outputs,boolean inPlace,Pooling3DConfig pooling3DConfig) {
        super(null,sameDiff, inputs, inPlace);
        this.config = pooling3DConfig;

        if(inputArrays != null) {
            getInputArguments().addAll(Arrays.asList(inputArrays));
        }

        if(outputs != null) {
            getOutputArguments().addAll(Arrays.asList(outputs));
        }
        addArgs();
    }


    private void addArgs() {
        getIArguments().add(config.getKT());
        getIArguments().add(config.getKW());
        getIArguments().add(config.getKH());
        getIArguments().add(config.getDT());
        getIArguments().add(config.getDW());
        getIArguments().add(config.getDH());
        getIArguments().add(config.getPT());
        getIArguments().add(config.getPW());
        getIArguments().add(config.getPH());
        getIArguments().add(config.getDilationT());
        getIArguments().add(config.getDilationW());
        getIArguments().add(config.getDilationH());

    }

    @Override
    public String opName() {
        return getPoolingPrefix() + "pool3d";
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Pooling3DDerivative pooling3DDerivative = Pooling3DDerivative.derivativeBuilder()
                .inPlace(inPlace)
                .sameDiff(sameDiff)
                .inputs(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .pooling3DConfig(config)
                .build();
        ret.addAll(Arrays.asList(pooling3DDerivative.getOutputFunctions()));

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

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for op " + opName());
    }

    @Override
    public String tensorflowName() {
      throw new NoOpNameFoundException("No op opName found for op " + opName());
    }

}
