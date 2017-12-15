package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
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
    public Pooling3D(SameDiff sameDiff, SDVariable[] inputs,INDArray[] inputArrays, INDArray[] outputs,boolean inPlace,Pooling3DConfig pooling3DConfig) {
        super(null,sameDiff, inputs, inPlace);
        this.config = pooling3DConfig;

        if(inputArrays != null) {
            addInputArgument(inputArrays);
        }

        if(outputs != null) {
            addOutputArgument(outputs);
        }
        addArgs();
    }


    private void addArgs() {
        addIArgument(config.getKT());
        addIArgument(config.getKW());
        addIArgument(config.getKH());
        addIArgument(config.getDT());
        addIArgument(config.getDW());
        addIArgument(config.getDH());
        addIArgument(config.getPT());
        addIArgument(config.getPW());
        addIArgument(config.getPH());
        addIArgument(config.getDilationT());
        addIArgument(config.getDilationW());
        addIArgument(config.getDilationH());

    }

    @Override
    public String opName() {
        return getPoolingPrefix() + "pool3d";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<SDVariable> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Pooling3DDerivative pooling3DDerivative = Pooling3DDerivative.derivativeBuilder()
                .inPlace(inPlace)
                .sameDiff(sameDiff)
                .inputs(inputs.toArray(new SDVariable[inputs.size()]))
                .pooling3DConfig(config)
                .build();
        ret.addAll(Arrays.asList(pooling3DDerivative.outputVariables()));

        return ret;
    }

    public String getPoolingPrefix() {
        if (config == null)
            return "pooling3d";

        switch(config.getType()) {
            case AVG:return "avg";
            case MAX: return "max";
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
