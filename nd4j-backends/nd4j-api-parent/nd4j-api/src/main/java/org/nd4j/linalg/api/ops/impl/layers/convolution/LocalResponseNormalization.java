package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;


/**
 * LocalResponseNormalization operation
 */
@Slf4j
@Getter
public class LocalResponseNormalization extends DynamicCustomOp {



    protected LocalResponseNormalizationConfig config;


    @Builder(builderMethodName = "builder")
    public LocalResponseNormalization(SameDiff sameDiff, DifferentialFunction[] inputFunctions,INDArray[] inputs, INDArray[] outputs,boolean inPlace,LocalResponseNormalizationConfig config) {
        super(null,sameDiff, inputFunctions, inPlace);
        this.config = config;
        if(inputs != null) {
            getInputArguments().addAll(Arrays.asList(inputs));
        }

        if(outputs!= null) {
            getOutputArguments().addAll(Arrays.asList(outputs));
        }

        addArgs();
    }


    public LocalResponseNormalization() {}


    private void addArgs() {
        getTArguments().add(config.getAlpha());
        getTArguments().add(config.getBeta());
        getTArguments().add(config.getBias());
        getTArguments().add(config.getDepth());
    }

    @Override
    public String opName() {
        return "lrn";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

        val aAlpha = nodeDef.getAttrOrThrow("alpha");
        val aBeta = nodeDef.getAttrOrThrow("beta");
        val aBias = nodeDef.getAttrOrThrow("bias");
        val aDepth = nodeDef.getAttrOrThrow("depth_radius");

        val alpha = aAlpha.getF();
        val beta = aBeta.getF();
        val bias = aBias.getF();
        val depth = aDepth.getF();

        LocalResponseNormalizationConfig localResponseNormalizationConfig = LocalResponseNormalizationConfig.builder()
                .alpha(alpha)
                .beta(beta)
                .bias(bias)
                .depth(depth)
                .build();
        this.config = localResponseNormalizationConfig;
        addArgs();
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val aAlpha = attributesForNode.get("alpha");
        val aBeta = attributesForNode.get("beta");
        val aBias = attributesForNode.get("bias");
        val aDepth = attributesForNode.get("size");

        val alpha = aAlpha.getF();
        val beta = aBeta.getF();
        val bias = aBias.getF();
        val depth = aDepth.getF();

        LocalResponseNormalizationConfig localResponseNormalizationConfig = LocalResponseNormalizationConfig.builder()
                .alpha(alpha)
                .beta(beta)
                .bias(bias)
                .depth(depth)
                .build();
        this.config = localResponseNormalizationConfig;
        addArgs();
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        LocalResponseNormalizationDerivative localResponseNormalizationDerivative = LocalResponseNormalizationDerivative.derivativeBuilder()
                .inPlace(inPlace)
                .sameDiff(sameDiff)
                .inputFunctions(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .config(config)
                .build();
        ret.addAll(Arrays.asList(localResponseNormalizationDerivative.getOutputFunctions()));

        return ret;
    }

    @Override
    public String onnxName() {
        return "LRN";
    }

    @Override
    public String tensorflowName() {
        return "LRN";
    }

}
