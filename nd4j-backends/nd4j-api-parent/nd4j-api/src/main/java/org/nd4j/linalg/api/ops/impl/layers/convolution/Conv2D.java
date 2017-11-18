package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;


/**
 * Conv2D operation
 */
@Slf4j
@Getter
public class Conv2D extends DynamicCustomOp {

   protected  Conv2DConfig conv2DConfig;

    @Builder(builderMethodName = "builder")
    public Conv2D(SameDiff sameDiff,
                  DifferentialFunction[] inputFunctions,
                  INDArray[] inputArrays, INDArray[] outputs,
                  Conv2DConfig conv2DConfig) {
        super(null,inputArrays,outputs);
        this.sameDiff = sameDiff;
        this.args = inputFunctions;
        this.conv2DConfig = conv2DConfig;
        addArgs();
    }

    public Conv2D() {}

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
        return "conv2d";
    }

    @Override
    public TOp asIntermediateRepresentation(OnnxProto3.NodeProto node, TGraph graph, Map<String, OnnxProto3.AttributeProto> attributesForNode) {
        return super.asIntermediateRepresentation(node, graph, attributesForNode);
    }

    @Override
    public TOp asIntermediateRepresentation(NodeDef tfNode, TGraph graph) {
        val tNode = buildBasicNode(tfNode, graph);

        val aStrides = tfNode.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();
        val sY = tfStrides.get(1);
        val sX = tfStrides.get(2);

        val aPadding = tfNode.getAttrOrDefault("padding", null);

        val paddingMode = aPadding.getS().toStringUtf8();

        // we know that second input to conv2d is weights array
        val weightsIndex = tNode.getInputs().get(1);
        val variable = graph.getVariableSpace().getVariable(weightsIndex);

        val kY = variable.getArray().size(0);
        val kX = variable.getArray().size(1);

        variable.setArray(variable.getArray().permute(3, 2, 0, 1).dup('c'));

        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

        if (!isSameMode)
            log.debug("Mode: {}", paddingMode);

        log.debug("Conv2D: k: [{}, {}]; s: [{}, {}]; padding: {}", kY, kX, sY, sX,  paddingMode);

        tNode.getOpState().setExtraBits(new int[] {kY, kX, sY.intValue(), sX.intValue(), 0, 0, 1, 1, isSameMode ? 1 : 0});

        return tNode;
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Conv2DDerivative conv2DDerivative = Conv2DDerivative.derivativeBuilder()
                .conv2DConfig(conv2DConfig)
                .outputs(getOutputArguments().toArray(new INDArray[getOutputArguments().size()]))
                .inputFunctions(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .build();
        ret.addAll(Arrays.asList(conv2DDerivative.getOutputFunctions()));
        return ret;
    }


    @Override
    public String onnxName() {
        return "Conv";
    }

    @Override
    public String tensorflowName() {
        return "conv2d";
    }
}
