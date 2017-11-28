package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
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
        if(inputFunctions != null && sameDiff != null)
            sameDiff.associateFunctionsAsArgs(inputFunctions,this);
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
    public void initWithArrays(Map<String, INDArray> arrayMap) {
        val var = sameDiff.getVariableForVertexId(vertexId);
        //place holder variable
        if (var.getArr() == null) {
            //assuming the array hasn't been initialized, setup the config
            //resolving the place holder variable.
            INDArray array = arrayMap.get(var.getVarName());
            array = (array.permute(3, 2, 0, 1).dup('c'));
            sameDiff.updateVariable(var.getVarName(), array);
            conv2DConfig.setKh(array.size(0));
            conv2DConfig.setKw(array.size(1));
        }
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val aStrides = nodeDef.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();
        val sY = tfStrides.get(1);
        val sX = tfStrides.get(2);

        val aPadding = nodeDef.getAttrOrDefault("padding", null);

        val paddingMode = aPadding.getS().toStringUtf8();
        int kY = 1;
        int kX = 1;
        val args = args();
        INDArray arr = sameDiff.getVariableForVertexId(args[1].resultVertexId()).getArr();
        if(arr == null) {
            arr = TFGraphMapper.getInstance().getNDArrayFromTensor(nodeDef.getInput(0), nodeDef, graph);
        }

        kY = arr.size(0);
        kX = arr.size(1);
        arr = (arr.permute(3, 2, 0, 1).dup('c'));
        val  varForOp = initWith.getVariableForVertexId(args[1].resultVertexId());
        initWith.associateArrayWithVariable(arr, varForOp);


        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");
        Conv2DConfig conv2DConfig = Conv2DConfig.builder()
                .kh(kY)
                .kw(kX)
                .sx(sX.intValue())
                .sy(sY.intValue())
                .isSameMode(isSameMode)
                .build();
        this.conv2DConfig = conv2DConfig;
        addArgs();

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val autoPad = !attributesForNode.containsKey("auto_pad") ? "VALID" : attributesForNode.get("auto_pad").getS().toStringUtf8();
        val dilations = attributesForNode.get("dilations");
        val dilationY = dilations == null ? 1 : dilations.getIntsList().get(0).intValue();
        val dilationX = dilations == null ? 1 : dilations.getIntsList().get(1).intValue();
        val group = attributesForNode.get("group");

        val kernelShape = attributesForNode.get("kernel_shape");
        int kY = kernelShape.getIntsList().get(0).intValue();
        int kX = kernelShape.getIntsList().size() < 2 ? kY : kernelShape.getIntsList().get(1).intValue();



        INDArray arr = sameDiff.getVariableForVertexId(vertexId).getArr();
        arr = (arr.permute(3, 2, 0, 1).dup('c'));
        initWith.associateArrayWithVariable(arr, initWith.getVariableForVertexId(vertexId));



        val strides = attributesForNode.get("strides");
        val sY = strides.getIntsList().get(0);
        val sX = strides.getIntsList().size() < 2 ? sY : strides.getIntsList().get(1);
        boolean isSameMode = autoPad
                .equalsIgnoreCase("SAME");
        Conv2DConfig conv2DConfig = Conv2DConfig.builder()
                .dh(dilationY)
                .dw(dilationX)
                .kh(kY)
                .kw(kX)
                .sx(sX.intValue())
                .sy(sY.intValue())
                .isSameMode(isSameMode)
                .build();
        this.conv2DConfig = conv2DConfig;
        addArgs();


    }

    @Override
    public String opName() {
        return "conv2d";
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
        return "Conv2D";
    }
}
