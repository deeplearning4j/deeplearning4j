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
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val aStrides = nodeDef.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();
        val sY = tfStrides.get(1);
        val sX = tfStrides.get(2);

        val aPadding = nodeDef.getAttrOrDefault("padding", null);

        val paddingMode = aPadding.getS().toStringUtf8();

        val kY = 0;
        val kX = 0;
      /*  val arr = TFGraphMapper.getInstance().getNDArrayFromTensor("input",nodeDef,graph);
        val kY = arr.size(0);
        val kX = arr.size(1);

        arr.assign(arr.permute(3, 2, 0, 1).dup('c'));
*/
        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");
        Conv2DConfig conv2DConfig = Conv2DConfig.builder()
                .kh((int) kY)
                .kw((int) kX)
                .sx(sX.intValue())
                .sy(sY.intValue())
                .isSameMode(isSameMode)
                .build();
        this.conv2DConfig = conv2DConfig;
        addArgs();

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val autoPad = attributesForNode.get("auto_pad");
        val dilations = attributesForNode.get("dilations");
        val dilationY = dilations.getIntsList().get(0);
        val dilationX = dilations.getIntsList().get(1);
        val group = attributesForNode.get("group");

        val kernelShape = attributesForNode.get("kernel_shape");
        val kY = kernelShape.getIntsList().get(0);
        val kX = kernelShape.getIntsList().get(1);


        val strides = attributesForNode.get("strides");
        val sY = strides.getIntsList().get(0);
        val sX = strides.getIntsList().get(1);
        boolean isSameMode = autoPad.getS().toStringUtf8()
                .equalsIgnoreCase("SAME");
        Conv2DConfig conv2DConfig = Conv2DConfig.builder()
                .dh(dilationY.intValue())
                .dw(dilationX.intValue())
                .kh(kY.intValue())
                .kw(kX.intValue())
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
