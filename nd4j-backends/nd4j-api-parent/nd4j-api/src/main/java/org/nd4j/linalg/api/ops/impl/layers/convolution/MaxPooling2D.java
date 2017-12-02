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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;


/**
 * Pooling2D operation
 */
@Slf4j
@Getter
public class MaxPooling2D extends DynamicCustomOp {

    protected Pooling2DConfig config;

    public MaxPooling2D() {}

    @Builder(builderMethodName = "builder")
    @SuppressWarnings("Used in lombok")
    public MaxPooling2D(SameDiff sameDiff, DifferentialFunction[] inputs, INDArray[] arrayInputs, INDArray[] arrayOutputs, Pooling2DConfig config) {
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
        ret.addAll(Arrays.asList(pooling2DDerivative.outputFunctions()));
        return ret;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val aStrides = nodeDef.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();
        val sY = tfStrides.get(1);
        val sX = tfStrides.get(2);

        val aKernels = nodeDef.getAttrOrThrow("ksize");
        val tfKernels = aKernels.getList().getIList();

        val kY = tfKernels.get(1);
        val kX = tfKernels.get(2);

        val aPadding = nodeDef.getAttrOrThrow("padding");
        val padding = aPadding.getList().getIList();

        val paddingMode = aPadding.getS().toStringUtf8().replaceAll("\"","");

        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

        int ph = padding.size() == 2 ? padding.get(0).intValue() : 0;
        int pw = padding.size() == 2 ? padding.get(1).intValue() : 0;

        if (!isSameMode)
            log.debug("Mode: {}", paddingMode);

        Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
                .sy(sY.intValue())
                .sx(sX.intValue())
                .type(Pooling2D.Pooling2DType.MAX)
                .isSameMode(isSameMode)
                .kh(kY.intValue())
                .kw(kX.intValue())
                .ph(ph)
                .pw(pw)
                .virtualWidth(1)
                .virtualHeight(1)
                .build();
        this.config = pooling2DConfig;
        addArgs();
        log.debug("Pooling: k: [{},{}]; s: [{}, {}], padding: {}", kY, kX, sY, sX, aPadding);


    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val paddingVal = !attributesForNode.containsKey("auto_pad") ? "VALID" : attributesForNode.get("auto_pad").getS().toStringUtf8();
        val isSameNode = paddingVal.equals("SAME");
        val kernelShape = attributesForNode.get("kernel_shape").getIntsList();
        val padding = attributesForNode.get("pads").getIntsList();
        val strides = attributesForNode.get("strides").getIntsList();

        Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
                .sy(strides.get(0).intValue())
                .sx(strides.size() < 2 ? strides.get(0).intValue() : strides.get(1).intValue())
                .type(Pooling2D.Pooling2DType.MAX)
                .isSameMode(isSameNode)
                .kh(kernelShape.get(0).intValue())
                .kw(kernelShape.size() < 2 ? kernelShape.get(0).intValue() : kernelShape.get(1).intValue())
                .ph(padding.get(0).intValue())
                .pw(padding.size() < 2 ? padding.get(0).intValue() : padding.get(1).intValue())
                .virtualWidth(1)
                .virtualHeight(1)
                .build();
        this.config = pooling2DConfig;
        addArgs();
    }



    @Override
    public String onnxName() {
        return "MaxPool";
    }

    @Override
    public String tensorflowName() {
        return "MaxPool";
    }



    public String getPoolingPrefix() {
        return "max";
    }


}
