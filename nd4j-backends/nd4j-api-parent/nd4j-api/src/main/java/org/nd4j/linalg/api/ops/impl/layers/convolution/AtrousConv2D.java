package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;


/**
 * An alias for tensorflow's atrous
 * convolution (vs being built in to pytorch)
 * operation
 */
@Slf4j
@Getter
public class AtrousConv2D extends Conv2D {


    @Builder(builderMethodName = "atrousBuilder")
    public AtrousConv2D(SameDiff sameDiff,
                        SDVariable[] inputFunctions,
                        INDArray[] inputArrays, INDArray[] outputs,
                        Conv2DConfig conv2DConfig) {
        super(sameDiff, inputFunctions, inputArrays, outputs, conv2DConfig);
        this.sameDiff = sameDiff;

        this.conv2DConfig = conv2DConfig;
        addArgs();
    }

    public AtrousConv2D() {
    }

    protected void addArgs() {
        addIArgument(conv2DConfig.getKh());
        addIArgument(conv2DConfig.getKw());
        addIArgument(conv2DConfig.getSy());
        addIArgument(conv2DConfig.getSx());
        addIArgument(conv2DConfig.getPh());
        addIArgument(conv2DConfig.getPw());
        addIArgument(conv2DConfig.getDh());
        addIArgument(conv2DConfig.getDw());
        addIArgument(ArrayUtil.fromBoolean(conv2DConfig.isSameMode()));

    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val aStrides = nodeDef.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();
        val sY = tfStrides.get(1);
        val sX = tfStrides.get(2);

        val aPadding = nodeDef.getAttrOrDefault("padding", null);

        val paddingMode = aPadding.getS().toStringUtf8();

        // we know that second input to conv2d is weights array
        //val tensorProto = mapper.getTensorFrom(attributesForNode.get("input"),graph);
        // val kY =tensorProto.getTensorShape().getDim(0).getSize();
        // val kX = tensorProto.getTensorShape().getDim(1).getSize();
        // val kY =tensorProto.getTensorShape().getDim(0).getSize();
        //val kX = tensorProto.getTensorShape().getDim(1).getSize();
        val kY = nodeDef.getAttrOrThrow("shape").getShape().getDim(0).getSize();
        val kX = nodeDef.getAttrOrThrow("shape").getShape().getDim(1).getSize();


        val rate = attributesForNode.get("rate").getI();

        //   variable.setArray(variable.getArray().permute(3, 2, 0, 1).dup('c'));

        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");
        Conv2DConfig conv2DConfig = Conv2DConfig.builder()
                .kh((int) kY)
                .kw((int) kX)
                .sx(sX.intValue())
                .sy(sY.intValue())
                .dw((int) rate)
                .dh((int) rate)
                .isSameMode(isSameMode)
                .build();
        this.conv2DConfig = conv2DConfig;

    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return conv2DConfig.toProperties();
    }

    @Override
    public String opName() {
        return "atrous_conv2d";
    }


    @Override
    public String onnxName() {
        return "Conv";
    }

    @Override
    public String tensorflowName() {
        return "Dilation2D";
    }
}
