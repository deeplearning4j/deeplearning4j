package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;


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

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }
    
    /**
     * Divisor mode for average pooling only. 3 modes are supported:
     * MODE_0:
     * EXCLUDE_PADDING:
     * INCLUDE_PADDING: Always do sum(window) / (kH*kW) even if padding is present.
     */
    public enum Divisor {
        EXCLUDE_PADDING, INCLUDE_PADDING
    }

    public Pooling2D() {}

    @Builder(builderMethodName = "builder")
    @SuppressWarnings("Used in lombok")
    public Pooling2D(SameDiff sameDiff, SDVariable[] inputs,INDArray[] arrayInputs, INDArray[] arrayOutputs,Pooling2DConfig config) {
        super(null,sameDiff, inputs, false);
       if(arrayInputs != null) {
           addInputArgument(arrayInputs);
       }

       if(arrayOutputs != null) {
           addOutputArgument(arrayOutputs);
       }

       this.config = config;


        addArgs();
    }

    @Override
    public void setValueFor(Field target, Object value) {
        config.setValueFor(target,value);
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        return config.toProperties();
    }

    private void addArgs() {
        addIArgument(config.getKh());
        addIArgument(config.getKw());
        addIArgument(config.getSy());
        addIArgument(config.getSx());
        addIArgument(config.getPh());
        addIArgument(config.getPw());
        addIArgument(config.getDh());
        addIArgument(config.getDw());
        addIArgument(ArrayUtil.fromBoolean(config.isSameMode()));
        addIArgument((config.getType() == Pooling2DType.AVG) ? config.getDivisor().ordinal() : (int)config.getExtra());
        addIArgument(ArrayUtil.fromBoolean(config.isNHWC()));
    }

    @Override
    public boolean isConfigProperties() {
        return true;
    }

    @Override
    public String configFieldName() {
        return "config";
    }

    @Override
    public String opName() {
        return getPoolingPrefix() + "pool2d";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<SDVariable> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Pooling2DDerivative pooling2DDerivative = Pooling2DDerivative.derivativeBuilder()
                .inputs(inputs.toArray(new SDVariable[inputs.size()]))
                .sameDiff(sameDiff)
                .config(config)
                .build();
        ret.addAll(Arrays.asList(pooling2DDerivative.outputVariables()));
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

        if (!isSameMode)
            log.debug("Mode: {}", paddingMode);

        Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
                .sy(sY.intValue())
                .sx(sX.intValue())
                .type(null)
                .isSameMode(isSameMode)
                .kh(kY.intValue())
                .kw(kX.intValue())
                .ph(padding.get(0).intValue())
                .pw(padding.get(1).intValue())
                .virtualWidth(1)
                .virtualHeight(1)
                .build();
        this.config = pooling2DConfig;
        addArgs();
        log.debug("Pooling: k: [{},{}]; s: [{}, {}], padding: {}", kY, kX, sY, sX, aPadding);


    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val isSameNode = attributesForNode.get("auto_pad").getS().equals("SAME");
        val kernelShape = attributesForNode.get("kernel_shape").getIntsList();
        val padding = attributesForNode.get("pads").getIntsList();
        val strides = attributesForNode.get("strides").getIntsList();

        Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
                .sy(strides.get(0).intValue())
                .sx(strides.get(1).intValue())
                .type(null)
                .isSameMode(isSameNode)
                .kh(kernelShape.get(0).intValue())
                .kw(kernelShape.get(1).intValue())
                .ph(padding.get(0).intValue())
                .pw(padding.get(1).intValue())
                .virtualWidth(1)
                .virtualHeight(1)
                .build();
        this.config = pooling2DConfig;
        addArgs();
    }


    public String getPoolingPrefix() {
        if (config == null)
            return "somepooling";

        switch(config.getType()) {
            case AVG:return "avg";
            case MAX: return "max";
            case PNORM: return "pnorm";
            default: throw new IllegalStateException("No pooling type found.");
        }
    }


    @Override
    public String onnxName() {
        return "Pooling";
    }

    @Override
    public String tensorflowName() {
        return "Pooling2D";
    }
}
