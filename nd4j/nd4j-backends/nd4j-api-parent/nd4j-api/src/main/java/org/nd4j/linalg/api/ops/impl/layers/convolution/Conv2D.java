package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.ConditionalFieldValueIntIndexArrayAdapter;
import org.nd4j.imports.descriptors.properties.adapters.ConditionalFieldValueNDArrayShapeAdapter;
import org.nd4j.imports.descriptors.properties.adapters.SizeThresholdIntArrayIntIndexAdpater;
import org.nd4j.imports.descriptors.properties.adapters.StringEqualsAdapter;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;


/**
 * Conv2D operation
 */
@Slf4j
@Getter
@NoArgsConstructor
public class Conv2D extends DynamicCustomOp {

    protected Conv2DConfig config;

    @Builder(builderMethodName = "builder")
    public Conv2D(SameDiff sameDiff,
                  SDVariable[] inputFunctions,
                  INDArray[] inputArrays, INDArray[] outputs,
                  Conv2DConfig config) {
        super(null, inputArrays, outputs);
        this.sameDiff = sameDiff;
        this.config = config;

        addArgs();
        sameDiff.putFunctionForId(this.getOwnName(), this);    //Normally called in DynamicCustomOp constructor, via setInstanceId - but sameDiff field is null at that point
        sameDiff.addArgsFor(inputFunctions, this);
    }

    protected void addArgs() {
        addIArgument(config.getkH(),
                config.getkW(),
                config.getsH(),
                config.getsW(),
                config.getpH(),
                config.getpW(),
                config.getdH(),
                config.getdW(),
                ArrayUtil.fromBoolean(config.isSameMode()),
                ArrayUtil.fromBoolean(config.isNHWC()));
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    @Override
    public Object getValue(Field property) {
        if (config == null) {
            config = Conv2DConfig.builder().build();
        }

        return config.getValue(property);
    }

    @Override
    public void setValueFor(Field target, Object value) {
        config.setValueFor(target, value);
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return config.toProperties();
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
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
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        OnnxGraphMapper.getInstance().initFunctionFromProperties(node.getOpType(), this, attributesForNode, node, graph);
        addArgs();
    }


    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String, Map<String, AttributeAdapter>> ret = new HashMap<>();
        Map<String, AttributeAdapter> tfMappings = new LinkedHashMap<>();
        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(this);


        tfMappings.put("kH", new ConditionalFieldValueNDArrayShapeAdapter("NCHW", 2, 0, fields.get("dataFormat")));
        tfMappings.put("kW", new ConditionalFieldValueNDArrayShapeAdapter("NCHW", 3, 1, fields.get("dataFormat")));
        tfMappings.put("sH", new ConditionalFieldValueIntIndexArrayAdapter("NCHW", 2, 1, fields.get("dataFormat")));
        tfMappings.put("sW", new ConditionalFieldValueIntIndexArrayAdapter("NCHW", 3, 2, fields.get("dataFormat")));
        tfMappings.put("isSameMode", new StringEqualsAdapter("SAME"));
        tfMappings.put("isNHWC", new StringEqualsAdapter("NHWC"));


        Map<String, AttributeAdapter> onnxMappings = new HashMap<>();
        onnxMappings.put("kH", new SizeThresholdIntArrayIntIndexAdpater(0, 2, 0));
        onnxMappings.put("kW", new SizeThresholdIntArrayIntIndexAdpater(1, 2, 0));
        onnxMappings.put("dH", new SizeThresholdIntArrayIntIndexAdpater(0, 2, 0));
        onnxMappings.put("dW", new SizeThresholdIntArrayIntIndexAdpater(1, 2, 0));
        onnxMappings.put("sH", new SizeThresholdIntArrayIntIndexAdpater(0, 2, 0));
        onnxMappings.put("sW", new SizeThresholdIntArrayIntIndexAdpater(1, 2, 0));
        onnxMappings.put("isSameMode", new StringEqualsAdapter("SAME"));
        onnxMappings.put("isNHWC", new StringEqualsAdapter("NHWC"));

        ret.put(tensorflowName(), tfMappings);
        ret.put(onnxName(), onnxMappings);
        return ret;
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();
        val strideMapping = PropertyMapping.builder()
                .tfAttrName("strides")
                .onnxAttrName("strides")
                .propertyNames(new String[]{"sW", "sH"})
                .build();


        val kernelMappingH = PropertyMapping.builder()
                .propertyNames(new String[]{"kH"})
                .tfInputPosition(1)
                .shapePosition(0)
                .onnxAttrName("kernel_shape")
                .build();

        val kernelMappingW = PropertyMapping.builder()
                .propertyNames(new String[]{"kW"})
                .tfInputPosition(1)
                .shapePosition(1)
                .onnxAttrName("kernel_shape")
                .build();

        val dilationMapping = PropertyMapping.builder()
                .onnxAttrName("dilations")
                .propertyNames(new String[]{"dW", "dH"})
                .tfAttrName("rates")
                .build();

        val dataFormat = PropertyMapping.builder()
                .onnxAttrName("data_format")
                .tfAttrName("data_format")
                .propertyNames(new String[]{"dataFormat"})
                .build();

        val nhwc = PropertyMapping.builder()
                .onnxAttrName("data_format")
                .tfAttrName("data_format")
                .propertyNames(new String[]{"isNHWC"})
                .build();

        val sameMode = PropertyMapping.builder()
                .onnxAttrName("auto_pad")
                .propertyNames(new String[]{"isSameMode"})
                .tfAttrName("padding")
                .build();

        val paddingWidthHeight = PropertyMapping.builder()
                .onnxAttrName("padding")
                .propertyNames(new String[]{"pH", "pW"})
                .build();


        map.put("sW", strideMapping);
        map.put("sH", strideMapping);
        map.put("kH", kernelMappingH);
        map.put("kW", kernelMappingW);
        map.put("dW", dilationMapping);
        map.put("dH", dilationMapping);
        map.put("isSameMode", sameMode);
        map.put("pH", paddingWidthHeight);
        map.put("pW", paddingWidthHeight);
        map.put("dataFormat", dataFormat);
        map.put("isNHWC", nhwc);

        try {
            ret.put(onnxName(), map);
        } catch (NoOpNameFoundException e) {
            //ignore
        }


        try {
            ret.put(tensorflowName(), map);
        } catch (NoOpNameFoundException e) {
            //ignore
        }

        return ret;
    }


    @Override
    public String opName() {
        return "conv2d";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> inputs = new ArrayList<>(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Conv2DDerivative conv2DDerivative = Conv2DDerivative.derivativeBuilder()
                .sameDiff(sameDiff)
                .config(config)
                .outputs(outputArguments())
                .inputFunctions(inputs.toArray(new SDVariable[inputs.size()]))
                .build();
        List<SDVariable> ret = Arrays.asList(conv2DDerivative.outputVariables());
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

    @Override
    public String[] tensorflowNames() {
        return new String[]{"Conv2D"};
    }
}
