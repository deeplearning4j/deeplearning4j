package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.IntArrayIntIndexAdpater;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * Depthwise convolution operation.
 *
 */
@NoArgsConstructor
public class DepthwiseConv2dNative extends Conv2D {

    @Builder(builderMethodName = "depthWiseBuilder")
    public DepthwiseConv2dNative(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputArrays, INDArray[] outputs,
                                 Conv2DConfig conv2DConfig) {
        super(sameDiff, inputFunctions, inputArrays, outputs, conv2DConfig);
    }


    @Override
    public String opName() {
        return "DepthwiseConv2dNative".toLowerCase();
    }



    @Override
    public String tensorflowName() {
        return "DepthwiseConv2dNative";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
    }


    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String, Map<String, AttributeAdapter>> ret = new HashMap<>();
        Map<String, AttributeAdapter> tfMappings = new LinkedHashMap<>();
        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(this);
        tfMappings.put("sx",new IntArrayIntIndexAdpater(0));
        tfMappings.put("sy",new IntArrayIntIndexAdpater(1));
        tfMappings.put("ph",new IntArrayIntIndexAdpater(0));
        tfMappings.put("pw",new IntArrayIntIndexAdpater(1));
        ret.put(tensorflowName(), tfMappings);
        return ret;
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();
        val strideMapping = PropertyMapping.builder()
                .tfAttrName("strides")
                .onnxAttrName("strides")
                .propertyNames(new String[]{"sx", "sy"})
                .build();


        val paddingWidthHeight = PropertyMapping.builder()
                .onnxAttrName("padding")
                .propertyNames(new String[]{"ph", "pw"})
                .build();

        map.put("sx", strideMapping);
        map.put("sy", strideMapping);
        map.put("ph", paddingWidthHeight);
        map.put("pw", paddingWidthHeight);
        try {
            ret.put(tensorflowName(), map);
        } catch (NoOpNameFoundException e) {
            //ignore
        }

        return ret;

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
    public void setValueFor(Field target, Object value) {
        config.setValueFor(target,value);
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return super.propertiesForFunction();
    }
}
