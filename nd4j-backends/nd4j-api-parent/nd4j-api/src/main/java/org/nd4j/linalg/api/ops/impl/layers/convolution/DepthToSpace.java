package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.val;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class DepthToSpace extends DynamicCustomOp {
    private String dataFormat;
    private int blockSize;

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> attrs = new LinkedHashMap<>();

        val dataFormatMapping = PropertyMapping.builder()
                .tfAttrName("data_format")
                .propertyNames(new String[]{"dataFormat"})
                .build();
        attrs.put("dataFormat",dataFormatMapping);

        val blockSize = PropertyMapping.builder()
                .tfAttrName("block_size")
                .propertyNames(new String[]{"blockSize"})
                .build();
        attrs.put("blockSize",blockSize);

        ret.put(tensorflowName(),attrs);
        return ret;
    }

    @Override
    public String opName() {
        return "depthtospace";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[] {"DepthToSpace"};
    }

    @Override
    public String tensorflowName() {
        return "DepthToSpace";
    }
}
