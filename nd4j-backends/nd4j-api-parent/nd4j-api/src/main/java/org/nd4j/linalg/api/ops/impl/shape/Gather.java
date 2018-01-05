package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.Map;

/**
 * Gather op conversion
 *
 * @author raver119@gmail.com
 */
@NoArgsConstructor
public class Gather extends DynamicCustomOp {

    private int broadcast,axis;


    @Override
    public String onnxName() {
        return "Gather";
    }

    @Override
    public String tensorflowName() {
        throw new ND4JIllegalStateException("No tensorflow op name found for " + opName());
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();
        val broadcast = PropertyMapping.builder()
                .onnxAttrName("broadcast")
                .propertyNames(new String[]{"broadcast"}).build();

        val axis = PropertyMapping.builder()
                .onnxAttrName("axis")
                .propertyNames(new String[]{"axis"}).build();

        map.put("broadcast",broadcast);
        map.put("axis",axis);

        ret.put(onnxName(),map);
        return ret;
    }

    @Override
    public String opName() {
        return "gather";
    }
}
