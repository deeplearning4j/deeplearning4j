package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Stack operation. Stacks n input tensors along provided axis.
 *
 * @author raver119@gmail.com
 */
public class Stack extends DynamicCustomOp {
    protected int axis;

    public Stack() {
    }

    public Stack(SameDiff sameDiff, SDVariable[] values, int axis) {
        super(null, sameDiff, values, false);
        this.axis = axis;
        addArgs();
    }

    public void addArgs() {
        addIArgument(axis);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "stack";
    }


    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"Pack", "Stack"};
    }

    @Override
    public String opName() {
        return "stack";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        throw new UnsupportedOperationException("No analog found for onnx for " + opName());
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();

        val axisMapping = PropertyMapping.builder()
                .onnxAttrName("axis")
                .tfAttrName("axis")
                .propertyNames(new String[]{"axis"})
                .build();

        map.put("axis", axisMapping);

        for (val name : tensorflowNames())
            ret.put(name, map);

        return ret;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Differentiation not supported yet.");
    }

}
