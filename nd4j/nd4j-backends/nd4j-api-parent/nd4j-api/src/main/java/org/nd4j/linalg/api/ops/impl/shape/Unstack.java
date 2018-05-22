package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Unstack op conversion
 *
 * @author raver119@gmail.com
 */
public class Unstack extends DynamicCustomOp {

    // TODO: libnd4j currently doesn't support "num", number of outputs is inferred.
    //    private int num;
    private int axis;

    public Unstack() {
    }

    public Unstack(SameDiff sameDiff, SDVariable value, int axis) {
        super(null, sameDiff, new SDVariable[]{value}, false);
        this.axis = axis;
        addArgs();
    }

    public void addArgs() {
        addIArgument(axis);
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"Unstack", "Unpack"};
    }

    @Override
    public String tensorflowName() {
        return "Unstack";
    }


    @Override
    public String opName() {
        return "unstack";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val attrAxis = nodeDef.getAttrOrThrow("axis");
        int axis = (int) attrAxis.getI();
        this.axis = axis;
        addArgs();
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("axis", axis);
        return ret;
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();

        val axisMapping = PropertyMapping.builder()
                .onnxAttrName("axis")
                .tfInputPosition(-1)
                .propertyNames(new String[]{"axis"})
                .build();

        map.put("axis", axisMapping);

        ret.put(tensorflowName(), map);

        return ret;
    }


    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        throw new UnsupportedOperationException("No analog found for onnx for " + opName());
    }

    /*
    This is required because number of outputs depends on the input shape.
     */
    @Override
    public List<long[]> calculateOutputShape() {
        val ret = new ArrayList<long[]>();
        val inputShape = arg().getShape();
        val outputShape = new long[inputShape.length - 1];
        for (int i = 0; i < axis; i++) {
            outputShape[i] = inputShape[i];
        }
        for (int i = axis + 1; i < inputShape.length; i++) {
            outputShape[i - 1] = inputShape[i];
        }
        for (int i = 0; i < inputShape[axis]; i++) {
            ret.add(outputShape);
        }
        return ret;
    }
}
