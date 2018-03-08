package org.nd4j.linalg.api.ops.impl.accum;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.BooleanAdapter;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

public class CumProd extends DynamicCustomOp {
    protected boolean exclusive = false;
    protected boolean reverse = false;

    public CumProd() {
    }

    public CumProd(SameDiff sameDiff, SDVariable x, int... dimension) {
        super(null, sameDiff, new SDVariable[]{x});
        this.sameDiff = sameDiff;
        this.dimensions = dimension;
        addArgs();
    }

    public CumProd(SameDiff sameDiff, SDVariable x, boolean exclusive, boolean reverse, int... dimension) {
        super(null, sameDiff, new SDVariable[]{x});
        this.sameDiff = sameDiff;
        this.dimensions = dimension;
        this.exclusive = exclusive;
        this.reverse = reverse;
        addArgs();
    }

    @Override
    public String opName() {
        return "cumprod";
    }


    @Override
    public String tensorflowName() {
        return "Cumprod";
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String, Map<String, AttributeAdapter>> ret = new HashMap<>();
        Map<String, AttributeAdapter> tfMappings = new LinkedHashMap<>();

        tfMappings.put("exclusive", new BooleanAdapter());
        tfMappings.put("reverse", new BooleanAdapter());


        ret.put(tensorflowName(), tfMappings);

        return ret;
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();

        val exclusiveMapper = PropertyMapping.builder()
                .tfAttrName("exclusive")
                .propertyNames(new String[]{"exclusive"})
                .build();

        val reverseMapper = PropertyMapping.builder()
                .tfAttrName("reverse")
                .propertyNames(new String[]{"reverse"})
                .build();


        map.put("exclusive", exclusiveMapper);
        map.put("reverse", reverseMapper);

        ret.put(tensorflowName(), map);

        return ret;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    protected void addArgs() {
        addIArgument(exclusive ? 1 : 0, reverse ? 1 : 0);
        if (dimensions != null && dimensions.length > 0)
            addIArgument(dimensions);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        // Output gradient is the reversed cumulative product of the reversed input gradient
        SDVariable gradient = sameDiff.setupFunction(grad.get(0));

        SDVariable reverseGrad = sameDiff.reverse(gradient, 1 - dimensions[0]);
        SDVariable ret = sameDiff.cumprod(reverseGrad, exclusive, reverse, dimensions);
        SDVariable reversedRet = sameDiff.reverse(ret, 1 - dimensions[0]);
        return Arrays.asList(reversedRet);
    }
}
