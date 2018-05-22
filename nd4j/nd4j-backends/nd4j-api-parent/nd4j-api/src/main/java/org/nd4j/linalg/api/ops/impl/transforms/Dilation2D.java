package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.*;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Dilation2D op wrapper
 *
 * @author raver119@gmail.com
 */
public class Dilation2D extends DynamicCustomOp {
    protected boolean isSameMode;

    // rates
    protected int r0, r1, r2, r3;

    // strides
    protected int s0, s1, s2, s3;


    public Dilation2D() {
    }

    public Dilation2D(SameDiff sameDiff, SDVariable[] inputAndWeights, int[] strides,
                      int[] rates, boolean isSameMode, boolean inPlace ) {
        super(null, sameDiff, inputAndWeights, inPlace);

        if (rates.length < 4)
            throw new IllegalArgumentException("Dilation rate length must be 4.");
        if (strides.length < 4)
            throw new IllegalArgumentException("Strides length must be 4.");

        r0 = rates[0];
        r1 = rates[1];
        r2 = rates[2];
        r3 = rates[3];
        s0 = strides[0];
        s1 = strides[1];
        s2 = strides[2];
        s3 = strides[3];
        this.isSameMode = isSameMode;

        addArgs();

    }

    public Dilation2D(INDArray[] inputArrays, INDArray[] outputs) {
        super(null, inputArrays, outputs);

    }

    protected void addArgs() {
        addIArgument(isSameMode ? 1 : 0,
                r0, r1, r2, r3,
                s0, s1, s2, s3);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode,nodeDef, graph);
        addArgs();
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val sameMode = PropertyMapping.builder()
                .tfAttrName("padding")
                .propertyNames(new String[]{"isSameMode"})
                .build();

        val ratesMapping = PropertyMapping.builder()
                .tfAttrName("rates")
                .propertyNames(new String[]{"r0", "r1", "r2", "r3"})
                .build();

        val stridesMapping = PropertyMapping.builder()
                .tfAttrName("strides")
                .propertyNames(new String[]{"s0", "s1", "s2", "s3"})
                .build();

        map.put("isSameMode", sameMode);

        map.put("r0", ratesMapping);
        map.put("r1", ratesMapping);
        map.put("r2", ratesMapping);
        map.put("r3", ratesMapping);

        map.put("s0", stridesMapping);
        map.put("s1", stridesMapping);
        map.put("s2", stridesMapping);
        map.put("s3", stridesMapping);

        try {
            ret.put(onnxName(), map);
        }catch(NoOpNameFoundException e) {
            //ignore, we dont care about onnx for this set of ops
        }


        try {
            ret.put(tensorflowName(),map);
        }catch(NoOpNameFoundException e) {
            throw new RuntimeException(e);
        }

        return ret;
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String, Map<String, AttributeAdapter>> ret = new HashMap<>();
        Map<String,AttributeAdapter> tfMappings = new LinkedHashMap<>();
        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(this);


        tfMappings.put("r0", new IntArrayIntIndexAdpater(0));
        tfMappings.put("r1", new IntArrayIntIndexAdpater(1));
        tfMappings.put("r2", new IntArrayIntIndexAdpater(2));
        tfMappings.put("r3", new IntArrayIntIndexAdpater(3));

        tfMappings.put("s0", new IntArrayIntIndexAdpater(0));
        tfMappings.put("s1", new IntArrayIntIndexAdpater(1));
        tfMappings.put("s2", new IntArrayIntIndexAdpater(2));
        tfMappings.put("s3", new IntArrayIntIndexAdpater(3));

        tfMappings.put("isSameMode",new StringEqualsAdapter("SAME"));

        // Onnx doesn't have this op i think?
        Map<String,AttributeAdapter> onnxMappings = new HashMap<>();
        onnxMappings.put("isSameMode",new StringEqualsAdapter("SAME"));

        ret.put(tensorflowName(), tfMappings);
        ret.put(onnxName(), onnxMappings);
        return ret;
    }

    @Override
    public boolean isConfigProperties() {
        return true;
    }


    @Override
    public String opName() {
        return "dilation2d";
    }

    @Override
    public String onnxName() {
        return "Dilation_2D";
    }

    @Override
    public String tensorflowName() {
        return "Dilation2D";
    }
}
