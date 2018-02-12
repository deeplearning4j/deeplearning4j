package org.nd4j.linalg.api.ops.impl.transforms.dtype;

import lombok.NonNull;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.DataTypeAdapter;
import org.nd4j.imports.descriptors.properties.adapters.IntArrayIntIndexAdpater;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;

/**
 * Cast op wrapper. This op changes data type of input array.
 *
 * @author raver119@gmail.com
 */
public class Cast extends BaseDynamicTransformOp {
    private DataBuffer.Type typeDst;

    public Cast() {
        //
    }

    public Cast(SameDiff sameDiff, SDVariable arg, @NonNull DataBuffer.Type dst) {
        super(sameDiff, new SDVariable[] {arg}, false);

        this.typeDst = dst;
        addArgs();
    }


    @Override
    public void setValueFor(Field target, Object value) {
        if(value == null) {
            throw new ND4JIllegalStateException("Unable to set field " + target + " using null value!");
        }

        // FIXME!
        if (!(value instanceof DataBuffer.Type))
            return;

        try {
            target.set(this, (DataBuffer.Type) value);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }

    protected void addArgs() {
        addIArgument(SameDiff.getDataTypeAsByte(typeDst));
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String,Map<String,AttributeAdapter>> ret = new LinkedHashMap<>();
        Map<String,AttributeAdapter> tfAdapters = new LinkedHashMap<>();

        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(this);

        tfAdapters.put("typeDst", new DataTypeAdapter());

        ret.put(tensorflowName(),tfAdapters);
        return ret;
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val dstMapping = PropertyMapping.builder()
                .tfAttrName("DstT")
                .propertyNames(new String[]{"typeDst"})
                .build();

        for(val propertyMapping : new PropertyMapping[] {dstMapping}) {
            for (val keys : propertyMapping.getPropertyNames())
                map.put(keys, propertyMapping);
        }

        ret.put(tensorflowName(),map);

        return ret;
    }

    @Override
    public String opName() {
        return "cast";
    }

    @Override
    public String tensorflowName() {
        return "Cast";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // FIXME: we'll just do reverse cast here, but we don't have sameDiff.cast() yet
        SDVariable gradient = sameDiff.setupFunction(i_v.get(0));
        throw new UnsupportedOperationException("Not implemented yet");
        //return Collections.singletonList(sameDiff.batchToSpace(gradient, blocks, padding));
    }
}
