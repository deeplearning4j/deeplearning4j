package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import lombok.val;
import onnx.OnnxMlProto3;
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
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Created by farizrahman4u on 3/28/18.
 */
@NoArgsConstructor
public class SequenceMask extends DynamicCustomOp {

    private int maxLen;
    private boolean is_static_maxlen = false;
    public SequenceMask(SameDiff sameDiff, SDVariable input, SDVariable maxLen) {
        super(null, sameDiff, new SDVariable[] {input, maxLen}, false);
    }

    public SequenceMask(SameDiff sameDiff, SDVariable input, int maxLen) {
        super(null, sameDiff, new SDVariable[] {input}, false);
        this.maxLen = maxLen;
        this.is_static_maxlen = true;
        addIArgument(maxLen);
    }

    public SequenceMask(SameDiff sameDiff, SDVariable input) {
        super(null, sameDiff, new SDVariable[] {input}, false);
    }
    

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val targetNode = TFGraphMapper.getInstance().getNodeWithNameFromGraph(graph, nodeDef.getInput(1));
        val maxlen = TFGraphMapper.getInstance().getNDArrayFromTensor("value", targetNode, graph);
        if (maxlen == null){
            // No 2nd input
            this.is_static_maxlen = true;
        }
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        if (is_static_maxlen) {
            addIArgument(this.maxLen);
        }
    }
    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> attrs = new LinkedHashMap<>();
        if (is_static_maxlen) {
            val maxLen = PropertyMapping.builder()
                    .propertyNames(new String[]{"maxLen"})
                    .tfAttrName("maxlen")
                    .build();
            attrs.put("maxLen", maxLen);
        }
        ret.put(tensorflowName(), attrs);
        return ret;
    }

    @Override
    public String opName() {
        return "sequence_mask";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx opName found for " + opName());
    }


    @Override
    public String tensorflowName() {
        return "SequenceMask";
    }
}
