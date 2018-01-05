package org.nd4j.linalg.api.ops.impl.layers.recurrent;

import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.SRUConfiguration;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;

/**
 * Simple recurrent unit
 *
 * @author Adam Gibson
 */
public class SRU extends DynamicCustomOp {

    private SRUConfiguration configuration;

    public SRU() { }

    public SRU(SameDiff sameDiff, SRUConfiguration configuration) {
        super(null, sameDiff, configuration.args());
        this.configuration = configuration;
    }

    @Override
    public String opName() {
        return "sru";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op name for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op name for " + opName());
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }


}
