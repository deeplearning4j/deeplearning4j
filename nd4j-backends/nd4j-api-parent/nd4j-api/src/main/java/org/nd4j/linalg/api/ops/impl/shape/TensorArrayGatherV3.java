package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

public class TensorArrayGatherV3 extends DynamicCustomOp {

    /*   @Override
       public TOp asIntermediateRepresentation(NodeDef node, TGraph graph) {
           val tNode = buildBasicNode(node, graph);

           val idd = tNode.getInputs().get(1);

           if (idd.getNode() < 0) {
               val idxArg = tNode.getInputs().remove(1);
               val variable = graph.getVariableSpace().getVariable(idxArg);

               int idx = variable.getArray().getInt(0);

               tNode.getOpState().setExtraBits(new int[]{idx});
           }

           return tNode;
       }

       @Override
       public TOp asIntermediateRepresentation(OnnxProto3.NodeProto node, TGraph graph, Map<String, OnnxProto3.AttributeProto> attributesForNode) {
           val tNode = buildBasicNode(node, graph);

           val idd = tNode.getInputs().get(1);

           if (idd.getNode() < 0) {
               val idxArg = tNode.getInputs().remove(1);
               val variable = graph.getVariableSpace().getVariable(idxArg);

               int idx = variable.getArray().getInt(0);

               tNode.getOpState().setExtraBits(new int[]{idx});
           }

           return tNode;
       }

       */@Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op name found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "TensorArrayGatherV3";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Differentiation not supported yet.");

    }

    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String opName() {
        return "tensorarraygatherv3";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val inputOne = nodeDef.getInput(1);
        val varFor = initWith.getVariable(inputOne);
        val var = TFGraphMapper.getInstance().getArrayFrom(nodeDef,graph);
        val idx = var.getInt(0);
        addIArgument(idx);

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }
}
