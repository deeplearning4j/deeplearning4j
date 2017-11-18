package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.imports.NoOpNameFoundException;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

/**
 * Stack op conversion
 *
 * @author raver119@gmail.com
 */
public class Stack  extends DifferentialFunction {
    @Override
    public TOp asIntermediateRepresentation(NodeDef node, TGraph graph) {
        val tNode = buildBasicNode(node, graph);

        val attrAxis = node.getAttrOrThrow("axis");
        int axis = (int) attrAxis.getI();

        tNode.getOpState().setExtraBits(new int[]{axis});

        return tNode;
    }

    @Override
    public TOp asIntermediateRepresentation(OnnxProto3.NodeProto node, TGraph graph, Map<String, OnnxProto3.AttributeProto> attributesForNode) {
        throw new UnsupportedOperationException();
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
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
      throw new UnsupportedOperationException("Differentiation not supported yet.");
    }

    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String opName() {
        return "stack";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith) {

    }
}
