package org.nd4j.imports.converters.tf;

import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

public class TensorArrayWriteV3 extends BaseTensorFlowNode {

    @Override
    public TNode asIntermediateRepresentation(NodeDef node, TGraph graph) {
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
    public String opName() {
        return "tensorarraywritev3";
    }
}
