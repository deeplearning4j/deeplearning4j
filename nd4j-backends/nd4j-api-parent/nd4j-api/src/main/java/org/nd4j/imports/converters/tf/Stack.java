package org.nd4j.imports.converters.tf;

import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

/**
 * Stack op conversion
 *
 * @author raver119@gmail.com
 */
public class Stack  extends BaseTensorFlowNode {
    @Override
    public TNode asIntermediateRepresentation(NodeDef node, TGraph graph) {
        val tNode = buildBasicNode(node, graph);

        val attrAxis = node.getAttrOrThrow("axis");
        int axis = (int) attrAxis.getI();

        tNode.getOpState().setExtraBits(new int[]{axis});

        return tNode;
    }

    @Override
    public String opName() {
        return "stack";
    }
}
