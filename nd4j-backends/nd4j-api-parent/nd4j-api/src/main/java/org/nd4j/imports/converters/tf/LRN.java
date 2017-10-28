package org.nd4j.imports.converters.tf;

import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

/**
 * Local Response Normalization
 *
 * @author raver119@gmail.com
 */
public class LRN extends BaseTensorFlowNode {

    @Override
    public TNode asIntermediateRepresentation(NodeDef node, TGraph graph) {
        val tNode = buildBasicNode(node, graph);

        val aAlpha = node.getAttrOrThrow("alpha");
        val aBeta = node.getAttrOrThrow("beta");
        val aBias = node.getAttrOrThrow("bias");
        val aDepth = node.getAttrOrThrow("depth_radius");

        val alpha = aAlpha.getF();
        val beta = aBeta.getF();
        val bias = aBias.getF();
        val depth = aDepth.getF();

        tNode.getOpState().setExtraArgs(new Object[]{alpha, beta, bias, depth});

        return tNode;
    }

    @Override
    public String opName() {
        return "lrn";
    }
}
