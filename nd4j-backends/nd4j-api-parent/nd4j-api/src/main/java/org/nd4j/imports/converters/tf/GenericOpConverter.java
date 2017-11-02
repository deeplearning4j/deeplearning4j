package org.nd4j.imports.converters.tf;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TIndex;
import org.nd4j.graph.intermediate.TNode;
import org.nd4j.graph.intermediate.TVariableSpace;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.NodeDef;

/**
 * This converter is used as default one, and used for ops that do not have own special converters
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class GenericOpConverter extends BaseTensorFlowNode {
    @Override
    public TNode asIntermediateRepresentation(NodeDef node, TGraph graph) {
        val tNode = buildBasicNode(node, graph);

        tNode.setOpState(getOpStateFromNodeDef(node, node.getInputCount(), tNode, graph.getVariableSpace()));

        return tNode;
    }

    @Override
    public String opName() {
        return "GenericOpConverter";
    }
}
