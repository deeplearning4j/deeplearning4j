package org.nd4j.imports.converters.tf;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TIndex;
import org.nd4j.graph.intermediate.TNode;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.NodeDef;

/**
 * Concat op implementation
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class Concat extends BaseTensorFlowNode {
    @Override
    public TNode asIntermediateRepresentation(NodeDef node, TGraph graph) {
        val tNode = buildBasicNode(node, graph);

        log.debug("TNode inputs: {}", tNode.getInputs());
        TIndex dimIndex;
        int idx = -1;
        int cnt = 0;
        int concatDimension = 0;
        for (val index:tNode.getInputs()) {
            log.debug("Trying to find node: [{}]", index);
            val variable = graph.getVariableSpace().getVariable(index);

            // concat dimension is only possible
            if (variable != null && variable.getId() < 0 && variable.getArray() == null) {
                idx = cnt;
                concatDimension = variable.getShape()[0];
                break;
            } else if (variable != null && variable.getId() < 0) {
                val arr = variable.getArray();
                if (arr.length() == 1) {
                    concatDimension = arr.getInt(0);
                    idx = cnt;
                    break;
                }
            }
            cnt++;
        }

        if (idx < 0)
            throw new ND4JIllegalStateException("Can't find dimension for concatenatiion");

        // deleting index of concat dimension
        tNode.getInputs().remove(idx);

        // if that's convolution graph, we should swap dimensions
        if (concatDimension == 3)
            concatDimension = 1;

        tNode.getOpState().setExtraBits(new int[]{concatDimension});
        log.debug("Concat dimension: {}", concatDimension);

        return tNode;
    }

    @Override
    public String opName() {
        return "concat";
    }
}
