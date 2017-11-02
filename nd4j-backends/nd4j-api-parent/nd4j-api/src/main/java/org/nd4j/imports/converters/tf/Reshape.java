package org.nd4j.imports.converters.tf;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;

/**
 * Reshape operation
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class Reshape extends BaseTensorFlowNode {
    @Override
    public TNode asIntermediateRepresentation(NodeDef node, TGraph graph) {
        val tNode = buildBasicNode(node, graph);

        // in reshape operation we replace second input, and replace it with extra args
        log.debug("TNode inputs: {}", tNode.getInputs());
        val shapeIndex = tNode.getInputs().remove(1);
        val variable = graph.getVariableSpace().getVariable(shapeIndex);

        assert variable != null;
        assert variable.getShape() != null;

        // we know that TF is always C order
        int[] args = ArrayUtils.add(variable.getShape(),  0, (int)'c');


        log.debug("Reshape node_{}, new shape: {}", tNode.getId(), Arrays.toString(args));

        // new shape goes here
        tNode.getOpState().setExtraBits(args);

        return tNode;
    }

    @Override
    public String opName() {
        return "reshape";
    }
}
