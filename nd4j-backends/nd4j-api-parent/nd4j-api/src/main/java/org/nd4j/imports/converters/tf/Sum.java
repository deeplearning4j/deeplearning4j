package org.nd4j.imports.converters.tf;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class Sum extends BaseTensorFlowNode{

    public Sum() {
        super();
    }

    public Sum(@NonNull NodeDef nodeDef, @NonNull TGraph graph) {
        super(nodeDef, graph);
    }

    @Override
    public String opName() {
        return "sum";
    }

    /**
     * This method returns given TF node as TNode
     *
     * @return
     */
    @Override
    public TNode asIntermediateRepresentation(@NonNull NodeDef node, @NonNull TGraph graph) {
        val tNode = buildBasicNode(node, graph);

        /**
         * 2 options here. We either have specific dimension, or not.
         * If not - that'll be reduceScalar, if yes - there will be reduceAlongDimension
         */

        log.debug("TNode inputs: {}", tNode.getInputs());
        val shapeIndex = tNode.getInputs().remove(1);

        val variable = graph.getVariableSpace().getVariable(shapeIndex);

        // reduce to scalar
        if (variable.getArray() == null && variable.getShape().length == 2 && variable.getShape()[0] == 1 && variable.getShape()[1] == 1)
            tNode.getOpState().setAxes(new int[]{Integer.MAX_VALUE});// we're going for scalar
        else {
            if (variable.getArray() != null) {
              val axes = variable.getArray().data().asInt();
              tNode.getOpState().setAxes(axes);
            } else
                tNode.getOpState().setAxes(variable.getShape());
        }

        return tNode;
    }
}
