package org.nd4j.imports.converters.tf;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

/**
 * Conv2D converter
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class Conv2D extends BaseTensorFlowNode {
    @Override
    public TNode asIntermediateRepresentation(NodeDef tfNode, TGraph graph) {
        val tNode = buildBasicNode(tfNode, graph);

        val aStrides = tfNode.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();
        val sY = tfStrides.get(1);
        val sX = tfStrides.get(2);

        val aPadding = tfNode.getAttrOrDefault("padding", null);

        val paddingMode = aPadding.getS().toStringUtf8();

        // we know that second input to conv2d is weights array
        val weightsIndex = tNode.getInputs().get(1);
        val variable = graph.getVariableSpace().getVariable(weightsIndex);

        val kY = variable.getArray().size(0);
        val kX = variable.getArray().size(1);

        variable.setArray(variable.getArray().permute(3, 2, 0, 1).dup('c'));

        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

        if (!isSameMode)
            log.debug("Mode: {}", paddingMode);

        log.debug("Conv2D: k: [{}, {}]; s: [{}, {}]; padding: {}", kY, kX, sY, sX,  paddingMode);

        tNode.getOpState().setExtraBits(new int[] {kY, kX, sY.intValue(), sX.intValue(), 0, 0, 1, 1, isSameMode ? 1 : 0});

        return tNode;
    }

    @Override
    public String opName() {
        return "conv2d";
    }
}
