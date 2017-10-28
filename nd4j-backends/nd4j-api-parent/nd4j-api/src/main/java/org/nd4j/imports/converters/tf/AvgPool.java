package org.nd4j.imports.converters.tf;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

/**
 * AvgPool2D converter
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class AvgPool extends BaseTensorFlowNode {
    @Override
    public TNode asIntermediateRepresentation(NodeDef tfNode, TGraph graph) {
        val tNode = buildBasicNode(tfNode, graph);

        val aStrides = tfNode.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();
        val sY = tfStrides.get(1);
        val sX = tfStrides.get(2);

        val aKernels = tfNode.getAttrOrThrow("ksize");
        val tfKernels = aKernels.getList().getIList();

        val kY = tfKernels.get(1);
        val kX = tfKernels.get(2);

        val aPadding = tfNode.getAttrOrThrow("padding");

        val paddingMode = aPadding.getS().toStringUtf8().replaceAll("\"","");

        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

        if (!isSameMode)
            log.debug("Mode: {}", paddingMode);

        log.debug("Pooling: k: [{},{}]; s: [{}, {}], padding: {}", kY, kX, sY, sX, aPadding);

        tNode.getOpState().setExtraBits(new int[] {kY.intValue(), kX.intValue(), sY.intValue(), sX.intValue(), 0, 0, 1, 1, isSameMode ? 1 : 0 });

        return tNode;
    }

    @Override
    public String opName() {
        return "avgpool";
    }
}
