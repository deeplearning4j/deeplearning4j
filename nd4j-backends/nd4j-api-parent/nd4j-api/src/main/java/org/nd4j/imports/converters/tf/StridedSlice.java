package org.nd4j.imports.converters.tf;

import com.google.common.primitives.Ints;
import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;

/**
 * @author raver119@gmail.com
 */
public class StridedSlice extends BaseTensorFlowNode {
    @Override
    public TNode asIntermediateRepresentation(NodeDef node, TGraph graph) {
        val tNode = buildBasicNode(node, graph);
        /*
            strided slice typically takes 4 tensor arguments:
            0) input, it's shape determines number of elements in other arguments
            1) begin indices
            2) end indices
            3) strides
         */

        val strides = graph.getVariableSpace().getVariable(tNode.getInputs().remove(3));
        val end = graph.getVariableSpace().getVariable(tNode.getInputs().remove(2));
        val begin = graph.getVariableSpace().getVariable(tNode.getInputs().remove(1));

        val iArgs = new ArrayList<Integer>();

        for (int e = 0; e < begin.getArray().length(); e++)
            iArgs.add((int) begin.getArray().getInt(e));

        for (int e = 0; e < end.getArray().length(); e++)
            iArgs.add((int) end.getArray().getInt(e));

        for (int e = 0; e < strides.getArray().length(); e++)
            iArgs.add((int) strides.getArray().getInt(e));


        val bits = Ints.toArray(iArgs);
        tNode.getOpState().setExtraBits(bits);

        return tNode;
    }

    @Override
    public String opName() {
        return "stridedslice";
    }
}
