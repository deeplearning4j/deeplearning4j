package org.nd4j.imports.converters.tf;

import com.google.common.primitives.Ints;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;

/**
 * @author raver119@gmail.com
 */
@Slf4j
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

        val inputBegin = tNode.getInputs().get(1);
        val inputEnd = tNode.getInputs().get(2);
        val inputStrides = tNode.getInputs().get(3);


        val iArgs = new ArrayList<Integer>();

        // bit masks for this slice
        val bm = node.getAttrOrThrow("begin_mask");
        val xm = node.getAttrOrThrow("ellipsis_mask");
        val em = node.getAttrOrThrow("end_mask");
        val nm = node.getAttrOrThrow("new_axis_mask");
        val sm = node.getAttrOrThrow("shrink_axis_mask");

        iArgs.add((int) bm.getI());
        iArgs.add((int) xm.getI());
        iArgs.add((int) em.getI());

        iArgs.add((int) nm.getI());
        iArgs.add((int) sm.getI());

        if (inputBegin.getNode() < 0 && inputEnd.getNode() < 0 && inputStrides.getNode() < 0) {

            // order matters, hehe
            val strides = graph.getVariableSpace().getVariable(tNode.getInputs().remove(3));
            val end = graph.getVariableSpace().getVariable(tNode.getInputs().remove(2));
            val begin = graph.getVariableSpace().getVariable(tNode.getInputs().remove(1));

            for (int e = 0; e < begin.getArray().length(); e++)
                iArgs.add((int) begin.getArray().getInt(e));

            for (int e = 0; e < end.getArray().length(); e++)
                iArgs.add((int) end.getArray().getInt(e));

            for (int e = 0; e < strides.getArray().length(); e++)
                iArgs.add((int) strides.getArray().getInt(e));
        } else {
            // do nothing
        }

        val bits = Ints.toArray(iArgs);
        tNode.getOpState().setExtraBits(bits);

        return tNode;
    }

    @Override
    public String opName() {
        return "stridedslice";
    }
}
