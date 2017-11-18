package org.nd4j.linalg.api.ops.impl.shape;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TIndex;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.NodeDef;

@Slf4j
public class Concat extends DynamicCustomOp {


    @Override
    public String opName() {
        return "concat";
    }

    @Override
    public TOp asIntermediateRepresentation(NodeDef node, TGraph graph) {
        val tNode = buildBasicNode(node, graph);

        log.debug("TOp inputs: {}", tNode.getInputs());
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
    public String onnxName() {
        return "Concat";
    }

    @Override
    public String tensorflowName() {
        return "concat";
    }



    @Override
    public Op.Type opType() {
        return Op.Type.SHAPE;
    }
}
