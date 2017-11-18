package org.nd4j.linalg.api.ops.impl.controlflow;

import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.tensorflow.framework.NodeDef;

/**
 * From the onnx docs:
 *   Given DATA tensor of rank r >= 1, and INDICES tensor of rank q, gather
 entries of the outer-most dimension of DATA indexed by INDICES, and concatenate
 them in an output tensor of rank q + (r - 1).

 Example:
 DATA  = [
 [1.0, 1.2],
 [2.3, 3.4],
 [4.5, 5.7],
 ]
 INDICES = [
 [0, 1],
 [1, 2],
 ]
 OUTPUT = [
 [
 [1.0, 1.2],
 [2.3, 3.4],
 ],
 [
 [2.3, 3.4],
 [4.5, 5.7],
 ],
 ]



 */
public class Gather extends DynamicCustomOp {
    @Override
    public String opName() {
        return "gather";
    }

    @Override
    public String onnxName() {
        return "Gather";
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No op name found for tensorflow " + opName());
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public TOp asIntermediateRepresentation(NodeDef node, TGraph graph) {
        return super.asIntermediateRepresentation(node, graph);
    }
}
