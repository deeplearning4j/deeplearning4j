package org.nd4j.linalg.api.ops.impl.grid;

import onnx.OnnxProto3;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.GridOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.grid.GridDescriptor;
import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseGridOp extends BaseOp implements GridOp {
    protected List<OpDescriptor> queuedOps = new ArrayList<>();
    protected List<GridPointers> grid = new ArrayList<>();

    public BaseGridOp() {

    }

    public BaseGridOp(INDArray x, INDArray y) {
        // no-op
    }

    protected BaseGridOp(Op... ops) {
        grid = new ArrayList<>(ops.length);
        for (Op op : ops) {
            queuedOps.add(new OpDescriptor(op, null));
            grid.add(null);
        }
    }

    protected BaseGridOp(OpDescriptor... descriptors) {
        for (OpDescriptor op : descriptors) {
            queuedOps.add(op);
            grid.add(null);
        }
    }

    protected BaseGridOp(GridPointers... pointers) {
        for (GridPointers ptr : pointers) {
            grid.add(ptr);
        }
    }

    protected BaseGridOp(List<Op> ops) {
        this(ops.toArray(new Op[0]));
    }


    @Override
    public GridDescriptor getGridDescriptor() {
        GridDescriptor descriptor = new GridDescriptor();
        descriptor.setGridDepth(grid.size());
        descriptor.setGridPointers(grid);
        return descriptor;
    }

    @Override
    public TOp asIntermediateRepresentation(OnnxProto3.NodeProto node, TGraph graph, Map<String, OnnxProto3.AttributeProto> attributesForNode) {
        return null;
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " + opName());
    }
}
