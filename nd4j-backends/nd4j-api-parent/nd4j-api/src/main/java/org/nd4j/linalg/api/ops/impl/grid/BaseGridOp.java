package org.nd4j.linalg.api.ops.impl.grid;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.GridOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.grid.GridDescriptor;
import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;

import java.util.ArrayList;
import java.util.List;

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



    /**
     * Pairwise op (applicable with an individual element in y)
     *
     * @param origin the origin number
     * @param other  the other number
     * @return the transformed output
     */
    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return null;
    }

    /**
     * Pairwise op (applicable with an individual element in y)
     *
     * @param origin the origin number
     * @param other  the other number
     * @return the transformed output
     */
    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return null;
    }

    /**
     * Pairwise op (applicable with an individual element in y)
     *
     * @param origin the origin number
     * @param other  the other number
     * @return the transformed output
     */
    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return null;
    }

    /**
     * Pairwise op (applicable with an individual element in y)
     *
     * @param origin the origin number
     * @param other  the other number
     * @return the transformed output
     */
    @Override
    public float op(float origin, float other) {
        return 0;
    }

    /**
     * Pairwise op (applicable with an individual element in y)
     *
     * @param origin the origin number
     * @param other  the other number
     * @return the transformed output
     */
    @Override
    public double op(double origin, double other) {
        return 0;
    }

    /**
     * Transform an individual element
     *
     * @param origin the origin element
     * @return the new element
     */
    @Override
    public double op(double origin) {
        return 0;
    }

    /**
     * Transform an individual element
     *
     * @param origin the origin element
     * @return the new element
     */
    @Override
    public float op(float origin) {
        return 0;
    }

    /**
     * Transform an individual element
     *
     * @param origin the origin element
     * @return the new element
     */
    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return null;
    }

    /**
     * A copy of this operation for a particular dimension of the input
     *
     * @param index     the index of the op to iterate over
     * @param dimension the dimension to ge the input for
     * @return the operation for that dimension
     */
    @Override
    public Op opForDimension(int index, int dimension) {
        return null;
    }

    /**
     * A copy of this operation for a particular dimension of the input
     *
     * @param index     the index of the op to iterate over
     * @param dimension the dimension to ge the input for
     * @return the operation for that dimension
     */
    @Override
    public Op opForDimension(int index, int... dimension) {
        return null;
    }


}
