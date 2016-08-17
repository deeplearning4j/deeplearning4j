package org.nd4j.linalg.api.ops.impl.grid;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.GridOp;
import org.nd4j.linalg.api.ops.MetaOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.grid.GridDescriptor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseGridOp extends BaseOp implements GridOp {
    private List<Op> queuedOps = new ArrayList<>();

    public BaseGridOp() {

    }

    public BaseGridOp(INDArray x, INDArray y) {
        // no-op
    }

    protected BaseGridOp(Op... ops) {
        for (Op op: ops) {
            queuedOps.add(op);
        }
    }

    protected BaseGridOp(List<Op> ops) {
        queuedOps.addAll(ops);
    }


    @Override
    public GridDescriptor getGridReference() {
        GridDescriptor descriptor = new GridDescriptor();
        descriptor.setGridDepth(queuedOps.size());
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
