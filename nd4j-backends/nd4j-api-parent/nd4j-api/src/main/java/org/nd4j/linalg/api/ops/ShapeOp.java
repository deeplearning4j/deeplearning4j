package org.nd4j.linalg.api.ops;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

public abstract class ShapeOp extends BaseOp {
    public ShapeOp() {}





    public ShapeOp(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }



    public ShapeOp(SameDiff sameDiff,DifferentialFunction i_v,boolean inPlace) {
        this(sameDiff,i_v,i_v.getResultShape(),inPlace,null);
    }

    public ShapeOp(SameDiff sameDiff,
                           DifferentialFunction i_v,
                           int[] shape,
                           boolean inPlace,
                           Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);
        this.shape = shape;

        if (i_v != null) {
            this.args = new DifferentialFunction[] {sameDiff.setupFunction(i_v)};
            f().validateFunctionReference(i_v);
            f().validateDifferentialFunctionsameDiff(i_v);
            addAsNewVertexId();
            f().addFunctionEdges(this);
        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
    }
    /**
     * Specify an alternative output array
     *
     * @param x the input
     * @param z the output
     * @param n the number of elements to iterate on
     */
    public ShapeOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public ShapeOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>();
        ret.add(shape);
        return ret;
    }

    @Override
    public Type opType() {
        return Type.SHAPE;
    }

    /**
     * An op for one ndarray
     *
     * @param x the ndarray
     */
    public ShapeOp(INDArray x) {
        super(x);
    }

    /**
     * Specify an alternative result array
     *
     * @param x the input
     * @param z the output array
     */
    public ShapeOp(INDArray x, INDArray z) {
        super(x, z);
    }
}
