package org.nd4j.linalg.api.ops;

import com.google.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A gradient op always makes the following assumptions:
 * there is always a y (beacuse of backpropagating
 * or using the chain rule)
 *
 * and that it is special exec (for now)
 *
 * This op type sis meant to be used
 * to build derivative operations.
 *
 *
 * @author Adam Gibson
 */
public abstract class BaseGradientOp extends BaseTransformOp implements GradientOp {

    public BaseGradientOp(INDArray x, INDArray z) {
        super(x, z);
        assertWrt();
    }

    public BaseGradientOp() {
    }

    public BaseGradientOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
        assertWrt();
    }

    public BaseGradientOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        assertWrt();
    }

    public BaseGradientOp(INDArray x) {
        super(x);
        assertWrt();
    }

    /**
     * The array
     * to the gradient with respect to
     *
     * @return
     */
    @Override
    public INDArray wrt() {
        return y();
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public boolean isPassThrough() {
        return true;
    }

    private void assertWrt() {
        Preconditions.checkState(y != null,"A gradient op must define a wrt variable as a Y. ");
    }

}
