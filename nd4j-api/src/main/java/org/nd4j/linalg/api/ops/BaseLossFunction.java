package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public abstract class BaseLossFunction extends BaseAccumulation implements LossFunction {
    public BaseLossFunction(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public BaseLossFunction(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public BaseLossFunction(INDArray x) {
        super(x);
    }

    public BaseLossFunction(INDArray x, INDArray y) {
        super(x, y);
    }

    public BaseLossFunction() {
    }

    @Override
    public INDArray input() {
        return x;
    }

    @Override
    public INDArray output() {
        return y;
    }
}
