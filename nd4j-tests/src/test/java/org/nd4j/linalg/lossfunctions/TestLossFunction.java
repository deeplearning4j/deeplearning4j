package org.nd4j.linalg.lossfunctions;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseLossFunction;
import org.nd4j.linalg.api.ops.Op;

/**
 * @author Adam Gibson
 */
public class TestLossFunction extends BaseLossFunction {
    public TestLossFunction(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public TestLossFunction(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public TestLossFunction(INDArray x) {
        super(x);
    }

    public TestLossFunction(INDArray x, INDArray y) {
        super(x, y);
    }

    public TestLossFunction() {
    }

    @Override
    public void update(Number result) {

    }

    @Override
    public void update(IComplexNumber result) {

    }

    @Override
    public String name() {
        return "test_loss_function";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        return null;
    }
}
