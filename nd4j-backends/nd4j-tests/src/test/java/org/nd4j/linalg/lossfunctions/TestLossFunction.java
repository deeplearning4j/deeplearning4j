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
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "test_loss_function";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        return null;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        return null;
    }

    @Override
    public double update(double accum, double x) {
        return 0;
    }

    @Override
    public double update(double accum, double x, double y) {
        return 0;
    }

    @Override
    public float update(float accum, float x) {
        return 0;
    }

    @Override
    public float update(float accum, float x, float y) {
        return 0;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x) {
        return null;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, double x, double y) {
        return null;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x) {
        return null;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, IComplexNumber y) {
        return null;
    }

    @Override
    public IComplexNumber update(IComplexNumber accum, IComplexNumber x, double y) {
        return null;
    }
}
