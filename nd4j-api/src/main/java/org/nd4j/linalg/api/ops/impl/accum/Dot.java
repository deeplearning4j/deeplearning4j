package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.Op;

/**
 * Dot product
 * @author Adam Gibson
 */
public class Dot extends BaseAccumulation {
    public Dot(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
    }

    public Dot(INDArray x, INDArray y, int n) {
        super(x, y, n);
    }

    public Dot(INDArray x) {
        super(x);
    }

    public Dot(INDArray x, INDArray y) {
        super(x, y);
    }

    @Override
    public String name() {
        return "dot";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);


        if (y() != null)
            return new Dot(xAlongDimension, y.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Dot(x.vectorAlongDimension(index, dimension));

    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        numProcessed++;
        return origin.mul(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        numProcessed++;
        return origin.mul(other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        numProcessed++;
        return origin.mul(other);
    }

    @Override
    public float op(float origin, float other) {
        numProcessed++;
        return origin  * other;
    }

    @Override
    public double op(double origin, double other) {
        numProcessed++;
        return origin  * other;
    }

    @Override
    public double op(double origin) {
        numProcessed++;
        return origin;
    }

    @Override
    public float op(float origin) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin;
    }

    @Override
    public void update(Number result) {
         currentResult = currentResult().doubleValue() + result.doubleValue();
    }

    @Override
    public void update(IComplexNumber result) {
         currentComplexResult = currentResultComplex().add(result);
    }
}
