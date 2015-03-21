package org.nd4j.linalg.api.ops.impl.transforms;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Set range to a particular set of values
 *
 * @author Adam Gibson
 */
public class SetRange extends BaseTransformOp {

    private double min, max;

    public SetRange(INDArray x, INDArray z, double min, double max) {
        super(x, z);
        this.min = min;
        this.max = max;
        init(x, y, z, n);
    }

    public SetRange(INDArray x, INDArray z, int n, double min, double max) {
        super(x, z, n);
        this.min = min;
        this.max = max;
        init(x, y, z, n);
    }

    public SetRange(INDArray x, INDArray y, INDArray z, int n, double min, double max) {
        super(x, y, z, n);
        this.min = min;
        this.max = max;
        init(x, y, z, n);
    }

    public SetRange(INDArray x, double min, double max) {
        super(x);
        this.min = min;
        this.max = max;
        init(x, y, z, n);
    }

    @Override
    public String name() {
        return "setrange";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return op(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return op(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return op(origin);
    }

    @Override
    public float op(float origin, float other) {
        return op(origin);
    }

    @Override
    public double op(double origin, double other) {
        return op(origin);
    }

    @Override
    public double op(double origin) {
        if (origin >= min && origin <= max)
            return origin;
        if (min == 0 && max == 1) {
            double val = 1 / (1 + FastMath.exp(-origin));
            return (FastMath.floor(val * (max - min)) + min);
        }

        double ret = (FastMath.floor(origin * (max - min)) + min);
        return ret;
    }

    @Override
    public float op(float origin) {
        return (float) op((double) origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return Nd4j.createComplexNumber(op(origin.realComponent().doubleValue()), op(origin.imaginaryComponent().doubleValue()));
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[]{min, max};
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new SetRange(x.vectorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(), min, max);
        else
            return new SetRange(x.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(), min, max);
    }
}
