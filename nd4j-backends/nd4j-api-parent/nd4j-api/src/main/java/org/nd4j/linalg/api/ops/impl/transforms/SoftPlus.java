package org.nd4j.linalg.api.ops.impl.transforms;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.util.ComplexUtil;

/**
 * @author Adam Gibson
 */
public class SoftPlus extends BaseTransformOp {
    public SoftPlus(INDArray x, INDArray z) {
        super(x, z);
    }

    public SoftPlus() {
    }

    public SoftPlus(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public SoftPlus(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public SoftPlus(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 13;
    }

    @Override
    public String name() {
        return "softplus";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return ComplexUtil.log(ComplexUtil.exp(origin).add(1));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return ComplexUtil.log(ComplexUtil.exp(origin).add(1));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return ComplexUtil.log(ComplexUtil.exp(origin).add(1));
    }

    @Override
    public float op(float origin, float other) {
        return (float) FastMath.log(1 + FastMath.exp(origin));
    }

    @Override
    public double op(double origin, double other) {
        return FastMath.log(1 + FastMath.exp(origin));
    }

    @Override
    public double op(double origin) {
        return  FastMath.log(1 + FastMath.exp(origin));
    }

    @Override
    public float op(float origin) {
        return (float) FastMath.log(1 + FastMath.exp(origin));
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return ComplexUtil.log(ComplexUtil.exp(origin).add(1));
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);
        if (y() != null)
            return new SoftPlus(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new SoftPlus(xAlongDimension, z.vectorAlongDimension(index, dimension), x.lengthLong());

    }


    @Override
    public TransformOp derivative() {
        return new Sigmoid(x,y,z,n);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new SoftPlus(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new SoftPlus(xAlongDimension, z.tensorAlongDimension(index, dimension), x.lengthLong());

    }
}
