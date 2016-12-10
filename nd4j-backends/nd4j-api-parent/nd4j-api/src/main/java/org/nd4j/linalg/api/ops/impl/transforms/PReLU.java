package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 12/5/16.
 */
public class PReLU extends BaseTransformOp {
    /*
    FIXME: need to add support for U and L for alphaA
     */
    private double u = 1/3.0;
    private double l = 1/8.0;
    private INDArray alphaA;
    private float alpha = 0.5f;
    public PReLU() {
    }

    public PReLU(INDArray x) {
        super(x);
        this.alphaA = Nd4j.rand(x.shape()); //this should be u and l uniform distribution
        setY(alphaA);
    }

    public PReLU(INDArray x, double l, double u) {
        super(x);
        this.l = l;
        this.u = u;
        this.alphaA = Nd4j.rand(x.shape(),l,u,Nd4j.getRandom());
        setY(alphaA);
    }

    public PReLU(INDArray x, INDArray y) {
        super(x);
        this.alphaA = y;
        setY(alphaA);
    }

    public PReLU(INDArray x, INDArray y, INDArray z) {
        super(x,y,z,x.lengthLong());
        this.alphaA = y;
    }

    public INDArray getAlpha() {
        return alphaA;
    }

    @Override
    public int opNum() {
        return 47;
    }

    @Override
    public String name() {
        return "prelu";
    }

    @Override
    public float op(float origin, float other) {
        //must fix
        return origin < 0 ? (float) alpha * origin : origin;
    }

    @Override
    public double op(double origin, double other) {
        //must fix
        return origin < 0 ?  alpha * origin : origin;
    }

    @Override
    public double op(double origin) {
        //must fix
        return origin < 0 ?  alpha * origin : origin;
    }

    @Override
    public float op(float origin) {
        //must fix
        return origin < 0 ? (float)alpha * origin : origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return origin;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new LeakyReLU(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(), alpha);
        else
            return new LeakyReLU(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length(), alpha);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new LeakyReLU(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(), alpha);
        else
            return new LeakyReLU(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length(), alpha);
    }

    @Override
    public TransformOp derivative() {
        //must fix
        return new LeakyReLUDerivative(x,y,z,n,1.0);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
    }
}
