package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

/**
 * [1, 2, 3, 1] -> [0, 0, 1, 0]
 * @author Adam Gibson
 */
public class IsMax extends BaseTransformOp {
    public IsMax(INDArray x, INDArray z) {
        super(x, z);
    }

    public IsMax() {
    }

    public IsMax(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public IsMax(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public IsMax(INDArray x) {
        super(x);
    }

    public IsMax(INDArray x, INDArray z, long n, int... dimensions) {
        super(x, z, n);
        this.extraArgs = new Object[dimensions.length + 1];
        this.extraArgs[0] = dimensions.length;
        for( int i = 0; i < dimensions.length; i++)
            this.extraArgs[i + 1] = dimensions[i];
    }

    public IsMax(INDArray x, INDArray y, INDArray z, long n, int... dimensions) {
        super(x, y, z, n);
        this.extraArgs = new Object[dimensions.length+1];
        this.extraArgs[0] = dimensions.length;
        for( int i = 0; i < dimensions.length; i++)
            this.extraArgs[i + 1] = dimensions[i];
    }

    public IsMax(INDArray x, int... dimensions) {
        super(x);
        this.extraArgs = new Object[dimensions.length + 1];
        this.extraArgs[0] = dimensions.length;
        for( int i = 0; i < dimensions.length; i++ )
            this.extraArgs[i + 1] = dimensions[i];
    }

    @Override
    public int opNum() {
        return 41;
    }

    @Override
    public String name() {
        return "ismax";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return null;
    }

    @Override
    public float op(float origin, float other) {
        return 0;
    }

    @Override
    public double op(double origin, double other) {
        return 0;
    }

    @Override
    public double op(double origin) {
        return 0;
    }

    @Override
    public float op(float origin) {
        return 0;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return null;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new IsMax(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new IsMax(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new IsMax(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new IsMax(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }
}
