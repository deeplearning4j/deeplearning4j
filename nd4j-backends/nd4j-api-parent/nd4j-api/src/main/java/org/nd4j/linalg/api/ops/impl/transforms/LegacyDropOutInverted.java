package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

/**
 * Inverted DropOut implementation as Op
 *
 * PLEASE NOTE: This is legacy DropOutInverted implementation, please consider using op with the same name from randomOps
 * @author raver119@gmail.com
 */
public class LegacyDropOutInverted extends BaseTransformOp {

    private double p;

    public LegacyDropOutInverted() {

    }

    public LegacyDropOutInverted(INDArray x, double p) {
        super(x);
        this.p = p;
        init(x, null, x, x.length());
    }

    public LegacyDropOutInverted(INDArray x, INDArray z, double p) {
        super(x, z);
        this.p = p;
        init(x, null, z, x.length());
    }

    public LegacyDropOutInverted(INDArray x, INDArray z, double p, long n) {
        super(x, z, n);
        this.p = p;
        init(x, null, z, n);
    }

    @Override
    public int opNum() {
        return 44;
    }

    @Override
    public String name() {
        return "legacy_dropout_inverted";
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
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new LegacyDropOutInverted(xAlongDimension, z.vectorAlongDimension(index, dimension), p,
                            xAlongDimension.length());
        else
            return new LegacyDropOutInverted(xAlongDimension, z.vectorAlongDimension(index, dimension), p,
                            xAlongDimension.length());
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new LegacyDropOutInverted(xAlongDimension, z.tensorAlongDimension(index, dimension), p,
                            xAlongDimension.length());
        else
            return new LegacyDropOutInverted(xAlongDimension, z.tensorAlongDimension(index, dimension), p,
                            xAlongDimension.length());

    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {p, (double) n};
    }
}
