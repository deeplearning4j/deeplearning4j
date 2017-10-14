package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.Collections;
import java.util.List;

/**
 * Set
 * @author Adam Gibson
 */
public class Set extends BaseTransformOp {
    public Set(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Set(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public Set(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public Set(INDArray x, INDArray z) {
        super(x, z);
    }

    public Set() {}

    public Set(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Set(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Set(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 16;
    }

    @Override
    public String name() {
        return "set";
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
        return other;
    }

    @Override
    public float op(float origin, float other) {
        return other;
    }

    @Override
    public double op(double origin, double other) {
        return other;
    }

    @Override
    public double op(double origin) {
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
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new Set(x.tensorAlongDimension(index, dimension), y.vectorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Set(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension),
                            xAlongDimension.length());

    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);
        if (y() != null)
            return new Set(x.tensorAlongDimension(index, dimension), y.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new Set(x.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension),
                            xAlongDimension.length());

    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ym1 = f().rsub(rarg(),f().one(getShape()));
        DifferentialFunction ret = f().mul(f().mul(rarg(),f().pow(larg(), 2.0)),larg());


        return Collections.singletonList(ret);
    }

}
