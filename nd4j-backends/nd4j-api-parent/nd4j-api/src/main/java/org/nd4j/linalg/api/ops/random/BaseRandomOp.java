package org.nd4j.linalg.api.ops.random;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.RandomOp;
import org.nd4j.linalg.api.shape.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseRandomOp extends BaseOp implements RandomOp {

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
        return null;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        return null;
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>(1);
        ret.add(this.shape);
        return ret;
    }


}
