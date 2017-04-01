package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author raver119@gmail.com
 */
public class Histogram extends BaseTransformOp {
    int numBins = 0;

    public Histogram() {
        //no-op
    }

    public Histogram(INDArray x, INDArray z) {
        setX(x);
        setZ(z);
        numBins = z.length();

        double max = x.maxNumber().doubleValue();
        double min = x.minNumber().doubleValue();

        this.extraArgs = new Object[] {(double) numBins, min, max};
    }

    public Histogram(INDArray x, int numberOfBins) {
        this(x, Nd4j.create(numberOfBins));
    }

    @Override
    public int opNum() {
        return 48;
    }

    @Override
    public String name() {
        return "histogram";
    }

    @Override
    public boolean isExecSpecial() {
        return true;
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
        return null;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        return null;
    }
}
