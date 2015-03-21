package org.nd4j.linalg.jcublas.rng.distribution;

import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.rng.JcudaRandom;

/**
 * Uniform distribution based on an upper
 * and lower bound
 *
 * @author Adam Gibson
 */
public class UniformDistribution extends BaseJCudaDistribution {
    private double upper, lower;

    public UniformDistribution(JcudaRandom random, double upper, double lower) {
        super(random);
        this.upper = upper;
        this.lower = lower;
    }

    @Override
    public double probability(double x) {
        return 0;
    }

    @Override
    public double density(double x) {
        return 0;
    }

    @Override
    public double cumulativeProbability(double x) {
        if (x <= lower) {
            return 0;
        }
        if (x >= upper) {
            return 1;
        }
        return (x - lower) / (upper - lower);
    }

    @Override
    public double cumulativeProbability(double x0, double x1) throws NumberIsTooLargeException {
        return 0;
    }

    @Override
    public double inverseCumulativeProbability(double p) throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }
        return p * (upper - lower) + lower;
    }

    @Override
    public double getNumericalMean() {
        return 0.5 * (lower + upper);
    }

    @Override
    public double getNumericalVariance() {
        double ul = upper - lower;
        return ul * ul / 12;
    }

    @Override
    public double getSupportLowerBound() {
        return lower;
    }

    @Override
    public double getSupportUpperBound() {
        return upper;
    }

    @Override
    public boolean isSupportLowerBoundInclusive() {
        return true;
    }

    @Override
    public boolean isSupportUpperBoundInclusive() {
        return true;
    }

    @Override
    public boolean isSupportConnected() {
        return true;
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public double sample() {
        final double u = random.nextDouble();
        return u * upper + (1 - u) * lower;
    }

    @Override
    public double[] sample(int sampleSize) {
        CudaDoubleDataBuffer buffer = new CudaDoubleDataBuffer(sampleSize);
        doSampleUniformDouble(buffer.pointer(), lower, upper, buffer.length());
        double[] buffer2 = buffer.asDouble();
        buffer.destroy();
        return buffer2;
    }

    @Override
    public INDArray sample(int[] shape) {
        INDArray ret = Nd4j.create(shape);
        JCudaBuffer buffer = (JCudaBuffer) ret.data();
        if (buffer.dataType() == DataBuffer.FLOAT)
            doSampleUniform(buffer.pointer(), (float) lower, (float) upper, buffer.length());
        else if (buffer.dataType() == DataBuffer.DOUBLE)
            doSampleUniformDouble(buffer.pointer(), lower, upper, buffer.length());

        return ret;
    }

    @Override
    public double probability(double x0, double x1) throws NumberIsTooLargeException {
        return 0;
    }
}
