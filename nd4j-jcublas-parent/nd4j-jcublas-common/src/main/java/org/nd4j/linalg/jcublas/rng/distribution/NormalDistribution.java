package org.nd4j.linalg.jcublas.rng.distribution;

import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.rng.JcudaRandom;

/**
 * Normal Distribution
 *
 * @author Adam Gibson
 */
public class NormalDistribution extends BaseJCudaDistribution {
    /**
     * Default inverse cumulative probability accuracy.
     *
     * @since 2.1
     */
    public static final double DEFAULT_INVERSE_ABSOLUTE_ACCURACY = 1e-9;
    /**
     * Serializable version identifier.
     */
    private static final long serialVersionUID = 8589540077390120676L;
    /**
     * &radic;(2 &pi;)
     */
    private static final double SQRT2PI = FastMath.sqrt(2 * FastMath.PI);
    /**
     * &radic;(2)
     */
    private static final double SQRT2 = FastMath.sqrt(2.0);
    /**
     * Standard deviation of this distribution.
     */
    private final double standardDeviation;
    /**
     * Mean of this distribution.
     */
    private double mean;
    //more than one mean
    private INDArray means;
    /**
     * Inverse cumulative probability accuracy.
     */
    private double solverAbsoluteAccuracy;

    /**
     * Create a normal distribution with mean equal to zero and standard
     * deviation equal to one.
     */
    public NormalDistribution() {
        this(0, 1);
    }


    public NormalDistribution(JcudaRandom random, INDArray means, double standardDeviation) {
        super(random);
        this.means = means;
        this.standardDeviation = standardDeviation;
    }

    /**
     * Create a normal distribution using the given mean and standard deviation.
     *
     * @param mean Mean for this distribution.
     * @param sd   Standard deviation for this distribution.
     * @throws org.apache.commons.math3.exception.NotStrictlyPositiveException if {@code sd <= 0}.
     */
    public NormalDistribution(double mean, double sd)
            throws NotStrictlyPositiveException {
        this(mean, sd, DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }

    /**
     * Create a normal distribution using the given mean, standard deviation and
     * inverse cumulative distribution accuracy.
     *
     * @param mean               Mean for this distribution.
     * @param sd                 Standard deviation for this distribution.
     * @param inverseCumAccuracy Inverse cumulative probability accuracy.
     * @throws NotStrictlyPositiveException if {@code sd <= 0}.
     * @since 2.1
     */
    public NormalDistribution(double mean, double sd, double inverseCumAccuracy)
            throws NotStrictlyPositiveException {
        this(Nd4j.getRandom(), mean, sd, inverseCumAccuracy);
    }

    /**
     * Creates a normal distribution.
     *
     * @param rng                Random number generator.
     * @param mean               Mean for this distribution.
     * @param sd                 Standard deviation for this distribution.
     * @param inverseCumAccuracy Inverse cumulative probability accuracy.
     * @throws NotStrictlyPositiveException if {@code sd <= 0}.
     * @since 3.1
     */
    public NormalDistribution(Random rng,
                              double mean,
                              double sd,
                              double inverseCumAccuracy)
            throws NotStrictlyPositiveException {
        super((JcudaRandom) rng);

        if (sd <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.STANDARD_DEVIATION, sd);
        }

        this.mean = mean;
        standardDeviation = sd;
        solverAbsoluteAccuracy = inverseCumAccuracy;
    }

    /**
     * Normal distribution with a matrix of means
     *
     * @param mean the means to use
     * @param std  the standard deviation
     */
    public NormalDistribution(INDArray mean, double std) {
        super((JcudaRandom) Nd4j.getRandom());
        this.means = mean;
        this.standardDeviation = std;
    }

    @Override
    public double probability(double x) {
        return 0;
    }

    @Override
    public double density(double x) {
        final double x0 = x - mean;
        final double x1 = x0 / standardDeviation;
        return FastMath.exp(-0.5 * x1 * x1) / (standardDeviation * SQRT2PI);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double probability(double x0,
                              double x1)
            throws NumberIsTooLargeException {
        if (x0 > x1) {
            throw new NumberIsTooLargeException(LocalizedFormats.LOWER_ENDPOINT_ABOVE_UPPER_ENDPOINT,
                    x0, x1, true);
        }
        final double denom = standardDeviation * SQRT2;
        final double v0 = (x0 - mean) / denom;
        final double v1 = (x1 - mean) / denom;
        return 0.5 * Erf.erf(v0, v1);
    }


    @Override
    public double cumulativeProbability(double x) {
        final double dev = x - mean;
        if (FastMath.abs(dev) > 40 * standardDeviation) {
            return dev < 0 ? 0.0d : 1.0d;
        }
        return 0.5 * (1 + Erf.erf(dev / (standardDeviation * SQRT2)));
    }

    @Override
    public double cumulativeProbability(double x0, double x1) throws NumberIsTooLargeException {
        return probability(x0, x1);
    }

    @Override
    public double inverseCumulativeProbability(double p) throws OutOfRangeException {
        if (p < 0.0 || p > 1.0) {
            throw new OutOfRangeException(p, 0, 1);
        }
        return mean + standardDeviation * SQRT2 * Erf.erfInv(2 * p - 1);
    }

    @Override
    public double getNumericalMean() {
        return mean;
    }

    @Override
    public double getNumericalVariance() {
        final double s = standardDeviation;
        return s * s;
    }

    @Override
    public double getSupportLowerBound() {
        return Double.NEGATIVE_INFINITY;
    }

    @Override
    public double getSupportUpperBound() {
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public boolean isSupportLowerBoundInclusive() {
        return false;
    }

    @Override
    public boolean isSupportUpperBoundInclusive() {
        return false;
    }

    @Override
    public boolean isSupportConnected() {
        return false;
    }


    @Override
    public double sample() {
        return standardDeviation * random.nextGaussian() + mean;
    }

    @Override
    public double[] sample(int sampleSize) {
        CudaDoubleDataBuffer buffer = new CudaDoubleDataBuffer(sampleSize);
        doSampleNormal(mean, standardDeviation, buffer.pointer(), sampleSize);
        double[] buffer2 = buffer.asDouble();
        buffer.destroy();
        return buffer2;
    }

    @Override
    public INDArray sample(int[] shape) {
        INDArray ret = Nd4j.create(shape);
        JCudaBuffer buffer = (JCudaBuffer) ret.data();
        if (means != null) {
            if (buffer.dataType() != DataBuffer.DOUBLE)
                doSampleNormal(buffer.pointer(), means, (float) standardDeviation);

            else
                doSampleNormalDouble(buffer.pointer(), means, standardDeviation);


        } else {
            if (buffer.dataType() == DataBuffer.FLOAT)
                doSampleNormal((float) mean, (float) standardDeviation, buffer.pointer(), buffer.length());
            else if (buffer.dataType() == DataBuffer.DOUBLE)
                doSampleNormal(mean, standardDeviation, buffer.pointer(), buffer.length());

        }

        return ret;
    }
}
