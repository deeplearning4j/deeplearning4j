package org.nd4j.linalg.jcublas.rng.distribution;

import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.special.Beta;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.SaddlePointExpansion;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.rng.JcudaRandom;

/**
 * Binomial distribution with cuda
 *
 * @author Adam Gibson
 */
public class BinomialDistribution extends BaseJCudaDistribution {

    /**
     * The number of trials.
     */
    private final int numberOfTrials;
    /**
     * The probability of success.
     */
    private double probabilityOfSuccess;
    private INDArray pNDArray;

    public BinomialDistribution(JcudaRandom random, INDArray pNDArray, int numberOfTrials) {
        super(random);
        this.pNDArray = pNDArray;
        this.numberOfTrials = numberOfTrials;
    }

    public BinomialDistribution(JcudaRandom random, int numberOfTrials, double probabilityOfSuccess) {
        super(random);
        this.numberOfTrials = numberOfTrials;
        this.probabilityOfSuccess = probabilityOfSuccess;
    }

    @Override
    public double probability(double x) {
        double ret;
        if (x < 0 || x > numberOfTrials) {
            ret = 0.0;
        } else {
            ret = FastMath.exp(SaddlePointExpansion.logBinomialProbability((int) x,
                    numberOfTrials, probabilityOfSuccess,
                    1.0 - probabilityOfSuccess));
        }
        return ret;
    }

    @Override
    public double density(double x) {
        return 0;
    }

    @Override
    public double cumulativeProbability(double x) {
        double ret;
        if (x < 0) {
            ret = 0.0;
        } else if (x >= numberOfTrials) {
            ret = 1.0;
        } else {
            ret = 1.0 - Beta.regularizedBeta(probabilityOfSuccess,
                    x + 1.0, numberOfTrials - x);
        }
        return ret;
    }

    @Override
    public double cumulativeProbability(double x0, double x1) throws NumberIsTooLargeException {
        return 0;
    }

    @Override
    public double inverseCumulativeProbability(double p) throws OutOfRangeException {
        return 0;
    }

    @Override
    public double getNumericalMean() {
        return numberOfTrials * probabilityOfSuccess;
    }

    @Override
    public double getNumericalVariance() {
        return probabilityOfSuccess < 1.0 ? 0 : numberOfTrials;
    }

    @Override
    public double getSupportLowerBound() {
        return probabilityOfSuccess < 1.0 ? 0 : numberOfTrials;
    }

    @Override
    public double getSupportUpperBound() {
        return probabilityOfSuccess > 0.0 ? numberOfTrials : 0;
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
        return true;
    }


    @Override
    public double[] sample(int sampleSize) {
        CudaDoubleDataBuffer buffer = new CudaDoubleDataBuffer(sampleSize);
        if (pNDArray != null) {

        } else {
            doBinomialDouble(probabilityOfSuccess, buffer.pointer(), numberOfTrials, buffer.length());
        }
        double[] buffer2 = buffer.asDouble();
        buffer.destroy();
        return buffer2;
    }

    @Override
    public INDArray sample(int[] shape) {
        INDArray ret = Nd4j.create(shape);
        JCudaBuffer buffer = (JCudaBuffer) ret.data();
        if (ret.data().dataType() == DataBuffer.DOUBLE) {
            if (pNDArray != null) {
                doBinomialDouble(pNDArray, buffer.pointer(), numberOfTrials, buffer.length());
            } else {
                doBinomialDouble(probabilityOfSuccess, buffer.pointer(), numberOfTrials, buffer.length());
            }
        } else {
            if (pNDArray != null) {
                doBinomial(pNDArray, buffer.pointer(), numberOfTrials, buffer.length());

            } else {
                doBinomial((float) probabilityOfSuccess, buffer.pointer(), numberOfTrials, buffer.length());

            }
        }
        return ret;
    }

    @Override
    public double probability(double x0, double x1) throws NumberIsTooLargeException {
        return 0.0;
    }
}
