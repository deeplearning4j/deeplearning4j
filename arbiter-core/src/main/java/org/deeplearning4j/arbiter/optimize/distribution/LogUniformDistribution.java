package org.deeplearning4j.arbiter.optimize.distribution;

import com.google.common.base.Preconditions;
import lombok.Getter;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;

import java.util.Random;

/**
 * Log uniform distribution, with support in range [min, max] for min > 0
 *
 * Reference: https://www.vosesoftware.com/riskwiki/LogUniformdistribution.php
 *
 * @author Alex Black
 */
public class LogUniformDistribution implements RealDistribution {

    @Getter private final double min;
    @Getter private final double max;

    private final double logMin;
    private final double logMax;

    private transient Random rng = new Random();

    /**
     *
     * @param min Minimum value
     * @param max Maximum value
     */
    public LogUniformDistribution(double min, double max) {
        Preconditions.checkArgument(min > 0, "Minimum must be > 0. Got: " + min);
        Preconditions.checkArgument(max > min, "Maximum must be > min. Got: (min, max)=("
                + min + "," + max + ")");
        this.min = min;
        this.max = max;

        this.logMin = Math.log(min);
        this.logMax = Math.log(max);
    }

    @Override
    public double probability(double x) {
        if(x < min || x > max){
            return 0;
        }

        return 1.0 / (x * (logMax - logMin));
    }

    @Override
    public double density(double x) {
        return probability(x);
    }

    @Override
    public double cumulativeProbability(double x) {
        if(x <= min){
            return 0.0;
        } else if(x >= max){
            return 1.0;
        }

        return (Math.log(x)-logMin)/(logMax-logMin);
    }

    @Override
    public double cumulativeProbability(double x0, double x1) throws NumberIsTooLargeException {
        return cumulativeProbability(x1) - cumulativeProbability(x0);
    }

    @Override
    public double inverseCumulativeProbability(double p) throws OutOfRangeException {
        Preconditions.checkArgument(p >= 0 && p <= 1, "Invalid input: " + p);
        return Math.exp(p * (logMax-logMin) + logMin);
    }

    @Override
    public double getNumericalMean() {
        return (max-min)/(logMax-logMin);
    }

    @Override
    public double getNumericalVariance() {
        double d1 = (logMax-logMin)*(max*max - min*min) - 2*(max-min)*(max-min);
        return d1 / (2*Math.pow(logMax-logMin, 2.0));
    }

    @Override
    public double getSupportLowerBound() {
        return min;
    }

    @Override
    public double getSupportUpperBound() {
        return max;
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

    @Override
    public void reseedRandomGenerator(long seed) {
        rng.setSeed(seed);
    }

    @Override
    public double sample() {
        return inverseCumulativeProbability(rng.nextDouble());
    }

    @Override
    public double[] sample(int sampleSize) {
        double[] d = new double[sampleSize];
        for( int i=0; i<sampleSize; i++ ){
            d[i] = sample();
        }
        return d;
    }

    @Override
    public String toString(){
        return "LogUniformDistribution(min=" + min + ",max=" + max + ")";
    }
}
