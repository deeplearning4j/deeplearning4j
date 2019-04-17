package org.nd4j.autodiff.samediff.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import static org.nd4j.autodiff.samediff.ops.SDValidation.validateInteger;

/**
 * SameDiff random number generator operations<br>
 * Accessible via {@link SameDiff#random()}
 *
 * @author Alex Black
 */
public class SDRandom extends SDOps {

    public SDRandom(SameDiff sd) {
        super(sd);
    }

    /**
     * @see #bernoulli(String, double, SDVariable)
     */
    public SDVariable bernoulli(double p, SDVariable shape) {
        return bernoulli(null, p, shape);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a Bernoulli distribution,
     * with the specified probability. Array values will have value 1 with probability P and value 0 with probability
     * 1-P.<br>
     * See {@link #bernoulli(String, double, long...)}  for the equivalent function where the shape is
     * specified as a long[] instead
     *
     * @param name  Name of the new SDVariable
     * @param p     Probability of value 1
     * @param shape Shape of the new random SDVariable, as a 1D array
     * @return New SDVariable
     */
    public SDVariable bernoulli(String name, double p, SDVariable shape) {
        validateInteger("bernoulli random", shape);
        SDVariable ret = f().randomBernoulli(p, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #bernoulli(String, double, long...)
     */
    public SDVariable bernoulli(double p, long... shape) {
        return bernoulli(null, p, shape);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a Bernoulli distribution,
     * with the specified probability. Array values will have value 1 with probability P and value 0 with probability
     * 1-P.<br>
     * See {@link #bernoulli(String, double, SDVariable)}  for the equivalent function where the shape is
     * specified as a SDVarible instead
     *
     * @param name  Name of the new SDVariable
     * @param p     Probability of value 1
     * @param shape Shape of the new random SDVariable, as a 1D array
     * @return New SDVariable
     */
    public SDVariable bernoulli(String name, double p, long... shape) {
        SDVariable ret = f().randomBernoulli(p, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a Binomial distribution,
     * with the specified number of trials and probability.
     *
     * @param nTrials Number of trials parameter for the binomial distribution
     * @param p       Probability of success for each trial
     * @param shape   Shape of the new random SDVariable, as a 1D array
     * @return New SDVariable
     */
    public SDVariable binomial(int nTrials, double p, long... shape) {
        return binomial(null, nTrials, p, shape);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a Binomial distribution,
     * with the specified number of trials and probability.
     *
     * @param name    Name of the new SDVariable
     * @param nTrials Number of trials parameter for the binomial distribution
     * @param p       Probability of success for each trial
     * @param shape   Shape of the new random SDVariable, as a 1D array
     * @return New SDVariable
     */
    public SDVariable binomial(String name, int nTrials, double p, long... shape) {
        SDVariable ret = f().randomBinomial(nTrials, p, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a exponential distribution:
     * P(x) = lambda * exp(-lambda * x)
     *
     * @param lambda Must be > 0
     * @param shape  Shape of the output
     * @return new SDVariable
     */
    public SDVariable exponential(double lambda, SDVariable shape) {
        return exponential(null, lambda, shape);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a exponential distribution:
     * P(x) = lambda * exp(-lambda * x)
     *
     * @param name   Name of the output variable
     * @param lambda Must be > 0
     * @param shape  Shape of the new variable
     * @return new SDVaribale
     */
    public SDVariable exponential(String name, double lambda, SDVariable shape) {
        validateInteger("exponential random", shape);
        SDVariable ret = f().randomExponential(lambda, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #logNormal(String, double, double, long...)
     */
    public SDVariable logNormal(double mean, double stddev, long... shape) {
        return logNormal(null, mean, stddev, shape);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a Log Normal distribution,
     * i.e., {@code log(x) ~ N(mean, stdev)}<br>
     *
     * @param name   Name of the new SDVariable
     * @param mean   Mean value for the random array
     * @param stddev Standard deviation for the random array
     * @param shape  Shape of the new random SDVariable
     * @return New SDVariable
     */
    public SDVariable logNormal(String name, double mean, double stddev, long... shape) {
        SDVariable ret = f().randomLogNormal(mean, stddev, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #normal(String, double, double, SDVariable)
     */
    public SDVariable normal(double mean, double stddev, SDVariable shape) {
        return normal(null, mean, stddev, shape);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a Gaussian (normal) distribution,
     * N(mean, stdev)<br>
     * See {@link #normal(String, double, double, long...)} for the equivalent function where the shape is
     * specified as a long[] instead
     *
     * @param name   Name of the new SDVariable
     * @param mean   Mean value for the random array
     * @param stddev Standard deviation for the random array
     * @param shape  Shape of the new random SDVariable, as a 1D array
     * @return New SDVariable
     */
    public SDVariable normal(String name, double mean, double stddev, SDVariable shape) {
        validateInteger("normal (Gaussian) random", shape);
        SDVariable ret = f().randomNormal(mean, stddev, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #normal(String, double, double, long...)
     */
    public SDVariable normal(double mean, double stddev, long... shape) {
        return normal(null, mean, stddev, shape);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a Gaussian (normal) distribution,
     * N(mean, stdev)<br>
     * See {@link #normal(String, double, double, SDVariable)} for the equivalent function where the shape is
     * specified as a long[] instead
     *
     * @param name   Name of the new SDVariable
     * @param mean   Mean value for the random array
     * @param stddev Standard deviation for the random array
     * @param shape  Shape of the new random SDVariable
     * @return New SDVariable
     */
    public SDVariable normal(String name, double mean, double stddev, long... shape) {
        SDVariable ret = f().randomNormal(mean, stddev, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #normalTruncated(String, double, double, long...)
     */
    public SDVariable normalTruncated(double mean, double stddev, long... shape) {
        return normalTruncated(null, mean, stddev, shape);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a Gaussian (normal) distribution,
     * N(mean, stdev). However, any values more than 1 standard deviation from the mean are dropped and re-sampled<br>
     *
     * @param name   Name of the new SDVariable
     * @param mean   Mean value for the random array
     * @param stddev Standard deviation for the random array
     * @param shape  Shape of the new random SDVariable
     * @return New SDVariable
     */
    public SDVariable normalTruncated(String name, double mean, double stddev, long... shape) {
        SDVariable ret = f().randomNormalTruncated(mean, stddev, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #uniform(String, double, double, SDVariable)
     */
    public SDVariable uniform(double min, double max, SDVariable shape) {
        return uniform(null, min, max, shape);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a uniform distribution,
     * U(min,max)<br>
     * See {@link #uniform(double, double, long...)} for the equivalent function where the shape is
     * specified as a long[] instead
     *
     * @param name  Name of the new SDVariable
     * @param min   Minimum value
     * @param max   Maximum value. Must satisfy max >= min
     * @param shape Shape of the new random SDVariable, as a 1D array
     * @return New SDVariable
     */
    public SDVariable uniform(String name, double min, double max, SDVariable shape) {
        validateInteger("uniform random", shape);
        SDVariable ret = f().randomUniform(min, max, shape);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #uniform(String, double, double, long...)
     */
    public SDVariable uniform(double min, double max, long... shape) {
        return uniform(null, min, max, shape);
    }

    /**
     * Generate a new random SDVariable, where values are randomly sampled according to a uniform distribution,
     * U(min,max)<br>
     * See {@link #uniform(double, double, long...)} for the equivalent function where the shape is
     * specified as a SDVariable instead
     *
     * @param name  Name of the new SDVariable
     * @param min   Minimum value
     * @param max   Maximum value. Must satisfy max >= min
     * @param shape Shape of the new random SDVariable
     * @return New SDVariable
     */
    public SDVariable uniform(String name, double min, double max, long... shape) {
        SDVariable ret = f().randomUniform(min, max, shape);
        return updateVariableNameAndReference(ret, name);
    }

}
