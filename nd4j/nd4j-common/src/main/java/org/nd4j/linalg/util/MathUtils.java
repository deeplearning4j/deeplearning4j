/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.util;


import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.primitives.Counter;
import org.nd4j.util.SetUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;


/**
 * This is a math jcuda.utils class.
 *
 * @author Adam Gibson
 */
public class MathUtils {



    /**
     * The natural logarithm of 2.
     */
    public static final double log2 = Math.log(2);
    /**
     * The small deviation allowed in double comparisons.
     */
    public static final double SMALL = 1e-6;


    public static double pow(double base, double exponent) {
        double result = 1;

        if (exponent == 0) {
            return result;
        }
        if (exponent < 0) {
            return 1 / pow(base, exponent * -1);
        }

        return FastMath.pow(base, exponent);
    }

    /**
     * Normalize a value
     * (val - min) / (max - min)
     *
     * @param val value to normalize
     * @param max max value
     * @param min min value
     * @return the normalized value
     */
    public static double normalize(double val, double min, double max) {
        if (max < min)
            throw new IllegalArgumentException("Max must be greather than min");

        return (val - min) / (max - min);
    }

    /**
     * Clamps the value to a discrete value
     *
     * @param value the value to clamp
     * @param min   min for the probability distribution
     * @param max   max for the probability distribution
     * @return the discrete value
     */
    public static int clamp(int value, int min, int max) {
        if (value < min)
            value = min;
        if (value > max)
            value = max;
        return value;
    }

    /**
     * Discretize the given value
     *
     * @param value    the value to discretize
     * @param min      the min of the distribution
     * @param max      the max of the distribution
     * @param binCount the number of bins
     * @return the discretized value
     */
    public static int discretize(double value, double min, double max, int binCount) {
        int discreteValue = (int) (binCount * normalize(value, min, max));
        return clamp(discreteValue, 0, binCount - 1);
    }

    /**
     * See: http://stackoverflow.com/questions/466204/rounding-off-to-nearest-power-of-2
     *
     * @param v the number to getFromOrigin the next power of 2 for
     * @return the next power of 2 for the passed in value
     */
    public static long nextPowOf2(long v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;

    }

    /**
     * Generates a binomial distributed number using
     * the given rng
     *
     * @param rng
     * @param n
     * @param p
     * @return
     */
    public static int binomial(RandomGenerator rng, int n, double p) {
        if ((p < 0) || (p > 1)) {
            return 0;
        }
        int c = 0;
        for (int i = 0; i < n; i++) {
            if (rng.nextDouble() < p) {
                c++;
            }
        }
        return c;
    }

    /**
     * Generate a uniform random number from the given rng
     *
     * @param rng the rng to use
     * @param min the min num
     * @param max the max num
     * @return a number uniformly distributed between min and max
     */
    public static double uniform(Random rng, double min, double max) {
        return rng.nextDouble() * (max - min) + min;
    }

    /**
     * Returns the correlation coefficient of two double vectors.
     *
     * @param residuals       residuals
     * @param targetAttribute target attribute vector
     * @return the correlation coefficient or r
     */
    public static double correlation(double[] residuals, double targetAttribute[]) {
        double[] predictedValues = new double[residuals.length];
        for (int i = 0; i < predictedValues.length; i++) {
            predictedValues[i] = targetAttribute[i] - residuals[i];
        }
        double ssErr = ssError(predictedValues, targetAttribute);
        double total = ssTotal(residuals, targetAttribute);
        return 1 - (ssErr / total);
    }//end correlation

    /**
     * 1 / 1 + exp(-x)
     *
     * @param x
     * @return
     */
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.pow(Math.E, -x));
    }

    /**
     * How much of the variance is explained by the regression
     *
     * @param residuals       error
     * @param targetAttribute data for target attribute
     * @return the sum squares of regression
     */
    public static double ssReg(double[] residuals, double[] targetAttribute) {
        double mean = sum(targetAttribute) / targetAttribute.length;
        double ret = 0;
        for (int i = 0; i < residuals.length; i++) {
            ret += Math.pow(residuals[i] - mean, 2);
        }
        return ret;
    }

    /**
     * How much of the variance is NOT explained by the regression
     *
     * @param predictedValues predicted values
     * @param targetAttribute data for target attribute
     * @return the sum squares of regression
     */
    public static double ssError(double[] predictedValues, double[] targetAttribute) {
        double ret = 0;
        for (int i = 0; i < predictedValues.length; i++) {
            ret += Math.pow(targetAttribute[i] - predictedValues[i], 2);
        }
        return ret;
    }

    /**
     * Calculate string similarity with tfidf weights relative to each character
     * frequency and how many times a character appears in a given string
     * @param strings the strings to calculate similarity for
     * @return the cosine similarity between the strings
     */
    public static double stringSimilarity(String... strings) {
        if (strings == null)
            return 0;
        Counter<String> counter = new Counter<>();
        Counter<String> counter2 = new Counter<>();

        for (int i = 0; i < strings[0].length(); i++)
            counter.incrementCount(String.valueOf(strings[0].charAt(i)), 1.0f);

        for (int i = 0; i < strings[1].length(); i++)
            counter2.incrementCount(String.valueOf(strings[1].charAt(i)), 1.0f);
        Set<String> v1 = counter.keySet();
        Set<String> v2 = counter2.keySet();


        Set<String> both = SetUtils.intersection(v1, v2);

        double sclar = 0, norm1 = 0, norm2 = 0;
        for (String k : both)
            sclar += counter.getCount(k) * counter2.getCount(k);
        for (String k : v1)
            norm1 += counter.getCount(k) * counter.getCount(k);
        for (String k : v2)
            norm2 += counter2.getCount(k) * counter2.getCount(k);
        return sclar / Math.sqrt(norm1 * norm2);
    }

    /**
     * Returns the vector length (sqrt(sum(x_i))
     *
     * @param vector the vector to return the vector length for
     * @return the vector length of the passed in array
     */
    public static double vectorLength(double[] vector) {
        double ret = 0;
        if (vector == null)
            return ret;
        else {
            for (int i = 0; i < vector.length; i++) {
                ret += Math.pow(vector[i], 2);
            }

        }
        return ret;
    }

    /**
     * Inverse document frequency: the total docs divided by the number of times the word
     * appeared in a document
     *
     * @param totalDocs                       the total documents for the data applyTransformToDestination
     * @param numTimesWordAppearedInADocument the number of times the word occurred in a document
     * @return log(10) (totalDocs/numTImesWordAppearedInADocument)
     */
    public static double idf(double totalDocs, double numTimesWordAppearedInADocument) {
        return totalDocs > 0 ? Math.log10(totalDocs / numTimesWordAppearedInADocument) : 0;
    }

    /**
     * Term frequency: 1+ log10(count)
     *
     * @param count the count of a word or character in a given string or document
     * @return 1+ log(10) count
     */
    public static double tf(int count) {
        return count > 0 ? 1 + Math.log10(count) : 0;
    }

    /**
     * Return td * idf
     *
     * @param td  the term frequency (assumed calculated)
     * @param idf inverse document frequency (assumed calculated)
     * @return td * idf
     */
    public static double tfidf(double td, double idf) {
        return td * idf;
    }

    private static int charForLetter(char c) {
        char[] chars = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                        't', 'u', 'v', 'w', 'x', 'y', 'z'};
        for (int i = 0; i < chars.length; i++)
            if (chars[i] == c)
                return i;
        return -1;

    }

    /**
     * Total variance in target attribute
     *
     * @param residuals       error
     * @param targetAttribute data for target attribute
     * @return Total variance in target attribute
     */
    public static double ssTotal(double[] residuals, double[] targetAttribute) {
        return ssReg(residuals, targetAttribute) + ssError(residuals, targetAttribute);
    }

    /**
     * This returns the sum of the given array.
     *
     * @param nums the array of numbers to sum
     * @return the sum of the given array
     */
    public static double sum(double[] nums) {

        double ret = 0;
        for (double d : nums)
            ret += d;

        return ret;
    }//end sum

    /**
     * This will merge the coordinates of the given coordinate system.
     *
     * @param x the x coordinates
     * @param y the y coordinates
     * @return a vector such that each (x,y) pair is at ret[i],ret[i+1]
     */
    public static double[] mergeCoords(double[] x, double[] y) {
        if (x.length != y.length)
            throw new IllegalArgumentException(
                            "Sample sizes must be the same for each data applyTransformToDestination.");
        double[] ret = new double[x.length + y.length];

        for (int i = 0; i < x.length; i++) {
            ret[i] = x[i];
            ret[i + 1] = y[i];
        }
        return ret;
    }//end mergeCoords

    /**
     * This will merge the coordinates of the given coordinate system.
     *
     * @param x the x coordinates
     * @param y the y coordinates
     * @return a vector such that each (x,y) pair is at ret[i],ret[i+1]
     */
    public static List<Double> mergeCoords(List<Double> x, List<Double> y) {
        if (x.size() != y.size())
            throw new IllegalArgumentException(
                            "Sample sizes must be the same for each data applyTransformToDestination.");

        List<Double> ret = new ArrayList<Double>();

        for (int i = 0; i < x.size(); i++) {
            ret.add(x.get(i));
            ret.add(y.get(i));
        }
        return ret;
    }//end mergeCoords

    /**
     * This returns the minimized loss values for a given vector.
     * It is assumed that  the x, y pairs are at
     * vector[i], vector[i+1]
     *
     * @param vector the vector of numbers to getFromOrigin the weights for
     * @return a double array with w_0 and w_1 are the associated indices.
     */
    public static double[] weightsFor(List<Double> vector) {
        /* split coordinate system */
        List<double[]> coords = coordSplit(vector);
        /* x vals */
        double[] x = coords.get(0);
        /* y vals */
        double[] y = coords.get(1);


        double meanX = sum(x) / x.length;
        double meanY = sum(y) / y.length;

        double sumOfMeanDifferences = sumOfMeanDifferences(x, y);
        double xDifferenceOfMean = sumOfMeanDifferencesOnePoint(x);

        double w_1 = sumOfMeanDifferences / xDifferenceOfMean;

        double w_0 = meanY - (w_1) * meanX;

        //double w_1=(n*sumOfProducts(x,y) - sum(x) * sum(y))/(n*sumOfSquares(x) - Math.pow(sum(x),2));

        //	double w_0=(sum(y) - (w_1 * sum(x)))/n;

        double[] ret = new double[vector.size()];
        ret[0] = w_0;
        ret[1] = w_1;

        return ret;
    }//end weightsFor

    /**
     * This will return the squared loss of the given
     * points
     *
     * @param x   the x coordinates to use
     * @param y   the y coordinates to use
     * @param w_0 the first weight
     * @param w_1 the second weight
     * @return the squared loss of the given points
     */
    public static double squaredLoss(double[] x, double[] y, double w_0, double w_1) {
        double sum = 0;
        for (int j = 0; j < x.length; j++) {
            sum += Math.pow((y[j] - (w_1 * x[j] + w_0)), 2);
        }
        return sum;
    }//end squaredLoss

    public static double w_1(double[] x, double[] y, int n) {
        return (n * sumOfProducts(x, y) - sum(x) * sum(y)) / (n * sumOfSquares(x) - Math.pow(sum(x), 2));
    }

    public static double w_0(double[] x, double[] y, int n) {
        double weight1 = w_1(x, y, n);

        return (sum(y) - (weight1 * sum(x))) / n;
    }

    /**
     * This returns the minimized loss values for a given vector.
     * It is assumed that  the x, y pairs are at
     * vector[i], vector[i+1]
     *
     * @param vector the vector of numbers to getFromOrigin the weights for
     * @return a double array with w_0 and w_1 are the associated indices.
     */
    public static double[] weightsFor(double[] vector) {

        /* split coordinate system */
        List<double[]> coords = coordSplit(vector);
        /* x vals */
        double[] x = coords.get(0);
        /* y vals */
        double[] y = coords.get(1);


        double meanX = sum(x) / x.length;
        double meanY = sum(y) / y.length;

        double sumOfMeanDifferences = sumOfMeanDifferences(x, y);
        double xDifferenceOfMean = sumOfMeanDifferencesOnePoint(x);

        double w_1 = sumOfMeanDifferences / xDifferenceOfMean;

        double w_0 = meanY - (w_1) * meanX;


        double[] ret = new double[vector.length];
        ret[0] = w_0;
        ret[1] = w_1;

        return ret;
    }//end weightsFor

    public static double errorFor(double actual, double prediction) {
        return actual - prediction;
    }

    /**
     * Used for calculating top part of simple regression for
     * beta 1
     *
     * @param vector  the x coordinates
     * @param vector2 the y coordinates
     * @return the sum of mean differences for the input vectors
     */
    public static double sumOfMeanDifferences(double[] vector, double[] vector2) {
        double mean = sum(vector) / vector.length;
        double mean2 = sum(vector2) / vector2.length;
        double ret = 0;
        for (int i = 0; i < vector.length; i++) {
            double vec1Diff = vector[i] - mean;
            double vec2Diff = vector2[i] - mean2;
            ret += vec1Diff * vec2Diff;
        }
        return ret;
    }//end sumOfMeanDifferences

    /**
     * Used for calculating top part of simple regression for
     * beta 1
     *
     * @param vector the x coordinates
     * @return the sum of mean differences for the input vectors
     */
    public static double sumOfMeanDifferencesOnePoint(double[] vector) {
        double mean = sum(vector) / vector.length;
        double ret = 0;
        for (int i = 0; i < vector.length; i++) {
            double vec1Diff = Math.pow(vector[i] - mean, 2);
            ret += vec1Diff;
        }
        return ret;
    }//end sumOfMeanDifferences

    /**
     * This returns the product of all numbers in the given array.
     *
     * @param nums the numbers to multiply over
     * @return the product of all numbers in the array, or 0
     * if the length is or or nums i null
     */
    public static double times(double[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        double ret = 1;
        for (int i = 0; i < nums.length; i++)
            ret *= nums[i];
        return ret;
    }//end times

    /**
     * This returns the sum of products for the given
     * numbers.
     *
     * @param nums the sum of products for the give numbers
     * @return the sum of products for the given numbers
     */
    public static double sumOfProducts(double[]... nums) {
        if (nums == null || nums.length < 1)
            return 0;
        double sum = 0;

        for (int i = 0; i < nums.length; i++) {
            /* The ith column for all of the rows */
            double[] column = column(i, nums);
            sum += times(column);

        }
        return sum;
    }//end sumOfProducts

    /**
     * This returns the given column over an n arrays
     *
     * @param column the column to getFromOrigin values for
     * @param nums   the arrays to extract values from
     * @return a double array containing all of the numbers in that column
     * for all of the arrays.
     * @throws IllegalArgumentException if the index is < 0
     */
    private static double[] column(int column, double[]... nums) throws IllegalArgumentException {

        double[] ret = new double[nums.length];

        for (int i = 0; i < nums.length; i++) {
            double[] curr = nums[i];
            ret[i] = curr[column];
        }
        return ret;
    }//end column

    /**
     * This returns the coordinate split in a list of coordinates
     * such that the values for ret[0] are the x values
     * and ret[1] are the y values
     *
     * @param vector the vector to split with x and y values/
     * @return a coordinate split for the given vector of values.
     * if null, is passed in null is returned
     */
    public static List<double[]> coordSplit(double[] vector) {

        if (vector == null)
            return null;
        List<double[]> ret = new ArrayList<double[]>();
        /* x coordinates */
        double[] xVals = new double[vector.length / 2];
        /* y coordinates */
        double[] yVals = new double[vector.length / 2];
        /* current points */
        int xTracker = 0;
        int yTracker = 0;
        for (int i = 0; i < vector.length; i++) {
            //even value, x coordinate
            if (i % 2 == 0)
                xVals[xTracker++] = vector[i];
            //y coordinate
            else
                yVals[yTracker++] = vector[i];
        }
        ret.add(xVals);
        ret.add(yVals);

        return ret;
    }//end coordSplit

    /**
     * This will partition the given whole variable data applyTransformToDestination in to the specified chunk number.
     *
     * @param arr   the data applyTransformToDestination to pass in
     * @param chunk the number to separate by
     * @return a partition data applyTransformToDestination relative to the passed in chunk number
     */
    public static List<List<Double>> partitionVariable(List<Double> arr, int chunk) {
        int count = 0;
        List<List<Double>> ret = new ArrayList<List<Double>>();


        while (count < arr.size()) {

            List<Double> sublist = arr.subList(count, count + chunk);
            count += chunk;
            ret.add(sublist);

        }
        //All data sets must be same size
        for (List<Double> lists : ret) {
            if (lists.size() < chunk)
                ret.remove(lists);
        }
        return ret;
    }//end partitionVariable

    /**
     * This returns the coordinate split in a list of coordinates
     * such that the values for ret[0] are the x values
     * and ret[1] are the y values
     *
     * @param vector the vector to split with x and y values
     *               Note that the list will be more stable due to the size operator.
     *               The array version will have extraneous values if not monitored
     *               properly.
     * @return a coordinate split for the given vector of values.
     * if null, is passed in null is returned
     */
    public static List<double[]> coordSplit(List<Double> vector) {

        if (vector == null)
            return null;
        List<double[]> ret = new ArrayList<double[]>();
        /* x coordinates */
        double[] xVals = new double[vector.size() / 2];
        /* y coordinates */
        double[] yVals = new double[vector.size() / 2];
        /* current points */
        int xTracker = 0;
        int yTracker = 0;
        for (int i = 0; i < vector.size(); i++) {
            //even value, x coordinate
            if (i % 2 == 0)
                xVals[xTracker++] = vector.get(i);
            //y coordinate
            else
                yVals[yTracker++] = vector.get(i);
        }
        ret.add(xVals);
        ret.add(yVals);

        return ret;
    }//end coordSplit

    /**
     * This returns the x values of the given vector.
     * These are assumed to be the even values of the vector.
     *
     * @param vector the vector to getFromOrigin the values for
     * @return the x values of the given vector
     */
    public static double[] xVals(double[] vector) {


        if (vector == null)
            return null;
        double[] x = new double[vector.length / 2];
        int count = 0;
        for (int i = 0; i < vector.length; i++) {
            if (i % 2 != 0)
                x[count++] = vector[i];
        }
        return x;
    }//end xVals

    /**
     * This returns the odd indexed values for the given vector
     *
     * @param vector the odd indexed values of rht egiven vector
     * @return the y values of the given vector
     */
    public static double[] yVals(double[] vector) {
        double[] y = new double[vector.length / 2];
        int count = 0;
        for (int i = 0; i < vector.length; i++) {
            if (i % 2 == 0)
                y[count++] = vector[i];
        }
        return y;
    }//end yVals

    /**
     * This returns the sum of squares for the given vector.
     *
     * @param vector the vector to obtain the sum of squares for
     * @return the sum of squares for this vector
     */
    public static double sumOfSquares(double[] vector) {
        double ret = 0;
        for (double d : vector)
            ret += Math.pow(d, 2);
        return ret;
    }

    /**
     * This returns the determination coefficient of two vectors given a length
     *
     * @param y1 the first vector
     * @param y2 the second vector
     * @param n  the length of both vectors
     * @return the determination coefficient or r^2
     */
    public static double determinationCoefficient(double[] y1, double[] y2, int n) {
        return Math.pow(correlation(y1, y2), 2);
    }

    /**
     * Returns the logarithm of a for base 2.
     *
     * @param a a double
     * @return the logarithm for base 2
     */
    public static double log2(double a) {
        if (a == 0)
            return 0.0;
        return Math.log(a) / log2;
    }

    /**
     * This returns the root mean squared error of two data sets
     *
     * @param real      the realComponent values
     * @param predicted the predicted values
     * @return the root means squared error for two data sets
     */
    public static double rootMeansSquaredError(double[] real, double[] predicted) {
        double ret = 1 / real.length;
        for (int i = 0; i < real.length; i++) {
            ret += Math.pow((real[i] - predicted[i]), 2);
        }
        return Math.sqrt(ret);
    }//end rootMeansSquaredError

    /**
     * This returns the entropy (information gain, or uncertainty of a random variable): -sum(x*log(x))
     *
     * @param vector the vector of values to getFromOrigin the entropy for
     * @return the entropy of the given vector
     */
    public static double entropy(double[] vector) {
        if (vector == null || vector.length == 0)
            return 0;
        else {
            double ret = 0;
            for (double d : vector)
                ret += d * Math.log(d);
            return -ret;

        }
    }//end entropy

    /**
     * This returns the kronecker delta of two doubles.
     *
     * @param i the first number to compare
     * @param j the second number to compare
     * @return 1 if they are equal, 0 otherwise
     */
    public static int kroneckerDelta(double i, double j) {
        return (i == j) ? 1 : 0;
    }

    /**
     * This calculates the adjusted r^2 including degrees of freedom.
     * Also known as calculating "strength" of a regression
     *
     * @param rSquared      the r squared value to calculate
     * @param numRegressors number of variables
     * @param numDataPoints size of the data applyTransformToDestination
     * @return an adjusted r^2 for degrees of freedom
     */
    public static double adjustedrSquared(double rSquared, int numRegressors, int numDataPoints) {
        double divide = (numDataPoints - 1) / (numDataPoints - numRegressors - 1);
        double rSquaredDiff = 1 - rSquared;
        return 1 - (rSquaredDiff * divide);
    }


    public static double[] normalizeToOne(double[] doubles) {
        normalize(doubles, sum(doubles));
        return doubles;
    }

    public static double min(double[] doubles) {
        double ret = doubles[0];
        for (double d : doubles)
            if (d < ret)
                ret = d;
        return ret;
    }

    public static double max(double[] doubles) {
        double ret = doubles[0];
        for (double d : doubles)
            if (d > ret)
                ret = d;
        return ret;
    }

    /**
     * Normalizes the doubles in the array using the given value.
     *
     * @param doubles the array of double
     * @param sum     the value by which the doubles are to be normalized
     * @throws IllegalArgumentException if sum is zero or NaN
     */
    public static void normalize(double[] doubles, double sum) {

        if (Double.isNaN(sum)) {
            throw new IllegalArgumentException("Can't normalize array. Sum is NaN.");
        }
        if (sum == 0) {
            // Maybe this should just be a return.
            throw new IllegalArgumentException("Can't normalize array. Sum is zero.");
        }
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] /= sum;
        }
    }//end normalize

    /**
     * Converts an array containing the natural logarithms of
     * probabilities stored in a vector back into probabilities.
     * The probabilities are assumed to sum to one.
     *
     * @param a an array holding the natural logarithms of the probabilities
     * @return the converted array
     */
    public static double[] logs2probs(double[] a) {

        double max = a[maxIndex(a)];
        double sum = 0.0;

        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = Math.exp(a[i] - max);
            sum += result[i];
        }

        normalize(result, sum);

        return result;
    }//end logs2probs

    /**
     * This returns the entropy for a given vector of probabilities.
     *
     * @param probabilities the probabilities to getFromOrigin the entropy for
     * @return the entropy of the given probabilities.
     */
    public static double information(double[] probabilities) {
        double total = 0.0;
        for (double d : probabilities) {
            total += (-1.0 * log2(d) * d);
        }
        return total;
    }//end information

    /**
     * Returns index of maximum element in a given
     * array of doubles. First maximum is returned.
     *
     * @param doubles the array of doubles
     * @return the index of the maximum element
     */
    public static /*@pure@*/ int maxIndex(double[] doubles) {

        double maximum = 0;
        int maxIndex = 0;

        for (int i = 0; i < doubles.length; i++) {
            if ((i == 0) || (doubles[i] > maximum)) {
                maxIndex = i;
                maximum = doubles[i];
            }
        }

        return maxIndex;
    }//end maxIndex

    /**
     * This will return the factorial of the given number n.
     *
     * @param n the number to getFromOrigin the factorial for
     * @return the factorial for this number
     */
    public static double factorial(double n) {
        if (n == 1 || n == 0)
            return 1;
        for (double i = n; i > 0; i--, n *= (i > 0 ? i : 1)) {
        }
        return n;
    }//end factorial

    /**
     * Returns the log-odds for a given probability.
     *
     * @param prob the probability
     * @return the log-odds after the probability has been mapped to
     * [Utils.SMALL, 1-Utils.SMALL]
     */
    public static /*@pure@*/ double probToLogOdds(double prob) {

        if (gr(prob, 1) || (sm(prob, 0))) {
            throw new IllegalArgumentException("probToLogOdds: probability must " + "be in [0,1] " + prob);
        }
        double p = SMALL + (1.0 - 2 * SMALL) * prob;
        return Math.log(p / (1 - p));
    }

    /**
     * Rounds a double to the next nearest integer value. The JDK version
     * of it doesn't work properly.
     *
     * @param value the double value
     * @return the resulting integer value
     */
    public static /*@pure@*/ int round(double value) {

        int roundedValue = value > 0 ? (int) (value + 0.5) : -(int) (Math.abs(value) + 0.5);

        return roundedValue;
    }//end round

    /**
     * This returns the permutation of n choose r.
     *
     * @param n the n to choose
     * @param r the number of elements to choose
     * @return the permutation of these numbers
     */
    public static double permutation(double n, double r) {
        double nFac = MathUtils.factorial(n);
        double nMinusRFac = MathUtils.factorial((n - r));
        return nFac / nMinusRFac;
    }//end permutation

    /**
     * This returns the combination of n choose r
     *
     * @param n the number of elements overall
     * @param r the number of elements to choose
     * @return the amount of possible combinations for this applyTransformToDestination of elements
     */
    public static double combination(double n, double r) {
        double nFac = MathUtils.factorial(n);
        double rFac = MathUtils.factorial(r);
        double nMinusRFac = MathUtils.factorial((n - r));

        return nFac / (rFac * nMinusRFac);
    }//end combination

    /**
     * sqrt(a^2 + b^2) without under/overflow.
     */
    public static double hypotenuse(double a, double b) {
        double r;
        if (Math.abs(a) > Math.abs(b)) {
            r = b / a;
            r = Math.abs(a) * Math.sqrt(1 + r * r);
        } else if (b != 0) {
            r = a / b;
            r = Math.abs(b) * Math.sqrt(1 + r * r);
        } else {
            r = 0.0;
        }
        return r;
    }//end hypotenuse

    /**
     * Rounds a double to the next nearest integer value in a probabilistic
     * fashion (e.g. 0.8 has a 20% chance of being rounded down to 0 and a
     * 80% chance of being rounded up to 1). In the limit, the average of
     * the rounded numbers generated by this procedure should converge to
     * the original double.
     *
     * @param value the double value
     * @param rand  the random number generator
     * @return the resulting integer value
     */
    public static int probRound(double value, Random rand) {

        if (value >= 0) {
            double lower = Math.floor(value);
            double prob = value - lower;
            if (rand.nextDouble() < prob) {
                return (int) lower + 1;
            } else {
                return (int) lower;
            }
        } else {
            double lower = Math.floor(Math.abs(value));
            double prob = Math.abs(value) - lower;
            if (rand.nextDouble() < prob) {
                return -((int) lower + 1);
            } else {
                return -(int) lower;
            }
        }
    }//end probRound

    /**
     * Rounds a double to the given number of decimal places.
     *
     * @param value             the double value
     * @param afterDecimalPoint the number of digits after the decimal point
     * @return the double rounded to the given precision
     */
    public static /*@pure@*/ double roundDouble(double value, int afterDecimalPoint) {

        double mask = Math.pow(10.0, (double) afterDecimalPoint);

        return (double) (Math.round(value * mask)) / mask;
    }//end roundDouble

    /**
     * Rounds a double to the given number of decimal places.
     *
     * @param value             the double value
     * @param afterDecimalPoint the number of digits after the decimal point
     * @return the double rounded to the given precision
     */
    public static /*@pure@*/ float roundFloat(float value, int afterDecimalPoint) {

        float mask = (float) Math.pow(10, (float) afterDecimalPoint);

        return (float) (Math.round(value * mask)) / mask;
    }//end roundDouble

    /**
     * This will return the bernoulli trial for the given event.
     * A bernoulli trial is a mechanism for detecting the probability
     * of a given event occurring k times in n independent trials
     *
     * @param n           the number of trials
     * @param k           the number of times the target event occurs
     * @param successProb the probability of the event happening
     * @return the probability of the given event occurring k times.
     */
    public static double bernoullis(double n, double k, double successProb) {

        double combo = MathUtils.combination(n, k);
        double p = successProb;
        double q = 1 - successProb;
        return combo * Math.pow(p, k) * Math.pow(q, n - k);
    }//end bernoullis

    /**
     * Tests if a is smaller than b.
     *
     * @param a a double
     * @param b a double
     */
    public static /*@pure@*/ boolean sm(double a, double b) {

        return (b - a > SMALL);
    }

    /**
     * Tests if a is greater than b.
     *
     * @param a a double
     * @param b a double
     */
    public static /*@pure@*/ boolean gr(double a, double b) {

        return (a - b > SMALL);
    }

    /**
     * This will take a given string and separator and convert it to an equivalent
     * double array.
     *
     * @param data      the data to separate
     * @param separator the separator to use
     * @return the new double array based on the given data
     */
    public static double[] fromString(String data, String separator) {
        String[] split = data.split(separator);
        double[] ret = new double[split.length];
        for (int i = 0; i < split.length; i++) {
            ret[i] = Double.parseDouble(split[i]);
        }
        return ret;
    }//end fromString

    /**
     * Computes the mean for an array of doubles.
     *
     * @param vector the array
     * @return the mean
     */
    public static /*@pure@*/ double mean(double[] vector) {

        double sum = 0;

        if (vector.length == 0) {
            return 0;
        }
        for (int i = 0; i < vector.length; i++) {
            sum += vector[i];
        }
        return sum / (double) vector.length;
    }//end mean

    /**
     * This will convert the given binary string to a decimal based
     * integer
     *
     * @param binary the binary string to convert
     * @return an equivalent base 10 number
     */
    public static int toDecimal(String binary) {
        long num = Long.parseLong(binary);
        long rem;
        /* Use the remainder method to ensure validity */
        while (num > 0) {
            rem = num % 10;
            num = num / 10;
            if (rem != 0 && rem != 1) {
                System.out.println("This is not a binary number.");
                System.out.println("Please try once again.");
                return -1;
            }
        }
        int i = Integer.parseInt(binary, 2);
        return i;
    }//end toDecimal

    /**
     * This will translate a vector in to an equivalent integer
     *
     * @param vector the vector to translate
     * @return a z value such that the value is the interleaved lsd to msd for each
     * double in the vector
     */
    public static int distanceFinderZValue(double[] vector) {
        StringBuilder binaryBuffer = new StringBuilder();
        List<String> binaryReps = new ArrayList<String>(vector.length);
        for (int i = 0; i < vector.length; i++) {
            double d = vector[i];
            int j = (int) d;
            String binary = Integer.toBinaryString(j);
            binaryReps.add(binary);
        }
        //append from left to right, the least to the most significant bit
        //till all strings are empty
        while (!binaryReps.isEmpty()) {
            for (int j = 0; j < binaryReps.size(); j++) {
                String curr = binaryReps.get(j);
                if (!curr.isEmpty()) {
                    char first = curr.charAt(0);
                    binaryBuffer.append(first);
                    curr = curr.substring(1);
                    binaryReps.set(j, curr);
                } else
                    binaryReps.remove(j);
            }
        }
        return Integer.parseInt(binaryBuffer.toString(), 2);

    }//end distanceFinderZValue

    /**
     * This returns the euclidean distance of two vectors
     * sum(i=1,n)   (q_i - p_i)^2
     *
     * @param p the first vector
     * @param q the second vector
     * @return the euclidean distance between two vectors
     */
    public static double euclideanDistance(double[] p, double[] q) {

        double ret = 0;
        for (int i = 0; i < p.length; i++) {
            double diff = (q[i] - p[i]);
            double sq = Math.pow(diff, 2);
            ret += sq;
        }
        return ret;

    }//end euclideanDistance

    /**
     * This returns the euclidean distance of two vectors
     * sum(i=1,n)   (q_i - p_i)^2
     *
     * @param p the first vector
     * @param q the second vector
     * @return the euclidean distance between two vectors
     */
    public static double euclideanDistance(float[] p, float[] q) {

        double ret = 0;
        for (int i = 0; i < p.length; i++) {
            double diff = (q[i] - p[i]);
            double sq = Math.pow(diff, 2);
            ret += sq;
        }
        return ret;

    }//end euclideanDistance

    /**
     * This will generate a series of uniformally distributed
     * numbers between l times
     *
     * @param l the number of numbers to generate
     * @return l uniformally generated numbers
     */
    public static double[] generateUniform(int l) {
        double[] ret = new double[l];
        Random rgen = new Random();
        for (int i = 0; i < l; i++) {
            ret[i] = rgen.nextDouble();
        }
        return ret;
    }//end generateUniform

    /**
     * This will calculate the Manhattan distance between two sets of points.
     * The Manhattan distance is equivalent to:
     * 1_sum_n |p_i - q_i|
     *
     * @param p the first point vector
     * @param q the second point vector
     * @return the Manhattan distance between two object
     */
    public static double manhattanDistance(double[] p, double[] q) {

        double ret = 0;
        for (int i = 0; i < p.length; i++) {
            double difference = p[i] - q[i];
            ret += Math.abs(difference);
        }
        return ret;
    }//end manhattanDistance

    public static double[] sampleDoublesInInterval(double[][] doubles, int l) {
        double[] sample = new double[l];
        for (int i = 0; i < l; i++) {
            int rand1 = randomNumberBetween(0, doubles.length - 1);
            int rand2 = randomNumberBetween(0, doubles[i].length);
            sample[i] = doubles[rand1][rand2];
        }

        return sample;
    }

    /**
     * Generates a random integer between the specified numbers
     *
     * @param begin the begin of the interval
     * @param end   the end of the interval
     * @param anchor the base number (assuming to be generated from an external rng)
     * @return an int between begin and end
     */
    public static int randomNumberBetween(double begin, double end,double anchor) {
        if (begin > end)
            throw new IllegalArgumentException("Begin must not be less than end");
        return (int) begin + (int) (anchor * ((end - begin) + 1));
    }


    /**
     * Generates a random integer between the specified numbers
     *
     * @param begin the begin of the interval
     * @param end   the end of the interval
     * @return an int between begin and end
     */
    public static int randomNumberBetween(double begin, double end) {
        if (begin > end)
            throw new IllegalArgumentException("Begin must not be less than end");
        return (int) begin + (int) (Math.random() * ((end - begin) + 1));
    }

    /**
     * Generates a random integer between the specified numbers
     *
     * @param begin the begin of the interval
     * @param end   the end of the interval
     * @return an int between begin and end
     */
    public static int randomNumberBetween(double begin, double end, RandomGenerator rng) {
        if (begin > end)
            throw new IllegalArgumentException("Begin must not be less than end");
        return (int) begin + (int) (rng.nextDouble() * ((end - begin) + 1));
    }

    public static float randomFloatBetween(float begin, float end) {
        float rand = (float) Math.random();
        return begin + (rand * ((end - begin)));
    }

    public static double randomDoubleBetween(double begin, double end) {
        return begin + (Math.random() * ((end - begin)));
    }

    /**
     * This returns the slope of the given points.
     *
     * @param x1 the first x to use
     * @param x2 the end x to use
     * @param y1 the begin y to use
     * @param y2 the end y to use
     * @return the slope of the given points
     */
    public double slope(double x1, double x2, double y1, double y2) {
        return (y2 - y1) / (x2 - x1);
    }//end slope

    public static void shuffleArray(int[] array, long rngSeed) {
        shuffleArray(array, new Random(rngSeed));
    }

    public static void shuffleArray(int[] array, Random rng) {
        //https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
        for (int i = array.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int temp = array[j];
            array[j] = array[i];
            array[i] = temp;
        }
    }

    /**
     * hashCode method, taken from Java 1.8 Double.hashCode(double) method
     *
     * @param value Double value to hash
     * @return Hash code for the double value
     */
    public static int hashCode(double value) {
        long bits = Double.doubleToLongBits(value);
        return (int) (bits ^ (bits >>> 32));
    }
}
