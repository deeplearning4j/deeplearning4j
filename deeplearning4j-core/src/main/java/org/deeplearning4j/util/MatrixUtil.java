package org.deeplearning4j.util;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.FloatDataSet;
import org.deeplearning4j.nn.linalg.*;
import org.jblas.*;
import org.jblas.ranges.Range;
import org.jblas.ranges.RangeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Matrix Ops
 *
 * @author Adam Gibson
 */
public class MatrixUtil {
    private static Logger log = LoggerFactory.getLogger(MatrixUtil.class);


    public static <E extends DoubleMatrix> void complainAboutMissMatchedMatrices(E d1, E d2) {
        if (d1 == null || d2 == null)
            throw new IllegalArgumentException("No null matrices allowed");
        if (d1.rows != d2.rows)
            throw new IllegalArgumentException("Matrices must have same rows");

    }


    /**
     * Cuts all numbers below a certain cut off
     *
     * @param minNumber the min number to check
     * @param matrix    the matrix to max by
     */
    public static  <E extends DoubleMatrix> void max(double minNumber, E matrix) {
        for (int i = 0; i < matrix.length; i++)
            matrix.put(i, Math.max(0, matrix.get(i)));

    }

    /**
     * Divides each row by its max
     *
     * @param toScale the matrix to divide by its row maxes
     */
    public static void scaleByMax(DoubleMatrix toScale) {
        DoubleMatrix scale = toScale.rowMaxs();
        for (int i = 0; i < toScale.rows; i++) {
            double scaleBy = scale.get(i, 0);
            toScale.putRow(i, toScale.getRow(i).divi(scaleBy));
        }
    }


    /**
     * Flips the dimensions of each slice of a tensor or 4d tensor
     * slice wise
     * @param input the input to flip
     * @param <E>
     * @return the flipped tensor
     */
    public static <E extends DoubleMatrix> E flipDimMultiDim(E input) {
        if(input instanceof FourDTensor) {
            FourDTensor t = (FourDTensor) input;
            DoubleMatrix ret = new DoubleMatrix(input.rows,input.columns);
            FourDTensor flipped = createBasedOn(ret,t);
            for(int i = 0; i < flipped.numTensors(); i++) {
                for(int j = 0; j < flipped.getTensor(i).slices(); j++) {
                    flipped.put(i,j,flipDim(t.getSliceOfTensor(i,j)));
                }
            }

            return createBasedOn(flipped,input);
        }
        else if(input instanceof Tensor) {
            Tensor t = (Tensor) input;
            DoubleMatrix ret = new DoubleMatrix(input.rows,input.columns);
            Tensor flipped = createBasedOn(ret,t);
            for(int j = 0; j < flipped.slices(); j++) {
                flipped.setSlice(j,flipDim(t.getSlice(j)));
            }
            return createBasedOn(ret,input);
        }

        else
            return (E) flipDim(input);
    }


    /**
     * Flips the dimensions of the given matrix.
     * [1,2]       [3,4]
     * [3,4] --->  [1,2]
     * @param flip the matrix to flip
     * @return the flipped matrix
     */
    public static DoubleMatrix flipDim(DoubleMatrix flip) {
        DoubleMatrix ret = new DoubleMatrix(flip.rows,flip.columns);
        CounterMap<Integer,Integer> dimsFlipped = new CounterMap<>();
        for(int j = 0; j < flip.columns; j++) {
            for(int i = 0; i < flip.rows;  i++) {
                for(int k = flip.rows - 1; k >= 0; k--) {
                    if(dimsFlipped.getCount(i,j) > 0 || dimsFlipped.getCount(k,j) > 0)
                        continue;
                    dimsFlipped.incrementCount(i,j,1.0);
                    dimsFlipped.incrementCount(k,j,1.0);
                    double first = flip.get(i,j);
                    double second = flip.get(k,j);
                    ret.put(k,j,first);
                    ret.put(i,j,second);
                }
            }
        }


        return ret;

    }



    /**
     * Cumulative sum
     *
     * @param sum the matrix to get the cumulative sum of
     * @return a matrix of the same dimensions such that the at i,j
     * is the cumulative sum of it + the predecessor elements in the column
     */
    public static  <E extends DoubleMatrix> E cumsum(E sum) {
        DoubleMatrix ret = new DoubleMatrix(sum.rows, sum.columns);
        for (int i = 0; i < ret.columns; i++) {
            for (int j = 0; j < ret.rows; j++) {
                int[] indices = new int[j + 1];
                for (int row = 0; row < indices.length; row++)
                    indices[row] = row;
                DoubleMatrix toSum = sum.get(indices, new int[]{i});
                double d = toSum.sum();
                ret.put(j, i, d);
            }
        }
        return (E) ret;
    }


    /**
     * Truncates a matrix down to size rows x columns
     *
     * @param toTruncate the matrix to truncate
     * @param rows       the rows to reduce to
     * @param columns    the columns to reduce to
     * @return a subset of the old matrix
     * with the specified dimensions
     */
    public static DoubleMatrix truncate(DoubleMatrix toTruncate, int rows, int columns) {
        if (rows >= toTruncate.rows && columns >= toTruncate.columns || rows < 1 || columns < 1)
            return toTruncate;

        DoubleMatrix ret = new DoubleMatrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                ret.put(i, j, toTruncate.get(i, j));
            }
        }
        return ret;
    }

    /**
     * Reshapes this matrix in to a 3d matrix
     * @param input the input matrix
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param numSlices the number of slices in the tensor
     * @return the reshaped matrix as a tensor
     */
    public static <E extends DoubleMatrix> Tensor reshape(E input,int rows, int columns,int numSlices) {
        DoubleMatrix ret = input.reshape(rows * numSlices,columns);
        Tensor retTensor = new Tensor(ret,false);
        retTensor.setSlices(numSlices);
        return retTensor;
    }


    /**
     * Rotates a matrix by reversing its input
     * If its  any kind of tensor each slice of each tensor
     * will be reversed
     * @param input the input to rotate
     * @param <E>
     * @return the rotated matrix or tensor
     */
    public static <E extends DoubleMatrix> E rot(E input) {
        if(input instanceof FourDTensor) {
            FourDTensor t = (FourDTensor) input;
            FourDTensor ret = new FourDTensor(t);
            for(int i = 0; i < ret.numTensors(); i++) {
                Tensor t1 = ret.getTensor(i);
                for(int j = 0; j < t1.slices(); j++) {
                    ret.setSlice(j,reverse(t1.getSlice(j)));
                }
            }

            return createBasedOn(ret,input);
        }

        else if(input instanceof Tensor) {
            Tensor t = (Tensor) input;
            Tensor ret = new Tensor(t);
            for(int j = 0; j < t.slices(); j++) {
                ret.setSlice(j,reverse(t.getSlice(j)));
            }
            return createBasedOn(ret,input);


        }

        else
            return reverse(input);

    }



    /**
     * Reshapes this matrix in to a 3d matrix
     * @param input the input matrix
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param numSlices the number of slices in the tensor
     * @return the reshaped matrix as a tensor
     */
    public static <E extends DoubleMatrix> FourDTensor reshape(E input,int rows, int columns,int numSlices,int numTensors) {
        DoubleMatrix ret = input.reshape(rows * numSlices,columns);
        FourDTensor retTensor = new FourDTensor(ret,false);
        retTensor.setSlices(numSlices);
        retTensor.setNumTensor(numTensors);
        return retTensor;
    }

    /**
     * Binarizes the matrix such that any number greater than cutoff is 1 otherwise zero
     * @param cutoff the cutoff point
     */
    public static void binarize(double cutoff,DoubleMatrix input) {
        for(int i = 0; i < input.length; i++)
            if(input.get(i) > cutoff)
                input.put(i,1);
            else
                input.put(i,0);
    }


    /**
     * Generate a new matrix which has the given number of replications of this.
     */
    public static ComplexDoubleMatrix repmat(ComplexDoubleMatrix matrix, int rowMult, int columnMult) {
        ComplexDoubleMatrix result = new ComplexDoubleMatrix(matrix.rows * rowMult, matrix.columns * columnMult);

        for (int c = 0; c < columnMult; c++) {
            for (int r = 0; r < rowMult; r++) {
                for (int i = 0; i < matrix.rows; i++) {
                    for (int j = 0; j < matrix.columns; j++) {
                        result.put(r * matrix.rows + i, c * matrix.columns + j, matrix.get(i, j));
                    }
                }
            }
        }
        return result;
    }

    public static DoubleMatrix shape(DoubleMatrix d) {
        return new DoubleMatrix(new double[]{d.rows, d.columns});
    }

    public static DoubleMatrix size(DoubleMatrix d) {
        return shape(d);
    }


    /**
     * Down sample a signal
     * @param data
     * @param stride
     * @param <E>
     * @return
     */
    public static <E extends DoubleMatrix> E downSample(E data, DoubleMatrix stride) {
        DoubleMatrix d = DoubleMatrix.ones((int) stride.get(0), (int) stride.get(1));
        d.divi(prod(stride));
        DoubleMatrix ret = Convolution.conv2d(data, d, Convolution.Type.VALID);
        ret = ret.get(RangeUtils.interval(0, (int) stride.get(0)), RangeUtils.interval(0, (int) stride.get(1)));
        return createBasedOn(ret,data);
    }

    /**
     * Takes the product of all the elements in the matrix
     *
     * @param product the matrix to get the product of elements of
     * @return the product of all the elements in the matrix
     */
    public static <E extends DoubleMatrix> double prod(E product) {
        double ret = 1.0;
        for (int i = 0; i < product.length; i++)
            ret *= product.get(i);
        return ret;
    }


    /**
     * Upsampling a signal
     * @param d
     * @param scale
     * @param <E>
     * @return
     */
    public static <E extends DoubleMatrix> E upSample(E d, E scale) {
        DoubleMatrix shape = size(d);

        DoubleMatrix idx = new DoubleMatrix(shape.length, 1);


        for (int i = 0; i < shape.length; i++) {
            DoubleMatrix tmp = DoubleMatrix.zeros((int) shape.get(i) * (int) scale.get(i), 1);
            int[] indices = indicesCustomRange(0, (int) scale.get(i), (int) scale.get(i) * (int) shape.get(i));
            tmp.put(indices, 1.0);
            idx.put(i, cumsum(tmp).sum());
        }
        return createBasedOn(idx,d);
    }

    public static int[] indicesCustomRange(int start, int increment, int end) {
        int len = end - start - increment;
        if (len >= end)
            len = end;
        int[] ret = new int[len];
        int idx = 0;
        int count = 0;
        for (int i = 0; i < len; i++) {
            ret[i] = idx;
            idx += increment;
            if (idx >= len)
                break;
            count++;
        }
        int[] realRet = new int[count];
        System.arraycopy(ret, 0, realRet, 0, realRet.length);
        return realRet;
    }


    public static DoubleMatrix rangeVector(double begin, double end) {
        int diff = (int) Math.abs(end - begin);
        DoubleMatrix ret = new DoubleMatrix(1, diff);
        for (int i = 0; i < ret.length; i++)
            ret.put(i, i);
        return ret;
    }

    public static ComplexDoubleMatrix complexRangeVector(double begin, double end) {
        int diff = (int) Math.abs(end - begin);
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(1, diff);
        for (int i = 0; i < ret.length; i++)
            ret.put(i, i);
        return ret.transpose();
    }


    public static double angle(ComplexDouble phase) {
        Complex c = new Complex(phase.real(), phase.imag());
        return c.atan().getReal();
    }

    /**
     * Implements matlab's compare of complex numbers:
     * compares the max(abs(d1),abs(d2))
     * if they are equal, compares their angles
     *
     * @param d1 the first number
     * @param d2 the second number
     * @return standard comparator interface
     */
    private static int compare(ComplexDouble d1, ComplexDouble d2) {
        if (d1.abs() > d2.abs())
            return 1;
        else if (d2.abs() > d1.abs())
            return -1;
        else {
            if (angle(d1) > angle(d2))
                return 1;
            else if (angle(d1) < angle(d2))
                return -1;
            return 0;
        }

    }


    public static ComplexDouble max(ComplexDoubleMatrix matrix) {
        ComplexDouble max = matrix.get(0);
        for (int i = 1; i < matrix.length; i++)
            if (compare(max, matrix.get(i)) > 0)
                max = matrix.get(i);
        return max;
    }


    /**
     * Divides each row by its max
     *
     * @param toScale the matrix to divide by its row maxes
     */
    public static void scaleByMax(ComplexDoubleMatrix toScale) {

        for (int i = 0; i < toScale.rows; i++) {
            ComplexDouble scaleBy = max(toScale.getRow(i));
            toScale.putRow(i, toScale.getRow(i).divi(scaleBy));
        }
    }

    public static <E extends DoubleMatrix>  E variance(E input) {
        DoubleMatrix means = (E) input.columnMeans();
        DoubleMatrix diff = MatrixFunctions.pow(input.subRowVector(means), 2);
        //avg of the squared differences from the mean
        DoubleMatrix variance =  diff.columnMeans().div(input.rows);
        return createBasedOn(variance,input);

    }


    public static <E extends DoubleMatrix> int[] toInts(E ints)  {
        int[] ret = new int[ints.length];
        for(int i = 0; i < ints.length; i++)
            ret[i] = (int) ints.get(i);
        return ret;
    }


    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     * @param toReverse the matrix to reverse
     * @param <E>
     * @return the reversed matrix
     */
    public static  <E extends DoubleMatrix> E reverse(E toReverse) {
        DoubleMatrix ret = new DoubleMatrix(toReverse.rows, toReverse.columns);
        int reverseIndex = 0;
        for (int i = toReverse.length - 1; i >= 0; i--) {
            ret.put(reverseIndex++, toReverse.get(i));
        }
        return createBasedOn(ret,toReverse);
    }

    /**
     * Utility method for creating a variety of matrices, useful for handling conversions.
     *
     * This method is always O(1) due to not needing to copy data.
     *
     * Note however, that this makes the assumption that the passed in matrix isn't being referenced
     * by anything else as this being a safe operation
     * @param result the result matrix to cast
     * @param input the input it was based on
     * @param <E> the type of matrix
     * @return the casted matrix
     */
    public static <E extends DoubleMatrix> E createBasedOn(DoubleMatrix result,E input) {
        if(input.getClass().equals(result.getClass()))
            return (E) result;

        else if(input instanceof FourDTensor) {
            FourDTensor tensor = new FourDTensor(result,false);
            FourDTensor casted = (FourDTensor) input;
            tensor.setSlices(casted.slices());
            tensor.setPerMatrixRows(casted.rows());
            tensor.setNumTensor(casted.getNumTensor());
            return (E) tensor;
        }

        else if(input instanceof Tensor) {
            Tensor ret = new Tensor(result,false);
            Tensor casted = (Tensor) input;
            ret.setPerMatrixRows(ret.rows());
            ret.columns = input.columns;
            ret.setSlices(casted.slices());
            return (E) ret;
        }
        else
            return (E) result;


    }


    /**
     * Returns the maximum dimension of the passed in matrix
     *
     * @param d the max dimension of the passed in matrix
     * @return the max dimension of the passed in matrix
     */
    public static double  length(ComplexDoubleMatrix d) {
        return Math.max(d.rows, d.columns);


    }

    /**
     * Floor function applied to the matrix
     * @param input
     * @param <E>
     * @return
     */
    public static <E extends DoubleMatrix> E floor(E input) {
        return createBasedOn(MatrixFunctions.floor(input),input);
    }




    /**
     * Returns the maximum dimension of the passed in matrix
     *
     * @param d the max dimension of the passed in matrix
     * @return the max dimension of the passed in matrix
     */
    public static <E extends DoubleMatrix> double length(E d) {
        if (d instanceof Tensor) {
            Tensor t = (Tensor) d;
            return MathUtils.max(new double[]{t.rows(), t.columns(), t.slices()});
        } else
            return Math.max(d.rows, d.columns);


    }


    public static DataSet xorData(int n) {

        DoubleMatrix x = DoubleMatrix.rand(n, 2);
        x = x.gti(0.5);

        DoubleMatrix y = DoubleMatrix.zeros(n, 2);
        for (int i = 0; i < x.rows; i++) {
            if (x.get(i, 0) == x.get(i, 1))
                y.put(i, 0, 1);
            else
                y.put(i, 1, 1);
        }

        return new DataSet(x, y);

    }

    public static DataSet xorData(int n, int columns) {

        DoubleMatrix x = DoubleMatrix.rand(n, columns);
        x = x.gti(0.5);

        DoubleMatrix x2 = DoubleMatrix.rand(n, columns);
        x2 = x2.gti(0.5);

        DoubleMatrix eq = x.eq(x2).eq(DoubleMatrix.zeros(n, columns));


        int median = columns / 2;

        DoubleMatrix outcomes = new DoubleMatrix(n, 2);
        for (int i = 0; i < outcomes.rows; i++) {
            DoubleMatrix left = eq.get(i, new org.jblas.ranges.IntervalRange(0, median));
            DoubleMatrix right = eq.get(i, new org.jblas.ranges.IntervalRange(median, columns));
            if (left.sum() > right.sum())
                outcomes.put(i, 0, 1);
            else
                outcomes.put(i, 1, 1);
        }


        return new DataSet(eq, outcomes);

    }

    public static <E extends DoubleMatrix>  double magnitude(E vec) {
        double sum_mag = 0;
        for (int i = 0; i < vec.length; i++)
            sum_mag = sum_mag + vec.get(i) * vec.get(i);

        return Math.sqrt(sum_mag);
    }


    /**
     * Exp with more generic casting
     * @param d the input
     * @param <E>
     * @return the exp of this matrix
     */
    public static <E extends DoubleMatrix> E exp(E d) {
        return createBasedOn(MatrixFunctions.exp(d),d);
    }

    /**
     * Flattens the matrix
     * @param d
     * @param <E>
     * @return
     */
    public static <E extends DoubleMatrix> E unroll(E d) {
        DoubleMatrix ret = new DoubleMatrix(1, d.length);
        for (int i = 0; i < d.length; i++)
            ret.put(i, d.get(i));
        return createBasedOn(ret,d);
    }


    public static <E extends DoubleMatrix> E outcomes(E d) {
        DoubleMatrix ret = new DoubleMatrix(d.rows, 1);
        for (int i = 0; i < d.rows; i++)
            ret.put(i, SimpleBlas.iamax(d.getRow(i)));
        return createBasedOn(ret,d);
    }

    /**
     *
     * @param d1
     * @param d2
     * @param <E>
     * @return
     */
    public static <E extends DoubleMatrix> double cosineSim(E d1, E d2) {
        d1 = unitVec(d1);
        d2 = unitVec(d2);
        double ret = d1.dot(d2);
        return ret;
    }

    /**
     * Normalizes the matrix by subtracting the min,
     * dividing by the max - min
     * @param input the input to normalize
     * @param <E>
     * @return the normalized matrix
     */
    public static <E extends DoubleMatrix> E normalize(E input) {
        double min = input.min();
        double max = input.max();
        return createBasedOn(input.subi(min).divi(max - min),input);
    }


    public static <E extends DoubleMatrix> double  cosine(E matrix) {
        return 1 * Math.sqrt(MatrixFunctions.pow(matrix, 2).sum());
    }


    public static <E extends DoubleMatrix> E unitVec(E toScale) {
        double length = toScale.norm2();
        if (length > 0)
            return createBasedOn((SimpleBlas.scal(1.0 / length, toScale)),toScale);
        return toScale;
    }

    /**
     * A uniform sample ranging from 0 to 1.
     *
     * @param rng     the rng to use
     * @param rows    the number of rows of the matrix
     * @param columns the number of columns of the matrix
     * @return a uniform sample of the given shape and size
     * with numbers between 0 and 1
     */
    public static DoubleMatrix uniform(RandomGenerator rng, int rows, int columns) {

        UniformRealDistribution uDist = new UniformRealDistribution(rng, 0, 1);
        DoubleMatrix U = new DoubleMatrix(rows, columns);
        for (int i = 0; i < U.rows; i++)
            for (int j = 0; j < U.columns; j++)
                U.put(i, j, uDist.sample());
        return U;
    }


    /**
     * Creates a matrix of row x columns with the assigned value
     * @param value the value to assign
     * @param rows the number of rows in the matrix
     * @param columns the number of columns of the matrix
     * @return the value matrix with the specified dimensions
     */
    public static DoubleMatrix valueMatrixOf(double value,int rows,int columns) {
        DoubleMatrix ret = new DoubleMatrix(rows,columns);
        assign(ret,value);
        return ret;
    }

    /**
     * A uniform sample ranging from 0 to sigma.
     *
     * @param rng   the rng to use
     * @return a uniform sample of the given shape and size
     * <p/>
     * with numbers between 0 and 1
     */
    public static DoubleMatrix randn(RandomGenerator rng,int rows,int columns) {
        DoubleMatrix U = new DoubleMatrix(rows, columns);
        for (int i = 0; i < U.rows; i++)
            for (int j = 0; j < U.columns; j++) {
                U.put(i, j, rng.nextGaussian());

            }
        return U;
    }


    /**
     * A uniform sample ranging from 0 to sigma.
     *
     * @param rng   the rng to use
     * @param mean, the matrix mean from which to generate values from
     * @param sigma the standard deviation to use to generate the gaussian noise
     * @return a uniform sample of the given shape and size
     * <p/>
     * with numbers between 0 and 1
     */
    public static <E extends DoubleMatrix> E normal(RandomGenerator rng, E mean, double sigma) {
        DoubleMatrix U = new DoubleMatrix(mean.rows, mean.columns);
        for (int i = 0; i < U.rows; i++)
            for (int j = 0; j < U.columns; j++) {
                RealDistribution reals = new NormalDistribution(rng,mean.get(i, j), FastMath.sqrt(sigma),NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
                U.put(i, j, reals.sample());

            }
        return createBasedOn(U,mean);
    }


    /**
     * A uniform sample ranging from 0 to sigma.
     *
     * @param rng      the rng to use
     * @param mean,    the matrix mean from which to generate values from
     * @param variance the variance matrix where each column is the variance
     *                 for the respective columns of the matrix
     * @return a uniform sample of the given shape and size
     * <p/>
     * with numbers between 0 and 1
     */
    public static <E extends DoubleMatrix> E normal(RandomGenerator rng, E mean, E variance) {
        DoubleMatrix std =  sqrt(variance);
        for (int i = 0; i < variance.length; i++)
            if (variance.get(i) <= 0)
                variance.put(i, 1e-4);

        DoubleMatrix U = new DoubleMatrix(mean.rows, mean.columns);
        for (int i = 0; i < U.rows; i++)
            for (int j = 0; j < U.columns; j++) {
                RealDistribution reals = new NormalDistribution(rng,mean.get(i, j), std.get(j),NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
                U.put(i, j, reals.sample());

            }
        return createBasedOn(U,mean);
    }


    /**
     * Sample from a normal distribution given a mean of zero and a matrix of standard deviations.
     *
     * @param rng the rng to use
     *            for the respective columns of the matrix
     * @return a uniform sample of the given shape and size
     */
    public static DoubleMatrix normal(RandomGenerator rng, DoubleMatrix standardDeviations) {

        DoubleMatrix U = new DoubleMatrix(standardDeviations.rows, standardDeviations.columns);
        for (int i = 0; i < U.rows; i++)
            for (int j = 0; j < U.columns; j++) {
                RealDistribution reals = new NormalDistribution(0, standardDeviations.get(i, j));
                U.put(i, j, reals.sample());

            }
        return U;
    }

    public static <E extends DoubleMatrix> boolean isValidOutcome(E out) {
        boolean found = false;
        for (int col = 0; col < out.length; col++) {
            if (out.get(col) > 0) {
                found = true;
                break;
            }
        }
        return found;
    }



    public static int[] toIndices(Range range) {
        int[] ret = new int[range.length()];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = range.value();
            range.next();
            ;
        }

        return ret;
    }


    public static ComplexDoubleMatrix exp(ComplexDoubleMatrix input) {
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(input.rows, input.columns);
        for (int i = 0; i < ret.length; i++) {
            ret.put(i, ComplexUtil.exp(input.get(i)));
        }
        return ret;
    }



    public static ComplexDoubleMatrix complexPadWithZeros(ComplexDoubleMatrix toPad, int rows, int cols) {
        ComplexDoubleMatrix ret = ComplexDoubleMatrix.zeros(rows, cols);
        for (int i = 0; i < toPad.rows; i++) {
            for (int j = 0; j < toPad.columns; j++) {
                ret.put(i, j, toPad.get(i, j));
            }
        }
        return ret;
    }


    public static ComplexNDArray complexPadWithZeros(NDArray toPad,int[] newShape) {
        ComplexNDArray ret = ComplexNDArray.zeros(newShape);
        for(int i = 0; i < toPad.length; i++)
            ret.put(i,toPad.get(i));
        return ret;
    }


    public static ComplexNDArray complexPadWithZeros(ComplexNDArray toPad,int[] newShape) {
        ComplexNDArray ret = ComplexNDArray.zeros(newShape);
        for(int i = 0; i < toPad.length; i++)
            ret.put(i,toPad.get(i));
        return ret;
    }



    public static ComplexDoubleMatrix complexPadWithZeros(DoubleMatrix toPad, int rows, int cols) {
        ComplexDoubleMatrix ret = ComplexDoubleMatrix.zeros(rows, cols);
        for (int i = 0; i < toPad.rows; i++) {
            for (int j = 0; j < toPad.columns; j++) {
                ret.put(i, j, toPad.get(i, j));
            }
        }
        return ret;
    }

    /**
     * Pads a matrix with zeros
     *
     * @param toPad the matrix to pad
     * @param rows  the number of rows to pad the matrix to
     * @param cols  the number of columns to pad the matrix to
     * @return
     */
    public static ComplexDoubleMatrix padWithZeros(ComplexDoubleMatrix toPad, int rows, int cols) {
        ComplexDoubleMatrix ret = ComplexDoubleMatrix.zeros(rows, cols);
        for (int i = 0; i < toPad.rows; i++) {
            for (int j = 0; j < toPad.columns; j++) {
                ret.put(i, j, toPad.get(i, j));
            }
        }
        return ret;
    }


    /**
     * Assigns every element in the matrix to the given value
     *
     * @param toAssign the matrix to modify
     * @param val      the value to assign
     */
    public static <E extends DoubleMatrix> void assign(E toAssign, double val) {
        for (int i = 0; i < toAssign.length; i++)
            toAssign.put(i, val);
    }

    /**
     * Rotate a matrix 90 degrees
     *
     * @param toRotate the matrix to rotate
     */
    public static <E extends DoubleMatrix> void rot90(E toRotate) {
        for (int i = 0; i < toRotate.rows; i++)
            for (int j = 0; j < toRotate.columns; j++)
                toRotate.put(i, j, toRotate.get(toRotate.columns - i - 1));

    }

    /**
     * Pads the matrix with zeros to surround the passed in matrix
     * with the given rows and columns
     *
     * @param toPad the matrix to pad
     * @param rows  the rows of the destination matrix
     * @param cols  the columns of the destination matrix
     * @return a new matrix with the elements of toPad with zeros or
     * a clone of toPad if the rows and columns are both greater in length than
     * rows and cols
     */
    public  static <E extends DoubleMatrix> E padWithZeros(E toPad, int rows, int cols) {
        if (rows < 1)
            throw new IllegalArgumentException("Illegal number of rows " + rows);
        if (cols < 1)
            throw new IllegalArgumentException("Illegal number of columns " + cols);
        DoubleMatrix ret = null;
        //nothing to pad
        if (toPad.rows >= rows) {
            if (toPad.columns >= cols)
                return createBasedOn(toPad.dup(),toPad);
            else
                ret = new DoubleMatrix(toPad.rows, cols);


        } else if (toPad.columns >= cols) {
            if (toPad.rows >= rows)
                return createBasedOn(toPad.dup(),toPad);
            else
                ret = new DoubleMatrix(rows, toPad.columns);
        } else
            ret = new DoubleMatrix(rows, cols);

        for (int i = 0; i < toPad.rows; i++) {
            for (int j = 0; j < toPad.columns; j++) {
                double d = toPad.get(i, j);
                ret.put(i, j, d);
            }
        }
        return createBasedOn(ret,toPad);
    }

    public static ComplexDoubleMatrix numDivideMatrix(ComplexDouble div, ComplexDoubleMatrix toDiv) {
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(toDiv.rows, toDiv.columns);

        for (int i = 0; i < ret.length; i++) {
            //prevent numerical underflow
            ComplexDouble curr = toDiv.get(i).addi(1e-6);
            ret.put(i, div.div(curr));
        }

        return ret;
    }


    /**
     * div / matrix. This also does padding of the matrix
     * so that no numbers are 0 by adding 1e-6 when dividing
     *
     * @param div   the number to divide by
     * @param toDiv the matrix to divide
     * @return the matrix such that each element i in the matrix
     * is div / num
     */
    public static <E extends DoubleMatrix> E numDivideMatrix(double div, E toDiv) {
        DoubleMatrix ret = new DoubleMatrix(toDiv.rows, toDiv.columns);

        for (int i = 0; i < ret.length; i++)
            //prevent numerical underflow
            ret.put(i, div / toDiv.get(i) + 1e-6);
        return createBasedOn(ret,toDiv);
    }

    /**
     * Returns true if any element in the matrix is Infinite
     * @param test the matrix to test
     * @return whether any element in the matrix is Infinite
     */
    public static <E extends DoubleMatrix> boolean isInfinite(E test) {
        DoubleMatrix nan = test.isInfinite();
        for (int i = 0; i < nan.length; i++) {
            if (nan.get(i) > 0)
                return true;
        }
        return false;
    }


    /**
     * Returns true if any element in the matrix is NaN
     * @param test the matrix to test
     * @return whether any element in the matrix is NaN
     */
    public static boolean isNaN(DoubleMatrix test) {
        for (int i = 0; i < test.length; i++) {
            if (Double.isNaN(test.get(i)))
                return true;
        }
        return false;
    }



    public static DoubleMatrix addColumnVector(DoubleMatrix addTo,DoubleMatrix add) {
        for(int i = 0; i < addTo.rows; i++) {
            addTo.putRow(i,addTo.getRow(i).addi(add.get(i)));
        }

        return addTo;
    }

    public static DoubleMatrix addRowVector(DoubleMatrix addTo,DoubleMatrix add) {
        for(int i = 0; i < addTo.rows; i++) {
            addTo.putRow(i,addTo.getRow(i).addi(add.get(i)));
        }

        return addTo;
    }


    public static void assertNaN(Collection<DoubleMatrix> matrices) {
        int count = 0;
        for(DoubleMatrix d : matrices) {
            assert isNaN(d) : " The matrix " + count + " was NaN";
            count++;

        }
    }

    public static void assertNaN(DoubleMatrix...matrices) {
        for(int i = 0; i < matrices.length; i++)
            assert isNaN(matrices[i]) : " The matrix " + i + " was NaN";
    }



    public static void discretizeColumns(DoubleMatrix toDiscretize, int numBins) {
        DoubleMatrix columnMaxes = toDiscretize.columnMaxs();
        DoubleMatrix columnMins = toDiscretize.columnMins();
        for (int i = 0; i < toDiscretize.columns; i++) {
            double min = columnMins.get(i);
            double max = columnMaxes.get(i);
            DoubleMatrix col = toDiscretize.getColumn(i);
            DoubleMatrix newCol = new DoubleMatrix(col.length);
            for (int j = 0; j < col.length; j++) {
                int bin = MathUtils.discretize(col.get(j), min, max, numBins);
                newCol.put(j, bin);
            }
            toDiscretize.putColumn(i, newCol);

        }
    }

    /**
     * Rounds the matrix to the number of specified by decimal places
     *
     * @param d   the matrix to round
     * @param num the number of decimal places to round to(example: pass 2 for the 10s place)
     * @return the rounded matrix
     */
    public static <E extends DoubleMatrix> E  roundToTheNearest(E d, int num) {
        DoubleMatrix ret = d.mul(num);
        for (int i = 0; i < d.rows; i++)
            for (int j = 0; j < d.columns; j++) {
                double d2 = d.get(i, j);
                double newNum = MathUtils.roundDouble(d2, num);
                ret.put(i, j, newNum);
            }
        return createBasedOn(ret,d);
    }


    /**
     * Rounds the matrix to the number of specified by decimal places
     *
     * @param d   the matrix to round
     * @return the rounded matrix
     */
    public static <E extends DoubleMatrix> E  round(E d) {
        DoubleMatrix ret = d;
        for (int i = 0; i < d.rows; i++)
            for (int j = 0; j < d.columns; j++) {
                double d2 = d.get(i, j);
                ret.put(i, j, FastMath.round(d2));
            }
        return createBasedOn(ret,d);
    }

    public static void columnNormalizeBySum(DoubleMatrix x) {
        for (int i = 0; i < x.columns; i++)
            x.putColumn(i, x.getColumn(i).div(x.getColumn(i).sum()));
    }


    /**
     * One dimensional filter on a signal
     * @param b
     * @param a
     * @param x
     * @return
     */
    public static DoubleMatrix oneDimensionalDigitalFilter(DoubleMatrix b, DoubleMatrix a, DoubleMatrix x) {
        return new IirFilter(x,a,b).filter();
    }



    public static DoubleMatrix toOutcomeVector(int index,int numOutcomes) {
        int[] nums = new int[numOutcomes];
        nums[index] = 1;
        return toMatrix(nums);
    }

    public static FloatMatrix toOutcomeVectorFloat(int index,int numOutcomes) {
        int[] nums = new int[numOutcomes];
        nums[index] = 1;
        return toFloatMatrix(nums);
    }




    public static DoubleMatrix toDoubleMatrix(FloatMatrix arr) {
        DoubleMatrix d = new DoubleMatrix(arr.rows,arr.columns);
        for(int i = 0; i < arr.rows; i++)
            for(int j = 0; j < arr.columns; j++)
                d.put(i,j,arr.get(i,j));
        return d;
    }



    public static FloatMatrix toFloatMatrix(DoubleMatrix arr) {
        FloatMatrix d = new FloatMatrix(arr.rows,arr.columns);
        for(int i = 0; i < arr.rows; i++)
            for(int j = 0; j < arr.columns; j++)
                d.put(i,j,(float)arr.get(i,j));
        return d;
    }



    public static FloatMatrix toFloatMatrix(double[] arr) {
        FloatMatrix d = new FloatMatrix(arr.length);
        for(int i = 0; i < arr.length; i++)
            d.put(i,(float)arr[i]);
        return d;
    }

    public static FloatMatrix toFloatMatrix(double[][] arr) {
        FloatMatrix d = new FloatMatrix(arr.length,arr[0].length);
        for(int i = 0; i < arr.length; i++)
            for(int j = 0; j < arr[i].length; j++)
                d.put(i,j,(float)arr[i][j]);
        return d;
    }

    public static FloatMatrix toFloatMatrix(int[] arr) {
        FloatMatrix d = new FloatMatrix(arr.length);
        for(int i = 0; i < arr.length; i++)
            d.put(i,arr[i]);
        return d;
    }

    public static FloatMatrix toFloatMatrix(int[][] arr) {
        FloatMatrix d = new FloatMatrix(arr.length,arr[0].length);
        for(int i = 0; i < arr.length; i++)
            for(int j = 0; j < arr[i].length; j++)
                d.put(i,j,arr[i][j]);
        return d;
    }

    public static DoubleMatrix add(DoubleMatrix a,DoubleMatrix b) {
        return a.addi(b);
    }


    /**
     * Soft max function
     * row_maxes is a row vector (max for each row)
     * row_maxes = rowmaxes(input)
     * diff = exp(input - max) / diff.rowSums()
     *
     * @param input the input for the softmax
     * @param row whether the row maxes should be taken or the column maxes,
     *            this is dependent on whether the features are column wise or row wise

     * @return the softmax output (a probability matrix) scaling each row to between
     * 0 and 1
     */
    public static <E extends FloatMatrix> E softmax(E input,boolean row) {
       if(row) {
           FloatMatrix max = input.rowMaxs();
           FloatMatrix diff = MatrixFunctions.exp(input.subColumnVector(max));
           diff.diviColumnVector(diff.rowSums());
           return createBasedOn(diff,input);

       }

        else {
           FloatMatrix max = input.columnMaxs();
           FloatMatrix diff = MatrixFunctions.exp(input.subRowVector(max));
           diff.diviRowVector(diff.columnSums());
           return createBasedOn(diff,input);

       }
      }

    /**
     * Soft max function
     * row_maxes is a row vector (max for each row)
     * row_maxes = rowmaxes(input)
     * diff = exp(input - max) / diff.rowSums()
     *
     * @param input the input for the softmax
     * @param row whether the row maxes should be taken or the column maxes,
     *            this is dependent on whether the features are column wise or row wise
     * @return the softmax output (a probability matrix) scaling each row to between
     * 0 and 1
     */
    public static <E extends DoubleMatrix> E softmax(E input,boolean row) {
        if(row) {
            DoubleMatrix max = input.rowMaxs();
            DoubleMatrix diff = MatrixFunctions.exp(input.subColumnVector(max));
            diff.diviColumnVector(diff.rowSums());
            return createBasedOn(diff,input);

        }

        else {
            DoubleMatrix max = input.columnMaxs();
            DoubleMatrix diff = MatrixFunctions.exp(input.subRowVector(max));
            diff.diviRowVector(diff.columnSums());
            return createBasedOn(diff,input);

        }  }


    /**
     * Soft max function
     * row_maxes is a row vector (max for each row)
     * row_maxes = rowmaxes(input)
     * diff = exp(input - max) / diff.rowSums()
     *
     * @param input the input for the softmax
     * @return the softmax output (a probability matrix) scaling each row to between
     * 0 and 1
     */
    public static <E extends FloatMatrix> E softmax(E input) {
        return softmax(input,false);
    }

    /**
     * Soft max function
     * row_maxes is a row vector (max for each row)
     * row_maxes = rowmaxes(input)
     * diff = exp(input - max) / diff.rowSums()
     *
     * @param input the input for the softmax
     * @return the softmax output (a probability matrix) scaling each row to between
     * 0 and 1
     */
    public static <E extends DoubleMatrix> E softmax(E input) {
        return softmax(input,false);
    }

    public static DoubleMatrix mean(DoubleMatrix input,int axis) {
        DoubleMatrix ret = new DoubleMatrix(input.rows,1);
        //column wise
        if(axis == 0) {
            return input.columnMeans();
        }
        //row wise
        else if(axis == 1) {
            return ret.rowMeans();
        }


        return ret;
    }


    public static DoubleMatrix sum(DoubleMatrix input,int axis) {
        DoubleMatrix ret = new DoubleMatrix(input.rows,1);
        //column wise
        if(axis == 0) {
            for(int i = 0; i < input.columns; i++) {
                ret.put(i,input.getColumn(i).sum());
            }
            return ret;
        }
        //row wise
        else if(axis == 1) {
            for(int i = 0; i < input.rows; i++) {
                ret.put(i,input.getRow(i).sum());
            }
            return ret;
        }

        for(int i = 0; i < input.rows; i++)
            ret.put(i,input.getRow(i).sum());
        return ret;
    }



    /**
     * Generate a binomial distribution based on the given rng,
     * a matrix of p values, and a max number.
     * @param p the p matrix to use
     * @param n the n to use
     * @param rng the rng to use
     * @return a binomial distribution based on the one n, the passed in p values, and rng
     */
    public static <E extends DoubleMatrix> E binomial(E p,int n,RandomGenerator rng) {
        DoubleMatrix ret = new DoubleMatrix(p.rows,p.columns);
        for(int i = 0; i < ret.length; i++) {
            ret.put(i,MathUtils.binomial(rng, n, p.get(i)));
        }
        return createBasedOn(ret,p);
    }



    public static DoubleMatrix rand(int rows, int columns,double min,double max,RandomGenerator rng) {
        DoubleMatrix ret = new DoubleMatrix(rows,columns);
        for(int i = 0; i < ret.length; i++) {
            ret.put(i,MathUtils.randomNumberBetween(min,max,rng));
        }
        return ret;
    }





    public static DoubleMatrix columnWiseMean(DoubleMatrix x,int axis) {
        DoubleMatrix ret = DoubleMatrix.zeros(x.columns);
        for(int i = 0; i < x.columns; i++) {
            ret.put(i,x.getColumn(axis).mean());
        }
        return ret;
    }


    public static DoubleMatrix toFlattened(Collection<DoubleMatrix> matrices) {
        int length = 0;
        for(DoubleMatrix m : matrices)  length += m.length;
        DoubleMatrix ret = new DoubleMatrix(1,length);
        int linearIndex = 0;
        for(DoubleMatrix d : matrices) {
            for(int i = 0; i < d.length; i++) {
                ret.put(linearIndex++,d.get(i));
            }
        }

        return ret;

    }


    public static DoubleMatrix toFlattened(int length,Iterator<? extends DoubleMatrix>...matrices) {

        DoubleMatrix ret = new DoubleMatrix(1,length);
        int linearIndex = 0;

        List<double[]> gradient = new ArrayList<>();
        for(Iterator<? extends DoubleMatrix> iter : matrices) {
            while(iter.hasNext()) {
                DoubleMatrix d = iter.next();
                gradient.add(d.data);
            }
        }




        DoubleMatrix ret2 = new DoubleMatrix(ArrayUtil.combine(gradient));
        return ret2.reshape(1,ret2.length);
    }


    public static DoubleMatrix toFlattened(DoubleMatrix...matrices) {
        int length = 0;
        for(DoubleMatrix m : matrices)  length += m.length;
        DoubleMatrix ret = new DoubleMatrix(1,length);
        int linearIndex = 0;
        for(DoubleMatrix d : matrices) {
            for(int i = 0; i < d.length; i++) {
                ret.put(linearIndex++,d.get(i));
            }
        }

        return ret;
    }

    public static DoubleMatrix avg(DoubleMatrix...matrices) {
        if(matrices == null)
            return null;
        if(matrices.length == 1)
            return matrices[0];
        else {
            DoubleMatrix ret = matrices[0];
            for(int i = 1; i < matrices.length; i++)
                ret = ret.add(matrices[i]);

            ret = ret.div(matrices.length);
            return ret;
        }
    }


    public static int maxIndex(DoubleMatrix matrix) {
        double max = matrix.max();
        for(int j = 0; j < matrix.length; j++) {
            if(matrix.get(j) == max)
                return j;
        }
        return -1;
    }


    /**
     * Takes the sigmoid of a given matrix
     * @param x the input
     * @param <E>
     * @return the input with the sigmoid function applied
     */
    public static <E extends DoubleMatrix> E sqrt(E x) {
        DoubleMatrix ret = new DoubleMatrix(x.rows,x.columns);
        for(int i = 0; i < ret.length; i++)
            ret.put(i, x.get(i) > 0 ? FastMath.sqrt(x.get(i)) : 0.0);

        return createBasedOn(ret,x);
    }


    /**
     * Takes the sigmoid of a given matrix
     * @param x the input
     * @param <E>
     * @return the input with the sigmoid function applied
     */
    public static <E extends DoubleMatrix> E sigmoid(E x) {
        DoubleMatrix ones = DoubleMatrix.ones(x.rows, x.columns);
        return createBasedOn(ones.div(ones.add(exp(x.neg()))),x);
    }

    public static DoubleMatrix dot(DoubleMatrix a,DoubleMatrix b) {
        boolean isScalar = a.isColumnVector() || a.isRowVector() && b.isColumnVector() || b.isRowVector();
        if(isScalar) {
            return DoubleMatrix.scalar(a.dot(b));
        }
        else {
            return  a.mmul(b);
        }
    }

    /**
     * Ensures numerical stability.
     * Clips values of input such that
     * exp(k * in) is within single numerical precision
     * @param input the input to trim
     * @param k the k (usually 1)
     * @param <E>
     * @return the stabilized input
     */
    public static <E extends DoubleMatrix> E stabilizeInput(E input,double k) {
        double realMin =  1.1755e-38;
        double cutOff = FastMath.log(realMin);
        for(int i = 0; i < input.length; i++) {
            if(input.get(i) * k > -cutOff)
                input.put(i,-cutOff / k);
            else if(input.get(i) * k < cutOff)
                input.put(i,cutOff / k);
        }

        return input;

    }


    public static DoubleMatrix out(DoubleMatrix a,DoubleMatrix b) {
        return a.mmul(b);
    }


    public static DoubleMatrix scalarMinus(double scalar,DoubleMatrix ep) {
        DoubleMatrix d = new DoubleMatrix(ep.rows,ep.columns);
        d.addi(scalar);
        return d.sub(ep);
    }

    public static <E extends DoubleMatrix> E oneMinus(E ep) {
        return createBasedOn(DoubleMatrix.ones(ep.rows, ep.columns).sub(ep),ep);
    }

    public static <E extends DoubleMatrix> E oneDiv(E ep) {
        for(int i = 0; i < ep.rows; i++) {
            for(int j = 0; j < ep.columns; j++) {
                if(ep.get(i,j) == 0) {
                    ep.put(i,j,0.01);
                }
            }
        }
        return createBasedOn(DoubleMatrix.ones(ep.rows, ep.columns).div(ep),ep);
    }


    /**
     * Normalizes the passed in matrix by subtracting the mean
     * and dividing by the standard deviation
     * @param toNormalize the matrix to normalize
     */
    public static <E extends DoubleMatrix> void normalizeZeroMeanAndUnitVariance(E toNormalize) {
        DoubleMatrix columnMeans = toNormalize.columnMeans();
        DoubleMatrix columnStds = columnStdDeviation(toNormalize);

        toNormalize.subiRowVector(columnMeans);
        columnStds.addi(1e-6);
        toNormalize.diviRowVector(columnStds);

    }




    /**
     * Column wise variance
     * @param input the input to get the variance for
     * @return the column wise variance of the input
     */
    public static DoubleMatrix columnVariance(DoubleMatrix input) {
        DoubleMatrix columnMeans = input.columnMeans();
        DoubleMatrix ret = new DoubleMatrix(1,columnMeans.columns);
        for(int i = 0;i < ret.columns; i++) {
            DoubleMatrix column = input.getColumn(i);
            double variance = StatUtils.variance(column.toArray(),columnMeans.get(i));
            if(variance == 0)
                variance = 1e-6;
            ret.put(i,variance);
        }
        return ret;
    }

    /**
     * Calculates the column wise standard deviations
     * of the matrix
     * @param m the matrix to use
     * @return the standard deviations of each column in the matrix
     * as a row matrix
     */
    public static DoubleMatrix columnStd(DoubleMatrix m) {
        DoubleMatrix ret = new DoubleMatrix(1,m.columns);
        StandardDeviation std = new StandardDeviation();

        for(int i = 0; i < m.columns; i++) {
            double result = std.evaluate(m.getColumn(i).data);
            ret.put(i,result);
        }

        ret.divi(m.rows);

        return ret;
    }

    /**
     * Calculates the column wise standard deviations
     * of the matrix
     * @param m the matrix to use
     * @return the standard deviations of each column in the matrix
     * as a row matrix
     */
    public static <E extends DoubleMatrix> E rowStd(E m) {
        StandardDeviation std = new StandardDeviation();

        DoubleMatrix ret = new DoubleMatrix(1,m.columns);
        for(int i = 0; i < m.rows; i++) {
            double result = std.evaluate(m.getRow(i).data);
            ret.put(i,result);
        }
        return createBasedOn(ret,m);
    }

    /**
     * Returns the mean squared error of the 2 matrices.
     * Note that the matrices must be the same length
     * or an {@link IllegalArgumentException} is thrown
     * @param input the first one
     * @param other the second one
     * @return the mean square error of the matrices
     */
    public static <E extends DoubleMatrix> double meanSquaredError(E input,E other) {
        if(input.length != other.length)
            throw new IllegalArgumentException("Matrices must be same length");
        SimpleRegression r = new SimpleRegression();
        r.addData(new double[][]{input.data,other.data});
        return r.getMeanSquareError();
    }

    /**
     * A log impl that prevents numerical underflow
     * Any number that's infinity or NaN is replaced by
     * 1e-6.
     * @param vals the vals to convert to log
     * @return the log of the numbers or 1e-6 for anomalies
     */
    public static <E extends DoubleMatrix> E log(E vals) {
        DoubleMatrix ret = new DoubleMatrix(vals.rows,vals.columns);
        for(int i = 0; i < vals.length; i++) {
            double logVal = Math.log(vals.get(i));
            if(!Double.isNaN(logVal) && !Double.isInfinite(logVal))
                ret.put(i,logVal);
            else
                ret.put(i,1e-6);
        }
        return createBasedOn(ret,vals);
    }

    /**
     * Returns the sum squared error of the 2 matrices.
     * Note that the matrices must be the same length
     * or an {@link IllegalArgumentException} is thrown
     * @param input the first one
     * @param other the second one
     * @return the sum square error of the matrices
     */
    public static <E extends DoubleMatrix> double sumSquaredError(E input,E other) {
        if(input.length != other.length)
            throw new IllegalArgumentException("Matrices must be same length");
        SimpleRegression r = new SimpleRegression();
        r.addData(new double[][]{input.data,other.data});
        return r.getSumSquaredErrors();
    }

    public static <E extends DoubleMatrix> void normalizeMatrix(E toNormalize) {
        DoubleMatrix columnMeans = toNormalize.columnMeans();
        toNormalize.subiRowVector(columnMeans);
        DoubleMatrix std = columnStd(toNormalize);
        std.addi(1e-6);
        toNormalize.diviRowVector(std);
    }


    public static <E extends DoubleMatrix> E normalizeByColumnSums(E m) {
        DoubleMatrix columnSums = m.columnSums();
        for(int i = 0; i < m.columns; i++) {
            m.putColumn(i,m.getColumn(i).div(columnSums.get(i)));

        }
        return m;
    }


    public static DoubleMatrix columnStdDeviation(DoubleMatrix m) {
        DoubleMatrix ret = new DoubleMatrix(1,m.columns);

        for(int i = 0; i < ret.length; i++) {
            StandardDeviation dev = new StandardDeviation();
            double std = dev.evaluate(m.getColumn(i).toArray());
            ret.put(i,std);
        }

        return ret;
    }

    /**
     * Divides the given matrix's columns
     * by each column's respective standard deviations
     * @param m the matrix to divide
     * @return the column divided by the standard deviation
     */
    public static <E extends DoubleMatrix> E divColumnsByStDeviation(E m) {
        DoubleMatrix std = columnStdDeviation(m);
        for(int i = 0; i < m.columns; i++) {
            m.putColumn(i,m.getColumn(i).div(std.get(i)));
        }
        return m;

    }

    /**
     * Subtracts by column mean.
     * This ensures a mean of zero.
     * This is part of normalizing inputs
     * for a neural net
     * @param m the matrix to normalize
     * @return the normalized matrix which each
     * column subtracted by its mean
     */
    public static <E extends DoubleMatrix> E normalizeByColumnMeans(E m) {
        DoubleMatrix columnMeans = m.columnMeans();
        for(int i = 0; i < m.columns; i++) {
            m.putColumn(i,m.getColumn(i).sub(columnMeans.get(i)));

        }
        return m;
    }

    public static <E extends DoubleMatrix> E  normalizeByRowSums(E m) {
        DoubleMatrix rowSums = m.rowSums();
        for(int i = 0; i < m.rows; i++) {
            m.putRow(i,m.getRow(i).div(rowSums.get(i)));
        }
        return m;
    }


    public static <E extends FloatMatrix> void complainAboutMissMatchedMatrices(E d1, E d2) {
        if (d1 == null || d2 == null)
            throw new IllegalArgumentException("No null matrices allowed");
        if (d1.rows != d2.rows)
            throw new IllegalArgumentException("Matrices must have same rows");

    }


    /**
     * Cuts all numbers below a certain cut off
     *
     * @param minNumber the min number to check
     * @param matrix    the matrix to max by
     */
    public static  <E extends FloatMatrix> void max(double minNumber, E matrix) {
        for (int i = 0; i < matrix.length; i++)
            matrix.put(i, Math.max(0, matrix.get(i)));

    }

    /**
     * Divides each row by its max
     *
     * @param toScale the matrix to divide by its row maxes
     */
    public static void scaleByMax(FloatMatrix toScale) {
        FloatMatrix scale = toScale.rowMaxs();
        for (int i = 0; i < toScale.rows; i++) {
            float scaleBy = scale.get(i, 0);
            toScale.putRow(i, toScale.getRow(i).divi(scaleBy));
        }
    }


    /**
     * Flips the dimensions of each slice of a tensor or 4d tensor
     * slice wise
     * @param input the input to flip
     * @param <E>
     * @return the flipped tensor
     */
    public static <E extends FloatMatrix> E flipDimMultiDim(E input) {
        if(input instanceof FloatFourDTensor) {
            FloatFourDTensor t = (FloatFourDTensor) input;
            FloatMatrix ret = new FloatMatrix(input.rows,input.columns);
            FloatFourDTensor flipped = createBasedOn(ret,t);
            for(int i = 0; i < flipped.numTensors(); i++) {
                for(int j = 0; j < flipped.getTensor(i).slices(); j++) {
                    flipped.put(i,j,flipDim(t.getSliceOfTensor(i,j)));
                }
            }

            return createBasedOn(flipped,input);
        }
        else if(input instanceof FloatTensor) {
            FloatTensor t = (FloatTensor) input;
            FloatMatrix ret = new FloatMatrix(input.rows,input.columns);
            FloatTensor flipped = createBasedOn(ret,t);
            for(int j = 0; j < flipped.slices(); j++) {
                flipped.setSlice(j,flipDim(t.getSlice(j)));
            }
            return createBasedOn(ret,input);
        }

        else
            return (E) flipDim(input);
    }


    /**
     * Flips the dimensions of the given matrix.
     * [1,2]       [3,4]
     * [3,4] --->  [1,2]
     * @param flip the matrix to flip
     * @return the flipped matrix
     */
    public static FloatMatrix flipDim(FloatMatrix flip) {
        FloatMatrix ret = new FloatMatrix(flip.rows,flip.columns);
        CounterMap<Integer,Integer> dimsFlipped = new CounterMap<>();
        for(int j = 0; j < flip.columns; j++) {
            for(int i = 0; i < flip.rows;  i++) {
                for(int k = flip.rows - 1; k >= 0; k--) {
                    if(dimsFlipped.getCount(i,j) > 0 || dimsFlipped.getCount(k,j) > 0)
                        continue;
                    dimsFlipped.incrementCount(i,j,1.0);
                    dimsFlipped.incrementCount(k,j,1.0);
                    float first = flip.get(i,j);
                    float second = flip.get(k,j);
                    ret.put(k,j,first);
                    ret.put(i,j,second);
                }
            }
        }


        return ret;

    }



    /**
     * Cumulative sum
     *
     * @param sum the matrix to get the cumulative sum of
     * @return a matrix of the same dimensions such that the at i,j
     * is the cumulative sum of it + the predecessor elements in the column
     */
    public static  <E extends FloatMatrix> E cumsum(E sum) {
        FloatMatrix ret = new FloatMatrix(sum.rows, sum.columns);
        for (int i = 0; i < ret.columns; i++) {
            for (int j = 0; j < ret.rows; j++) {
                int[] indices = new int[j + 1];
                for (int row = 0; row < indices.length; row++)
                    indices[row] = row;
                FloatMatrix toSum = sum.get(indices, new int[]{i});
                float d = toSum.sum();
                ret.put(j, i, d);
            }
        }
        return (E) ret;
    }


    /**
     * Truncates a matrix down to size rows x columns
     *
     * @param toTruncate the matrix to truncate
     * @param rows       the rows to reduce to
     * @param columns    the columns to reduce to
     * @return a subset of the old matrix
     * with the specified dimensions
     */
    public static FloatMatrix truncate(FloatMatrix toTruncate, int rows, int columns) {
        if (rows >= toTruncate.rows && columns >= toTruncate.columns || rows < 1 || columns < 1)
            return toTruncate;

        FloatMatrix ret = new FloatMatrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                ret.put(i, j, toTruncate.get(i, j));
            }
        }
        return ret;
    }

    /**
     * Reshapes this matrix in to a 3d matrix
     * @param input the input matrix
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param numSlices the number of slices in the tensor
     * @return the reshaped matrix as a tensor
     */
    public static <E extends FloatMatrix> FloatTensor reshape(E input,int rows, int columns,int numSlices) {
        FloatMatrix ret = input.reshape(rows * numSlices,columns);
        FloatFourDTensor retTensor = new FloatFourDTensor(ret,false);
        retTensor.setSlices(numSlices);
        return retTensor;
    }


    /**
     * Rotates a matrix by reversing its input
     * If its  any kind of tensor each slice of each tensor
     * will be reversed
     * @param input the input to rotate
     * @param <E>
     * @return the rotated matrix or tensor
     */
    public static <E extends FloatMatrix> E rot(E input) {
        if(input instanceof FloatFourDTensor) {
            FloatFourDTensor t = (FloatFourDTensor) input;
            FloatFourDTensor ret = new FloatFourDTensor(t);
            for(int i = 0; i < ret.numTensors(); i++) {
                FloatTensor t1 = ret.getTensor(i);
                for(int j = 0; j < t1.slices(); j++) {
                    ret.setSlice(j,reverse(t1.getSlice(j)));
                }
            }

            return createBasedOn(ret,input);
        }

        else if(input instanceof FloatTensor) {
            FloatTensor t = (FloatTensor) input;
            FloatTensor ret = new FloatTensor(t);
            for(int j = 0; j < t.slices(); j++) {
                ret.setSlice(j,reverse(t.getSlice(j)));
            }
            return createBasedOn(ret,input);


        }

        else
            return reverse(input);

    }



    /**
     * Reshapes this matrix in to a 3d matrix
     * @param input the input matrix
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param numSlices the number of slices in the tensor
     * @return the reshaped matrix as a tensor
     */
    public static <E extends FloatMatrix> FloatFourDTensor reshape(E input,int rows, int columns,int numSlices,int numTensors) {
        FloatMatrix ret = input.reshape(rows * numSlices,columns);
        FloatFourDTensor retTensor = new FloatFourDTensor(ret,false);
        retTensor.setSlices(numSlices);
        retTensor.setNumTensor(numTensors);
        return retTensor;
    }

    /**
     * Binarizes the matrix such that any number greater than cutoff is 1 otherwise zero
     * @param cutoff the cutoff point
     */
    public static void binarize(double cutoff,FloatMatrix input) {
        for(int i = 0; i < input.length; i++)
            if(input.get(i) > cutoff)
                input.put(i,1);
            else
                input.put(i,0);
    }


    /**
     * Generate a new matrix which has the given number of replications of this.
     */
    public static ComplexFloatMatrix repmat(ComplexFloatMatrix matrix, int rowMult, int columnMult) {
        ComplexFloatMatrix result = new ComplexFloatMatrix(matrix.rows * rowMult, matrix.columns * columnMult);

        for (int c = 0; c < columnMult; c++) {
            for (int r = 0; r < rowMult; r++) {
                for (int i = 0; i < matrix.rows; i++) {
                    for (int j = 0; j < matrix.columns; j++) {
                        result.put(r * matrix.rows + i, c * matrix.columns + j, matrix.get(i, j));
                    }
                }
            }
        }
        return result;
    }

    public static FloatMatrix shape(FloatMatrix d) {
        return new FloatMatrix(new float[]{d.rows, d.columns});
    }

    public static FloatMatrix size(FloatMatrix d) {
        return shape(d);
    }


    /**
     * Down sample a signal
     * @param data
     * @param stride
     * @param <E>
     * @return
     */
    public static <E extends FloatMatrix> E downSample(E data, FloatMatrix stride) {
        FloatMatrix d = FloatMatrix.ones((int) stride.get(0), (int) stride.get(1));
        d.divi(prod(stride));
        FloatMatrix ret = Convolution.conv2d(data, d, Convolution.Type.VALID);
        ret = ret.get(RangeUtils.interval(0, (int) stride.get(0)), RangeUtils.interval(0, (int) stride.get(1)));
        return createBasedOn(ret,data);
    }

    /**
     * Takes the product of all the elements in the matrix
     *
     * @param product the matrix to get the product of elements of
     * @return the product of all the elements in the matrix
     */
    public static <E extends FloatMatrix> float prod(E product) {
        float ret = 1.0f;
        for (int i = 0; i < product.length; i++)
            ret *= product.get(i);
        return ret;
    }


    /**
     * Upsampling a signal
     * @param d
     * @param scale
     * @param <E>
     * @return
     */
    public static <E extends FloatMatrix> E upSample(E d, E scale) {
        FloatMatrix shape = size(d);

        FloatMatrix idx = new FloatMatrix(shape.length, 1);


        for (int i = 0; i < shape.length; i++) {
            FloatMatrix tmp = FloatMatrix.zeros((int) shape.get(i) * (int) scale.get(i), 1);
            int[] indices = indicesCustomRange(0, (int) scale.get(i), (int) scale.get(i) * (int) shape.get(i));
            tmp.put(indices, 1.0f);
            idx.put(i, cumsum(tmp).sum());
        }
        return createBasedOn(idx,d);
    }

    public static FloatMatrix rangeVector(float begin, float end) {
        int diff = (int) Math.abs(end - begin);
        FloatMatrix ret = new FloatMatrix(1, diff);
        for (int i = 0; i < ret.length; i++)
            ret.put(i, i);
        return ret;
    }

    public static ComplexFloatMatrix complexRangeVector(float begin, float end) {
        int diff = (int) Math.abs(end - begin);
        ComplexFloatMatrix ret = new ComplexFloatMatrix(1, diff);
        for (int i = 0; i < ret.length; i++)
            ret.put(i, i);
        return ret.transpose();
    }


    public static double angle(ComplexFloat phase) {
        Complex c = new Complex(phase.real(), phase.imag());
        return c.atan().getReal();
    }

    /**
     * Implements matlab's compare of complex numbers:
     * compares the max(abs(d1),abs(d2))
     * if they are equal, compares their angles
     *
     * @param d1 the first number
     * @param d2 the second number
     * @return standard comparator interface
     */
    private static int compare(ComplexFloat d1, ComplexFloat d2) {
        if (d1.abs() > d2.abs())
            return 1;
        else if (d2.abs() > d1.abs())
            return -1;
        else {
            if (angle(d1) > angle(d2))
                return 1;
            else if (angle(d1) < angle(d2))
                return -1;
            return 0;
        }

    }


    public static ComplexFloat max(ComplexFloatMatrix matrix) {
        ComplexFloat max = matrix.get(0);
        for (int i = 1; i < matrix.length; i++)
            if (compare(max, matrix.get(i)) > 0)
                max = matrix.get(i);
        return max;
    }


    /**
     * Divides each row by its max
     *
     * @param toScale the matrix to divide by its row maxes
     */
    public static void scaleByMax(ComplexFloatMatrix toScale) {

        for (int i = 0; i < toScale.rows; i++) {
            ComplexFloat scaleBy = max(toScale.getRow(i));
            toScale.putRow(i, toScale.getRow(i).divi(scaleBy));
        }
    }

    public static <E extends FloatMatrix>  E variance(E input) {
        FloatMatrix means = (E) input.columnMeans();
        FloatMatrix diff = MatrixFunctions.pow(input.subRowVector(means), 2);
        //avg of the squared differences from the mean
        FloatMatrix variance =  diff.columnMeans().div(input.rows);
        return createBasedOn(variance,input);

    }


    public static <E extends FloatMatrix> int[] toInts(E ints)  {
        int[] ret = new int[ints.length];
        for(int i = 0; i < ints.length; i++)
            ret[i] = (int) ints.get(i);
        return ret;
    }


    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     * @param toReverse the matrix to reverse
     * @param <E>
     * @return the reversed matrix
     */
    public static  <E extends FloatMatrix> E reverse(E toReverse) {
        FloatMatrix ret = new FloatMatrix(toReverse.rows, toReverse.columns);
        int reverseIndex = 0;
        for (int i = toReverse.length - 1; i >= 0; i--) {
            ret.put(reverseIndex++, toReverse.get(i));
        }
        return createBasedOn(ret,toReverse);
    }

    /**
     * Utility method for creating a variety of matrices, useful for handling conversions.
     *
     * This method is always O(1) due to not needing to copy data.
     *
     * Note however, that this makes the assumption that the passed in matrix isn't being referenced
     * by anything else as this being a safe operation
     * @param result the result matrix to cast
     * @param input the input it was based on
     * @param <E> the type of matrix
     * @return the casted matrix
     */
    public static <E extends FloatMatrix> E createBasedOn(FloatMatrix result,E input) {
        if(input.getClass().equals(result.getClass()))
            return (E) result;

        else if(input instanceof FloatFourDTensor) {
            FloatFourDTensor tensor = new FloatFourDTensor(result,false);
            FloatFourDTensor casted = (FloatFourDTensor) input;
            tensor.setSlices(casted.slices());
            tensor.setPerMatrixRows(casted.rows());
            tensor.setNumTensor(casted.getNumTensor());
            return (E) tensor;
        }

        else if(input instanceof FloatTensor) {
            FloatTensor ret = new FloatTensor(result,false);
            FloatTensor casted = (FloatTensor) input;
            ret.setPerMatrixRows(ret.rows());
            ret.setSlices(casted.slices());
            return (E) ret;
        }
        else
            return (E) result;


    }


    /**
     * Returns the maximum dimension of the passed in matrix
     *
     * @param d the max dimension of the passed in matrix
     * @return the max dimension of the passed in matrix
     */
    public static float  length(ComplexFloatMatrix d) {
        return (float) Math.max(d.rows, d.columns);


    }

    /**
     * Floor function applied to the matrix
     * @param input
     * @param <E>
     * @return
     */
    public static <E extends FloatMatrix> E floor(E input) {
        return createBasedOn(MatrixFunctions.floor(input),input);
    }




    /**
     * Returns the maximum dimension of the passed in matrix
     *
     * @param d the max dimension of the passed in matrix
     * @return the max dimension of the passed in matrix
     */
    public static <E extends FloatMatrix> float length(E d) {
        if (d instanceof FloatTensor) {
            FloatTensor t = (FloatTensor) d;
            return (float) MathUtils.max(new double[]{t.rows(), t.columns(), t.slices()});
        } else
            return Math.max(d.rows, d.columns);


    }


    public static FloatDataSet xorFloatData(int n) {

        FloatMatrix x = FloatMatrix.rand(n, 2);
        x = x.gti(0.5f);

        FloatMatrix y = FloatMatrix.zeros(n, 2);
        for (int i = 0; i < x.rows; i++) {
            if (x.get(i, 0) == x.get(i, 1))
                y.put(i, 0, 1);
            else
                y.put(i, 1, 1);
        }

        return new FloatDataSet(x, y);

    }

    public static FloatDataSet xorFloatData(int n, int columns) {

        FloatMatrix x = FloatMatrix.rand(n, columns);
        x = x.gti(0.5f);

        FloatMatrix x2 = FloatMatrix.rand(n, columns);
        x2 = x2.gti(0.5f);

        FloatMatrix eq = x.eq(x2).eq(FloatMatrix.zeros(n, columns));


        int median = columns / 2;

        FloatMatrix outcomes = new FloatMatrix(n, 2);
        for (int i = 0; i < outcomes.rows; i++) {
            FloatMatrix left = eq.get(i, new org.jblas.ranges.IntervalRange(0, median));
            FloatMatrix right = eq.get(i, new org.jblas.ranges.IntervalRange(median, columns));
            if (left.sum() > right.sum())
                outcomes.put(i, 0, 1);
            else
                outcomes.put(i, 1, 1);
        }


        return new FloatDataSet(eq, outcomes);

    }

    public static <E extends FloatMatrix>  double magnitude(E vec) {
        double sum_mag = 0;
        for (int i = 0; i < vec.length; i++)
            sum_mag = sum_mag + vec.get(i) * vec.get(i);

        return Math.sqrt(sum_mag);
    }


    /**
     * Exp with more generic casting
     * @param d the input
     * @param <E>
     * @return the exp of this matrix
     */
    public static <E extends FloatMatrix> E exp(E d) {
        return createBasedOn(MatrixFunctions.exp(d),d);
    }

    /**
     * Flattens the matrix
     * @param d
     * @param <E>
     * @return
     */
    public static <E extends FloatMatrix> E unroll(E d) {
        FloatMatrix ret = new FloatMatrix(1, d.length);
        for (int i = 0; i < d.length; i++)
            ret.put(i, d.get(i));
        return createBasedOn(ret,d);
    }


    public static <E extends FloatMatrix> E outcomes(E d) {
        FloatMatrix ret = new FloatMatrix(d.rows, 1);
        for (int i = 0; i < d.rows; i++)
            ret.put(i, SimpleBlas.iamax(d.getRow(i)));
        return createBasedOn(ret,d);
    }

    /**
     *
     * @param d1
     * @param d2
     * @param <E>
     * @return
     */
    public static <E extends FloatMatrix> double cosineSim(E d1, E d2) {
        d1 = unitVec(d1);
        d2 = unitVec(d2);
        double ret = d1.dot(d2);
        return ret;
    }

    /**
     * Normalizes the matrix by subtracting the min,
     * dividing by the max - min
     * @param input the input to normalize
     * @param <E>
     * @return the normalized matrix
     */
    public static <E extends FloatMatrix> E normalize(E input) {
        float min = input.min();
        float max = input.max();
        return createBasedOn(input.subi(min).divi(max - min),input);
    }


    public static <E extends FloatMatrix> double  cosine(E matrix) {
        return 1 * Math.sqrt(MatrixFunctions.pow(matrix, 2).sum());
    }


    public static <E extends FloatMatrix> E unitVec(E toScale) {
        float length = toScale.norm2();
        if (length > 0)
            return createBasedOn(SimpleBlas.scal(1.0f / length, toScale),toScale);
        return toScale;
    }

    /**
     * A uniform sample ranging from 0 to 1.
     *
     * @param rng     the rng to use
     * @param rows    the number of rows of the matrix
     * @param columns the number of columns of the matrix
     * @return a uniform sample of the given shape and size
     * with numbers between 0 and 1
     */
    public static FloatMatrix uniformFloat(RandomGenerator rng, int rows, int columns) {

        UniformRealDistribution uDist = new UniformRealDistribution(rng, 0, 1);
        FloatMatrix U = new FloatMatrix(rows, columns);
        for (int i = 0; i < U.rows; i++)
            for (int j = 0; j < U.columns; j++)
                U.put(i, j, (float) uDist.sample());
        return U;
    }


    /**
     * A uniform sample ranging from 0 to sigma.
     *
     * @param rng   the rng to use
     * @param mean, the matrix mean from which to generate values from
     * @param sigma the standard deviation to use to generate the gaussian noise
     * @return a uniform sample of the given shape and size
     * <p/>
     * with numbers between 0 and 1
     */
    public static <E extends FloatMatrix> E normal(RandomGenerator rng, E mean, double sigma) {
        FloatMatrix U = new FloatMatrix(mean.rows, mean.columns);
        for (int i = 0; i < U.rows; i++)
            for (int j = 0; j < U.columns; j++) {
                RealDistribution reals = new NormalDistribution(rng,mean.get(i, j), FastMath.sqrt(sigma),NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
                U.put(i, j, (float) reals.sample());

            }
        return createBasedOn(U,mean);
    }


    /**
     * A uniform sample ranging from 0 to sigma.
     *
     * @param rng      the rng to use
     * @param mean,    the matrix mean from which to generate values from
     * @param variance the variance matrix where each column is the variance
     *                 for the respective columns of the matrix
     * @return a uniform sample of the given shape and size
     * <p/>
     * with numbers between 0 and 1
     */
    public static <E extends FloatMatrix> E normal(RandomGenerator rng, E mean, E variance) {
        FloatMatrix std =  sqrt(variance);
        for (int i = 0; i < variance.length; i++)
            if (variance.get(i) <= 0)
                variance.put(i, (float) 1e-4);

        FloatMatrix U = new FloatMatrix(mean.rows, mean.columns);
        for (int i = 0; i < U.rows; i++)
            for (int j = 0; j < U.columns; j++) {
                RealDistribution reals = new NormalDistribution(rng,mean.get(i, j), std.get(j),NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
                U.put(i, j, (float) reals.sample());

            }
        return createBasedOn(U,mean);
    }


    /**
     * Sample from a normal distribution given a mean of zero and a matrix of standard deviations.
     *
     * @param rng the rng to use
     *            for the respective columns of the matrix
     * @return a uniform sample of the given shape and size
     */
    public static FloatMatrix normal(RandomGenerator rng, FloatMatrix standardDeviations) {

        FloatMatrix U = new FloatMatrix(standardDeviations.rows, standardDeviations.columns);
        for (int i = 0; i < U.rows; i++)
            for (int j = 0; j < U.columns; j++) {
                RealDistribution reals = new NormalDistribution(0, standardDeviations.get(i, j));
                U.put(i, j, (float) reals.sample());

            }
        return U;
    }

    public static <E extends FloatMatrix> boolean isValidOutcome(E out) {
        boolean found = false;
        for (int col = 0; col < out.length; col++) {
            if (out.get(col) > 0) {
                found = true;
                break;
            }
        }
        return found;
    }





    public static ComplexFloatMatrix exp(ComplexFloatMatrix input) {
        ComplexFloatMatrix ret = new ComplexFloatMatrix(input.rows, input.columns);
        for (int i = 0; i < ret.length; i++) {
            ret.put(i, ComplexUtil.exp(input.get(i)));
        }
        return ret;
    }


    public static ComplexFloatMatrix complexPadWithZeros(FloatMatrix toPad, int rows, int cols) {
        ComplexFloatMatrix ret = ComplexFloatMatrix.zeros(rows, cols);
        for (int i = 0; i < toPad.rows; i++) {
            for (int j = 0; j < toPad.columns; j++) {
                ret.put(i, j, toPad.get(i, j));
            }
        }
        return ret;
    }

    /**
     * Pads a matrix with zeros
     *
     * @param toPad the matrix to pad
     * @param rows  the number of rows to pad the matrix to
     * @param cols  the number of columns to pad the matrix to
     * @return
     */
    public static ComplexFloatMatrix padWithZeros(ComplexFloatMatrix toPad, int rows, int cols) {
        ComplexFloatMatrix ret = ComplexFloatMatrix.zeros(rows, cols);
        for (int i = 0; i < toPad.rows; i++) {
            for (int j = 0; j < toPad.columns; j++) {
                ret.put(i, j, toPad.get(i, j));
            }
        }
        return ret;
    }


    /**
     * Assigns every element in the matrix to the given value
     *
     * @param toAssign the matrix to modify
     * @param val      the value to assign
     */
    public static <E extends FloatMatrix> void assign(E toAssign, float val) {
        for (int i = 0; i < toAssign.length; i++)
            toAssign.put(i, val);
    }

    /**
     * Rotate a matrix 90 degrees
     *
     * @param toRotate the matrix to rotate
     */
    public static <E extends FloatMatrix> void rot90(E toRotate) {
        for (int i = 0; i < toRotate.rows; i++)
            for (int j = 0; j < toRotate.columns; j++)
                toRotate.put(i, j, toRotate.get(toRotate.columns - i - 1));

    }

    /**
     * Pads the matrix with zeros to surround the passed in matrix
     * with the given rows and columns
     *
     * @param toPad the matrix to pad
     * @param rows  the rows of the destination matrix
     * @param cols  the columns of the destination matrix
     * @return a new matrix with the elements of toPad with zeros or
     * a clone of toPad if the rows and columns are both greater in length than
     * rows and cols
     */
    public  static <E extends FloatMatrix> E padWithZeros(E toPad, int rows, int cols) {
        if (rows < 1)
            throw new IllegalArgumentException("Illegal number of rows " + rows);
        if (cols < 1)
            throw new IllegalArgumentException("Illegal number of columns " + cols);
        FloatMatrix ret = null;
        //nothing to pad
        if (toPad.rows >= rows) {
            if (toPad.columns >= cols)
                return createBasedOn(toPad.dup(),toPad);
            else
                ret = new FloatMatrix(toPad.rows, cols);


        } else if (toPad.columns >= cols) {
            if (toPad.rows >= rows)
                return createBasedOn(toPad.dup(),toPad);
            else
                ret = new FloatMatrix(rows, toPad.columns);
        } else
            ret = new FloatMatrix(rows, cols);

        for (int i = 0; i < toPad.rows; i++) {
            for (int j = 0; j < toPad.columns; j++) {
                float d = toPad.get(i, j);
                ret.put(i, j, d);
            }
        }
        return createBasedOn(ret,toPad);
    }

    public static ComplexFloatMatrix numDivideMatrix(ComplexFloat div, ComplexFloatMatrix toDiv) {
        ComplexFloatMatrix ret = new ComplexFloatMatrix(toDiv.rows, toDiv.columns);

        for (int i = 0; i < ret.length; i++) {
            //prevent numerical underflow
            ComplexFloat curr = toDiv.get(i).addi((float) 1e-6);
            ret.put(i, div.div(curr));
        }

        return ret;
    }


    /**
     * div / matrix. This also does padding of the matrix
     * so that no numbers are 0 by adding 1e-6 when dividing
     *
     * @param div   the number to divide by
     * @param toDiv the matrix to divide
     * @return the matrix such that each element i in the matrix
     * is div / num
     */
    public static <E extends FloatMatrix> E numDivideMatrix(float div, E toDiv) {
        FloatMatrix ret = new FloatMatrix(toDiv.rows, toDiv.columns);

        for (int i = 0; i < ret.length; i++)
            //prevent numerical underflow
            ret.put(i, div / toDiv.get(i) + 1e-6f);
        return createBasedOn(ret,toDiv);
    }


    public static <E extends FloatMatrix> boolean isInfinite(E test) {
        FloatMatrix nan = test.isInfinite();
        for (int i = 0; i < nan.length; i++) {
            if (nan.get(i) > 0)
                return true;
        }
        return false;
    }

    public static boolean isNaN(FloatMatrix test) {
        for (int i = 0; i < test.length; i++) {
            if (Double.isNaN(test.get(i)))
                return true;
        }
        return false;
    }


    public static void discretizeColumns(FloatMatrix toDiscretize, int numBins) {
        FloatMatrix columnMaxes = toDiscretize.columnMaxs();
        FloatMatrix columnMins = toDiscretize.columnMins();
        for (int i = 0; i < toDiscretize.columns; i++) {
            double min = columnMins.get(i);
            double max = columnMaxes.get(i);
            FloatMatrix col = toDiscretize.getColumn(i);
            FloatMatrix newCol = new FloatMatrix(col.length);
            for (int j = 0; j < col.length; j++) {
                int bin = MathUtils.discretize(col.get(j), min, max, numBins);
                newCol.put(j, bin);
            }
            toDiscretize.putColumn(i, newCol);

        }
    }

    /**
     * Rounds the matrix to the number of specified by decimal places
     *
     * @param d   the matrix to round
     * @param num the number of decimal places to round to(example: pass 2 for the 10s place)
     * @return the rounded matrix
     */
    public static <E extends FloatMatrix> E  roundToTheNearest(E d, int num) {
        FloatMatrix ret = d.mul(num);
        for (int i = 0; i < d.rows; i++)
            for (int j = 0; j < d.columns; j++) {
                float d2 = d.get(i, j);
                float newNum = MathUtils.roundFloat(d2, num);
                ret.put(i, j, newNum);
            }
        return createBasedOn(ret,d);
    }


    /**
     * Rounds the matrix to the number of specified by decimal places
     *
     * @param d   the matrix to round
     * @return the rounded matrix
     */
    public static <E extends FloatMatrix> E  round(E d) {
        FloatMatrix ret = d;
        for (int i = 0; i < d.rows; i++)
            for (int j = 0; j < d.columns; j++) {
                double d2 = d.get(i, j);
                ret.put(i, j, FastMath.round(d2));
            }
        return createBasedOn(ret,d);
    }

    public static void columnNormalizeBySum(FloatMatrix x) {
        for (int i = 0; i < x.columns; i++)
            x.putColumn(i, x.getColumn(i).div(x.getColumn(i).sum()));
    }


    /**
     * One dimensional filter on a signal
     * @param b
     * @param a
     * @param x
     * @return
     */
    public static FloatMatrix oneDimensionalDigitalFilter(FloatMatrix b, FloatMatrix a, FloatMatrix x) {
        return new IirFilterFloat(x,a,b).filter();
    }


    //public static FloatMatrix oneDimensionalDigitalFilter()

    public static FloatMatrix toOutcomeFloatVector(int index,int numOutcomes) {
        int[] nums = new int[numOutcomes];
        nums[index] = 1;
        return toMatrixFloat(nums);
    }


    public static DoubleMatrix toMatrix(int[][] arr) {
        DoubleMatrix d = new DoubleMatrix(arr.length,arr[0].length);
        for(int i = 0; i < arr.length; i++)
            for(int j = 0; j < arr[i].length; j++)
                d.put(i,j,arr[i][j]);
        return d;
    }

    public static FloatMatrix toMatrixFloat(int[][] arr) {
        FloatMatrix d = new FloatMatrix(arr.length,arr[0].length);
        for(int i = 0; i < arr.length; i++)
            for(int j = 0; j < arr[i].length; j++)
                d.put(i,j,arr[i][j]);
        return d;
    }

    public static DoubleMatrix toMatrix(int[] arr) {
        DoubleMatrix d = new DoubleMatrix(arr.length);
        for(int i = 0; i < arr.length; i++)
            d.put(i,arr[i]);
        d.reshape(1, d.length);
        return d;
    }
    public static FloatMatrix toMatrixFloat(int[] arr) {
        FloatMatrix d = new FloatMatrix(arr.length);
        for(int i = 0; i < arr.length; i++)
            d.put(i,arr[i]);
        d.reshape(1, d.length);
        return d;
    }

    public static FloatMatrix add(FloatMatrix a,FloatMatrix b) {
        return a.addi(b);
    }




    /**
     * Number of dimensions for a given matrix
     * @param input the input matrix
     * @return the number of dimensions for a matrix
     */
    public static int numDims(FloatMatrix input) {
        if(input instanceof FloatFourDTensor)
            return 4;

        if(input instanceof FloatTensor)
            return 3;
        else
            return 2;
    }

    /**
     * The number of dimensions for a given matrix
     * @param input the input matrix
     * @return the number of dimensions for a matrix
     */
    public static int numDims(DoubleMatrix input) {
        if(input instanceof FourDTensor)
            return 4;

        if(input instanceof Tensor)
            return 3;
        else
            return 2;
    }


    public static FloatMatrix mean(FloatMatrix input,int axis) {
        FloatMatrix ret = new FloatMatrix(input.rows,1);
        //column wise
        if(axis == 0) {
            return input.columnMeans();
        }
        //row wise
        else if(axis == 1) {
            return ret.rowMeans();
        }


        return ret;
    }


    public static FloatMatrix sum(FloatMatrix input,int axis) {
        FloatMatrix ret = new FloatMatrix(input.rows,1);
        //column wise
        if(axis == 0) {
            for(int i = 0; i < input.columns; i++) {
                ret.put(i,input.getColumn(i).sum());
            }
            return ret;
        }
        //row wise
        else if(axis == 1) {
            for(int i = 0; i < input.rows; i++) {
                ret.put(i,input.getRow(i).sum());
            }
            return ret;
        }

        for(int i = 0; i < input.rows; i++)
            ret.put(i,input.getRow(i).sum());
        return ret;
    }


    public static <E extends DoubleMatrix> E appendBias(E...vectors) {
        int size = 0;
        for (E vector : vectors) {
            size += vector.rows;
        }
        // one extra for the bias
        size++;

        DoubleMatrix result = new DoubleMatrix(size, 1);
        int index = 0;
        for (E vector : vectors) {
            result.put(RangeUtils.interval(index,index + vector.rows),RangeUtils.all(),vector);
            index += vector.rows;
        }

        result.put(RangeUtils.interval(index,result.rows),RangeUtils.all(),DoubleMatrix.ones(1,result.columns));
        return createBasedOn(result,vectors[0]);
    }



    public static <E extends FloatMatrix> E appendBias(E...vectors) {
        int size = 0;
        for (E vector : vectors) {
            size += vector.rows;
        }
        // one extra for the bias
        size++;

        FloatMatrix result = new FloatMatrix(size, vectors[0].columns);
        int index = 0;
        for (E vector : vectors) {
            result.put(RangeUtils.interval(index,index + vector.rows),RangeUtils.all(),vector);
            index += vector.rows;
        }

        result.put(RangeUtils.interval(index,result.rows),RangeUtils.all(),FloatMatrix.ones(1,result.columns));
        return createBasedOn(result,vectors[0]);
    }

    /**
     * Generate a binomial distribution based on the given rng,
     * a matrix of p values, and a max number.
     * @param p the p matrix to use
     * @param n the n to use
     * @param rng the rng to use
     * @return a binomial distribution based on the one n, the passed in p values, and rng
     */
    public static <E extends FloatMatrix> E binomial(E p,int n,RandomGenerator rng) {
        FloatMatrix ret = new FloatMatrix(p.rows,p.columns);
        for(int i = 0; i < ret.length; i++) {
            ret.put(i,MathUtils.binomial(rng, n, p.get(i)));
        }
        return createBasedOn(ret,p);
    }



    /**
     * Generates a random matrix between min and max
     * @param rows the number of rows of the matrix
     * @param columns the number of columns in the matrix
     * @param min the minimum number
     * @param max the maximum number
     * @param rng the rng to use
     * @return a drandom matrix of the specified shape and range
     */
    public static DoubleMatrix randDouble(int rows, int columns,float min,float max,RandomGenerator rng) {
        DoubleMatrix ret = new DoubleMatrix(rows,columns);
        float r = max - min;
        for(int i = 0; i < ret.length; i++) {
            ret.put(i, r * rng.nextFloat() + min);
        }
        return ret;
    }



    /**
     * Generates a random matrix between min and max
     * @param rows the number of rows of the matrix
     * @param columns the number of columns in the matrix
     * @param min the minimum number
     * @param max the maximum number
     * @param rng the rng to use
     * @return a drandom matrix of the specified shape and range
     */
    public static FloatMatrix rand(int rows, int columns,float min,float max,RandomGenerator rng) {
        FloatMatrix ret = new FloatMatrix(rows,columns);
        float r = max - min;
        for(int i = 0; i < ret.length; i++) {
            ret.put(i, r * rng.nextFloat() + min);
        }
        return ret;
    }




    public static FloatMatrix columnWiseMean(FloatMatrix x,int axis) {
        FloatMatrix ret = FloatMatrix.zeros(x.columns);
        for(int i = 0; i < x.columns; i++) {
            ret.put(i,x.getColumn(axis).mean());
        }
        return ret;
    }


    public static FloatMatrix toFlattenedFloat(Collection<FloatMatrix> matrices) {
        int length = 0;
        for(FloatMatrix m : matrices)  length += m.length;
        FloatMatrix ret = new FloatMatrix(1,length);
        int linearIndex = 0;
        for(FloatMatrix d : matrices) {
            for(int i = 0; i < d.length; i++) {
                ret.put(linearIndex++,d.get(i));
            }
        }

        return ret;

    }


    public static FloatMatrix toFlattenedFloat(int length,Iterator<? extends FloatMatrix>...matrices) {
        List<float[]> gradient = new ArrayList<>();
        for(Iterator<? extends FloatMatrix> iter : matrices) {
            while(iter.hasNext()) {
                FloatMatrix d = iter.next();
                gradient.add(d.data);
            }
        }




        FloatMatrix ret2 = new FloatMatrix(ArrayUtil.combineFloat(gradient));
        return ret2.reshape(1,ret2.length);
    }


    public static FloatMatrix toFlattened(FloatMatrix...matrices) {
        int length = 0;
        for(FloatMatrix m : matrices)  length += m.length;
        FloatMatrix ret = new FloatMatrix(1,length);
        int linearIndex = 0;
        for(FloatMatrix d : matrices) {
            for(int i = 0; i < d.length; i++) {
                ret.put(linearIndex++,d.get(i));
            }
        }

        return ret;
    }

    public static FloatMatrix avg(FloatMatrix...matrices) {
        if(matrices == null)
            return null;
        if(matrices.length == 1)
            return matrices[0];
        else {
            FloatMatrix ret = matrices[0];
            for(int i = 1; i < matrices.length; i++)
                ret = ret.add(matrices[i]);

            ret = ret.div(matrices.length);
            return ret;
        }
    }


    public static int maxIndex(FloatMatrix matrix) {
        double max = matrix.max();
        for(int j = 0; j < matrix.length; j++) {
            if(matrix.get(j) == max)
                return j;
        }
        return -1;
    }


    /**
     * Takes the sigmoid of a given matrix
     * @param x the input
     * @param <E>
     * @return the input with the sigmoid function applied
     */
    public static <E extends FloatMatrix> E sqrt(E x) {
        FloatMatrix ret = new FloatMatrix(x.rows,x.columns);
        for(int i = 0; i < ret.length; i++)
            ret.put(i, x.get(i) > 0 ? (float) FastMath.sqrt(x.get(i)) : 0.0f);

        return createBasedOn(ret,x);
    }


    /**
     * Takes the sigmoid of a given matrix
     * @param x the input
     * @param <E>
     * @return the input with the sigmoid function applied
     */
    public static <E extends FloatMatrix> E sigmoid(E x) {
        FloatMatrix ones = FloatMatrix.ones(x.rows, x.columns);
        return createBasedOn(ones.div(ones.add(exp(x.neg()))),x);
    }

    public static FloatMatrix dot(FloatMatrix a,FloatMatrix b) {
        boolean isScalar = a.isColumnVector() || a.isRowVector() && b.isColumnVector() || b.isRowVector();
        if(isScalar) {
            return FloatMatrix.scalar(a.dot(b));
        }
        else {
            return  a.mmul(b);
        }
    }

    /**
     * Ensures numerical stability.
     * Clips values of input such that
     * exp(k * in) is within single numerical precision
     * @param input the input to trim
     * @param k the k (usually 1)
     * @param <E>
     * @return the stabilized input
     */
    public static <E extends FloatMatrix> E stabilizeInput(E input,float k) {
        float realMin =  (float) 1.1755e-38;
        float cutOff = (float) FastMath.log(realMin);
        for(int i = 0; i < input.length; i++) {
            if(input.get(i) * k > -cutOff)
                input.put(i, (float) -cutOff / k);
            else if(input.get(i) * k < cutOff)
                input.put(i,(float) cutOff / k);
        }

        return input;

    }


    public static FloatMatrix out(FloatMatrix a,FloatMatrix b) {
        return a.mmul(b);
    }


    public static FloatMatrix scalarMinus(float scalar,FloatMatrix ep) {
        FloatMatrix d = new FloatMatrix(ep.rows,ep.columns);
        d.addi(scalar);
        return d.sub(ep);
    }

    public static <E extends FloatMatrix> E oneMinus(E ep) {
        return createBasedOn(FloatMatrix.ones(ep.rows, ep.columns).sub(ep),ep);
    }

    public static <E extends FloatMatrix> E oneDiv(E ep) {
        for(int i = 0; i < ep.rows; i++) {
            for(int j = 0; j < ep.columns; j++) {
                if(ep.get(i,j) == 0) {
                    ep.put(i,j,0.01f);
                }
            }
        }
        return createBasedOn(FloatMatrix.ones(ep.rows, ep.columns).div(ep),ep);
    }


    /**
     * Normalizes the passed in matrix by subtracting the mean
     * and dividing by the standard deviation
     * @param toNormalize the matrix to normalize
     */
    public static <E extends FloatMatrix> void normalizeZeroMeanAndUnitVariance(E toNormalize) {
        FloatMatrix columnMeans = toNormalize.columnMeans();
        FloatMatrix columnStds = columnStdDeviation(toNormalize);

        toNormalize.subiRowVector(columnMeans);
        columnStds.addi(1e-6f);
        toNormalize.diviRowVector(columnStds);

    }


    /**
     * Column wise variance
     * @param input the input to get the variance for
     * @return the column wise variance of the input
     */
    public static FloatMatrix columnVariance(FloatMatrix input) {
        FloatMatrix columnMeans = input.columnMeans();
        FloatMatrix ret = new FloatMatrix(1,columnMeans.columns);
        for(int i = 0;i < ret.columns; i++) {
            FloatMatrix column = input.getColumn(i);
            float variance = (float) StatUtils.variance(fromFloatArr(column.toArray()),(double) columnMeans.get(i));
            if(variance == 0)
                variance = 1e-6f;
            ret.put(i,variance);
        }
        return ret;
    }

    /**
     * Calculates the column wise standard deviations
     * of the matrix
     * @param m the matrix to use
     * @return the standard deviations of each column in the matrix
     * as a row matrix
     */
    public static FloatMatrix columnStd(FloatMatrix m) {
        FloatMatrix ret = new FloatMatrix(1,m.columns);
        StandardDeviation std = new StandardDeviation();

        for(int i = 0; i < m.columns; i++) {
            float result = (float) std.evaluate(fromFloatArr(m.getColumn(i).data));
            ret.put(i,result);
        }

        ret.divi(m.rows);

        return ret;
    }


    private static float[] fromDoubleArr(double[] d) {
        float[] ret = new float[d.length];
        for(int i = 0; i < d.length; i++)
            ret[i] = (float) d[i];
        return ret;
    }


    private static double[] fromFloatArr(float[] d) {
        double[] ret = new double[d.length];
        for(int i = 0; i < d.length; i++)
            ret[i] =  d[i];
        return ret;
    }

    /**
     * Calculates the column wise standard deviations
     * of the matrix
     * @param m the matrix to use
     * @return the standard deviations of each column in the matrix
     * as a row matrix
     */
    public static <E extends FloatMatrix> E rowStd(E m) {
        StandardDeviation std = new StandardDeviation();

        FloatMatrix ret = new FloatMatrix(1,m.columns);
        for(int i = 0; i < m.rows; i++) {
            float result = (float) std.evaluate(fromFloatArr(m.getRow(i).data));
            ret.put(i,result);
        }
        return createBasedOn(ret,m);
    }


    /**
     * A log impl that prevents numerical underflow
     * Any number that's infinity or NaN is replaced by
     * 1e-6.
     * @param vals the vals to convert to log
     * @return the log of the numbers or 1e-6 for anomalies
     */
    public static <E extends FloatMatrix> E log(E vals) {
        FloatMatrix ret = new FloatMatrix(vals.rows,vals.columns);
        for(int i = 0; i < vals.length; i++) {
            float logVal = (float) Math.log(vals.get(i));
            if(!Double.isNaN(logVal) && !Double.isInfinite(logVal))
                ret.put(i,logVal);
            else
                ret.put(i,1e-6f);
        }
        return createBasedOn(ret,vals);
    }


    public static <E extends FloatMatrix> void normalizeMatrix(E toNormalize) {
        FloatMatrix columnMeans = toNormalize.columnMeans();
        toNormalize.subiRowVector(columnMeans);
        FloatMatrix std = columnStd(toNormalize);
        std.addi(1e-6f);
        toNormalize.diviRowVector(std);
    }


    public static <E extends FloatMatrix> E normalizeByColumnSums(E m) {
        FloatMatrix columnSums = m.columnSums();
        for(int i = 0; i < m.columns; i++) {
            m.putColumn(i,m.getColumn(i).div(columnSums.get(i)));

        }
        return m;
    }


    public static FloatMatrix columnStdDeviation(FloatMatrix m) {
        FloatMatrix ret = new FloatMatrix(1,m.columns);

        for(int i = 0; i < ret.length; i++) {
            StandardDeviation dev = new StandardDeviation();
            double[] vals = fromFloatArr(m.getColumn(i).toArray());
            float std = (float) dev.evaluate(vals);
            ret.put(i,std);
        }

        return ret;
    }

    /**
     * Divides the given matrix's columns
     * by each column's respective standard deviations
     * @param m the matrix to divide
     * @return the column divided by the standard deviation
     */
    public static <E extends FloatMatrix> E divColumnsByStDeviation(E m) {
        FloatMatrix std = columnStdDeviation(m);
        for(int i = 0; i < m.columns; i++) {
            m.putColumn(i,m.getColumn(i).div(std.get(i)));
        }
        return m;

    }

    /**
     * Subtracts by column mean.
     * This ensures a mean of zero.
     * This is part of normalizing inputs
     * for a neural net
     * @param m the matrix to normalize
     * @return the normalized matrix which each
     * column subtracted by its mean
     */
    public static <E extends FloatMatrix> E normalizeByColumnMeans(E m) {
        FloatMatrix columnMeans = m.columnMeans();
        for(int i = 0; i < m.columns; i++) {
            m.putColumn(i,m.getColumn(i).sub(columnMeans.get(i)));

        }
        return m;
    }

    public static <E extends FloatMatrix> E  normalizeByRowSums(E m) {
        FloatMatrix rowSums = m.rowSums();
        for(int i = 0; i < m.rows; i++) {
            m.putRow(i,m.getRow(i).div(rowSums.get(i)));
        }
        return m;
    }



}
