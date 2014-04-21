package org.deeplearning4j.util;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.nn.Tensor;
import org.jblas.*;
import org.jblas.ranges.Range;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class MatrixUtil {
    private static Logger log = LoggerFactory.getLogger(MatrixUtil.class);



    public static void complainAboutMissMatchedMatrices(DoubleMatrix d1,DoubleMatrix d2) {
        if(d1 == null || d2 == null)
            throw new IllegalArgumentException("No null matrices allowed");
        if(d1.rows != d2.rows)
            throw new IllegalArgumentException("Matrices must have same rows");

    }


    /**
     * Cuts all numbers below a certain cut off
     * @param minNumber the min number to check
     * @param matrix the matrix to max by
     */
    public static void max(double minNumber,DoubleMatrix matrix) {
        for(int i = 0; i < matrix.length; i++)
            matrix.put(i,Math.max(0,matrix.get(i)));

    }

    /**
     * Divides each row by its max
     * @param toScale the matrix to divide by its row maxes
     */
    public static void scaleByMax(DoubleMatrix toScale) {
        DoubleMatrix scale = toScale.rowMaxs();
        for(int i = 0; i < toScale.rows; i++) {
            double scaleBy = scale.get(i,0);
            toScale.putRow(i,toScale.getRow(i).divi(scaleBy));
        }
    }



    /** Generate a new matrix which has the given number of replications of this. */
    public static ComplexDoubleMatrix repmat(ComplexDoubleMatrix matrix,int rowMult, int columnMult) {
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



    public static DoubleMatrix rangeVector(double begin, double end) {
        int diff = (int) Math.abs(end - begin);
        DoubleMatrix ret = new DoubleMatrix(1,diff);
        for(int i = 0; i < ret.length; i++)
            ret.put(i,i);
        return ret;
    }

    public static ComplexDoubleMatrix complexRangeVector(double begin, double end) {
        int diff = (int) Math.abs(end - begin);
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(1,diff);
        for(int i = 0; i < ret.length; i++)
            ret.put(i,i);
        return ret.transpose();
    }


    public static double angle(ComplexDouble phase) {
        Complex c = new Complex(phase.real(),phase.imag());
        return c.atan().getReal();
    }

    /**
     * Implements matlab's compare of complex numbers:
     * compares the max(abs(d1),abs(d2))
     * if they are equal, compares their angles
     * @param d1 the first number
     * @param d2 the second number
     * @return standard comparator interface
     */
    private static int compare(ComplexDouble d1,ComplexDouble d2) {
        if(d1.abs() > d2.abs())
            return 1;
        else if(d2.abs() > d1.abs())
            return -1;
        else {
            if(angle(d1) > angle(d2))
                return 1;
            else if(angle(d1) < angle(d2))
                return -1;
            return 0;
        }

    }


    public static ComplexDouble max(ComplexDoubleMatrix matrix) {
        ComplexDouble max = matrix.get(0);
        for(int i = 1; i < matrix.length; i++)
            if(compare(max,matrix.get(i)) > 0)
                max = matrix.get(i);
        return max;
    }



    /**
     * Divides each row by its max
     * @param toScale the matrix to divide by its row maxes
     */
    public static void scaleByMax(ComplexDoubleMatrix toScale) {

        for(int i = 0; i < toScale.rows; i++) {
            ComplexDouble scaleBy = max(toScale.getRow(i));
            toScale.putRow(i,toScale.getRow(i).divi(scaleBy));
        }
    }

    public static DoubleMatrix variance(DoubleMatrix input) {
        DoubleMatrix means = input.columnMeans();
        DoubleMatrix diff = MatrixFunctions.pow(input.subRowVector(means),2);
        //avg of the squared differences from the mean
        DoubleMatrix variance = diff.columnMeans().div(input.rows);
        return variance;

    }


    /**
     * Takes an image (grey-levels) and a kernel and a position,
     * applies the convolution at that position and returns the
     * new pixel value.
     *
     * @param input The 2D double array representing the image.
     * @param x The x coordinate for the position of the convolution.
     * @param y The y coordinate for the position of the convolution.
     * @param k The 2D array representing the kernel.
     * @param kernelWidth The width of the kernel.
     * @param kernelHeight The height of the kernel.
     * @return The new pixel value after the convolution.
     */
    public static double singlePixelConvolution(DoubleMatrix input,
                                                int x, int y,
                                                DoubleMatrix k,
                                                int kernelWidth,
                                                int kernelHeight) {
        double output = 0;

        for (int i = 0; i < kernelWidth; ++i) {
            for (int j = 0; j < kernelHeight; ++j) {
                output += (input.get(x + i,y + j) * k.get(i,j));
            }
        }
        return output;
    }


    public static DoubleMatrix reverse(DoubleMatrix toReverse) {
        DoubleMatrix ret = new DoubleMatrix(toReverse.rows,toReverse.columns);
        int reverseIndex = 0;
        for(int i = toReverse.length - 1; i >= 0; i--) {
            ret.put(reverseIndex++,toReverse.get(i));
        }
        return ret;
    }


    /**
     * Takes a 2D array of grey-levels and a kernel and applies the convolution
     * over the area of the image specified by width and height.
     *
     * @param input the 2D double array representing the image
     * @param width the width of the image
     * @param height the height of the image
     * @param kernel the 2D array representing the kernel
     * @param kernelWidth the width of the kernel
     * @param kernelHeight the height of the kernel
     * @return the 2D array representing the new image
     */
    public static DoubleMatrix convolution2D(DoubleMatrix input,
                                             int width, int height,
                                             DoubleMatrix kernel,
                                             int kernelWidth,
                                             int kernelHeight) {
        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        DoubleMatrix output = DoubleMatrix.zeros(smallWidth,smallHeight);

        for (int i = 0; i < smallWidth; ++i) {
            for (int j = 0; j < smallHeight; ++j) {
                output.put(i,j,singlePixelConvolution(input, i, j, kernel,
                        kernelWidth, kernelHeight));
            }
        }
        return output;
    }


    /**
     * Returns the maximum dimension of the passed in matrix
     * @param d the max dimension of the passed in matrix
     * @return the max dimension of the passed in matrix
     */
    public static double length(ComplexDoubleMatrix d) {
        return Math.max(d.rows,d.columns);


    }

    /**
     * Returns the maximum dimension of the passed in matrix
     * @param d the max dimension of the passed in matrix
     * @return the max dimension of the passed in matrix
     */
    public static double length(DoubleMatrix d) {
        if(d instanceof Tensor) {
            Tensor t = (Tensor) d;
            return MathUtils.max(new double[]{t.rows(),t.columns(),t.slices()});
        }
        else
            return Math.max(d.rows,d.columns);


    }

    /**
     * Takes a 2D array of grey-levels and a kernel and applies the convolution
     * over the area of the image specified by width and height.
     * The specified height is implied from the input and given kernel
     *
     * @param input the 2D double array representing the image
     @param kernel the 2D array representing the kernel
      * @return the 2D array representing the new image
     */
    public static DoubleMatrix convolution2D(DoubleMatrix input,
                                             DoubleMatrix kernel) {
        return convolution2D(input,input.rows,input.columns,kernel,kernel.rows,kernel.columns);
    }


    public static DataSet xorData(int n) {

        DoubleMatrix x = DoubleMatrix.rand(n,2);
        x = x.gti(0.5);

        DoubleMatrix y = DoubleMatrix.zeros(n,2);
        for(int i = 0; i < x.rows; i++) {
            if(x.get(i,0) == x.get(i,1))
                y.put(i,0,1);
            else
                y.put(i,1,1);
        }

        return new DataSet(x,y);

    }

    public static DataSet xorData(int n, int columns) {

        DoubleMatrix x = DoubleMatrix.rand(n,columns);
        x = x.gti(0.5);

        DoubleMatrix x2 = DoubleMatrix.rand(n,columns);
        x2 = x2.gti(0.5);

        DoubleMatrix eq = x.eq(x2).eq(DoubleMatrix.zeros(n,columns));


        int median = columns / 2;

        DoubleMatrix outcomes = new DoubleMatrix(n,2);
        for(int i = 0; i < outcomes.rows; i++) {
            DoubleMatrix left = eq.get(i,new org.jblas.ranges.IntervalRange(0,median));
            DoubleMatrix right = eq.get(i,new org.jblas.ranges.IntervalRange(median,columns));
            if(left.sum() > right.sum())
                outcomes.put(i,0,1);
            else
                outcomes.put(i,1,1);
        }


        return new DataSet(eq,outcomes);

    }

    public static double magnitude(DoubleMatrix vec) {
        double sum_mag = 0;
        for(int i = 0; i < vec.length;i++)
            sum_mag = sum_mag + vec.get(i) * vec.get(i);

        return Math.sqrt(sum_mag);
    }


    public static DoubleMatrix unroll(DoubleMatrix d) {
        DoubleMatrix ret = new DoubleMatrix(1,d.length);
        for(int i = 0; i < d.length; i++)
            ret.put(i,d.get(i));
        return ret;
    }


    public static DoubleMatrix outcomes(DoubleMatrix d) {
        DoubleMatrix ret = new DoubleMatrix(d.rows,1);
        for(int i = 0; i < d.rows; i++)
            ret.put(i,SimpleBlas.iamax(d.getRow(i)));
        return ret;
    }

    public static double cosineSim(DoubleMatrix d1,DoubleMatrix d2) {
        d1 = MatrixUtil.unitVec(d1);
        d2 = MatrixUtil.unitVec(d2);
        double ret = d1.dot(d2);
        return ret;
    }


    public static DoubleMatrix normalize(DoubleMatrix input) {
        double min = input.min();
        double max = input.max();
        return input.subi(min).divi(max - min);
    }


    public static double cosine(DoubleMatrix matrix) {
        //1.0 * math.sqrt(sum(val * val for val in vec1.itervalues()))
        return 1 * Math.sqrt(MatrixFunctions.pow(matrix, 2).sum());
    }


    public static DoubleMatrix unitVec(DoubleMatrix toScale) {
        double length = toScale.norm2();
        if(length > 0)
            return SimpleBlas.scal(1.0 / length, toScale);
        return toScale;
    }

    /**
     * A uniform sample ranging from 0 to 1.
     * @param rng the rng to use
     * @param rows the number of rows of the matrix
     * @param columns the number of columns of the matrix
     * @return a uniform sample of the given shape and size
     * with numbers between 0 and 1
     */
    public static DoubleMatrix uniform(RandomGenerator rng,int rows,int columns) {

        UniformRealDistribution uDist = new UniformRealDistribution(rng,0,1);
        DoubleMatrix U = new DoubleMatrix(rows,columns);
        for(int i = 0; i < U.rows; i++)
            for(int j = 0; j < U.columns; j++)
                U.put(i,j,uDist.sample());
        return U;
    }


    /**
     * A uniform sample ranging from 0 to sigma.
     * @param rng the rng to use
     * @param mean, the matrix mean from which to generate values from
     * @param sigma the standard deviation to use to generate the gaussian noise
     * @return a uniform sample of the given shape and size
     *
     * with numbers between 0 and 1
     */
    public static DoubleMatrix normal(RandomGenerator rng,DoubleMatrix mean,double sigma) {
        DoubleMatrix U = new DoubleMatrix(mean.rows,mean.columns);
        for(int i = 0; i < U.rows; i++)
            for(int j = 0; j < U.columns; j++)  {
                RealDistribution reals = new NormalDistribution(mean.get(i,j),Math.sqrt(sigma));
                U.put(i,j,reals.sample());

            }
        return U;
    }



    /**
     * A uniform sample ranging from 0 to sigma.
     * @param rng the rng to use
     * @param mean, the matrix mean from which to generate values from
     * @param variance the variance matrix where each column is the variance
     * for the respective columns of the matrix
     * @return a uniform sample of the given shape and size
     *
     * with numbers between 0 and 1
     */
    public static DoubleMatrix normal(RandomGenerator rng,DoubleMatrix mean,DoubleMatrix variance) {
        DoubleMatrix std = MatrixFunctions.sqrt(variance);
        for(int i = 0;i < variance.length; i++)
            if(variance.get(i) <= 0)
                variance.put(i,1e-4);

        DoubleMatrix U = new DoubleMatrix(mean.rows,mean.columns);
        for(int i = 0; i < U.rows; i++)
            for(int j = 0; j < U.columns; j++)  {
                RealDistribution reals = new NormalDistribution(mean.get(i,j),std.get(j));
                U.put(i,j,reals.sample());

            }
        return U;
    }


    /**
     * Sample from a normal distribution given a mean of zero and a matrix of standard deviations.
     * @param rng the rng to use
     * for the respective columns of the matrix
     * @return a uniform sample of the given shape and size
     */
    public static DoubleMatrix normal(RandomGenerator rng,DoubleMatrix standardDeviations) {

        DoubleMatrix U = new DoubleMatrix(standardDeviations.rows,standardDeviations.columns);
        for(int i = 0; i < U.rows; i++)
            for(int j = 0; j < U.columns; j++)  {
                RealDistribution reals = new NormalDistribution(0,standardDeviations.get(i,j));
                U.put(i,j,reals.sample());

            }
        return U;
    }

    public static boolean isValidOutcome(DoubleMatrix out) {
        boolean found = false;
        for(int col = 0; col < out.length; col++) {
            if(out.get(col) > 0) {
                found = true;
                break;
            }
        }
        return found;
    }


    public static double min(DoubleMatrix matrix) {
        double ret = matrix.get(0);
        for(int i = 0; i < matrix.length; i++) {
            if(matrix.get(i) < ret)
                ret = matrix.get(i);
        }
        return ret;
    }

    public static double max(DoubleMatrix matrix) {
        double ret = matrix.get(0);
        for(int i = 0; i < matrix.length; i++) {
            if(matrix.get(i) > ret)
                ret = matrix.get(i);
        }
        return ret;
    }

    public static void ensureValidOutcomeMatrix(DoubleMatrix out) {
        boolean found = false;
        for(int col = 0; col < out.length; col++) {
            if(out.get(col) > 0) {
                found = true;
                break;
            }
        }
        if(!found) {
            log.warn("Found invalid matrix assuming; nothing which means adding a 1 to the first spot");
            out.put(0,1.0);
        }

    }

    public static void assertIntMatrix(DoubleMatrix matrix) {
        for(int i = 0; i < matrix.length; i++) {
            int cast = (int) matrix.get(i);
            if(cast != matrix.get(i))
                throw new IllegalArgumentException("Found something that is not an integer at linear index " + i);
        }
    }


    public static int[] toIndices(Range range) {
        int[] ret = new int[range.length()];
        for(int i = 0;i  < ret.length; i++) {
            ret[i] = range.value();
            range.next();;
        }

        return ret;
    }


    public static ComplexDoubleMatrix exp(ComplexDoubleMatrix input) {
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(input.rows,input.columns);
        for(int i = 0; i < ret.length; i++) {
            ret.put(i,ComplexUtil.exp(input.get(i)));
        }
        return ret;
    }


    public static ComplexDoubleMatrix complexPadWithZeros(DoubleMatrix toPad, int rows, int cols) {
        ComplexDoubleMatrix ret = ComplexDoubleMatrix.zeros(rows,cols);
        for(int i = 0; i < toPad.rows; i++) {
            for(int j = 0; j < toPad.columns; j++) {
                ret.put(i,j,toPad.get(i,j));
            }
        }
        return ret;
    }

    public static ComplexDoubleMatrix padWithZeros(ComplexDoubleMatrix toPad, int rows, int cols) {
        ComplexDoubleMatrix ret = ComplexDoubleMatrix.zeros(rows,cols);
        for(int i = 0; i < toPad.rows; i++) {
            for(int j = 0; j < toPad.columns; j++) {
                ret.put(i,j,toPad.get(i,j));
            }
        }
        return ret;
    }


    public static void assign(DoubleMatrix toAssign,double val) {
        for(int i = 0; i < toAssign.length; i++)
            toAssign.put(i,val);
    }

    /**
     * Pads the matrix with zeros to surround the passed in matrix
     * with the given rows and columns
     * @param toPad the matrix to pad
     * @param rows the rows of the destination matrix
     * @param cols the columns of the destination matrix
     * @return a new matrix with the elements of toPad with zeros or
     * a clone of toPad if the rows and columns are both greater in length than
     * rows and cols
     */
    public static DoubleMatrix padWithZeros(DoubleMatrix toPad, int rows, int cols) {
        if(rows < 1)
            throw new IllegalArgumentException("Illegal number of rows " + rows);
        if(cols < 1)
            throw new IllegalArgumentException("Illegal number of columns " + cols);
        DoubleMatrix ret = null;
        //nothing to pad
        if(toPad.rows >= rows) {
            if(toPad.columns >= cols)
                return toPad.dup();
            else
                ret = new DoubleMatrix(toPad.rows, cols);


        }
        else if(toPad.columns >= cols) {
            if(toPad.rows >= rows)
                return toPad.dup();
            else
                ret = new DoubleMatrix(rows,toPad.columns);
        }
        else
            ret = new DoubleMatrix(rows, cols);

        for(int i = 0; i < toPad.rows; i++) {
            for(int j = 0; j < toPad.columns; j++) {
                double d = toPad.get(i,j);
                ret.put(i,j,d);
            }
        }
        return ret;
    }

    public static ComplexDoubleMatrix numDivideMatrix(ComplexDouble div,ComplexDoubleMatrix toDiv) {
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(toDiv.rows,toDiv.columns);

        for(int i = 0; i < ret.length; i++) {
            //prevent numerical underflow
            ComplexDouble curr = toDiv.get(i).addi(1e-6);
            ret.put(i,div.div(curr));
        }

        return ret;
    }


    public static DoubleMatrix numDivideMatrix(double div,DoubleMatrix toDiv) {
        DoubleMatrix ret = new DoubleMatrix(toDiv.rows,toDiv.columns);

        for(int i = 0; i < ret.length; i++)
            //prevent numerical underflow
            ret.put(i,div / toDiv.get(i) +1e-6 );
        return ret;
    }


    public static boolean isInfinite(DoubleMatrix test) {
        DoubleMatrix nan = test.isInfinite();
        for(int i = 0; i < nan.length; i++) {
            if(nan.get(i) > 0)
                return true;
        }
        return false;
    }

    public static boolean isNaN(DoubleMatrix test) {
        for(int i = 0; i < test.length; i++) {
            if(Double.isNaN(test.get(i)))
                return true;
        }
        return false;
    }




    public static void discretizeColumns(DoubleMatrix toDiscretize,int numBins) {
        DoubleMatrix columnMaxes = toDiscretize.columnMaxs();
        DoubleMatrix columnMins = toDiscretize.columnMins();
        for(int i = 0; i < toDiscretize.columns; i++) {
            double min = columnMins.get(i);
            double max = columnMaxes.get(i);
            DoubleMatrix col = toDiscretize.getColumn(i);
            DoubleMatrix newCol = new DoubleMatrix(col.length);
            for(int j = 0; j < col.length; j++) {
                int bin = MathUtils.discretize(col.get(j), min, max, numBins);
                newCol.put(j,bin);
            }
            toDiscretize.putColumn(i,newCol);

        }
    }

    /**
     * Rounds the matrix to the number of specified by decimal places
     * @param d the matrix to round
     * @param num the number of decimal places to round to(example: pass 2 for the 10s place)
     * @return the rounded matrix
     */
    public static DoubleMatrix roundToTheNearest(DoubleMatrix d,int num) {
        DoubleMatrix ret = d.mul(num);
        for(int i = 0; i < d.rows; i++)
            for(int j = 0; j < d.columns; j++) {
                double d2 = d.get(i,j);
                double newNum = MathUtils.roundDouble(d2, num);
                ret.put(i,j,newNum);
            }
        return ret;
    }


    public static void columnNormalizeBySum(DoubleMatrix x) {
        for(int i = 0; i < x.columns; i++)
            x.putColumn(i, x.getColumn(i).div(x.getColumn(i).sum()));
    }


    public static DoubleMatrix toOutcomeVector(int index,int numOutcomes) {
        int[] nums = new int[numOutcomes];
        nums[index] = 1;
        return toMatrix(nums);
    }

    public static DoubleMatrix toMatrix(int[][] arr) {
        DoubleMatrix d = new DoubleMatrix(arr.length,arr[0].length);
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
     * @return the softmax output (a probability matrix) scaling each row to between
     * 0 and 1
     */
    public static DoubleMatrix softmax(DoubleMatrix input) {
        DoubleMatrix max = input.rowMaxs();
        DoubleMatrix diff = MatrixFunctions.exp(input.subColumnVector(max));
        diff.diviColumnVector(diff.rowSums());
        return diff;
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
    public static DoubleMatrix binomial(DoubleMatrix p,int n,RandomGenerator rng) {
        DoubleMatrix ret = new DoubleMatrix(p.rows,p.columns);
        for(int i = 0; i < ret.length; i++) {
            ret.put(i,MathUtils.binomial(rng, n, p.get(i)));
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



    public static DoubleMatrix sigmoid(DoubleMatrix x) {
        DoubleMatrix ones = DoubleMatrix.ones(x.rows, x.columns);
        return ones.div(ones.add(MatrixFunctions.exp(x.neg())));
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

    public static DoubleMatrix out(DoubleMatrix a,DoubleMatrix b) {
        return a.mmul(b);
    }


    public static DoubleMatrix scalarMinus(double scalar,DoubleMatrix ep) {
        DoubleMatrix d = new DoubleMatrix(ep.rows,ep.columns);
        d.addi(scalar);
        return d.sub(ep);
    }

    public static DoubleMatrix oneMinus(DoubleMatrix ep) {
        return DoubleMatrix.ones(ep.rows, ep.columns).sub(ep);
    }

    public static DoubleMatrix oneDiv(DoubleMatrix ep) {
        for(int i = 0; i < ep.rows; i++) {
            for(int j = 0; j < ep.columns; j++) {
                if(ep.get(i,j) == 0) {
                    ep.put(i,j,0.01);
                }
            }
        }
        return DoubleMatrix.ones(ep.rows, ep.columns).div(ep);
    }


    /**
     * Normalizes the passed in matrix by subtracting the mean
     * and dividing by the standard deviation
     * @param toNormalize the matrix to normalize
     */
    public static void normalizeZeroMeanAndUnitVariance(DoubleMatrix toNormalize) {
        DoubleMatrix columnMeans = toNormalize.columnMeans();
        DoubleMatrix columnStds = MatrixUtil.columnStdDeviation(toNormalize);

        toNormalize.subiRowVector(columnMeans);
        columnStds.addi(1e-6);
        toNormalize.diviRowVector(columnStds);

    }

    public static DoubleMatrix columnVariance(DoubleMatrix input) {
        DoubleMatrix columnMeans = input.columnMeans();
        DoubleMatrix ret = new DoubleMatrix(1,columnMeans.columns);
        for(int i = 0;i < ret.columns; i++) {
            DoubleMatrix column = input.getColumn(i);
            double variance = StatUtils.variance(column.toArray(),columnMeans.get(i));
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
    public static DoubleMatrix rowStd(DoubleMatrix m) {
        StandardDeviation std = new StandardDeviation();

        DoubleMatrix ret = new DoubleMatrix(1,m.columns);
        for(int i = 0; i < m.rows; i++) {
            double result = std.evaluate(m.getRow(i).data);
            ret.put(i,result);
        }
        return ret;
    }

    /**
     * Returns the mean squared error of the 2 matrices.
     * Note that the matrices must be the same length
     * or an {@link IllegalArgumentException} is thrown
     * @param input the first one
     * @param other the second one
     * @return the mean square error of the matrices
     */
    public static double meanSquaredError(DoubleMatrix input,DoubleMatrix other) {
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
    public static DoubleMatrix log(DoubleMatrix vals) {
        DoubleMatrix ret = new DoubleMatrix(vals.rows,vals.columns);
        for(int i = 0; i < vals.length; i++) {
            double logVal = Math.log(vals.get(i));
            if(!Double.isNaN(logVal) && !Double.isInfinite(logVal))
                ret.put(i,logVal);
            else
                ret.put(i,1e-6);
        }
        return ret;
    }

    /**
     * Returns the sum squared error of the 2 matrices.
     * Note that the matrices must be the same length
     * or an {@link IllegalArgumentException} is thrown
     * @param input the first one
     * @param other the second one
     * @return the sum square error of the matrices
     */
    public static double sumSquaredError(DoubleMatrix input,DoubleMatrix other) {
        if(input.length != other.length)
            throw new IllegalArgumentException("Matrices must be same length");
        SimpleRegression r = new SimpleRegression();
        r.addData(new double[][]{input.data,other.data});
        return r.getSumSquaredErrors();
    }

    public static void normalizeMatrix(DoubleMatrix toNormalize) {
        DoubleMatrix columnMeans = toNormalize.columnMeans();
        toNormalize.subiRowVector(columnMeans);
        DoubleMatrix std = columnStd(toNormalize);
        std.addi(1e-6);
        toNormalize.diviRowVector(std);
    }


    public static DoubleMatrix normalizeByColumnSums(DoubleMatrix m) {
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
    public static DoubleMatrix divColumnsByStDeviation(DoubleMatrix m) {
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
    public static DoubleMatrix normalizeByColumnMeans(DoubleMatrix m) {
        DoubleMatrix columnMeans = m.columnMeans();
        for(int i = 0; i < m.columns; i++) {
            m.putColumn(i,m.getColumn(i).sub(columnMeans.get(i)));

        }
        return m;
    }

    public static DoubleMatrix normalizeByRowSums(DoubleMatrix m) {
        DoubleMatrix rowSums = m.rowSums();
        for(int i = 0; i < m.rows; i++) {
            m.putRow(i,m.getRow(i).div(rowSums.get(i)));
        }
        return m;
    }

}
