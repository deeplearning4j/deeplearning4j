package org.deeplearning4j.util;


import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.util.FastMath;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

import static org.deeplearning4j.util.MatrixUtil.exp;



/**
 * Convolution is the code for applying the convolution operator.
 *  http://www.inf.ufpr.br/danielw/pos/ci724/20102/HIPR2/flatjavasrc/Convolution.java
 * @author: Simon Horne
 */
public class Convolution {

    private static Logger log = LoggerFactory.getLogger(Convolution.class);

    /**
     *
     *
     * Default no-arg constructor.
     */
    private Convolution() {
    }

    public static enum Type {
        FULL,VALID,SAME
    }


    public static DoubleMatrix conv2d(DoubleMatrix input,DoubleMatrix kernel,Type type) {

        DoubleMatrix xShape = new DoubleMatrix(1,2);
        xShape.put(0,input.rows);
        xShape.put(1,input.columns);


        DoubleMatrix yShape = new DoubleMatrix(1,2);
        yShape.put(0,kernel.rows);
        yShape.put(1,kernel.columns);


        DoubleMatrix zShape = xShape.add(yShape).sub(1);
        int retRows = (int) zShape.get(0);
        int retCols = (int) zShape.get(1);

        log.info("Conv 2d with dims " + retRows + " x " + retCols);
        ComplexDoubleMatrix fftInput = complexDisceteFourierTransform(input, retRows, retCols);
        ComplexDoubleMatrix fftKernel = complexDisceteFourierTransform(kernel, retRows, retCols);
        ComplexDoubleMatrix mul = fftKernel.mul(fftInput);
        ComplexDoubleMatrix retComplex = complexInverseDisceteFourierTransform(mul);
        DoubleMatrix ret = retComplex.getReal();
        if(type == Type.VALID) {

            DoubleMatrix validShape = xShape.subi(yShape).add(1);

            DoubleMatrix start = zShape.sub(validShape).div(2);
            DoubleMatrix end = start.add(validShape);

            ret = ret.get(RangeUtils.interval((int) start.get(0),(int) end.get(0)),RangeUtils.interval((int) start.get(1),(int) end.get(1)));




        }

        return ret;
    }



    public static ComplexDoubleMatrix complexInverseDisceteFourierTransform(ComplexDoubleMatrix input,int rows,int cols) {
        ComplexDoubleMatrix base = null;

        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.padWithZeros(input, rows, cols);
            //truncation
        else if(input.rows > rows || input.columns > cols) {
            base = input.dup();
            base = base.get(MatrixUtil.toIndices(RangeUtils.interval(0,rows)),MatrixUtil.toIndices(RangeUtils.interval(0,cols)));
        }

        else
            base = input.dup();

        ComplexDoubleMatrix temp = new ComplexDoubleMatrix(base.rows,base.columns);
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexDoubleMatrix column = base.getColumn(i);
            temp.putColumn(i,complexInverseDisceteFourierTransform1d(column));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexDoubleMatrix row = temp.getRow(i);
            ret.putRow(i,complexInverseDisceteFourierTransform1d(row));
        }

        return ret;

    }





    /**
     * Performs an inverse discrete fourier transform with the solution
     * being the number of rows and number of columns.
     * See matlab's iftt2 for more examples
     * @param input the input to transform
     * @param rows the number of rows for the transform
     * @param cols the number of columns for the transform
     * @return the 2d inverse discrete fourier transform
     */
    public static ComplexDoubleMatrix complexInverseDisceteFourierTransform(DoubleMatrix input,int rows,int cols) {
        ComplexDoubleMatrix base = null;
        StopWatch watch = new StopWatch();
        watch.start();
        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.complexPadWithZeros(input, rows, cols);
            //truncation
        else if(input.rows > rows || input.columns > cols) {
            base = new ComplexDoubleMatrix(input);
            base = base.get(MatrixUtil.toIndices(RangeUtils.interval(0,rows)),MatrixUtil.toIndices(RangeUtils.interval(0,cols)));
        }

        else
            base = new ComplexDoubleMatrix(input);

        ComplexDoubleMatrix temp = new ComplexDoubleMatrix(base.rows,base.columns);
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexDoubleMatrix column = base.getColumn(i);
            temp.putColumn(i,complexInverseDisceteFourierTransform1d(column));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexDoubleMatrix row = temp.getRow(i);
            ret.putRow(i,complexInverseDisceteFourierTransform1d(row));
        }
        watch.stop();
        log.info("Took " + watch.getTime());
        return ret;
    }

    /**
     * 1d inverse discrete fourier transform
     * see matlab's fft2 for more examples.
     * Note that this will throw an exception if the input isn't a vector
     * @param input the input to transform
     * @return the inverse fourier transform of the passed in input
     */
    public static ComplexDoubleMatrix complexInverseDisceteFourierTransform1d(DoubleMatrix input) {
        return complexInverseDisceteFourierTransform1d(new ComplexDoubleMatrix((input)));
    }

    /**
     * 1d inverse discrete fourier transform
     * see matlab's fft2 for more examples.
     * Note that this will throw an exception if the input isn't a vector
     * @param inputC the input to transform
     * @return the inverse fourier transform of the passed in input
     */
    public static ComplexDoubleMatrix complexInverseDisceteFourierTransform1d(ComplexDoubleMatrix inputC) {
        if(inputC.rows != 1 && inputC.columns != 1)
            throw new IllegalArgumentException("Illegal input: Must be a vector");
        double len = MatrixUtil.length(inputC);
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexDoubleMatrix range = MatrixUtil.complexRangeVector(0, (int) len);
        ComplexDoubleMatrix div2 = range.transpose().mul(c2);
        ComplexDoubleMatrix div3 = range.mmul(div2).negi();
        ComplexDoubleMatrix matrix = exp(div3).div(len);
        ComplexDoubleMatrix complexRet = inputC.isRowVector() ? matrix.mmul(inputC) : inputC.mmul(matrix);

        return complexRet;
    }


    /**
     * 1d discrete fourier transform, note that this will throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexDoubleMatrix complexDiscreteFourierTransform1d(ComplexDoubleMatrix inputC) {
        if(inputC.rows != 1 && inputC.columns != 1)
            throw new IllegalArgumentException("Illegal input: Must be a vector");

        double len = Math.max(inputC.rows,inputC.columns);
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexDoubleMatrix range = MatrixUtil.complexRangeVector(0, len);
        ComplexDoubleMatrix matrix = exp(range.mmul(range.transpose().mul(c2)));
        ComplexDoubleMatrix complexRet = inputC.isRowVector() ? matrix.mmul(inputC) : inputC.mmul(matrix);
        return complexRet;
    }




    /**
     * 1d discrete fourier transform, note that this will throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param input the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexDoubleMatrix complexDiscreteFourierTransform1d(DoubleMatrix input) {
        return complexDiscreteFourierTransform1d(new ComplexDoubleMatrix(input));
    }

    /**
     * Discrete fourier transform 2d
     * @param input the input to transform
     * @param rows the number of rows in the transformed output matrix
     * @param cols the number of columns in the transformed output matrix
     * @return the discrete fourier transform of the input
     */
    public static ComplexDoubleMatrix complexDisceteFourierTransform(DoubleMatrix input,int rows,int cols) {
        ComplexDoubleMatrix base = null;

        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.complexPadWithZeros(input, rows, cols);
            //truncation
        else if(input.rows > rows || input.columns > cols) {
            base = new ComplexDoubleMatrix(input);
            base = base.get(MatrixUtil.toIndices(RangeUtils.interval(0,rows)),MatrixUtil.toIndices(RangeUtils.interval(0,cols)));
        }
        else
            base = new ComplexDoubleMatrix(input);

        ComplexDoubleMatrix temp = new ComplexDoubleMatrix(base.rows,base.columns);
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexDoubleMatrix column = base.getColumn(i);
            temp.putColumn(i,complexDiscreteFourierTransform1d(column));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexDoubleMatrix row = temp.getRow(i);
            ret.putRow(i,complexDiscreteFourierTransform1d(row));
        }
        return ret;

    }

    /**
     * Returns the real component of an inverse 2d fourier transform
     * @param input the input to transform
     * @param rows the number of rows of the solution
     * @param cols the number of columns of the solution
     * @return the real component of an inverse discrete fourier transform
     */
    public static DoubleMatrix inverseDisceteFourierTransform(DoubleMatrix input,int rows,int cols) {
        return complexInverseDisceteFourierTransform(input,rows,cols).getReal();

    }


    /**
     * Returns the real component of a 2d fourier transform
     * @param input the input to transform
     * @param rows the number of rows of the solution
     * @param cols the number of columns of the solution
     * @return the real component of a discrete fourier transform
     */
    public static DoubleMatrix disceteFourierTransform(DoubleMatrix input,int rows,int cols) {
        return complexDisceteFourierTransform(input,rows,cols).getReal();

    }


    /**
     * The inverse discrete fourier transform with the dimension size
     * being the same as the passed in input.
     *
     * @param inputC the input to transform
     * @return the
     */
    public static ComplexDoubleMatrix complexInverseDisceteFourierTransform(ComplexDoubleMatrix inputC) {
        return complexInverseDisceteFourierTransform(inputC,inputC.rows,inputC.columns);

    }



    public static ComplexDoubleMatrix complexInverseDisceteFourierTransform(DoubleMatrix input) {
        return complexInverseDisceteFourierTransform(input,input.rows,input.columns);
    }



    public static ComplexDoubleMatrix complexDisceteFourierTransform(DoubleMatrix input) {
        return complexDisceteFourierTransform(input,input.rows,input.columns);

    }


    public static DoubleMatrix inverseDisceteFourierTransform(DoubleMatrix input) {
        return complexInverseDisceteFourierTransform(input).getReal();

    }



    public static DoubleMatrix disceteFourierTransform(DoubleMatrix input) {
        return complexDisceteFourierTransform(input).getReal();

    }


}
