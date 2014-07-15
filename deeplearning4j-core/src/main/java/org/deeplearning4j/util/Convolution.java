package org.deeplearning4j.util;


import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.ComplexNDArray;
import org.deeplearning4j.nn.NDArray;
import org.jblas.*;
import org.jblas.ranges.Range;
import org.jblas.ranges.RangeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.deeplearning4j.util.MatrixUtil.exp;



/**
 * Convolution is the code for applying the convolution operator.
 *  http://www.inf.ufpr.br/danielw/pos/ci724/20102/HIPR2/flatjavasrc/Convolution.java
 *
 * @author Adam Gibson
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


    private static Range rangeFor(int m,int n,Type type) {
        switch (type) {
            case SAME:
                return RangeUtils.interval((int) Math.ceil(n / 2), m);
            case FULL:
                return RangeUtils.interval(0, m + n);
            case VALID:
                return RangeUtils.interval(m, n);
            default:
                throw new IllegalStateException("This should never happen");
        }

    }


    public static NDArray convn(NDArray input,NDArray kernel,Type type) {
        int dims = Math.max(input.shape().length,kernel.shape().length);
        List<Range> results = new ArrayList<>();
        ComplexNDArray inputComplex = new ComplexNDArray(input);
        ComplexNDArray kernelComplex = new ComplexNDArray(kernel);
        for(int i = 0; i < dims; i++) {
            int m = input.size(i);
            int n = kernel.size(i);
            int l = m + n - 1;
            if(i == 0) {
                inputComplex = ComplexNDArray.wrap(inputComplex, fft(input, l,i));
                kernelComplex = ComplexNDArray.wrap(kernelComplex, fft(kernel, l,i));

            }
            else {
                //size should change but doesn't on the next time around
                inputComplex = ComplexNDArray.wrap(inputComplex, fft(inputComplex, l,i));
                kernelComplex = ComplexNDArray.wrap(kernelComplex, fft(kernelComplex, l,i));
            }

            Range r = rangeFor(m,n,type);
            results.add(r);

        }


        inputComplex.muli(kernelComplex);

        for(int i = 0; i < input.shape().length; i++) {
            input = NDArray.wrap(input,ifft(inputComplex, inputComplex.rows, i).getReal());
        }




        //expand the ranges to get the data for each slice
        for(int i = 0; i < input.shape().length; i++) {
            NDArray slice = input.slice(i);

        }



        return input;
    }


    public static ComplexDoubleMatrix fft(DoubleMatrix transform) {
        return complexDisceteFourierTransform(transform,transform.rows,transform.columns);
    }

    public static ComplexDoubleMatrix ifft(DoubleMatrix transform) {
        return complexInverseDisceteFourierTransform(transform,transform.rows,transform.columns);
    }

    public static ComplexDoubleMatrix fft(NDArray transform,int numElements,int dimension) {
        NDArray r = transform.slice(dimension);
        return complexDisceteFourierTransform(r,r.shape());
    }

    public static ComplexDoubleMatrix fft(ComplexNDArray transform,int numElements,int dimension) {
        ComplexNDArray r = transform.slice(dimension);
        return complexDisceteFourierTransform(r,r.shape());
    }


    public static ComplexDoubleMatrix ifft(ComplexNDArray transform,int numElements,int dimension) {
        return complexInverseDisceteFourierTransform(transform.slice(dimension,0), numElements, transform.columns);
    }


    public static ComplexDoubleMatrix fft(NDArray transform,int numElements) {
        return complexDisceteFourierTransform(transform,numElements,transform.columns);
    }

    public static ComplexDoubleMatrix ifft(ComplexNDArray transform,int numElements) {
        return complexInverseDisceteFourierTransform(transform, numElements, transform.columns);
    }

    public static ComplexDoubleMatrix fft(DoubleMatrix transform,int numElements) {
        return complexDisceteFourierTransform(transform,numElements,transform.columns);
    }

    public static ComplexDoubleMatrix ifft(DoubleMatrix transform,int numElements) {
        return complexInverseDisceteFourierTransform(transform, numElements, transform.columns);
    }


    public static DoubleMatrix conv2d(DoubleMatrix input,DoubleMatrix kernel,Type type) {

        DoubleMatrix xShape = new DoubleMatrix(1,2);
        xShape.put(0,input.rows);
        xShape.put(1,input.columns);


        DoubleMatrix yShape = new DoubleMatrix(1,2);
        yShape.put(0,kernel.rows);
        yShape.put(1,kernel.columns);


        DoubleMatrix zShape = xShape.addi(yShape).subi(1);
        int retRows = (int) zShape.get(0);
        int retCols = (int) zShape.get(1);

        ComplexDoubleMatrix fftInput = complexDisceteFourierTransform(input, retRows, retCols);
        ComplexDoubleMatrix fftKernel = complexDisceteFourierTransform(kernel, retRows, retCols);
        ComplexDoubleMatrix mul = fftKernel.muli(fftInput);
        ComplexDoubleMatrix retComplex = complexInverseDisceteFourierTransform(mul);
        DoubleMatrix ret = retComplex.getReal();

        if(type == Type.VALID) {

            DoubleMatrix validShape = xShape.subi(yShape).addi(1);

            DoubleMatrix start = zShape.subi(validShape).divi(2);
            DoubleMatrix end = start.addi(validShape);
            if(start.get(0) < 1 || start.get(1) < 1)
                throw new IllegalStateException("Illegal row index " + start);
            if(end.get(0) < 1 || end.get(1) < 1)
                throw new IllegalStateException("Illegal column index " + end);

            ret = ret.get(RangeUtils.interval((int) start.get(0),(int) end.get(0)),RangeUtils.interval((int) start.get(1),(int) end.get(1)));




        }

        return ret;
    }






    public static ComplexDoubleMatrix complexInverseDisceteFourierTransform(ComplexDoubleMatrix input,int rows,int cols) {
        ComplexDoubleMatrix base;

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
        ComplexDoubleMatrix base;
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
     * @param input the input to transform
     * @return the inverse fourier transform of the passed in input
     */
    public static ComplexDoubleMatrix complexInverseDisceteFourierTransform1d(NDArray input) {
        return complexInverseDisceteFourierTransform1d(new ComplexNDArray((input)));
    }



    /**
     * 1d inverse discrete fourier transform
     * see matlab's fft2 for more examples.
     * Note that this will throw an exception if the input isn't a vector
     * @param inputC the input to transform
     * @return the inverse fourier transform of the passed in input
     */
    public static ComplexNDArray complexInverseDisceteFourierTransform1d(ComplexNDArray inputC) {
        if(inputC.shape().length != 1)
            throw new IllegalArgumentException("Illegal input: Must be a vector");
        double len = MatrixUtil.length(inputC);
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexDoubleMatrix range = MatrixUtil.complexRangeVector(0,len);
        ComplexDoubleMatrix div2 = range.transpose().mul(c2);
        ComplexDoubleMatrix div3 = range.mmul(div2).negi();
        ComplexDoubleMatrix matrix = exp(div3).div(len);
        ComplexDoubleMatrix complexRet = matrix.mmul(inputC);

        return ComplexNDArray.wrap(inputC,complexRet);
    }


    /**
     * 1d discrete fourier transform, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexNDArray complexDiscreteFourierTransform1d(ComplexNDArray inputC) {
        if(inputC.shape().length != 1)
            throw new IllegalArgumentException("Illegal input: Must be a vector");

        double len = Math.max(inputC.rows,inputC.columns);
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexDoubleMatrix range = MatrixUtil.complexRangeVector(0, len);
        ComplexDoubleMatrix matrix = exp(range.mmul(range.transpose().mul(c2)));
        ComplexDoubleMatrix complexRet = matrix.mmul(inputC);
        return ComplexNDArray.wrap(inputC,complexRet);
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
        ComplexDoubleMatrix range = MatrixUtil.complexRangeVector(0,len);
        ComplexDoubleMatrix div2 = range.transpose().mul(c2);
        ComplexDoubleMatrix div3 = range.mmul(div2).negi();
        ComplexDoubleMatrix matrix = exp(div3).div(len);
        ComplexDoubleMatrix complexRet = inputC.isRowVector() ? matrix.mmul(inputC) : inputC.mmul(matrix);

        return complexRet;
    }


    /**
     * 1d discrete fourier transform, note that this will
     * throw an exception if the passed in input
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
    public static ComplexDoubleMatrix complexDisceteFourierTransform(ComplexDoubleMatrix input,int rows,int cols) {
        ComplexDoubleMatrix base;

        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.complexPadWithZeros(input, rows, cols);
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
            temp.putColumn(i,complexDiscreteFourierTransform1d(column));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexDoubleMatrix row = temp.getRow(i);
            ret.putRow(i,complexDiscreteFourierTransform1d(row));
        }
        return ret;

    }


    /**
     * Discrete fourier transform 2d
     * @param input the input to transform
     * @param shape the shape of the output matrix
     * @return the discrete fourier transform of the input
     */
    public static ComplexDoubleMatrix complexDisceteFourierTransform(ComplexNDArray input,int[] shape) {
        ComplexNDArray base;

        //pad
        if(ArrayUtil.anyLess(input.shape(),shape))
            base = MatrixUtil.complexPadWithZeros(input,shape);
            //truncation
        else if(ArrayUtil.anyMore(input.shape(),shape)) {
            base = new ComplexNDArray(shape);
            for(int i = 0; i < ArrayUtil.prod(shape); i++)
                base.put(i,input.get(i));
        }
        else
            base = input;

        ComplexNDArray temp = new ComplexNDArray(shape);
        ComplexNDArray ret = new ComplexNDArray(shape);


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
     * Discrete fourier transform 2d
     * @param input the input to transform
     * @param shape the shape of the output matrix
     * @return the discrete fourier transform of the input
     */
    public static ComplexDoubleMatrix complexDisceteFourierTransform(NDArray input,int[] shape) {
        ComplexNDArray base;

        //pad
        if(ArrayUtil.anyLess(input.shape(),shape))
            base = MatrixUtil.complexPadWithZeros(input,shape);
            //truncation
        else if(ArrayUtil.anyMore(input.shape(),shape)) {
            base = new ComplexNDArray(shape);
            for(int i = 0; i < ArrayUtil.prod(shape); i++)
                base.put(i,input.get(i));
        }
        else
            base = new ComplexNDArray(input);

        ComplexNDArray temp = new ComplexNDArray(shape);
        ComplexNDArray ret = new ComplexNDArray(shape);


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
     * Discrete fourier transform 2d
     * @param input the input to transform
     * @param rows the number of rows in the transformed output matrix
     * @param cols the number of columns in the transformed output matrix
     * @return the discrete fourier transform of the input
     */
    public static ComplexDoubleMatrix complexDisceteFourierTransform(DoubleMatrix input,int rows,int cols) {
        ComplexDoubleMatrix base;

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

    public static FloatMatrix conv2d(FloatMatrix input,FloatMatrix kernel,Type type) {

        FloatMatrix xShape = new FloatMatrix(1,2);
        xShape.put(0,input.rows);
        xShape.put(1,input.columns);


        FloatMatrix yShape = new FloatMatrix(1,2);
        yShape.put(0,kernel.rows);
        yShape.put(1,kernel.columns);


        FloatMatrix zShape = xShape.add(yShape).sub(1);
        int retRows = (int) zShape.get(0);
        int retCols = (int) zShape.get(1);

        ComplexFloatMatrix fftInput = complexDisceteFourierTransform(input, retRows, retCols);
        ComplexFloatMatrix fftKernel = complexDisceteFourierTransform(kernel, retRows, retCols);
        ComplexFloatMatrix mul = fftKernel.mul(fftInput);
        ComplexFloatMatrix retComplex = complexInverseDisceteFourierTransform(mul);

        FloatMatrix ret = retComplex.getReal();

        if(type == Type.VALID) {

            FloatMatrix validShape = xShape.subi(yShape).add(1);

            FloatMatrix start = zShape.sub(validShape).div(2);
            FloatMatrix end = start.add(validShape);
            if(start.get(0) < 1 || start.get(1) < 1)
                throw new IllegalStateException("Illegal row index " + start);
            if(end.get(0) < 1 || end.get(1) < 1)
                throw new IllegalStateException("Illegal column index " + end);

            ret = ret.get(RangeUtils.interval((int) start.get(0),(int) end.get(0)),RangeUtils.interval((int) start.get(1),(int) end.get(1)));




        }

        return ret;
    }



    public static ComplexFloatMatrix complexInverseDisceteFourierTransform(ComplexFloatMatrix input,int rows,int cols) {
        ComplexFloatMatrix base;

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

        ComplexFloatMatrix temp = new ComplexFloatMatrix(base.rows,base.columns);
        ComplexFloatMatrix ret = new ComplexFloatMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexFloatMatrix column = base.getColumn(i);
            temp.putColumn(i,complexInverseDisceteFourierTransform1d(column));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexFloatMatrix row = temp.getRow(i);
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
    public static ComplexFloatMatrix complexInverseDisceteFourierTransform(FloatMatrix input,int rows,int cols) {
        ComplexFloatMatrix base = null;
        StopWatch watch = new StopWatch();
        watch.start();
        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.complexPadWithZeros(input, rows, cols);
            //truncation
        else if(input.rows > rows || input.columns > cols) {
            base = new ComplexFloatMatrix(input);
            base = base.get(MatrixUtil.toIndices(RangeUtils.interval(0,rows)),MatrixUtil.toIndices(RangeUtils.interval(0,cols)));
        }

        else
            base = new ComplexFloatMatrix(input);

        ComplexFloatMatrix temp = new ComplexFloatMatrix(base.rows,base.columns);
        ComplexFloatMatrix ret = new ComplexFloatMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexFloatMatrix column = base.getColumn(i);
            temp.putColumn(i,complexInverseDisceteFourierTransform1d(column));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexFloatMatrix row = temp.getRow(i);
            ret.putRow(i,complexInverseDisceteFourierTransform1d(row));
        }
        watch.stop();
        return ret;
    }

    /**
     * 1d inverse discrete fourier transform
     * see matlab's fft2 for more examples.
     * Note that this will throw an exception if the input isn't a vector
     * @param input the input to transform
     * @return the inverse fourier transform of the passed in input
     */
    public static ComplexFloatMatrix complexInverseDisceteFourierTransform1d(FloatMatrix input) {
        return complexInverseDisceteFourierTransform1d(new ComplexFloatMatrix((input)));
    }

    /**
     * 1d inverse discrete fourier transform
     * see matlab's fft2 for more examples.
     * Note that this will throw an exception if the input isn't a vector
     * @param inputC the input to transform
     * @return the inverse fourier transform of the passed in input
     */
    public static ComplexFloatMatrix complexInverseDisceteFourierTransform1d(ComplexFloatMatrix inputC) {
        if(inputC.rows != 1 && inputC.columns != 1)
            throw new IllegalArgumentException("Illegal input: Must be a vector");
        float len = MatrixUtil.length(inputC);
        ComplexFloat c2 = new ComplexFloat(0,-2).muli((float) FastMath.PI).divi(len);
        ComplexFloatMatrix range = MatrixUtil.complexRangeVector(0, (int) len);
        ComplexFloatMatrix div2 = range.transpose().mul(c2);
        ComplexFloatMatrix div3 = range.mmul(div2).negi();
        ComplexFloatMatrix matrix = exp(div3).div(len);
        ComplexFloatMatrix complexRet = inputC.isRowVector() ? matrix.mmul(inputC) : inputC.mmul(matrix);

        return complexRet;
    }


    /**
     * 1d discrete fourier transform, note that this will throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexFloatMatrix complexDiscreteFourierTransform1d(ComplexFloatMatrix inputC) {
        if(inputC.rows != 1 && inputC.columns != 1)
            throw new IllegalArgumentException("Illegal input: Must be a vector");

        float len = Math.max(inputC.rows,inputC.columns);
        ComplexFloat c2 = new ComplexFloat(0,-2).muli((float) FastMath.PI).divi(len);
        ComplexFloatMatrix range = MatrixUtil.complexRangeVector(0, len);
        ComplexFloatMatrix matrix = exp(range.mmul(range.transpose().mul(c2)));
        ComplexFloatMatrix complexRet = inputC.isRowVector() ? matrix.mmul(inputC) : inputC.mmul(matrix);
        return complexRet;
    }




    /**
     * 1d discrete fourier transform, note that this will throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param input the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexFloatMatrix complexDiscreteFourierTransform1d(FloatMatrix input) {
        return complexDiscreteFourierTransform1d(new ComplexFloatMatrix(input));
    }

    /**
     * Discrete fourier transform 2d
     * @param input the input to transform
     * @param rows the number of rows in the transformed output matrix
     * @param cols the number of columns in the transformed output matrix
     * @return the discrete fourier transform of the input
     */
    public static ComplexFloatMatrix complexDisceteFourierTransform(FloatMatrix input,int rows,int cols) {
        ComplexFloatMatrix base;

        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.complexPadWithZeros(input, rows, cols);
            //truncation
        else if(input.rows > rows || input.columns > cols) {
            base = new ComplexFloatMatrix(input);
            base = base.get(MatrixUtil.toIndices(RangeUtils.interval(0,rows)),MatrixUtil.toIndices(RangeUtils.interval(0,cols)));
        }
        else
            base = new ComplexFloatMatrix(input);

        ComplexFloatMatrix temp = new ComplexFloatMatrix(base.rows,base.columns);
        ComplexFloatMatrix ret = new ComplexFloatMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexFloatMatrix column = base.getColumn(i);
            temp.putColumn(i,complexDiscreteFourierTransform1d(column));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexFloatMatrix row = temp.getRow(i);
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
    public static FloatMatrix inverseDisceteFourierTransform(FloatMatrix input,int rows,int cols) {
        return complexInverseDisceteFourierTransform(input,rows,cols).getReal();

    }


    /**
     * Returns the real component of a 2d fourier transform
     * @param input the input to transform
     * @param rows the number of rows of the solution
     * @param cols the number of columns of the solution
     * @return the real component of a discrete fourier transform
     */
    public static FloatMatrix disceteFourierTransform(FloatMatrix input,int rows,int cols) {
        return complexDisceteFourierTransform(input,rows,cols).getReal();

    }


    /**
     * The inverse discrete fourier transform with the dimension size
     * being the same as the passed in input.
     *
     * @param inputC the input to transform
     * @return the
     */
    public static ComplexFloatMatrix complexInverseDisceteFourierTransform(ComplexFloatMatrix inputC) {
        return complexInverseDisceteFourierTransform(inputC,inputC.rows,inputC.columns);

    }



    public static ComplexFloatMatrix complexInverseDisceteFourierTransform(FloatMatrix input) {
        return complexInverseDisceteFourierTransform(input,input.rows,input.columns);
    }



    public static ComplexFloatMatrix complexDisceteFourierTransform(FloatMatrix input) {
        return complexDisceteFourierTransform(input,input.rows,input.columns);

    }


    public static FloatMatrix inverseDisceteFourierTransform(FloatMatrix input) {
        return complexInverseDisceteFourierTransform(input).getReal();

    }



    public static FloatMatrix disceteFourierTransform(FloatMatrix input) {
        return complexDisceteFourierTransform(input).getReal();

    }


}
