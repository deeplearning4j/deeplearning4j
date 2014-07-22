package org.deeplearning4j.fft;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;

import static org.deeplearning4j.util.MatrixUtil.exp;

/**
 * FFT and IFFT
 * @author Adam Gibson
 */
public class FFT {




    public static ComplexNDArray ifftn(ComplexNDArray transform,int dimension,int numElements) {
        ComplexNDArray result = transform.dup();
        result.iterateOverDimension(dimension,new IFFTSliceOp(transform,numElements));
        return result;
    }


    public static ComplexNDArray ifftn(ComplexNDArray transform) {
        return fftn(transform,0,transform.length);
    }



    public static ComplexNDArray ifftn(NDArray transform,int dimension,int numElements) {
        ComplexNDArray result = new ComplexNDArray(transform);
        result.iterateOverDimension(dimension,new IFFTSliceOp(transform,numElements));
        return result;
    }


    public static ComplexNDArray ifftn(NDArray transform) {
        return fftn(transform,0,transform.length);
    }


    public static ComplexNDArray fftn(ComplexNDArray transform,int dimension,int numElements) {
        ComplexNDArray result = transform.dup();
        result.iterateOverDimension(dimension,new IFFTSliceOp(transform,numElements));
        return result;
    }


    public static ComplexNDArray fftn(ComplexNDArray transform) {
        return fftn(transform,0,transform.length);
    }



    public static ComplexNDArray fftn(NDArray transform,int dimension,int numElements) {
        ComplexNDArray result = new ComplexNDArray(transform);
        result.iterateOverDimension(dimension,new FFTSliceOp(transform,numElements));
        return result;
    }


    public static ComplexNDArray fftn(NDArray transform) {
        return fftn(transform,0,transform.length);
    }




    public static ComplexDoubleMatrix ifft(NDArray transform) {
        return ifft(new ComplexNDArray(transform),transform.length);
    }


    public static ComplexDoubleMatrix ifft(NDArray transform,int numElements) {
        return ifft(new ComplexNDArray(transform),numElements);
    }



    /**
     * 1d discrete fourier transform, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexDoubleMatrix ifft(ComplexDoubleMatrix inputC) {
        return ifft(inputC, inputC.length);
    }


    /**
     * 1d discrete fourier transform, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexDoubleMatrix ifft(ComplexNDArray inputC, int n) {
        return fft((ComplexDoubleMatrix) inputC,n);
    }





    public static ComplexDoubleMatrix fft(NDArray transform,int numElements) {
        return fft(new ComplexNDArray(transform),numElements);
    }



    /**
     * 1d discrete fourier transform, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexDoubleMatrix fft(ComplexDoubleMatrix inputC) {
        return fft(inputC, inputC.length);
    }


    /**
     * 1d discrete fourier transform, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexDoubleMatrix fft(ComplexNDArray inputC, int n) {
        return fft((ComplexDoubleMatrix) inputC,n);
    }




    /**
     * 1d inverse discrete fourier transform
     * see matlab's fft2 for more examples.
     * Note that this will throw an exception if the input isn't a vector
     * @param inputC the input to transform
     * @return the inverse fourier transform of the passed in input
     */
    public static ComplexDoubleMatrix ifft(ComplexDoubleMatrix inputC,int n) {
        if(inputC.rows != 1 && inputC.columns != 1)
            throw new IllegalArgumentException("Illegal input: Must be a vector");
        double len = MatrixUtil.length(inputC);
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexDoubleMatrix range = MatrixUtil.complexRangeVector(0,len);
        ComplexDoubleMatrix div2 = range.transpose().mul(c2);
        ComplexDoubleMatrix div3 = range.mmul(div2).negi();
        ComplexDoubleMatrix matrix = exp(div3).div(len);
        ComplexDoubleMatrix complexRet = inputC.isRowVector() ? matrix.mmul(inputC) : inputC.mmul(matrix);


        if(n < complexRet.length) {
            ComplexDoubleMatrix newRet = new ComplexDoubleMatrix(1,n);
            for(int i = 0; i < n; i++)
                newRet.put(i,complexRet.get(i));
            return newRet;
        }

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
    public static  ComplexDoubleMatrix fft(ComplexDoubleMatrix inputC, int n) {
        if(inputC.rows != 1 && inputC.columns != 1)
            throw new IllegalArgumentException("Illegal input: Must be a vector");

        double len = inputC.length;
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexDoubleMatrix range = MatrixUtil.complexRangeVector(0, len);
        ComplexDoubleMatrix matrix = exp(range.mmul(range.transpose().mul(c2)));
        ComplexDoubleMatrix complexRet = inputC.isRowVector() ? matrix.mmul(inputC) : inputC.mmul(matrix);
        if(n < complexRet.length) {
            ComplexDoubleMatrix newRet = new ComplexDoubleMatrix(1,n);
            for(int i = 0; i < n; i++)
                newRet.put(i,complexRet.get(i));
            return newRet;
        }


        return complexRet;
    }



}