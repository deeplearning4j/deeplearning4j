package org.deeplearning4j.fft;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.nn.linalg.Shape;
import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.ComplexNDArrayUtil;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;

import java.util.Arrays;

import static org.deeplearning4j.util.MatrixUtil.exp;

/**
 * FFT and IFFT
 * @author Adam Gibson
 */
public class FFT {



    /**
     * ND IFFT, computes along the first on singleton dimension of
     * transform
     * @param transform the ndarray to transform
     * @param dimension the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     * @return the reverse ifft of the passed in array
     */
    public static ComplexNDArray ifftn(NDArray transform,int dimension,int numElements) {
        if(numElements < 1)
            throw new IllegalArgumentException("No elements specified");

        int[] finalShape = ArrayUtil.replace(transform.shape(), dimension, numElements);


        if(dimension == 0 && transform.shape().length <= 1)
            return new ComplexNDArray(transform);
        ComplexNDArray result = new ComplexNDArray(transform);
        int desiredElementsAlongDimension = result.size(dimension);

        if(numElements > desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.padWithZeros(result,finalShape);
        }

        else if(numElements < desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.truncate(result,finalShape);
        }

        result.iterateOverDimension(dimension,new IFFTSliceOp(result,numElements));

        assert Shape.shapeEquals(result.shape(),finalShape);


        return result;
    }



    /**
     * ND IFFT
     * @param transform the ndarray to transform
     * @param dimension the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     * @return the transformed array
     */
    public static ComplexNDArray ifftn(ComplexNDArray transform,int dimension,int numElements) {
        if(numElements < 1)
            throw new IllegalArgumentException("No elements specified");
        int[] finalShape = ArrayUtil.replace(transform.shape(), dimension, numElements);


        ComplexNDArray result = transform.dup();
        if(dimension == 0 && transform.shape().length <= 1)
            return result;



        int desiredElementsAlongDimension = result.size(dimension);

        if(numElements > desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.padWithZeros(result,finalShape);
        }

        else if(numElements < desiredElementsAlongDimension) {

            result = ComplexNDArrayUtil.truncate(result,finalShape);
        }


        result.iterateOverDimension(dimension,new IFFTSliceOp(transform,numElements));

        assert Shape.shapeEquals(result.shape(),finalShape);

        return result;
    }


    /**
     * Performs FFT along the first non singleton dimension of
     * transform. This means
     * @param transform the ndarray to transform
     * @param dimension the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     *                    along each dimension from each slice (note: each slice)
     * @return the transformed array
     */
    public static ComplexNDArray fftn(ComplexNDArray transform,int dimension,int numElements) {
        if(numElements < 1)
            throw new IllegalArgumentException("No elements specified");

        int[] finalShape = ArrayUtil.replace(transform.shape(), dimension, numElements);

        ComplexNDArray result = transform.dup();
        //do along the first non singleton dimension when the number of dimensions is
        //greater than 1
        if(dimension == 0 && result.shape().length <= 1)
            return result;
        int desiredElementsAlongDimension = result.size(dimension);

        if(numElements > desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.padWithZeros(result,finalShape);
        }

        else if(numElements < desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.truncate(result,finalShape);
        }


        result.iterateOverDimension(dimension,new FFTSliceOp(result,numElements));

        assert Shape.shapeEquals(result.shape(),finalShape);


        return result;
    }


    /**
     * Computes the fft along the first non singleton dimension of transform
     * when it is a matrix
     * @param transform the ndarray to transform
     * @param dimension the dimension to do fft along
     * @param numElements the desired number of elements in each fft
     * @return the fft of the specified ndarray
     */
    public static ComplexNDArray fftn(NDArray transform,int dimension,int numElements) {
        if(numElements < 1)
            throw new IllegalArgumentException("No elements specified");

        int[] finalShape = Shape.squeeze(ArrayUtil.replace(transform.shape(), dimension, numElements));


        ComplexNDArray result = new ComplexNDArray(transform);

        int desiredElementsAlongDimension = result.size(dimension);

        if(numElements > desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.padWithZeros(result,finalShape);
        }

        else if(numElements < desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.truncate(result,finalShape);
        }



        result.iterateOverDimension(dimension,new FFTSliceOp(result,numElements));
        assert Shape.shapeEquals(result.shape(),finalShape);

        return result;
    }

    /**
     * FFT on the whole array (n is equal the first dimension shape)
     * @param transform the matrix to transform
     * @return the ffted array
     */
    public static ComplexNDArray fftn(NDArray transform) {
        return fftn(transform,0,transform.shape()[0]);
    }



    /**
     * IFFT on the whole array (n is equal the first dimension shape)
     * @param transform the matrix to transform
     * @return the iffted array
     */
    public static ComplexDoubleMatrix ifft(NDArray transform) {
        return ifft(new ComplexNDArray(transform), transform.length);
    }

    /**
     * FFT on the whole array (n is equal the first dimension shape)
     * @param transform the matrix to transform
     * @param numElements the number of elements per dimension for fft
     * @return the ffted array
     */
    public static ComplexDoubleMatrix ifft(NDArray transform,int numElements) {
        return ifft(new ComplexNDArray(transform), numElements);
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



    /**
     * FFT on the whole array (n is equal the first dimension shape)
     * @param transform the matrix to transform
     * @return the ffted array
     */
    public static ComplexNDArray fftn(ComplexNDArray transform) {
        return fftn(transform,0,transform.shape()[0]);
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

    public static ComplexNDArray ifftn(ComplexNDArray transform,int dimension) {
        return fftn(transform,dimension,transform.shape()[0]);
    }


    public static ComplexNDArray ifftn(ComplexNDArray transform) {
        return fftn(transform, 0, transform.length);
    }


    public static ComplexNDArray ifftn(NDArray transform) {
        return fftn(transform, 0, transform.length);
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


        if(n != complexRet.length) {
            ComplexDoubleMatrix newRet = new ComplexDoubleMatrix(1,n);
            for(int i = 0; i < n; i++) {
                if(i >= complexRet.length)
                    break;

                newRet.put(i, complexRet.get(i));
            }
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
        if(n != complexRet.length) {
            ComplexDoubleMatrix newRet = new ComplexDoubleMatrix(1,n);
            for(int i = 0; i < n; i++) {
                if(i >= complexRet.length)
                    break;
                newRet.put(i, complexRet.get(i));
            }
            return newRet;
        }



        return complexRet;
    }



}