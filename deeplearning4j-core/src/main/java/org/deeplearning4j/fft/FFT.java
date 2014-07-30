package org.deeplearning4j.fft;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.nn.linalg.Shape;
import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.ComplexNDArrayUtil;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
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
            result = ComplexNDArrayUtil.truncate(result,numElements,dimension);
        }

        result.iterateOverDimension(dimension,new IFFTSliceOp(numElements));

        assert Shape.shapeEquals(result.shape(),finalShape);


        return result;
    }


    public static ComplexNDArray irfftn(ComplexNDArray arr) {
        int[] shape = arr.shape();
        ComplexNDArray ret = arr.dup();
        for(int i = 0; i < shape.length - 1; i++) {
            ret = FFT.ifftn(ret,i,shape[i]);
        }


        return irfft(ret, 0);
    }



    public static ComplexNDArray irfft(ComplexNDArray arr,int dimension) {
        return fftn(arr, arr.size(dimension), dimension);
    }

    public static ComplexNDArray irfft(ComplexNDArray arr) {
        return arr;
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
        FloatMatrix s1 = MatrixUtil.toFloatMatrix(transform.shape());
        FloatMatrix s2 = MatrixUtil.toFloatMatrix(finalShape);
        FloatMatrix shape = s1.sub(s2).addi(1);
        finalShape = MatrixUtil.toInts(shape);

        ComplexNDArray result = transform.dup();
        if(dimension == 0 && transform.shape().length <= 1)
            return result;

        result.iterateOverDimension(dimension,new IFFTSliceOp(numElements));


        int desiredElementsAlongDimension = result.size(dimension);

        if(numElements > desiredElementsAlongDimension)
            result = ComplexNDArrayUtil.padWithZeros(result,finalShape);


        else if(numElements < desiredElementsAlongDimension)

            result = ComplexNDArrayUtil.truncate(result,numElements,dimension);




        assert Shape.shapeEquals(result.shape(),finalShape) : "Shape was " + Arrays.toString(result.shape()) + " when should have been " + Arrays.toString(finalShape);

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



        for(int i = transform.shape().length - 1; i >= 0; i--) {
            result.iterateOverDimension(dimension,new FFTSliceOp(result.size(i)));
        }

        //do along the first non singleton dimension when the number of dimensions is
        //greater than 1
        if(dimension == 0 && result.shape().length <= 1)
            return result;
        int desiredElementsAlongDimension = result.size(dimension);

        if(numElements > desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.padWithZeros(result,finalShape);
        }

        else if(numElements < desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.truncate(result,numElements,dimension);
        }



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

        int[] finalShape = ArrayUtil.replace(transform.shape(), dimension, numElements);


        ComplexNDArray result = new ComplexNDArray(transform);

        int desiredElementsAlongDimension = result.size(dimension);

        if(numElements > desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.padWithZeros(result,finalShape);
        }

        else if(numElements < desiredElementsAlongDimension)
            result = ComplexNDArrayUtil.truncate(result,numElements,dimension);




        result.iterateOverDimension(dimension,new FFTSliceOp(numElements));
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
        if(!transform.isVector())
            throw new IllegalArgumentException("Input to this function must be a vector");

        return ifft(new ComplexNDArray(transform),transform.size(0));
    }

    /**
     * FFT on the whole array (n is equal the first dimension shape)
     * @param transform the matrix to transform
     * @param numElements the number of elements per dimension for fft
     * @return the ffted array
     */
    public static ComplexDoubleMatrix ifft(NDArray transform,int numElements) {
        if(!transform.isVector())
            throw new IllegalArgumentException("Input to this function must be a vector");

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
        if(!inputC.isVector())
            throw new IllegalArgumentException("Input to this function must be a vector");

        return ifft(inputC,inputC.rows);
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
        if(!inputC.isVector())
            throw new IllegalArgumentException("Input to this function must be a vector");

        return ifft((ComplexDoubleMatrix) inputC, n);
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
    public static  ComplexNDArray fft(ComplexNDArray inputC) {
        if(inputC.isVector())
            return fft(ComplexNDArray.wrap(inputC), inputC.shape()[0]);
        else {
            inputC.iterateOverAllRows(new FFTSliceOp(inputC.shape()[0]));
            return inputC;
        }
    }




    public static ComplexNDArray ifftn(ComplexNDArray transform,int dimension) {
        return ifftn(transform, dimension, transform.shape()[0]);
    }


    public static ComplexNDArray ifftn(ComplexNDArray transform) {
        return ifftn(transform, 0, transform.size(0));
    }


    public static ComplexNDArray ifftn(NDArray transform) {
        return ifftn(transform, 0, transform.length);
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


    public static Pair<int[],int[]> cookNdArgs(NDArray arr,int[] shape,int[] axes) {
        if(shape == null)
            shape = ArrayUtil.copy(arr.shape());
        if(axes == null) {
            axes = ArrayUtil.range(-shape.length,0);
        }

        if(shape.length != axes.length)
            throw new IllegalArgumentException("Shape and axes must be same length");

        return new Pair<>(shape,axes);
    }


    /**
     * 1d discrete fourier transform, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexNDArray fft(ComplexNDArray inputC, int n) {
        if(!inputC.isVector())
            throw new IllegalArgumentException("Illegal input: Must be a vector");

        double len = inputC.length;
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexNDArray range = ComplexNDArray.wrap(MatrixUtil.complexRangeVector(0, len));
        ComplexNDArray matrix = ComplexNDArray.wrap(ComplexNDArrayUtil.exp(range.mmul(range.mul(c2))));
        ComplexNDArray complexRet =  inputC.mmul(matrix);
        return ComplexNDArray.wrap(complexRet);
    }


    //underlying ifftn
    public static ComplexNDArray rawifftn(ComplexNDArray transform,int[] shape,int[] axes) {
        ComplexNDArray result = transform.dup();



        for(int i = transform.shape().length - 1; i >= 0; i--)
            result.iterateOverDimension(axes[i],new IFFTSliceOp(shape[i]));

        return result;
    }

    //underlying fftn
    public static ComplexNDArray rawfftn(ComplexNDArray transform,int[] shape,int[] axes) {
        ComplexNDArray result = transform.dup();



        for(int i = transform.shape().length - 1; i >= 0; i--) {
            if(i < transform.shape().length) {
                result = result.swapAxes(axes[axes.length - 1],i);
            }
            result.iterateOverAllRows(new FFTSliceOp(shape[i]));
        }


        return result;
    }





}