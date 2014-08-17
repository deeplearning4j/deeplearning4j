package org.deeplearning4j.linalg.fft;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.nn.linalg.Shape;
import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.ComplexNDArrayUtil;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.FloatMatrix;


/**
 * FFT and IFFT
 * @author Adam Gibson
 */
public class FFT {



    /**
     * FFT along a particular dimension
     * @param transform the ndarray to applyTransformToOrigin
     * @param numElements the desired number of elements in each fft
     * @return the ffted output
     */
    public static ComplexNDArray fft(NDArray transform,int numElements) {
        ComplexNDArray inputC = new ComplexNDArray(transform);
        if(inputC.isVector())
            return  new VectorFFT(inputC.length).apply(inputC);
        else {
            return rawfft(inputC,numElements,inputC.shape().length - 1);
        }
    }



    /**
     * 1d discrete fourier applyTransformToOrigin, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to applyTransformToOrigin
     * @return the the discrete fourier applyTransformToOrigin of the passed in input
     */
    public static  ComplexNDArray fft(ComplexNDArray inputC) {
        if(inputC.isVector())
            return  new VectorFFT(inputC.length).apply(inputC);
        else {
            return rawfft(inputC,inputC.size(inputC.shape().length - 1),inputC.shape().length - 1);
        }
    }

    /**
     * 1d discrete fourier applyTransformToOrigin, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param input the input to applyTransformToOrigin
     * @return the the discrete fourier applyTransformToOrigin of the passed in input
     */
    public static  ComplexNDArray fft(NDArray input) {
        ComplexNDArray inputC = new ComplexNDArray(input);
        return fft(inputC);
    }



    /**
     * FFT along a particular dimension
     * @param transform the ndarray to applyTransformToOrigin
     * @param numElements the desired number of elements in each fft
     * @return the ffted output
     */
    public static ComplexNDArray fft(NDArray transform,int numElements,int dimension) {
        ComplexNDArray inputC = new ComplexNDArray(transform);
        if(inputC.isVector())
            return  new VectorFFT(numElements).apply(inputC);
        else {
            return rawfft(inputC,numElements,dimension);
        }
    }


    /**
     * 1d discrete fourier applyTransformToOrigin, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to applyTransformToOrigin
     * @return the the discrete fourier applyTransformToOrigin of the passed in input
     */
    public static  ComplexNDArray fft(ComplexNDArray inputC,int numElements) {
        return fft(inputC,numElements,inputC.shape().length - 1);
    }


    /**
     * 1d discrete fourier applyTransformToOrigin, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to applyTransformToOrigin
     * @return the the discrete fourier applyTransformToOrigin of the passed in input
     */
    public static  ComplexNDArray fft(ComplexNDArray inputC,int numElements,int dimension) {
        if(inputC.isVector())
            return  new VectorFFT(numElements).apply(inputC);
        else {
            return rawfft(inputC,numElements,dimension);
        }
    }



    /**
     * IFFT along a particular dimension
     * @param transform the ndarray to applyTransformToOrigin
     * @param numElements the desired number of elements in each fft
     * @param dimension the dimension to do fft along
     * @return the iffted output
     */
    public static ComplexNDArray ifft(NDArray transform,int numElements,int dimension) {
        ComplexNDArray inputC = new ComplexNDArray(transform);
        if(inputC.isVector())
            return  new VectorIFFT(numElements).apply(inputC);
        else {
            return rawifft(inputC, numElements, dimension);
        }
    }



    /**
     * 1d discrete fourier applyTransformToOrigin, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to applyTransformToOrigin
     * @return the the discrete fourier applyTransformToOrigin of the passed in input
     */
    public static  ComplexNDArray ifft(ComplexNDArray inputC) {
        if(inputC.isVector())
            return  new VectorIFFT(inputC.length).apply(inputC);
        else {
            return rawifft(inputC, inputC.size(inputC.shape().length - 1), inputC.shape().length - 1);
        }
    }



    /**
     * FFT along a particular dimension
     * @param transform the ndarray to applyTransformToOrigin
     * @param numElements the desired number of elements in each fft
     * @return the ffted output
     */
    public static ComplexNDArray ifft(NDArray transform,int numElements) {
        ComplexNDArray inputC = new ComplexNDArray(transform);
        if(inputC.isVector())
            return  new VectorIFFT(numElements).apply(inputC);
        else {
            return rawifft(inputC,numElements,inputC.shape().length - 1);
        }
    }



    /**
     * 1d discrete fourier applyTransformToOrigin, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to applyTransformToOrigin
     * @return the the discrete fourier applyTransformToOrigin of the passed in input
     */
    public static  ComplexNDArray ifft(ComplexNDArray inputC,int numElements,int dimension) {
        if(inputC.isVector())
            return  new VectorIFFT(numElements).apply(inputC);
        else {
            return rawifft(inputC,numElements,dimension);
        }
    }





    /**
     * ND IFFT, computes along the first on singleton dimension of
     * applyTransformToOrigin
     * @param transform the ndarray to applyTransformToOrigin
     * @param dimension the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     * @return the reverse ifft of the passed in array
     */
    public static ComplexNDArray ifftn(NDArray transform,int dimension,int numElements) {
           return ifftn(new ComplexNDArray(transform),dimension,numElements);
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
     * @param transform the ndarray to applyTransformToOrigin
     * @param dimension the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     * @return the transformed array
     */
    public static ComplexNDArray ifftn(ComplexNDArray transform,int dimension,int numElements) {
        if(numElements < 1)
            throw new IllegalArgumentException("No elements specified");

        int[] finalShape = ArrayUtil.replace(transform.shape(), dimension, numElements);
        int[] axes = ArrayUtil.range(0, finalShape.length);

        ComplexNDArray result = transform.dup();

        int desiredElementsAlongDimension = result.size(dimension);

        if(numElements > desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.padWithZeros(result,finalShape);
        }

        else if(numElements < desiredElementsAlongDimension)
            result = ComplexNDArrayUtil.truncate(result,numElements,dimension);

        return rawifftn(result, finalShape, axes);
    }


    /**
     * Performs FFT along the first non singleton dimension of
     * applyTransformToOrigin. This means
     * @param transform the ndarray to applyTransformToOrigin
     * @param dimension the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     *                    along each dimension from each slice (note: each slice)
     * @return the transformed array
     */
    public static ComplexNDArray fftn(ComplexNDArray transform,int dimension,int numElements) {
        if(numElements < 1)
            throw new IllegalArgumentException("No elements specified");

        int[] finalShape = ArrayUtil.replace(transform.shape(), dimension, numElements);
        int[] axes = ArrayUtil.range(0, finalShape.length);

        ComplexNDArray result = transform.dup();

        int desiredElementsAlongDimension = result.size(dimension);

        if(numElements > desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.padWithZeros(result,finalShape);
        }

        else if(numElements < desiredElementsAlongDimension)
            result = ComplexNDArrayUtil.truncate(result,numElements,dimension);

        return rawfftn(result,finalShape,axes);
    }


    /**
     * Computes the fft along the first non singleton dimension of applyTransformToOrigin
     * when it is a matrix
     * @param transform the ndarray to applyTransformToOrigin
     * @param dimension the dimension to do fft along
     * @param numElements the desired number of elements in each fft
     * @return the fft of the specified ndarray
     */
    public static ComplexNDArray fftn(NDArray transform,int dimension,int numElements) {
        return fftn(new ComplexNDArray(transform),dimension,numElements);
    }

    /**
     * FFT on the whole array (n is equal the first dimension shape)
     * @param transform the matrix to applyTransformToOrigin
     * @return the ffted array
     */
    public static ComplexNDArray fftn(NDArray transform) {
        return fftn(transform,transform.shape().length - 1,transform.shape()[transform.shape().length - 1]);
    }








    /**
     * FFT on the whole array (n is equal the first dimension shape)
     * @param transform the matrix to applyTransformToOrigin
     * @return the ffted array
     */
    public static ComplexNDArray fftn(ComplexNDArray transform) {
        return fftn(transform,transform.shape().length - 1,transform.shape()[transform.shape().length - 1]);
    }






    public static ComplexNDArray ifftn(ComplexNDArray transform,int dimension) {
        return ifftn(transform, dimension, transform.shape()[dimension]);
    }


    public static ComplexNDArray ifftn(ComplexNDArray transform) {
        return ifftn(transform, transform.shape().length - 1,transform.size(transform.shape().length - 1));
    }


    public static ComplexNDArray ifftn(NDArray transform) {
        return ifftn(transform, transform.shape().length - 1, transform.size(transform.shape().length - 1));
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


    //underlying ifftn
    public static ComplexNDArray rawifftn(ComplexNDArray transform,int[] shape,int[] axes) {
        assert shape.length > 0 : "Shape length must be > 0";
        assert shape.length == axes.length : "Axes and shape must be the same length";

        ComplexNDArray result = transform.dup();



        for(int i =  shape.length - 1; i >= 0; i--) {
            result = FFT.ifft(result,shape[i],axes[i]);
        }


        return result;
    }

    //underlying fftn
    public static ComplexNDArray rawfftn(ComplexNDArray transform,int[] shape,int[] axes) {
        ComplexNDArray result = transform.dup();



        for(int i = shape.length - 1; i >= 0; i--) {
            result = FFT.fft(result, shape[i], axes[i]);
        }


        return result;
    }

    //underlying fftn

    /**
     * Underlying fft algorithm
     * @param transform the ndarray to transform
     * @param n the desired number of elements
     * @param dimension the dimension to do fft along
     * @return the transformed ndarray
     */
    public static ComplexNDArray rawfft(ComplexNDArray transform,int n,int dimension) {
        ComplexNDArray result = transform.dup();

        if(transform.size(dimension) != n) {
            int[] shape = ArrayUtil.copy(result.shape());
            shape[dimension] = n;
            if(transform.size(dimension) > n) {
                result = ComplexNDArrayUtil.truncate(result,n,dimension);
            }
            else
                result = ComplexNDArrayUtil.padWithZeros(result,shape);

        }


        if(dimension != result.shape().length - 1)
            result = result.swapAxes(result.shape().length - 1,dimension);

        result.iterateOverAllRows(new FFTSliceOp(result.size(result.shape().length - 1)));

        if(dimension != result.shape().length - 1)
            result = result.swapAxes(result.shape().length - 1,dimension);

        return result;
    }




    //underlying fftn
    public static ComplexNDArray rawifft(ComplexNDArray transform,int n,int dimension) {
        ComplexNDArray result = transform.dup();

        if(transform.size(dimension) != n) {
            int[] shape = ArrayUtil.copy(result.shape());
            shape[dimension] = n;
            if(transform.size(dimension) > n) {
                result = ComplexNDArrayUtil.truncate(result,n,dimension);
            }
            else
                result = ComplexNDArrayUtil.padWithZeros(result,shape);

        }


        if(dimension != result.shape().length - 1)
            result = result.swapAxes(result.shape().length - 1,dimension);

        result.iterateOverAllRows(new IFFTSliceOp(result.size(result.shape().length - 1)));

        if(dimension != result.shape().length - 1)
            result = result.swapAxes(result.shape().length - 1,dimension);

        return result;
    }

    //underlying fftn
    public static ComplexNDArray rawifft(ComplexNDArray transform,int dimension) {
        return rawifft(transform,transform.shape()[dimension],dimension);
    }







}