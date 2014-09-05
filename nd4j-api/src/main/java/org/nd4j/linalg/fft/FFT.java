package org.nd4j.linalg.fft;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.ComplexNDArrayUtil;


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
    public static IComplexNDArray fft(INDArray transform,int numElements) {
        IComplexNDArray inputC = NDArrays.createComplex(transform);
        if(inputC.isVector())
            return  new VectorFFT(inputC.length()).apply(inputC);
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
    public static  IComplexNDArray fft(IComplexNDArray inputC) {
        if(inputC.isVector())
            return  new VectorFFT(inputC.length()).apply(inputC);
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
    public static  IComplexNDArray fft(INDArray input) {
        IComplexNDArray inputC = NDArrays.createComplex(input);
        return fft(inputC);
    }



    /**
     * FFT along a particular dimension
     * @param transform the ndarray to applyTransformToOrigin
     * @param numElements the desired number of elements in each fft
     * @return the ffted output
     */
    public static IComplexNDArray fft(INDArray transform,int numElements,int dimension) {
        IComplexNDArray inputC = NDArrays.createComplex(transform);
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
    public static  IComplexNDArray fft(IComplexNDArray inputC,int numElements) {
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
    public static  IComplexNDArray fft(IComplexNDArray inputC,int numElements,int dimension) {
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
    public static IComplexNDArray ifft(INDArray transform,int numElements,int dimension) {
        IComplexNDArray inputC = NDArrays.createComplex(transform);
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
    public static  IComplexNDArray ifft(IComplexNDArray inputC) {
        if(inputC.isVector())
            return  new VectorIFFT(inputC.length()).apply(inputC);
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
    public static IComplexNDArray ifft(INDArray transform,int numElements) {
        IComplexNDArray inputC = NDArrays.createComplex(transform);
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
    public static  IComplexNDArray ifft(IComplexNDArray inputC,int numElements,int dimension) {
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
    public static IComplexNDArray ifftn(INDArray transform,int dimension,int numElements) {
           return ifftn(NDArrays.createComplex(transform),dimension,numElements);
    }


    public static IComplexNDArray irfftn(IComplexNDArray arr) {
        int[] shape = arr.shape();
        IComplexNDArray ret = arr.dup();
        for(int i = 0; i < shape.length - 1; i++) {
            ret = FFT.ifftn(ret,i,shape[i]);
        }


        return irfft(ret, 0);
    }



    public static IComplexNDArray irfft(IComplexNDArray arr,int dimension) {
        return fftn(arr, arr.size(dimension), dimension);
    }

    public static IComplexNDArray irfft(IComplexNDArray arr) {
        return arr;
    }

    /**
     * ND IFFT
     * @param transform the ndarray to applyTransformToOrigin
     * @param dimension the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     * @return the transformed array
     */
    public static IComplexNDArray ifftn(IComplexNDArray transform,int dimension,int numElements) {
        if(numElements < 1)
            throw new IllegalArgumentException("No elements specified");

        int[] finalShape = ArrayUtil.replace(transform.shape(), dimension, numElements);
        int[] axes = ArrayUtil.range(0, finalShape.length);

        IComplexNDArray result = transform.dup();

        int desiredElementsAlongDimension = result.size(dimension);

        if(numElements > desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.padWithZeros(result, finalShape);
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
    public static IComplexNDArray fftn(IComplexNDArray transform,int dimension,int numElements) {
        if(numElements < 1)
            throw new IllegalArgumentException("No elements specified");

        int[] finalShape = ArrayUtil.replace(transform.shape(), dimension, numElements);
        int[] axes = ArrayUtil.range(0, finalShape.length);

        IComplexNDArray result = transform.dup();

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
    public static IComplexNDArray fftn(INDArray transform,int dimension,int numElements) {
        return fftn(NDArrays.createComplex(transform),dimension,numElements);
    }

    /**
     * FFT on the whole array (n is equal the first dimension shape)
     * @param transform the matrix to applyTransformToOrigin
     * @return the ffted array
     */
    public static IComplexNDArray fftn(INDArray transform) {
        return fftn(transform,transform.shape().length - 1,transform.shape()[transform.shape().length - 1]);
    }








    /**
     * FFT on the whole array (n is equal the first dimension shape)
     * @param transform the matrix to applyTransformToOrigin
     * @return the ffted array
     */
    public static IComplexNDArray fftn(IComplexNDArray transform) {
        return fftn(transform,transform.shape().length - 1,transform.shape()[transform.shape().length - 1]);
    }






    public static IComplexNDArray ifftn(IComplexNDArray transform,int dimension) {
        return ifftn(transform, dimension, transform.shape()[dimension]);
    }


    public static IComplexNDArray ifftn(IComplexNDArray transform) {
        return ifftn(transform, transform.shape().length - 1,transform.size(transform.shape().length - 1));
    }


    public static IComplexNDArray ifftn(INDArray transform) {
        return ifftn(transform, transform.shape().length - 1, transform.size(transform.shape().length - 1));
    }







    //underlying ifftn
    public static IComplexNDArray rawifftn(IComplexNDArray transform,int[] shape,int[] axes) {
        assert shape.length > 0 : "Shape length must be > 0";
        assert shape.length == axes.length : "Axes and shape must be the same length";

        IComplexNDArray result = transform.dup();



        for(int i =  shape.length - 1; i >= 0; i--) {
            result = FFT.ifft(result,shape[i],axes[i]);
        }


        return result;
    }

    //underlying fftn
    public static IComplexNDArray rawfftn(IComplexNDArray transform,int[] shape,int[] axes) {
        IComplexNDArray result = transform.dup();



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
    public static IComplexNDArray rawfft(IComplexNDArray transform,int n,int dimension) {
        IComplexNDArray result = transform.dup();

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
    public static IComplexNDArray rawifft(IComplexNDArray transform,int n,int dimension) {
        IComplexNDArray result = transform.dup();

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
    public static IComplexNDArray rawifft(IComplexNDArray transform,int dimension) {
        return rawifft(transform,transform.shape()[dimension],dimension);
    }







}