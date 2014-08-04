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

        result.iterateOverDimension(dimension,new IFFTSliceOp(numElements),true);

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

        result.iterateOverDimension(dimension,new IFFTSliceOp(numElements),true);


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
            result.iterateOverDimension(dimension,new FFTSliceOp(result.size(i)),true);
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
        int[] axes = ArrayUtil.reverseCopy(ArrayUtil.range(0,dimension));

        ComplexNDArray result = new ComplexNDArray(transform);

        int desiredElementsAlongDimension = result.size(dimension);

        if(numElements > desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.padWithZeros(result,finalShape);
        }

        else if(numElements < desiredElementsAlongDimension)
            result = ComplexNDArrayUtil.truncate(result,numElements,dimension);

        return rawfftn(result,finalShape,axes);
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
     * FFT on the whole array (n is equal the first dimension shape)
     * @param transform the matrix to transform
     * @return the ffted array
     */
    public static ComplexNDArray fftn(ComplexNDArray transform) {
        return fftn(transform,0,transform.shape()[0]);
    }



    public static ComplexNDArray fft(NDArray transform,int numElements) {
        return new VectorFFT(numElements).apply(new ComplexNDArray(transform));
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
            return  new VectorFFT(inputC.length).apply(inputC);
        else {
            return rawfft(inputC,inputC.size(inputC.shape().length - 1),inputC.shape().length - 1);
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
        ComplexNDArray result = transform.dup();



        for(int i = transform.shape().length - 1; i >= 0; i--) {
            result = FFT.rawifft(result,shape[i],axes[i]);
        }


        return result;
    }

    //underlying fftn
    public static ComplexNDArray rawfftn(ComplexNDArray transform,int[] shape,int[] axes) {
        ComplexNDArray result = transform.dup();



        for(int i = transform.shape().length - 1; i >= 0; i--) {
              result = FFT.rawfft(result,shape[i],axes[i]);
        }


        return result;
    }

    //underlying fftn
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