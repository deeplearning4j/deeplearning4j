/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.fft;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.VectorFFT;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.ComplexNDArrayUtil;

/**
 * Abstract FFT Instance mostly handling basic things that shouldn't change
 * such as method overloading.
 *
 * @author Adam Gibson
 */
public abstract class BaseFFTInstance implements FFTInstance {

    /**
     * FFT along a particular dimension
     *
     * @param transform   the ndarray to op
     * @param numElements the desired number of elements in each fft
     * @return the ffted output
     */
    @Override
    public IComplexNDArray fft(INDArray transform, int numElements) {
        IComplexNDArray inputC = Nd4j.createComplex(transform);
        if (inputC.isVector())
            return (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorFFT(inputC,inputC.length()));
        else {
            return rawfft(inputC, numElements, inputC.shape().length - 1);
        }
    }


    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param inputC the input to op
     * @return the the discrete fourier op of the passed in input
     */
    @Override
    public IComplexNDArray fft(IComplexNDArray inputC) {
        if (inputC.isVector())
            return (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorFFT(inputC,inputC.length()));
        else {
            return rawfft(inputC, inputC.size(inputC.shape().length - 1), inputC.shape().length - 1);
        }
    }

    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param input the input to op
     * @return the the discrete fourier op of the passed in input
     */
    @Override
    public IComplexNDArray fft(INDArray input) {
        IComplexNDArray inputC = Nd4j.createComplex(input);
        return fft(inputC);
    }


    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param inputC the input to op
     * @return the the discrete fourier op of the passed in input
     */
    @Override
    public IComplexNDArray fft(IComplexNDArray inputC, int numElements) {
        return fft(inputC, numElements, inputC.shape().length - 1);
    }


    /**
     * ND IFFT, computes along the first on singleton dimension of
     * op
     *
     * @param transform   the ndarray to op
     * @param dimension   the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     * @return the reverse ifft of the passed in array
     */
    @Override
    public IComplexNDArray ifftn(INDArray transform, int dimension, int numElements) {
        return ifftn(Nd4j.createComplex(transform), dimension, numElements);
    }

    @Override
    public IComplexNDArray irfftn(IComplexNDArray arr) {
        int[] shape = arr.shape();
        IComplexNDArray ret = arr.dup();
        for (int i = 0; i < shape.length - 1; i++) {
            ret = ifftn(ret, i, shape[i]);
        }


        return irfft(ret, 0);
    }


    @Override
    public IComplexNDArray irfft(IComplexNDArray arr, int dimension) {
        return fftn(arr, arr.size(dimension), dimension);
    }

    @Override
    public IComplexNDArray irfft(IComplexNDArray arr) {
        return arr;
    }

    /**
     * ND IFFT
     *
     * @param transform   the ndarray to op
     * @param dimension   the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     * @return the transformed array
     */
    @Override
    public IComplexNDArray ifftn(IComplexNDArray transform, int dimension, int numElements) {
        if (numElements < 1)
            throw new IllegalArgumentException("No elements specified");

        int[] finalShape = ArrayUtil.replace(transform.shape(), dimension, numElements);
        int[] axes = ArrayUtil.range(0, finalShape.length);

        IComplexNDArray result = transform.dup();

        int desiredElementsAlongDimension = result.size(dimension);

        if (numElements > desiredElementsAlongDimension) {
            result = ComplexNDArrayUtil.padWithZeros(result, finalShape);
        } else if (numElements < desiredElementsAlongDimension)
            result = ComplexNDArrayUtil.truncate(result, numElements, dimension);

        return rawifftn(result, finalShape, axes);
    }


    /**
     * Performs FFT along the first non singleton dimension of
     * op. This means
     *
     * @param transform   the ndarray to op
     * @param dimension   the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     *                    along each dimension from each slice (note: each slice)
     * @return the transformed array
     */
    @Override
    public IComplexNDArray fftn(IComplexNDArray transform, int dimension, int numElements) {
        if (numElements < 1)
            throw new IllegalArgumentException("No elements specified");

        int[] finalShape = ArrayUtil.replace(transform.shape(), dimension, numElements);
        int[] axes = ArrayUtil.range(0, finalShape.length);

        IComplexNDArray result = transform.dup();

        int desiredElementsAlongDimension = result.size(dimension);

        if (numElements > desiredElementsAlongDimension)
            result = ComplexNDArrayUtil.padWithZeros(result, finalShape);
        else if (numElements < desiredElementsAlongDimension)
            result = ComplexNDArrayUtil.truncate(result, numElements, dimension);

        return rawfftn(result, finalShape, axes);
    }


    /**
     * Computes the fft along the first non singleton dimension of op
     * when it is a matrix
     *
     * @param transform   the ndarray to op
     * @param dimension   the dimension to do fft along
     * @param numElements the desired number of elements in each fft
     * @return the fft of the specified ndarray
     */
    @Override
    public IComplexNDArray fftn(INDArray transform, int dimension, int numElements) {
        return fftn(Nd4j.createComplex(transform), dimension, numElements);
    }

    /**
     * FFT on the whole array (n is equal the first dimension shape)
     *
     * @param transform the matrix to op
     * @return the ffted array
     */
    @Override
    public IComplexNDArray fftn(INDArray transform) {
        return fftn(transform, transform.shape().length - 1, transform.shape()[transform.shape().length - 1]);
    }

    /**
     * FFT on the whole array (n is equal the first dimension shape)
     *
     * @param transform the matrix to op
     * @return the ffted array
     */
    @Override
    public IComplexNDArray fftn(IComplexNDArray transform) {
        return rawfftn(transform, null, null);
    }

    @Override
    public IComplexNDArray ifftn(IComplexNDArray transform, int dimension) {
        return ifftn(transform, dimension, transform.shape()[dimension]);
    }

    @Override
    public IComplexNDArray ifftn(IComplexNDArray transform) {
        return rawifftn(transform, null, null);
    }

    @Override
    public IComplexNDArray ifftn(INDArray transform) {
        return ifftn(transform, transform.shape().length - 1, transform.size(transform.shape().length - 1));
    }

    //underlying ifftn
    @Override
    public IComplexNDArray rawifftn(IComplexNDArray transform, int[] shape, int[] axes) {
        return doFFt(transform, shape, axes, true);

    }

    //underlying fftn
    @Override
    public IComplexNDArray rawfftn(IComplexNDArray transform, int[] shape, int[] axes) {
        return doFFt(transform,shape,axes,false);
    }

    private IComplexNDArray fixShape(IComplexNDArray x,int[] shape,int axis, int n) {
        if(shape[axis] > n) {
            int[] newShape = ArrayUtil.copy(shape);
            newShape[axis] = n;
            x = ComplexNDArrayUtil.truncate(x,n,axis);
        }
        else {
            int[] newShape = ArrayUtil.copy(shape);
            newShape[axis] = n;
            x = ComplexNDArrayUtil.padWithZeros(x,newShape);
            return x;

        }
        return x;
    }


    //underlying fftn
    @Override
    public IComplexNDArray rawifft(IComplexNDArray transform, int dimension) {
        return rawifft(transform, transform.shape()[dimension], dimension);
    }

    protected IComplexNDArray doFFt(IComplexNDArray transform,int[] shape,int[] axes,boolean inverse) {
        IComplexNDArray result = transform.dup();
        if(shape == null)
            shape = ArrayUtil.copy(result.shape());
        boolean noAxes = false;
        if(axes == null || axes.length < 1) {
            noAxes = true;
            axes = ArrayUtil.range(0,shape.length);
            axes = ArrayUtil.reverseCopy(axes);
        }

        if(noAxes) {
            for(int i : axes) {
                if(i < 0)
                    i = shape.length + i;
                transform = fixShape(transform,shape,i,shape[i]);
            }
        }

        if(ArrayUtil.prod(shape) > ArrayUtil.prod(result.shape()))
            result = ComplexNDArrayUtil.padWithZeros(result,shape);


        return doInnerFft(result,shape,axes,inverse);
    }

    //the inner loop for an fft or ifft
    protected IComplexNDArray doInnerFft(IComplexNDArray result,int[] shape,int[] axes,boolean inverse) {
        for(int i = 0; i < axes.length; i++) {
            result = inverse ? ifft(result,shape[axes[i]],axes[i]) : fft(result,shape[axes[i]],axes[i]);
        }

        return result;
    }


}
