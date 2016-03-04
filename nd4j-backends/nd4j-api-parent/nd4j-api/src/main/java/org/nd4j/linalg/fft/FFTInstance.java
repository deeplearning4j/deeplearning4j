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


/**
 * Base line fft methods
 *
 * @author Adam Gibson
 */
public interface FFTInstance {

    /**
     * FFT along a particular dimension
     *
     * @param transform   the ndarray to op
     * @param numElements the desired number of elements in each fft
     * @return the ffted output
     */
    IComplexNDArray fft(INDArray transform, int numElements);


    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param inputC the input to op
     * @return the the discrete fourier op of the passed in input
     */
    IComplexNDArray fft(IComplexNDArray inputC);

    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param input the input to op
     * @return the the discrete fourier op of the passed in input
     */
    IComplexNDArray fft(INDArray input);


    /**
     * FFT along a particular dimension
     *
     * @param transform   the ndarray to op
     * @param numElements the desired number of elements in each fft
     * @return the ffted output
     */
    IComplexNDArray fft(INDArray transform, int numElements, int dimension);


    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param inputC the input to op
     * @return the the discrete fourier op of the passed in input
     */
    IComplexNDArray fft(IComplexNDArray inputC, int numElements);


    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param inputC the input to op
     * @return the the discrete fourier op of the passed in input
     */
    IComplexNDArray fft(IComplexNDArray inputC, int numElements, int dimension);


    /**
     * IFFT along a particular dimension
     *
     * @param transform   the ndarray to op
     * @param numElements the desired number of elements in each fft
     * @param dimension   the dimension to do fft along
     * @return the iffted output
     */
    IComplexNDArray ifft(INDArray transform, int numElements, int dimension);


    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param inputC the input to op
     * @return the the discrete fourier op of the passed in input
     */
    IComplexNDArray ifft(IComplexNDArray inputC);


    /**
     * FFT along a particular dimension
     *
     * @param transform   the ndarray to op
     * @param numElements the desired number of elements in each fft
     * @return the ffted output
     */
    IComplexNDArray ifft(INDArray transform, int numElements);


    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param inputC the input to op
     * @return the the discrete fourier op of the passed in input
     */
    IComplexNDArray ifft(IComplexNDArray inputC, int numElements, int dimension);


    /**
     * ND IFFT, computes along the first on singleton dimension of
     * op
     *
     * @param transform   the ndarray to op
     * @param dimension   the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     * @return the reverse ifft of the passed in array
     */
    IComplexNDArray ifftn(INDArray transform, int dimension, int numElements);

    IComplexNDArray irfftn(IComplexNDArray arr);


    IComplexNDArray irfft(IComplexNDArray arr, int dimension);

    IComplexNDArray irfft(IComplexNDArray arr);

    /**
     * ND IFFT
     *
     * @param transform   the ndarray to op
     * @param dimension   the dimension to iterate along
     * @param numElements the desired number of elements in each fft
     * @return the transformed array
     */
    IComplexNDArray ifftn(IComplexNDArray transform, int dimension, int numElements);

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
    IComplexNDArray fftn(IComplexNDArray transform, int dimension, int numElements);


    /**
     * Computes the fft along the first non singleton dimension of op
     * when it is a matrix
     *
     * @param transform   the ndarray to op
     * @param dimension   the dimension to do fft along
     * @param numElements the desired number of elements in each fft
     * @return the fft of the specified ndarray
     */
    IComplexNDArray fftn(INDArray transform, int dimension, int numElements);

    /**
     * FFT on the whole array (n is equal the first dimension shape)
     *
     * @param transform the matrix to op
     * @return the ffted array
     */
    IComplexNDArray fftn(INDArray transform);

    /**
     * FFT on the whole array (n is equal the first dimension shape)
     *
     * @param transform the matrix to op
     * @return the ffted array
     */
    IComplexNDArray fftn(IComplexNDArray transform);

    IComplexNDArray ifftn(IComplexNDArray transform, int dimension);

    IComplexNDArray ifftn(IComplexNDArray transform);


    IComplexNDArray ifftn(INDArray transform);

    //underlying ifftn
    IComplexNDArray rawifftn(IComplexNDArray transform, int[] shape, int[] axes);

    //underlying fftn
    IComplexNDArray rawfftn(IComplexNDArray transform, int[] shape, int[] axes);


    /**
     * Underlying fft algorithm
     *
     * @param transform the ndarray to op
     * @param n         the desired number of elements
     * @param dimension the dimension to do fft along
     * @return the transformed ndarray
     */
    IComplexNDArray rawfft(IComplexNDArray transform, int n, int dimension);

    /**
     * Underlying ifft impl
     *
     * @param transform the ndarray to op
     * @param n         the desired number of elements
     * @param dimension the dimension to do fft along
     * @return
     */
    IComplexNDArray rawifft(IComplexNDArray transform, int n, int dimension);

    /**
     * Underlying ifft impl
     *
     * @param transform the ndarray to op
     * @param n         the desired number of elements
     * @param dimension the dimension to do fft along
     * @return
     */
    IComplexNDArray rawifft(IComplexNDArray transform, int dimension);

}
