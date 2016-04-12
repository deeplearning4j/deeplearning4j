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

package org.nd4j.linalg.convolution;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.FFT;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ComplexNDArrayUtil;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.NDArrayUtil;

import java.util.Arrays;

/**
 * Default convolution instance (FFT based)
 *
 * @author Adam Gibson
 */
public class DefaultConvolutionInstance extends BaseConvolution {

    /**
     * ND Convolution
     *
     * @param input  the input to op
     * @param kernel the kernel to op with
     * @param type   the type of convolution
     * @param axes   the axes to do the convolution along
     * @return the convolution of the given input and kernel
     */
    @Override
    public IComplexNDArray convn(IComplexNDArray input, IComplexNDArray kernel, Convolution.Type type, int[] axes) {

        if (kernel.isScalar() && input.isScalar())
            return kernel.mul(input);

        INDArray shape = NDArrayUtil.toNDArray(Shape.sizeForAxes(axes, input.shape())).add(NDArrayUtil.toNDArray(Shape.sizeForAxes(axes, kernel.shape()))).subi(1);
        int[] intShape = NDArrayUtil.toInts(shape);

        IComplexNDArray ret = FFT.rawifftn(FFT.rawfftn(input, intShape, axes).muli(FFT.rawfftn(kernel, intShape, axes)), intShape, axes);


        switch (type) {
            case FULL:
                return ret;
            case SAME:
                return ComplexNDArrayUtil.center(ret, input.shape());
            case VALID:
                return ComplexNDArrayUtil.center(ret, NDArrayUtil.toInts(Transforms.abs(NDArrayUtil.toNDArray(input.shape()).sub(NDArrayUtil.toNDArray(kernel.shape())).addi(1))));

        }

        return ret;
    }

    /**
     * ND Convolution
     *
     * @param input  the input to op
     * @param kernel the kernel to op with
     * @param type   the type of convolution
     * @param axes   the axes to do the convolution along
     * @return the convolution of the given input and kernel
     */
    @Override
    public INDArray convn(INDArray input, INDArray kernel, Convolution.Type type, int[] axes) {
        if (input.shape().length != kernel.shape().length) {
            int[] newShape = new int[Math.max(input.shape().length, kernel.shape().length)];
            Arrays.fill(newShape, 1);
            int lengthDelta = Math.abs(input.shape().length - kernel.shape().length);
            if (input.shape().length < kernel.shape().length) {
                for (int i = input.shape().length - 1; i >= 0; i--)
                    newShape[i + lengthDelta] = input.shape()[i];
                input = input.reshape(newShape);


            } else {
                if (kernel.shape().length < input.shape().length) {
                    for (int i = kernel.shape().length - 1; i >= 0; i--)
                        newShape[i + lengthDelta] = kernel.shape()[i];

                    kernel = kernel.reshape(newShape);
                }

            }
        }

        if (kernel.isScalar() && input.isScalar())
            return kernel.mul(input);
        INDArray shape = NDArrayUtil.toNDArray(input.shape()).add(NDArrayUtil.toNDArray(kernel.shape())).subi(1);

        int[] intShape = NDArrayUtil.toInts(shape);


        IComplexNDArray fftedInput = FFT.rawfftn(Nd4j.createComplex(input), intShape, axes);
        IComplexNDArray fftedKernel = FFT.rawfftn(Nd4j.createComplex(kernel), intShape, axes);
        //broadcast to be same shape
        if (!Arrays.equals(fftedInput.shape(), fftedKernel.shape())) {
            if (fftedInput.length() < fftedKernel.length())
                fftedInput = ComplexNDArrayUtil.padWithZeros(fftedInput, fftedKernel.shape());
            else
                fftedKernel = ComplexNDArrayUtil.padWithZeros(fftedKernel, fftedInput.shape());

        }

        IComplexNDArray inputTimesKernel = fftedInput.muli(fftedKernel);
        IComplexNDArray convolution = FFT.ifftn(inputTimesKernel);


        switch (type) {
            case FULL:
                return convolution.getReal();
            case SAME:
                return ComplexNDArrayUtil.center(convolution, input.shape()).getReal();
            case VALID:
                int[] shape2 = NDArrayUtil.toInts(Transforms.abs(NDArrayUtil.toNDArray(input.shape()).sub(NDArrayUtil.toNDArray(kernel.shape())).addi(1)));
                return ComplexNDArrayUtil.center(convolution, shape2).getReal();

        }

        return convolution.getReal();
    }

}
