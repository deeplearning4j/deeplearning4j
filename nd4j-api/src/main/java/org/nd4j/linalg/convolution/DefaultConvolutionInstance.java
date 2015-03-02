/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.convolution;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.FFT;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.ComplexNDArrayUtil;
import org.nd4j.linalg.util.Shape;

import java.util.Arrays;

/**
 * Created by agibsonccc on 1/6/15.
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

        INDArray shape = ArrayUtil.toNDArray(Shape.sizeForAxes(axes, input.shape())).add(ArrayUtil.toNDArray(Shape.sizeForAxes(axes, kernel.shape()))).subi(1);
        int[] intShape = ArrayUtil.toInts(shape);

        IComplexNDArray ret = FFT.rawifftn(FFT.rawfftn(input, intShape, axes).muli(FFT.rawfftn(kernel, intShape, axes)), intShape, axes);


        switch (type) {
            case FULL:
                return ret;
            case SAME:
                return ComplexNDArrayUtil.center(ret, input.shape());
            case VALID:
                return ComplexNDArrayUtil.center(ret, ArrayUtil.toInts(Transforms.abs(ArrayUtil.toNDArray(input.shape()).sub(ArrayUtil.toNDArray(kernel.shape())).addi(1))));

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
        if (kernel.isScalar() && input.isScalar())
            return kernel.mul(input);
        INDArray shape = ArrayUtil.toNDArray(Shape.sizeForAxes(axes, input.shape())).add(ArrayUtil.toNDArray(Shape.sizeForAxes(axes, kernel.shape()))).subi(1);

        int[] intShape = ArrayUtil.toInts(shape);

        IComplexNDArray fftedInput = FFT.rawfftn(Nd4j.createComplex(input), intShape, axes);
        IComplexNDArray fftedKernel = FFT.rawfftn(Nd4j.createComplex(kernel), intShape, axes);
        //broadcast to be same shape
        if (!Arrays.equals(fftedInput.shape(), fftedKernel.shape())) {
            if (fftedInput.length() < fftedKernel.length()) {
                fftedInput = fftedInput.broadcast(fftedKernel.shape());
            } else {
                fftedKernel = fftedKernel.broadcast(fftedInput.shape());
            }
        }
        IComplexNDArray inputTimesKernel = fftedInput.muli(fftedKernel);

        IComplexNDArray convolution = FFT.ifftn(inputTimesKernel);


        switch (type) {
            case FULL:
                return convolution.getReal();
            case SAME:
                return ComplexNDArrayUtil.center(convolution, input.shape()).getReal();
            case VALID:
                int[] shape2 = ArrayUtil.toInts(Transforms.abs(ArrayUtil.toNDArray(input.shape()).sub(ArrayUtil.toNDArray(kernel.shape())).addi(1)));
                return ComplexNDArrayUtil.center(convolution, shape2).getReal();

        }


        return convolution.getReal();
    }


}
