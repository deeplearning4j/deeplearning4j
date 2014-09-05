package org.nd4j.linalg.convolution;


import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.FFT;

import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.ComplexNDArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Convolution is the code for applying the convolution operator.
 *  http://www.inf.ufpr.br/danielw/pos/ci724/20102/HIPR2/flatjavasrc/Convolution.java
 *
 * @author Adam Gibson
 */
public class Convolution {

    private static Logger log = LoggerFactory.getLogger(Convolution.class);

    /**
     *
     *
     * Default no-arg constructor.
     */
    private Convolution() {
    }

    public static enum Type {
        FULL,VALID,SAME
    }




    /**
     * ND Convolution
     * @param input the input to applyTransformToOrigin
     * @param kernel the kernel to applyTransformToOrigin with
     * @param type the type of convolution
     * @return the convolution of the given input and kernel
     */
    public static INDArray convn(INDArray input,INDArray kernel,Type type) {
        if(kernel.isScalar() && input.isScalar())
            return kernel.mul(input);
        INDArray shape = ArrayUtil.toNDArray(input.shape()).add(ArrayUtil.toNDArray(kernel.shape())).subi(1);

        int[] intShape = ArrayUtil.toInts(shape);
        int[] axes = ArrayUtil.range(0,intShape.length);

        IComplexNDArray fftedInput = FFT.rawfftn(Nd4j.createComplex(input),intShape,axes);
        IComplexNDArray fftedKernel = FFT.rawfftn(Nd4j.createComplex(kernel), intShape, axes);
        IComplexNDArray inputTimesKernel = fftedInput.muli(fftedKernel);

        IComplexNDArray convolution = FFT.ifftn(inputTimesKernel);




        switch(type) {
            case FULL:
                return convolution.getReal();
            case SAME:
                return ComplexNDArrayUtil.center(convolution, input.shape()).getReal();
            case VALID:
                return ComplexNDArrayUtil.center(convolution,ArrayUtil.toInts(ArrayUtil.toNDArray(input.shape()).sub(ArrayUtil.toNDArray(kernel.shape())).addi(1))).getReal();

        }


        return convolution.getReal();
    }




    /**
     * ND Convolution
     * @param input the input to applyTransformToOrigin
     * @param kernel the kernel to applyTransformToOrigin with
     * @param type the type of convolution
     * @return the convolution of the given input and kernel
     */
    public static IComplexNDArray convn(IComplexNDArray input,IComplexNDArray kernel,Type type) {

        if(kernel.isScalar() && input.isScalar())
            return kernel.mul(input);

        INDArray shape = ArrayUtil.toNDArray(input.shape()).add(ArrayUtil.toNDArray(kernel.shape())).subi(1);
        int[] intShape = ArrayUtil.toInts(shape);
        int[] axes = ArrayUtil.range(0, intShape.length);

        IComplexNDArray ret = FFT.rawifftn(FFT.rawfftn(input, intShape, axes).muli(FFT.rawfftn(kernel, intShape, axes)), intShape, axes);


        switch(type) {
            case FULL:
                 return ret;
            case SAME:
                return ComplexNDArrayUtil.center(ret,input.shape());
            case VALID:
                return ComplexNDArrayUtil.center(ret,ArrayUtil.toInts(ArrayUtil.toNDArray(input.shape()).sub(ArrayUtil.toNDArray(kernel.shape())).addi(1)));

        }

        return ret;
    }






















}
