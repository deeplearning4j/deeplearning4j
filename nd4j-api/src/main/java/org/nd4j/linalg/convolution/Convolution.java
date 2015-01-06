package org.nd4j.linalg.convolution;


import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.FFT;

import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.ComplexNDArrayUtil;
import org.nd4j.linalg.util.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;


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
     * 2d convolution (aka the last 2 dimensions
     * @param input the input to transform
     * @param kernel the kernel to convolve with
     * @param type
     * @return
     */
    public static INDArray conv2d(INDArray input,INDArray kernel,Type type) {
        return Nd4j.getConvolution().conv2d(input,kernel,type);
    }


    public static INDArray conv2d(IComplexNDArray input,IComplexNDArray kernel,Type type) {
        return Nd4j.getConvolution().conv2d(input,kernel,type);
    }

    /**
     * ND Convolution
     * @param input the input to transform
     * @param kernel the kernel to transform with
     * @param type the type of convolution
     * @param axes  the axes to do the convolution along
     * @return the convolution of the given input and kernel
     */
    public static INDArray convn(INDArray input,INDArray kernel,Type type,int[] axes) {
        return Nd4j.getConvolution().convn(input,kernel,type,axes);
    }




    /**
     * ND Convolution
     * @param input the input to transform
     * @param kernel the kernel to transform with
     * @param type the type of convolution
     * @param axes the axes to do the convolution along
     * @return the convolution of the given input and kernel
     */
    public static IComplexNDArray convn(IComplexNDArray input,IComplexNDArray kernel,Type type,int[] axes) {
        return Nd4j.getConvolution().convn(input,kernel,type,axes);
    }


    /**
     * ND Convolution
     * @param input the input to applyTransformToOrigin
     * @param kernel the kernel to applyTransformToOrigin with
     * @param type the type of convolution
     * @return the convolution of the given input and kernel
     */
    public static INDArray convn(INDArray input,INDArray kernel,Type type) {
        return Nd4j.getConvolution().convn(input,kernel,type);
    }

    /**
     * ND Convolution
     * @param input the input to applyTransformToOrigin
     * @param kernel the kernel to applyTransformToOrigin with
     * @param type the type of convolution
     * @return the convolution of the given input and kernel
     */
    public static IComplexNDArray convn(IComplexNDArray input,IComplexNDArray kernel,Type type) {
        return Nd4j.getConvolution().convn(input,kernel,type);
    }

}
