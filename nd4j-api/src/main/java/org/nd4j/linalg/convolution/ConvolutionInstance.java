package org.nd4j.linalg.convolution;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Convolution instance. Implementations of convolution algorithms
 *
 * @author Adam Gibson
 */
public interface ConvolutionInstance {
    /**
     * 2d convolution (aka the last 2 dimensions
     * @param input the input to transform
     * @param kernel the kernel to convolve with
     * @param type
     * @return
     */
    public INDArray conv2d(INDArray input,INDArray kernel,Convolution.Type type);


    public INDArray conv2d(IComplexNDArray input,IComplexNDArray kernel,Convolution.Type type);

    /**
     * ND Convolution
     * @param input the input to transform
     * @param kernel the kernel to transform with
     * @param type the type of convolution
     * @param axes  the axes to do the convolution along
     * @return the convolution of the given input and kernel
     */
    public INDArray convn(INDArray input,INDArray kernel,Convolution.Type type,int[] axes);



    /**
     * ND Convolution
     * @param input the input to transform
     * @param kernel the kernel to transform with
     * @param type the type of convolution
     * @param axes the axes to do the convolution along
     * @return the convolution of the given input and kernel
     */
    public IComplexNDArray convn(IComplexNDArray input,IComplexNDArray kernel,Convolution.Type type,int[] axes);

    /**
     * ND Convolution
     * @param input the input to applyTransformToOrigin
     * @param kernel the kernel to applyTransformToOrigin with
     * @param type the type of convolution
     * @return the convolution of the given input and kernel
     */
    public INDArray convn(INDArray input,INDArray kernel,Convolution.Type type);

    /**
     * ND Convolution
     * @param input the input to applyTransformToOrigin
     * @param kernel the kernel to applyTransformToOrigin with
     * @param type the type of convolution
     * @return the convolution of the given input and kernel
     */
    public IComplexNDArray convn(IComplexNDArray input,IComplexNDArray kernel,Convolution.Type type);
}
