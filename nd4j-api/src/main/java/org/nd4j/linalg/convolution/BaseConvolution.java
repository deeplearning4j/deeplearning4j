package org.nd4j.linalg.convolution;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;


/**
 * Base convolution implementation
 * @author Adam Gibson
 */
public abstract  class BaseConvolution implements ConvolutionInstance {
    /**
     * 2d convolution (aka the last 2 dimensions
     * @param input the input to transform
     * @param kernel the kernel to convolve with
     * @param type
     * @return
     */
    @Override
    public INDArray conv2d(INDArray input,INDArray kernel,Convolution.Type type) {
        int[] shape = input.shape().length < 2 ? ArrayUtil.range(0, 1) : ArrayUtil.range(input.shape().length - 2,input.shape().length);
        return convn(input,kernel,type,shape);
    }

    @Override
    public INDArray conv2d(IComplexNDArray input,IComplexNDArray kernel,Convolution.Type type) {
        int[] shape = input.shape().length < 2 ? ArrayUtil.range(0,1) : ArrayUtil.range(input.shape().length - 2,input.shape().length);
        return convn(input,kernel,type,shape);
    }






    /**
     * ND Convolution
     * @param input the input to applyTransformToOrigin
     * @param kernel the kernel to applyTransformToOrigin with
     * @param type the type of convolution
     * @return the convolution of the given input and kernel
     */
    @Override
    public INDArray convn(INDArray input,INDArray kernel,Convolution.Type type) {
        return convn(input,kernel,type,ArrayUtil.range(0,input.shape().length));
    }

    /**
     * ND Convolution
     * @param input the input to applyTransformToOrigin
     * @param kernel the kernel to applyTransformToOrigin with
     * @param type the type of convolution
     * @return the convolution of the given input and kernel
     */
    @Override
    public IComplexNDArray convn(IComplexNDArray input,IComplexNDArray kernel,Convolution.Type type) {
        return convn(input,kernel,type,ArrayUtil.range(0,input.shape().length));
    }
    
}
