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
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskFactoryProvider;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Convolution is the
 * code for applying
 * the convolution operator.
 *
 *
 * @author Adam Gibson
 */
public class Convolution {

    private static Logger log = LoggerFactory.getLogger(Convolution.class);

    public enum Type {
        FULL, VALID, SAME
    }


    /**
     * Default no-arg constructor.
     */
    private Convolution() {
    }

    /**
     *
     * @param col
     * @param stride
     * @param padding
     * @param height
     * @param width
     * @return
     */
    public static INDArray col2im(INDArray col, int[] stride, int[] padding, int height, int width) {
        return col2im(col, stride[0], stride[1], padding[0], padding[1], height, width);
    }

    /**
     * Rearrange matrix
     * columns into blocks

     * @param col the column
     *            transposed image to convert
     * @param sy stride y
     * @param sx stride x
     * @param ph padding height
     * @param pw padding width
     * @param h height
     * @param w width
     * @return
     */
    public static INDArray col2im(INDArray col, int sy, int sx, int ph, int pw, int h, int w) {
        Task<INDArray> task = Nd4j.getTaskFactory().getCol2ImTask(col,sy,sx,ph,pw,h,w);
        return task.invokeBlocking();
    }

    /**
     *
     * @param img
     * @param kernel
     * @param stride
     * @param padding
     * @return
     */
    public static INDArray im2col(INDArray img, int[] kernel, int[] stride, int[] padding) {
        return im2col(img, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], 0, false);
    }

    /**
     * Implement column formatted images
     * @param img the image to process
     * @param kh the kernel height
     * @param kw the kernel width
     * @param sy the stride along y
     * @param sx the stride along x
     * @param ph the padding width
     * @param pw the padding height
     * @param coverAll whether to cover the whole image or not
     * @return the column formatted image
     *
     */
    public static INDArray im2col(INDArray img, int kh, int kw, int sy, int sx, int ph, int pw, boolean coverAll) {
        Task<INDArray> task = Nd4j.getTaskFactory().getIm2ColTask(img, kh, kw, sy, sx, ph, pw, coverAll);
        return task.invokeBlocking();
    }

    /**
     * Implement column formatted images
     * @param img the image to process
     * @param kh the kernel height
     * @param kw the kernel width
     * @param sy the stride along y
     * @param sx the stride along x
     * @param ph the padding width
     * @param pw the padding height
     * @param pval the padding value (not used)
     * @param coverAll whether to cover the whole image or not
     * @return the column formatted image
     *
     */
    public static INDArray im2col(INDArray img, int kh, int kw, int sy, int sx, int ph, int pw, int pval, boolean coverAll) {
        Task<INDArray> task = Nd4j.getTaskFactory().getIm2ColTask(img, kh, kw, sy, sx, ph, pw, coverAll);
        return task.invokeBlocking();
    }

    /**
     *
     * The out size for a convolution
     * @param size
     * @param k
     * @param s
     * @param p
     * @param coverAll
     * @return
     */
    public static int outSize(int size,int k,int s,int p, boolean coverAll) {
        if (coverAll)
            return (size + p * 2 - k + s - 1) / s + 1;
        else
            return (size + p * 2 - k) / s + 1;
    }


    /**
     * 2d convolution (aka the last 2 dimensions
     *
     * @param input  the input to op
     * @param kernel the kernel to convolve with
     * @param type
     * @return
     */
    public static INDArray conv2d(INDArray input, INDArray kernel, Type type) {
        return Nd4j.getConvolution().conv2d(input, kernel, type);
    }

    /**
     *
     * @param input
     * @param kernel
     * @param type
     * @return
     */
    public static INDArray conv2d(IComplexNDArray input, IComplexNDArray kernel, Type type) {
        return Nd4j.getConvolution().conv2d(input, kernel, type);
    }

    /**
     * ND Convolution
     *
     * @param input  the input to op
     * @param kernel the kerrnel to op with
     * @param type   the type of convolution
     * @param axes   the axes to do the convolution along
     * @return the convolution of the given input and kernel
     */
    public static INDArray convn(INDArray input, INDArray kernel, Type type, int[] axes) {
        return Nd4j.getConvolution().convn(input, kernel, type, axes);
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
    public static IComplexNDArray convn(IComplexNDArray input, IComplexNDArray kernel, Type type, int[] axes) {
        return Nd4j.getConvolution().convn(input, kernel, type, axes);
    }

    /**
     * ND Convolution
     *
     * @param input  the input to op
     * @param kernel the kernel to op with
     * @param type   the type of convolution
     * @return the convolution of the given input and kernel
     */
    public static INDArray convn(INDArray input, INDArray kernel, Type type) {
        return Nd4j.getConvolution().convn(input, kernel, type);
    }

    /**
     * ND Convolution
     *
     * @param input  the input to op
     * @param kernel the kernel to op with
     * @param type   the type of convolution
     * @return the convolution of the given input and kernel
     */
    public static IComplexNDArray convn(IComplexNDArray input, IComplexNDArray kernel, Type type) {
        return Nd4j.getConvolution().convn(input, kernel, type);
    }


}
