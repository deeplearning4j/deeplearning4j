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
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Convolution is the code for applying the convolution operator.
 * http://www.inf.ufpr.br/danielw/pos/ci724/20102/HIPR2/flatjavasrc/Convolution.java
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
     * Implement column formatted images
     * @param img the image to process
     * @param kh the kernel height
     * @param kw the kernel width
     * @param sy the stride along y
     * @param sx the stride along x
     * @param ph the padding width
     * @param pw the padding height
     * @param pval the padding value
     * @param coverAll whether to cover the whole image or not
     * @return the column formatted image
     *
     */
    public static INDArray im2col(INDArray img,int kh, int kw, int sy, int sx, int ph, int pw, int pval, boolean coverAll) {
        int n = img.size(0);
        int c = img.size(1);
        int h = img.size(2);
        int w = img.size(3);
        int outWidth = outSize(h, kh, sy, ph, coverAll);
        int outHeight = outSize(w, kw, sx, pw, coverAll);
        INDArray padded = Nd4j.pad(img, new int[][]{
                {0, 0}
                , {0, 0}
                , {ph, ph + sy - 1}, {pw, pw + sx - 1}}, Nd4j.PadMode.CONSTANT);
        INDArray ret =   Nd4j.create(n, c, kh, kw, outHeight, outWidth);
        for(int i = 0; i < kh; i++) {
           int i_lim = i + sy * outHeight;
            for(int j = 0; j < kw; j++) {
                int  jLim = j + sx * outWidth;
                ret.put(new NDArrayIndex[]{NDArrayIndex.all()
                        ,NDArrayIndex.all()
                        ,new NDArrayIndex(i)
                        ,new NDArrayIndex(j),NDArrayIndex.all(),NDArrayIndex.all()}, padded.get(
                        NDArrayIndex.all()
                        , NDArrayIndex.all()
                        , NDArrayIndex.interval(i, i_lim)
                        , NDArrayIndex.interval(j,jLim),
                        NDArrayIndex.all(),NDArrayIndex.all()));
            }
        }
        return ret;
    }

    /**
     *
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
     * @param kernel the kernel to op with
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
