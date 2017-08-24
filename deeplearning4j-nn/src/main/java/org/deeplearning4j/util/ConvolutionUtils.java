/*-
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
 */

package org.deeplearning4j.util;


import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Convolutional shape utilities
 *
 * @author Adam Gibson
 */
public class ConvolutionUtils {

    private static final int[] ONES = new int[]{1,1};


    private ConvolutionUtils() {}

    public static int[] getOutputSize(INDArray inputData, int[] kernel, int[] strides, int[] padding,
                                      ConvolutionMode convolutionMode) {
        return getOutputSize(inputData, kernel, strides, padding, convolutionMode, ONES);
    }

    /**
     * Get the output size (height/width) for the given inpud data and CNN configuration
     *
     * @param inputData    Input data
     * @param kernel       Kernel size (height/width)
     * @param strides      Strides (height/width)
     * @param padding      Padding (height/width)
     * @return             Output size: int[2] with output height/width
     */
    public static int[] getOutputSize(INDArray inputData, int[] kernel, int[] strides, int[] padding,
                    ConvolutionMode convolutionMode, int[] dilation) {
        int inH = inputData.size(2);
        int inW = inputData.size(3);

        //Determine the effective kernel size, accounting for dilation
        //http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions
        int[] eKernel = effectiveKernelSize(kernel, dilation);
        boolean atrous = (eKernel == kernel);

        if (convolutionMode != ConvolutionMode.Same && (eKernel[0] <= 0 || eKernel[0] > inH + 2 * padding[0])) {
            StringBuilder sb = new StringBuilder();
            sb.append("Invalid input data or configuration: ");
            if(atrous) sb.append("effective ");
            sb.append("kernel height and input height must satisfy 0 < ");
            if(atrous) sb.append("effective ");
            sb.append("kernel height <= input height + 2 * padding height. \nGot ");
            if(atrous) sb.append("effective ");
            sb.append("kernel height = ").append(eKernel[0]).append(", input height = ").append(inH)
                    .append(" and padding height = ").append(padding[0]).append(" which do not satisfy 0 < ")
                    .append(eKernel[0]).append(" <= ").append(inH + 2 * padding[0])
                    .append(getCommonErrorMsg(inputData, eKernel, strides, padding, dilation));

            throw new DL4JInvalidInputException(sb.toString());
        }

        if (convolutionMode != ConvolutionMode.Same && (eKernel[1] <= 0 || eKernel[1] > inW + 2 * padding[1])) {

            StringBuilder sb = new StringBuilder();
            sb.append("Invalid input data or configuration: ");
            if(atrous) sb.append("effective ");
            sb.append("kernel width and input width must satisfy  0 < kernel width <= input width + 2 * padding width. ");
            sb.append("\nGot ");
            if(atrous) sb.append("effective ");
            sb.append("kernel width = ").append(eKernel[1]).append(", input width = ").append(inW)
                    .append(" and padding width = ").append(padding[1]).append(" which do not satisfy 0 < ")
                    .append(eKernel[1]).append(" <= ").append(inW + 2 * padding[1])
                    .append("\nInput size: [numExamples,inputDepth,inputHeight,inputWidth]=")
                    .append(Arrays.toString(inputData.shape()))
                    .append(getCommonErrorMsg(inputData, eKernel, strides, padding, dilation));

            throw new DL4JInvalidInputException(sb.toString());
        }

        if (convolutionMode == ConvolutionMode.Strict) {
            if ((inH - eKernel[0] + 2 * padding[0]) % strides[0] != 0) {
                double d = (inH - eKernel[0] + 2 * padding[0]) / ((double) strides[0]) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int) d;
                int sameSize = (int) Math.ceil(inH / ((double) strides[0]));

                StringBuilder sb = new StringBuilder();
                sb.append("Invalid input data or configuration: Combination of kernel size, stride and padding are not valid for given input height, using ConvolutionMode.Strict\n")
                        .append("ConvolutionMode.Strict requires: output height = (input height - kernelSize + 2*padding)/stride + 1 to be an integer. Got: (")
                        .append(inH).append(" - ").append(eKernel[0]).append(" + 2*").append(padding[0]).append(")/").append(strides[0]).append(" + 1 = ")
                        .append(str).append("\n").append("See \"Constraints on strides\" at http://cs231n.github.io/convolutional-networks/ and ConvolutionType enumeration Javadoc.\n")
                        .append("To truncate/crop the input, such that output height = floor(").append(str).append(") = ")
                        .append(truncated).append(", use ConvolutionType.Truncate.\n")
                        .append("Alternatively use ConvolutionType.Same, which will use padding to give an output height of ceil(")
                        .append(inH).append("/").append(strides[0]).append(")=").append(sameSize).append(getCommonErrorMsg(inputData, eKernel, strides, padding, dilation));

                throw new DL4JInvalidConfigException(sb.toString());
            }

            if ((inW - eKernel[1] + 2 * padding[1]) % strides[1] != 0) {
                double d = (inW - eKernel[1] + 2 * padding[1]) / ((double) strides[1]) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int) d;
                int sameSize = (int) Math.ceil(inW / ((double) strides[1]));
                StringBuilder sb = new StringBuilder();
                sb.append("Invalid input data or configuration: Combination of kernel size, stride and padding are not valid for given input width, using ConvolutionMode.Strict\n")
                        .append("ConvolutionMode.Strict requires: output width = (input - kernelSize + 2*padding)/stride + 1 to be an integer. Got: (")
                        .append(inW).append(" - ").append(eKernel[1]).append(" + 2*").append(padding[1])
                        .append(")/").append(strides[1]).append(" + 1 = ").append(str).append("\n")
                        .append("See \"Constraints on strides\" at http://cs231n.github.io/convolutional-networks/ and ConvolutionType enumeration Javadoc.\n")
                        .append("To truncate/crop the input, such that output width = floor(").append(str).append(") = ")
                        .append(truncated).append(", use ConvolutionType.Truncate.\n")
                        .append("Alternatively use ConvolutionType.Same, which will use padding to give an output width of ceil(")
                        .append(inW).append("/").append(strides[1]).append(")=").append(sameSize)
                        .append(getCommonErrorMsg(inputData, eKernel, strides, padding, dilation));
                throw new DL4JInvalidConfigException(
                                sb.toString());
            }
        } else if (convolutionMode == ConvolutionMode.Same) {
            //'Same' padding mode:
            //outH = ceil(inHeight / strideH)           decimal division
            //outW = ceil(inWidth / strideW)            decimal division

            //padHeightSum = ((outH - 1) * strideH + kH - inHeight)
            //padTop = padHeightSum / 2                 integer division
            //padBottom = padHeghtSum - padTop

            //padWidthSum = ((outW - 1) * strideW + kW - inWidth)
            //padLeft = padWidthSum / 2                 integer division
            //padRight = padWidthSum - padLeft

            int outH = (int) Math.ceil(inH / ((double) strides[0]));
            int outW = (int) Math.ceil(inW / ((double) strides[1]));

            return new int[] {outH, outW};
        }

        int hOut = (inH - eKernel[0] + 2 * padding[0]) / strides[0] + 1;
        int wOut = (inW - eKernel[1] + 2 * padding[1]) / strides[1] + 1;

        return new int[] {hOut, wOut};
    }

    public static int[] effectiveKernelSize(int[] kernel, int[] dilation){
        //Determine the effective kernel size, accounting for dilation
        //http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions
        if(dilation[0] == 1 && dilation[1] == 1){
            return kernel;
        } else {
            return new int[]{ kernel[0] + (kernel[0]-1)*(dilation[0]-1), kernel[1] + (kernel[1]-1)*(dilation[1]-1)};
        }
    }

    private static String getCommonErrorMsg(INDArray inputData, int[] kernel, int[] strides, int[] padding, int[] dilation) {
        String s = "\nInput size: [numExamples,inputDepth,inputHeight,inputWidth]=" + Arrays.toString(inputData.shape())
                        + ", inputKernel=" + Arrays.toString(kernel);
        if(dilation[0] != 1 || dilation[1] != 1){
            int[] effectiveKernel = effectiveKernelSize(kernel, dilation);
            s += ", effectiveKernelGivenDilation=" + Arrays.toString(effectiveKernel);
        }
        return s + ", strides=" + Arrays.toString(strides) + ", padding="
                        + Arrays.toString(padding) + ", dilation=" + Arrays.toString(dilation);
    }

    /**
     * Get top and left padding for same mode only.
     *
     * @param outSize
     * @param inSize
     * @param kernel
     * @param strides
     * @return
     */
    public static int[] getSameModeTopLeftPadding(int[] outSize, int[] inSize, int[] kernel, int[] strides, int[] dilation) {
        int[] eKernel = effectiveKernelSize(kernel, dilation);
        int[] outPad = new int[2];
        outPad[0] = ((outSize[0] - 1) * strides[0] + eKernel[0] - inSize[0]) / 2; //Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
        outPad[1] = ((outSize[1] - 1) * strides[1] + eKernel[1] - inSize[1]) / 2; //As above
        return outPad;
    }

    /**
     * Get bottom and right padding for same mode only.
     *
     * @param outSize
     * @param inSize
     * @param kernel
     * @param strides
     * @return
     */
    public static int[] getSameModeBottomRightPadding(int[] outSize, int[] inSize, int[] kernel, int[] strides, int[] dilation) {
        int[] eKernel = effectiveKernelSize(kernel, dilation);
        int[] outPad = new int[2];
        outPad[0] = ((outSize[0] - 1) * strides[0] + eKernel[0] - inSize[0] + 1) / 2; //Note that padTop is 1 smaller than this if bracketed term is not divisible by 2
        outPad[1] = ((outSize[1] - 1) * strides[1] + eKernel[1] - inSize[1] + 1) / 2; //As above
        return outPad;
    }

    /**
     * Get the height and width
     * from the configuration
     * @param conf the configuration to get height and width from
     * @return the configuration to get height and width from
     */
    public static int[] getHeightAndWidth(NeuralNetConfiguration conf) {
        return getHeightAndWidth(
                        ((org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer()).getKernelSize());
    }


    /**
     * @param conf the configuration to get
     *             the number of kernels from
     * @return the number of kernels/filters to apply
     */
    public static int numFeatureMap(NeuralNetConfiguration conf) {
        return ((org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer()).getNOut();
    }

    /**
     * Get the height and width
     * for an image
     * @param shape the shape of the image
     * @return the height and width for the image
     */
    public static int[] getHeightAndWidth(int[] shape) {
        if (shape.length < 2)
            throw new IllegalArgumentException("No width and height able to be found: array must be at least length 2");
        return new int[] {shape[shape.length - 1], shape[shape.length - 2]};
    }

    /**
     * Returns the number of
     * feature maps for a given shape (must be at least 3 dimensions
     * @param shape the shape to get the
     *              number of feature maps for
     * @return the number of feature maps
     * for a particular shape
     */
    public static int numChannels(int[] shape) {
        if (shape.length < 4)
            return 1;
        return shape[1];
    }

    /**
     * Perform validation on the CNN layer kernel/stride/padding. Expect 2d int[], with values > 0 for kernel size and
     * stride, and values >= 0 for padding.
     *
     * @param kernelSize Kernel size array to check
     * @param stride     Stride array to check
     * @param padding    Padding array to check
     */
    public static void validateCnnKernelStridePadding(int[] kernelSize, int[] stride, int[] padding) {
        if (kernelSize == null || kernelSize.length != 2) {
            throw new IllegalStateException("Invalid kernel size: expected int[] of length 2, got "
                            + (kernelSize == null ? null : Arrays.toString(kernelSize)));
        }

        if (stride == null || stride.length != 2) {
            throw new IllegalStateException("Invalid stride configuration: expected int[] of length 2, got "
                            + (stride == null ? null : Arrays.toString(stride)));
        }

        if (padding == null || padding.length != 2) {
            throw new IllegalStateException("Invalid padding configuration: expected int[] of length 2, got "
                            + (padding == null ? null : Arrays.toString(padding)));
        }

        if (kernelSize[0] <= 0 || kernelSize[1] <= 0) {
            throw new IllegalStateException(
                            "Invalid kernel size: values must be positive (> 0) for all dimensions. Got: "
                                            + Arrays.toString(kernelSize));
        }

        if (stride[0] <= 0 || stride[1] <= 0) {
            throw new IllegalStateException(
                            "Invalid stride configuration: values must be positive (> 0) for all dimensions. Got: "
                                            + Arrays.toString(stride));
        }

        if (padding[0] < 0 || padding[1] < 0) {
            throw new IllegalStateException(
                            "Invalid padding configuration: values must be >= 0 for all dimensions. Got: "
                                            + Arrays.toString(padding));
        }
    }
}
