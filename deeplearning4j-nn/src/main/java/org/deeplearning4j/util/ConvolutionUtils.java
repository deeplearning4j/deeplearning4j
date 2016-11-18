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
 */

package org.deeplearning4j.util;


import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Convolutional shape utilities
 *
 * @author Adam Gibson
 */
public class ConvolutionUtils {


    private ConvolutionUtils() {
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
    public static int[] getOutputSize(INDArray inputData, int[] kernel, int[] strides, int[] padding, ConvolutionMode convolutionMode){
        int inH = inputData.size(2);
        int inW = inputData.size(3);

        if (convolutionMode != ConvolutionMode.Same && (kernel[0] <= 0 || kernel[0] > inH + 2 * padding[0])) {
            throw new DL4JInvalidInputException(
                    "Invalid input data or configuration: kernel height and input height must satisfy 0 < kernel height <= input height + 2 * padding height. "
                    + "\nGot kernel height = " + kernel[0] + ", input height = " + inH + " and padding height = " + padding[0] +
                            " which do not satisfy 0 < " + kernel[0] + " <= " + (inH + 2 * padding[0])
                    + getCommonErrorMsg(inputData, kernel, strides, padding));
        }

        if (convolutionMode != ConvolutionMode.Same && (kernel[1] <= 0 || kernel[1] > inW + 2 * padding[1])) {
            throw new DL4JInvalidInputException(
                    "Invalid input data or configuration: kernel width and input width must satisfy  0 < kernel width <= input width + 2 * padding width. "
                            + "\nGot kernel width = " + kernel[1] + ", input width = " + inW + " and padding width = " + padding[1]
                            + " which do not satisfy 0 < " + kernel[1] + " <= " + (inW + 2 * padding[1])
                            + "\nInput size: [numExamples,inputDepth,inputHeight,inputWidth]=" + Arrays.toString(inputData.shape())
                            + getCommonErrorMsg(inputData, kernel, strides, padding));
        }

        if(convolutionMode == ConvolutionMode.Strict) {
            if ((inH - kernel[0] + 2 * padding[0]) % strides[0] != 0) {
                double d = (inH - kernel[0] + 2 * padding[0]) / ((double) strides[0]) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int)d;
                int sameSize = (int)Math.ceil(inH / ((double)strides[0]));
                throw new DL4JInvalidConfigException(
                        "Invalid input data or configuration: Combination of kernel size, stride and padding are not valid for given input height, using ConvolutionMode.Strict\n"
                                + "ConvolutionMode.Strict requires: output height = (input height - kernelSize + 2*padding)/stride + 1 to be an integer. Got: ("
                                + inH + " - " + kernel[0] + " + 2*" + padding[0] + ")/" + strides[0] + " + 1 = " + str + "\n"
                                + "See \"Constraints on strides\" at http://cs231n.github.io/convolutional-networks/ and ConvolutionType enumeration Javadoc.\n"
                                + "To truncate/crop the input, such that output height = floor(" + str + ") = " + truncated + ", use ConvolutionType.Truncate.\n"
                                + "Alternatively use ConvolutionType.Same, which will use padding to give an output height of ceil(" + inH + "/" + strides[0] + ")=" + sameSize
                                + getCommonErrorMsg(inputData, kernel, strides, padding));
            }

            if ((inW - kernel[1] + 2 * padding[1]) % strides[1] != 0) {
                double d = (inW - kernel[1] + 2 * padding[1]) / ((double) strides[1]) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int)d;
                int sameSize = (int)Math.ceil(inW / ((double)strides[1]));
                throw new DL4JInvalidConfigException(
                        "Invalid input data or configuration: Combination of kernel size, stride and padding are not valid for given input width, using ConvolutionMode.Strict\n"
                                + "ConvolutionMode.Strict requires: output width = (input - kernelSize + 2*padding)/stride + 1 to be an integer. Got: ("
                                + inW + " - " + kernel[1] + " + 2*" + padding[1] + ")/" + strides[1] + " + 1 = " + str + "\n"
                                + "See \"Constraints on strides\" at http://cs231n.github.io/convolutional-networks/ and ConvolutionType enumeration Javadoc.\n"
                                + "To truncate/crop the input, such that output width = floor(" + str + ") = " + truncated + ", use ConvolutionType.Truncate.\n"
                                + "Alternatively use ConvolutionType.Same, which will use padding to give an output width of ceil(" + inW + "/" + strides[1] + ")=" + sameSize
                                + getCommonErrorMsg(inputData, kernel, strides, padding));
            }
        } else if(convolutionMode == ConvolutionMode.Same){
            //'Same' padding mode:
            //outH = ceil(inHeight / strideH)           decimal division
            //outW = ceil(inWidth / strideW)            decimal division

            //padHeightSum = ((outH - 1) * strideH + kH - inHeight)
            //padTop = padHeightSum / 2                 integer division
            //padBottom = padHeghtSum - padTop

            //padWidthSum = ((outW - 1) * strideW + kW - inWidth)
            //padLeft = padWidthSum / 2                 integer division
            //padRight = padWidthSum - padLeft

            int outH = (int)Math.ceil(inH / ((double)strides[0]));
            int outW = (int)Math.ceil(inW / ((double)strides[1]));

            return new int[]{outH, outW};
        }

        int hOut = (inH - kernel[0] + 2 * padding[0]) / strides[0] + 1;
        int wOut = (inW - kernel[1] + 2 * padding[1]) / strides[1] + 1;

        return new int[]{hOut, wOut};
    }

    private static String getCommonErrorMsg(INDArray inputData, int[] kernel, int[] strides, int[] padding){
        return "\nInput size: [numExamples,inputDepth,inputHeight,inputWidth]=" + Arrays.toString(inputData.shape())
                + ", kernel=" + Arrays.toString(kernel) + ", strides=" + Arrays.toString(strides) + ", padding=" + Arrays.toString(padding);
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
    public static int[] getSameModeTopLeftPadding(int[] outSize, int[] inSize, int[] kernel, int[] strides){
        int[] outPad = new int[2];
        outPad[0] = ((outSize[0]-1)*strides[0] + kernel[0] - inSize[0])/2;      //Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
        outPad[1] = ((outSize[1]-1)*strides[1] + kernel[1] - inSize[1])/2;      //As above
        return outPad;
    }

    /**
     * Get the height and width
     * from the configuration
     * @param conf the configuration to get height and width from
     * @return the configuration to get height and width from
     */
    public static int[] getHeightAndWidth(NeuralNetConfiguration conf) {
        return getHeightAndWidth(((org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf.getLayer()).getKernelSize());
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
        if(shape.length < 2)
            throw new IllegalArgumentException("No width and height able to be found: array must be at least length 2");
        return new int[] {shape[shape.length - 1],shape[shape.length - 2]};
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
        if(shape.length < 4)
            return 1;
        return shape[1];
    }



}
