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


import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

import static org.deeplearning4j.util.ConvolutionUtils.effectiveKernelSize;

/**
 * Shape utilities for 3D convolution layers
 *
 * @author Max Pumperla
 */
public class Convolution3DUtils {

    private static final int[] ONES = new int[]{1, 1};


    private Convolution3DUtils() {
    }

    /**
     * Get the output size (depth/height/width) for the given input data and CNN3D configuration
     *
     * @param inputData       Input data
     * @param kernel          Kernel size (depth/height/width)
     * @param strides         Strides (depth/height/width)
     * @param padding         Padding (depth/height/width)
     * @param convolutionMode Convolution mode (Same, Strict, Truncate)
     * @param dilation        Kernel dilation (depth/height/width)
     * @return Output size: int[3] with output depth/height/width
     */
    public static int[] get3DOutputSize(INDArray inputData, int[] kernel, int[] strides, int[] padding,
                                        ConvolutionMode convolutionMode, int[] dilation, boolean isNCDHW) {

        // NCDHW vs. NDHWC
        int inD = (int) (isNCDHW ? inputData.size(2) : inputData.size(1));
        int inH = (int) (isNCDHW ? inputData.size(3) : inputData.size(2));
        int inW = (int) (isNCDHW ? inputData.size(4) : inputData.size(3));

        int[] eKernel = effectiveKernelSize(kernel, dilation);
        boolean atrous = (eKernel == kernel);

        // FIXME: int cast
        val inShape = new int[]{inD, inH, inW};
        validateShapes(ArrayUtil.toInts(inputData.shape()), eKernel, strides, padding, convolutionMode, dilation, inShape, atrous);

        if (convolutionMode == ConvolutionMode.Same) {
            int outD = (int) Math.ceil(inD / ((double) strides[0]));
            int outH = (int) Math.ceil(inH / ((double) strides[1]));
            int outW = (int) Math.ceil(inW / ((double) strides[2]));

            return new int[]{outD, outH, outW};
        }

        int outD = (inD - eKernel[0] + 2 * padding[0]) / strides[0] + 1;
        int outH = (inH - eKernel[1] + 2 * padding[1]) / strides[1] + 1;
        int outW = (inW - eKernel[2] + 2 * padding[2]) / strides[2] + 1;

        return new int[]{outD, outH, outW};
    }


    private static void validateShapes(int[] inputDataShape, int[] eKernel, int[] strides, int[] padding,
                                      ConvolutionMode convolutionMode, int[] dilation, int[] inShape,
                                      boolean atrous) {

        String[] dims = new String[]{"depth", "height", "width"};

        if (convolutionMode != ConvolutionMode.Same) {
            for (int i = 0; i < 3; i++) {
                if ((eKernel[i] <= 0 || eKernel[i] > inShape[i] + 2 * padding[i])) {
                    StringBuilder sb = new StringBuilder();
                    sb.append("Invalid input data or configuration: ");
                    if (atrous) sb.append("effective ");
                    sb.append("kernel ").append(dims[i]).append(" and input ")
                            .append(dims[i]).append(" must satisfy 0 < ");
                    if (atrous) sb.append("effective ");
                    sb.append("kernel ").append(dims[i]).append(" <= input ")
                            .append(dims[i]).append(" + 2 * padding ").append(dims[i]).append(". \nGot ");
                    if (atrous) sb.append("effective ");
                    sb.append("kernel = ").append(eKernel[i]).append(", input ").append(dims[i]).append(" = ")
                            .append(inShape[i]).append(" and padding ").append(dims[i]).append(" = ")
                            .append(padding[i]).append(" which do not satisfy 0 < ")
                            .append(eKernel[i]).append(" <= ").append(inShape[i] + 2 * padding[i])
                            .append(getCommonErrorMsg(inputDataShape, eKernel, strides, padding, dilation));

                    throw new DL4JInvalidInputException(sb.toString());
                }
            }
        }
        if (convolutionMode == ConvolutionMode.Strict) {
            for (int j = 0; j < 3; j++) {
                if ((inShape[j] - eKernel[0] + 2 * padding[0]) % strides[0] != 0) {
                    double d = (inShape[j] - eKernel[0] + 2 * padding[0]) / ((double) strides[0]) + 1.0;
                    String str = String.format("%.2f", d);
                    int truncated = (int) d;
                    int sameSize = (int) Math.ceil(inShape[j] / ((double) strides[0]));

                    StringBuilder sb = new StringBuilder();
                    sb.append("Invalid input data or configuration: Combination of kernel size, stride and padding ")
                            .append("are not valid for given input height, using ConvolutionMode.Strict\n")
                            .append("ConvolutionMode.Strict requires: output height = (input height - kernelSize + ")
                            .append( "2*padding)/stride + 1 to be an integer. Got: (")
                            .append(inShape[j]).append(" - ").append(eKernel[0]).append(" + 2*")
                            .append(padding[0]).append(")/").append(strides[0]).append(" + 1 = ")
                            .append(str).append("\n")
                            .append("See \"Constraints on strides\" at http://cs231n.github.io/convolutional-networks/ ")
                            .append("and ConvolutionType enumeration Javadoc.\n")
                            .append("To truncate/crop the input, such that output height = floor(").append(str)
                            .append(") = ").append(truncated).append(", use ConvolutionType.Truncate.\n")
                            .append("Alternatively use ConvolutionType.Same, which will use padding to give ")
                            .append("an output height of ceil(")
                            .append(inShape[j]).append("/").append(strides[0]).append(")=").append(sameSize)
                            .append(getCommonErrorMsg(inputDataShape, eKernel, strides, padding, dilation));

                    throw new DL4JInvalidConfigException(sb.toString());
                }
            }
        }
    }


    private static String getCommonErrorMsg(int[] inputDatashape, int[] kernel, int[] strides, int[] padding, int[] dilation) {
        String s = "\nInput size: [numExamples, inputDepth, inputHeight, inputWidth]=" + Arrays.toString(inputDatashape)
                + ", inputKernel=" + Arrays.toString(kernel);
        if (dilation[0] != 1 || dilation[1] != 1) {
            int[] effectiveKernel = effectiveKernelSize(kernel, dilation);
            s += ", effectiveKernelGivenDilation=" + Arrays.toString(effectiveKernel);
        }
        return s + ", strides=" + Arrays.toString(strides) + ", padding="
                + Arrays.toString(padding) + ", dilation=" + Arrays.toString(dilation);
    }

    /**
     * Get top and left padding for same mode only for 3d convolutions
     *
     * @param outSize
     * @param inSize
     * @param kernel
     * @param strides
     * @return
     */
    public static int[] get3DSameModeTopLeftPadding(int[] outSize, int[] inSize, int[] kernel, int[] strides,
                                                    int[] dilation) {
        int[] eKernel = effectiveKernelSize(kernel, dilation);
        int[] outPad = new int[3];
        outPad[0] = ((outSize[0] - 1) * strides[0] + eKernel[0] - inSize[0]) / 2;
        outPad[1] = ((outSize[1] - 1) * strides[1] + eKernel[1] - inSize[1]) / 2;
        outPad[2] = ((outSize[2] - 1) * strides[2] + eKernel[2] - inSize[2]) / 2;
        return outPad;
    }

    /**
     * Perform validation on the CNN3D layer kernel/stride/padding. Expect 3d int[], with values > 0 for kernel size and
     * stride, and values >= 0 for padding.
     *
     * @param kernelSize Kernel size array to check
     * @param stride     Stride array to check
     * @param padding    Padding array to check
     */
    public static void validateCnn3DKernelStridePadding(int[] kernelSize, int[] stride, int[] padding) {
        if (kernelSize == null || kernelSize.length != 3) {
            throw new IllegalStateException("Invalid kernel size: expected int[] of length 3, got "
                    + (kernelSize == null ? null : Arrays.toString(kernelSize)));
        }

        if (stride == null || stride.length != 3) {
            throw new IllegalStateException("Invalid stride configuration: expected int[] of length 3, got "
                    + (stride == null ? null : Arrays.toString(stride)));
        }

        if (padding == null || padding.length != 3) {
            throw new IllegalStateException("Invalid padding configuration: expected int[] of length 3, got "
                    + (padding == null ? null : Arrays.toString(padding)));
        }

        if (kernelSize[0] <= 0 || kernelSize[1] <= 0 || kernelSize[2] <= 0) {
            throw new IllegalStateException(
                    "Invalid kernel size: values must be positive (> 0) for all dimensions. Got: "
                            + Arrays.toString(kernelSize));
        }

        if (stride[0] <= 0 || stride[1] <= 0 || stride[2] <= 0) {
            throw new IllegalStateException(
                    "Invalid stride configuration: values must be positive (> 0) for all dimensions. Got: "
                            + Arrays.toString(stride));
        }

        if (padding[0] < 0 || padding[1] < 0 || padding[2] < 0) {
            throw new IllegalStateException(
                    "Invalid padding configuration: values must be >= 0 for all dimensions. Got: "
                            + Arrays.toString(padding));
        }
    }

}
