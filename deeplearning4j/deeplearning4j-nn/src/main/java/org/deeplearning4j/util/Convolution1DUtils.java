/*
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.util;


import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Shape utilities for 1D convolution layers
 *
 * @author Max Pumperla
 */
public class Convolution1DUtils {

    private static final int ONE = 1;


    private Convolution1DUtils() {
    }

    public static int getOutputSize(INDArray inputData, int kernel, int strides, int padding,
                                    ConvolutionMode convolutionMode) {
        return getOutputSize(inputData, kernel, strides, padding, convolutionMode, ONE);
    }

    /**
     * Get the output size (height) for the given input data and CNN1D configuration
     *
     * @param inputData       Input data
     * @param kernel          Kernel size
     * @param strides         Stride
     * @param padding         Padding
     * @param convolutionMode Convolution mode (Same, Strict, Truncate)
     * @param dilation        Kernel dilation
     * @return Output size (width)
     */
    public static int getOutputSize(INDArray inputData, int kernel, int strides, int padding,
                                    ConvolutionMode convolutionMode, int dilation) {
        // FIXME: int cast
        int inH = (int) inputData.size(2);
        int eKernel = effectiveKernelSize(kernel, dilation);
        boolean atrous = (eKernel == kernel);
        validateShapes(inputData, eKernel, strides, padding, convolutionMode, dilation, inH, atrous);

        if (convolutionMode == ConvolutionMode.Same) {
            int outH = (int) Math.ceil(inH / ((double) strides));
            return outH;
        }

        int outH = (inH - eKernel + 2 * padding) / strides + 1;
        return outH;
    }

    public static void validateShapes(INDArray inputData, int eKernel, int strides, int padding,
                                      ConvolutionMode convolutionMode, int dilation, int inShape,
                                      boolean atrous) {

        int inH = inShape;

        if (convolutionMode != ConvolutionMode.Same && (eKernel <= 0 || eKernel > inH + 2 * padding)) {
            StringBuilder sb = new StringBuilder();
            sb.append("Invalid input data or configuration: ");
            if (atrous) sb.append("effective ");
            sb.append("kernel height and input height must satisfy 0 < ");
            if (atrous) sb.append("effective ");
            sb.append("kernel height <= input height + 2 * padding height. \nGot ");
            if (atrous) sb.append("effective ");
            sb.append("kernel height = ").append(eKernel).append(", input height = ").append(inH)
                    .append(" and padding height = ").append(padding).append(" which do not satisfy 0 < ")
                    .append(eKernel).append(" <= ").append(inH + 2 * padding)
                    .append(getCommonErrorMsg(inputData, eKernel, strides, padding, dilation));

            throw new DL4JInvalidInputException(sb.toString());
        }


        if (convolutionMode == ConvolutionMode.Strict) {
            if ((inH - eKernel + 2 * padding) % strides != 0) {
                double d = (inH - eKernel + 2 * padding) / ((double) strides) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int) d;
                int sameSize = (int) Math.ceil(inH / ((double) strides));

                StringBuilder sb = new StringBuilder();
                sb.append("Invalid input data or configuration: Combination of kernel size, " +
                        "stride and padding are not " +
                        "valid for given input height, using ConvolutionMode.Strict\n")
                        .append("ConvolutionMode.Strict requires: output height = (input height - kernelSize + " +
                                "2*padding)/stride + 1 to be an integer. Got: (")
                        .append(inH).append(" - ").append(eKernel).append(" + 2*").append(padding).append(")/")
                        .append(strides).append(" + 1 = ")
                        .append(str).append("\n").append("See \"Constraints on strides\" at http://cs231n.github." +
                        "io/convolutional-networks/ and ConvolutionType enumeration Javadoc.\n")
                        .append("To truncate/crop the input, such that output height = floor(")
                        .append(str).append(") = ")
                        .append(truncated).append(", use ConvolutionType.Truncate.\n")
                        .append("Alternatively use ConvolutionType.Same, which will use padding to give an " +
                                "output height of ceil(")
                        .append(inH).append("/").append(strides).append(")=").append(sameSize)
                        .append(getCommonErrorMsg(inputData, eKernel, strides, padding, dilation));

                throw new DL4JInvalidConfigException(sb.toString());
            }
        }

    }

    public static int effectiveKernelSize(int kernel, int dilation) {
        //Determine the effective kernel size, accounting for dilation
        //http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions
        if (dilation == 1) {
            return kernel;
        } else {
            return kernel + (kernel - 1) * (dilation - 1);
        }
    }

    private static String getCommonErrorMsg(INDArray inputData, int kernel, int strides, int padding, int dilation) {
        String s = "\nInput size: [numExamples,inputDepth,inputHeight,inputWidth]=" + Arrays.toString(inputData.shape())
                + ", inputKernel=" + kernel;
        if (dilation != 1) {
            int effectiveKernel = effectiveKernelSize(kernel, dilation);
            s += ", effectiveKernelGivenDilation=" + effectiveKernel;
        }
        return s + ", stride=" + strides + ", padding=" + padding + ", dilation=" + dilation;
    }


    /**
     * Check that the convolution mode is consistent with the padding specification
     */
    public static void validateConvolutionModePadding(ConvolutionMode mode, int padding) {
        if (mode == ConvolutionMode.Same) {
            boolean nullPadding = true;
            if (padding != 0) nullPadding = false;
            if (!nullPadding)
                throw new IllegalArgumentException("Padding cannot be used when using the `same' convolution mode");

        }
    }

    /**
     * Get top padding for same mode only.
     *
     * @param outSize  Output size (length 2 array, height dimension first)
     * @param inSize   Input size (length 2 array, height dimension first)
     * @param kernel   Kernel size (length 2 array, height dimension first)
     * @param strides  Strides  (length 2 array, height dimension first)
     * @param dilation Dilation (length 2 array, height dimension first)
     * @return Top left padding (length 2 array, height dimension first)
     */
    public static int getSameModeTopLeftPadding(int outSize, int inSize, int kernel, int strides, int dilation) {
        int eKernel = effectiveKernelSize(kernel, dilation);
        //Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
        int outPad = ((outSize - 1) * strides + eKernel - inSize) / 2;
        Preconditions.checkState(outPad >= 0, "Invalid padding values calculated: %s - " +
                        "layer configuration is invalid? Input size %s, output size %s, kernel %s, " +
                        "strides %s, dilation %s", outPad, inSize, outSize, kernel, strides, dilation);
        return outPad;
    }

    /**
     * Perform validation on the CNN layer kernel/stride/padding. Expect int, with values > 0 for kernel size and
     * stride, and values >= 0 for padding.
     *
     * @param kernel  Kernel size  to check
     * @param stride  Stride to check
     * @param padding Padding to check
     */
    public static void validateCnn1DKernelStridePadding(int kernel, int stride, int padding) {

        if (kernel <= 0) {
            throw new IllegalStateException("Invalid kernel size: value must be positive (> 0). Got: " + kernel);
        }
        if (stride <= 0) {
            throw new IllegalStateException("Invalid kernel size: value must be positive (> 0). Got: " + stride);

        }
        if (padding < 0) {
            throw new IllegalStateException("Invalid kernel size: value must be positive (> 0). Got: " + padding);
        }
    }


}
