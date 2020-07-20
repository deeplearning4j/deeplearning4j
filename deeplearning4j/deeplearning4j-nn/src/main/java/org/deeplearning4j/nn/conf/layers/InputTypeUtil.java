/*******************************************************************************
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

package org.deeplearning4j.nn.conf.layers;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;

import java.util.Arrays;

/**
 * Utilities for calculating input types
 *
 * @author Alex Black
 */
@Slf4j
public class InputTypeUtil {

    private InputTypeUtil(){ }

    public static InputType getOutputTypeDeconvLayer(InputType inputType, int[] kernelSize, int[] stride, int[] padding,
                    int[] dilation, ConvolutionMode convolutionMode, long outputDepth, long layerIdx, String layerName,
                    Class<?> layerClass) {
        InputType.InputTypeConvolutional i = (InputType.InputTypeConvolutional) inputType;

        val hIn = i.getHeight();
        val wIn = i.getWidth();

        int padH = (padding == null ? 0 : padding[0]); //May be null for ConvolutionMode.Same
        int padW = (padding == null ? 0 : padding[1]);
        int kH = kernelSize[0];
        int kW = kernelSize[1];
        if (dilation[0] != 1) {
            kH = kH + (kH - 1) * (dilation[0] - 1);
        }
        if (dilation[1] != 1) {
            kW = kW + (kW - 1) * (dilation[1] - 1);
        }

        int sH = stride[0];
        int sW = stride[1];

        if (sH <= 0 || sW <= 0) {
            throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, sH <= 0)
                            + " Invalid strides: strides must be > 0 (strideH = " + sH + ", strideW = " + sW + ")"
                            + "\n" + getConfigErrorCommonLastLine(inputType, kernelSize, stride, padding, outputDepth,
                                            convolutionMode));
        }

        if (convolutionMode == ConvolutionMode.Same) {
            long hOut = stride[0] * hIn;
            long wOut = stride[1] * wIn;
            return InputType.convolutional(hOut, wOut, outputDepth, i.getFormat());
        }

        long hOut = sH * (hIn - 1) + kH - 2 * padH;
        long wOut = sW * (wIn - 1) + kW - 2 * padW;

        return InputType.convolutional(hOut, wOut, outputDepth, i.getFormat());
    }

    public static InputType getOutputTypeDeconv3dLayer(InputType inputType, int[] kernelSize, int[] stride, int[] padding,
                                                       int[] dilation, ConvolutionMode convolutionMode, Convolution3D.DataFormat dataFormat,
                                                       long outputDepth, long layerIdx, String layerName, Class<?> layerClass) {
        InputType.InputTypeConvolutional3D i = (InputType.InputTypeConvolutional3D) inputType;

        long hIn = i.getHeight();
        long wIn = i.getWidth();
        long dIn = i.getDepth();


        int padH = (padding == null ? 0 : padding[0]); //May be null for ConvolutionMode.Same
        int padW = (padding == null ? 0 : padding[1]);
        int padD = (padding == null ? 0 : padding[2]);
        int kH = kernelSize[0];
        int kW = kernelSize[1];
        int kD = kernelSize[2];
        if (dilation[0] != 1) {
            kH = kH + (kH - 1) * (dilation[0] - 1);
        }
        if (dilation[1] != 1) {
            kW = kW + (kW - 1) * (dilation[1] - 1);
        }
        if (dilation[2] != 1) {
            kD = kD + (kD - 1) * (dilation[2] - 1);
        }

        int sH = stride[0];
        int sW = stride[1];
        int sD = stride[2];

        if (sH <= 0 || sW <= 0 || sD <= 0) {
            throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, sH <= 0)
                    + " Invalid strides: strides must be > 0 (strideH = " + sH + ", strideW = " + sW + ", stride = " + sD + ")"
                    + "\n" + getConfigErrorCommonLastLine(inputType, kernelSize, stride, padding, outputDepth,
                    convolutionMode));
        }

        if (convolutionMode == ConvolutionMode.Same) {
            long hOut = stride[0] * hIn;
            long wOut = stride[1] * wIn;
            long dOut = stride[2] * dIn;
            return InputType.convolutional3D(dataFormat, dOut, hOut, wOut, outputDepth);
        }

        long hOut = sH * (hIn - 1) + kH - 2 * padH;
        long wOut = sW * (wIn - 1) + kW - 2 * padW;
        long dOut = sD * (dIn - 1) + kD - 2 * padD;

        return InputType.convolutional3D(dataFormat, dOut, hOut, wOut, outputDepth);
    }

    public static InputType getOutputTypeCnn3DLayers(InputType inputType, Convolution3D.DataFormat dataFormat, int[] kernelSize, int[] stride, int[] padding,
                                                     int[] dilation, ConvolutionMode convolutionMode, long outputChannels, long layerIdx,
                                                     String layerName, Class<?> layerClass) {
        if (convolutionMode == null) {
            String name = layerName == null ? "(not named)" : layerName;
            throw new DL4JInvalidConfigException("Invalid configuration: convolution mode is null for layer (idx="
                            + layerIdx + ", name=" + name + ", type=" + layerClass.getName() + ")");
        }

        InputType.InputTypeConvolutional3D i = (InputType.InputTypeConvolutional3D) inputType;

        long inDepth = i.getDepth();
        long inHeight = i.getHeight();
        long inWidth = i.getWidth();

        int padD = (padding == null ? 0 : padding[0]);
        int padH = (padding == null ? 0 : padding[1]);
        int padW = (padding == null ? 0 : padding[2]);

        int kD = kernelSize[0];
        int kH = kernelSize[1];
        int kW = kernelSize[2];


        if (dilation[0] != 1) {
            //Use *effective* kernel size, accounting for dilation
            kD = kD + (kD - 1) * (dilation[0] - 1);
        }
        if (dilation[1] != 1) {
            kH = kH + (kH - 1) * (dilation[1] - 1);
        }
        if (dilation[2] != 1) {
            kW = kW + (kW - 1) * (dilation[2] - 1);
        }

        int sD = stride[0];
        int sH = stride[1];
        int sW = stride[1];

        if (sH <= 0 || sW <= 0 || sD <= 0) {
            throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, sH <= 0)
                            + " Invalid strides: strides must be > 0 (strideH = " + sH + ", strideW = " + sW
                            + ", strideD = " + sD + ")" + "\n" + getConfigErrorCommonLastLine(inputType, kernelSize,
                                            stride, padding, outputChannels, convolutionMode));
        }

        if (kH <= 0 || kH > inHeight + 2 * padH) {
            throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, true)
                            + " Invalid input configuration for kernel height. Require 0 < kH <= inHeight + 2*padH; got (kH="
                            + kH + ", inHeight=" + inHeight + ", padH=" + padH + ")\n" + getConfigErrorCommonLastLine(
                                            inputType, kernelSize, stride, padding, outputChannels, convolutionMode));
        }

        if (kW <= 0 || kW > inWidth + 2 * padW) {
            throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, false)
                            + " Invalid input configuration for kernel width. Require 0 < kW <= inWidth + 2*padW; got (kW="
                            + kW + ", inWidth=" + inWidth + ", padW=" + padW + ")\n" + getConfigErrorCommonLastLine(
                                            inputType, kernelSize, stride, padding, outputChannels, convolutionMode));
        }
        if (kD <= 0 || kD > inDepth + 2 * padD) {
            throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, false)
                            + " Invalid input configuration for kernel channels. Require 0 < kD <= inDepth + 2*padD; got (kD="
                            + kD + ", inDepth=" + inDepth + ", padD=" + padD + ")\n" + getConfigErrorCommonLastLine(
                                            inputType, kernelSize, stride, padding, outputChannels, convolutionMode));
        }

        //Strict mode: require exactly the right size...
        if (convolutionMode == ConvolutionMode.Strict) {
            if ((inHeight - kH + 2 * padH) % sH != 0) {
                double d = (inHeight - kH + 2 * padH) / ((double) sH) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int) d;
                int sameSize = (int) Math.ceil(inHeight / ((double) stride[0]));
                throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, true)
                                + "\nCombination of kernel size, stride and padding are not valid for given input height, using ConvolutionMode.Strict\n"
                                + "ConvolutionMode.Strict requires: output height = (input height - kernelSize + 2*padding)/stride + 1 in height dimension to be an integer. Got: ("
                                + inHeight + " - " + kH + " + 2*" + padH + ")/" + sH + " + 1 = " + str + "\n"
                                + "See ConvolutionType enumeration Javadoc and \"Constraints on strides\" at http://cs231n.github.io/convolutional-networks/\n"
                                + "To truncate/crop the input, such that output height = floor(" + str + ") = "
                                + truncated + ", use ConvolutionType.Truncate.\n"
                                + "Alternatively use ConvolutionType.Same, which will use padding to give an output height of ceil("
                                + inHeight + "/" + stride[0] + ")=" + sameSize + "\n"
                                + getConfigErrorCommonLastLine(inputType, kernelSize, stride, padding, outputChannels,
                                                convolutionMode));
            }

            if ((inWidth - kW + 2 * padW) % sW != 0) {
                double d = (inWidth - kW + 2 * padW) / ((double) sW) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int) d;
                int sameSize = (int) Math.ceil(inWidth / ((double) stride[1]));
                throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, false)
                                + "\nCombination of kernel size, stride and padding are not valid for given input width, using ConvolutionMode.Strict\n"
                                + "ConvolutionMode.Strict requires: output width = (input width - kernelSize + 2*padding)/stride + 1 in width dimension to be an integer. Got: ("
                                + inWidth + " - " + kW + " + 2*" + padW + ")/" + sW + " + 1 = " + str + "\n"
                                + "See \"Constraints on strides\" at http://cs231n.github.io/convolutional-networks/ and ConvolutionType enumeration Javadoc.\n"
                                + "To truncate/crop the input, such that output width = floor(" + str + ") = "
                                + truncated + ", use ConvolutionType.Truncate.\n"
                                + "Alternatively use ConvolutionType.Same, which will use padding to give an output width of ceil("
                                + inWidth + "/" + stride[1] + ")=" + sameSize + "\n"
                                + getConfigErrorCommonLastLine(inputType, kernelSize, stride, padding, outputChannels,
                                                convolutionMode));
            }

            if ((inDepth - kD + 2 * padD) % sD != 0) {
                double d = (inDepth - kD + 2 * padD) / ((double) sD) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int) d;
                int sameSize = (int) Math.ceil(inDepth / ((double) stride[2]));
                throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, false)
                                + "\nCombination of kernel size, stride and padding are not valid for given input width, using ConvolutionMode.Strict\n"
                                + "ConvolutionMode.Strict requires: output channels = (input channels - kernelSize + 2*padding)/stride + 1 in width dimension to be an integer. Got: ("
                                + inDepth + " - " + kD + " + 2*" + padD + ")/" + sD + " + 1 = " + str + "\n"
                                + "See \"Constraints on strides\" at http://cs231n.github.io/convolutional-networks/ and ConvolutionType enumeration Javadoc.\n"
                                + "To truncate/crop the input, such that output width = floor(" + str + ") = "
                                + truncated + ", use ConvolutionType.Truncate.\n"
                                + "Alternatively use ConvolutionType.Same, which will use padding to give an output width of ceil("
                                + inDepth + "/" + stride[2] + ")=" + sameSize + "\n"
                                + getConfigErrorCommonLastLine(inputType, kernelSize, stride, padding, outputChannels,
                                                convolutionMode));
            }
        } else if (convolutionMode == ConvolutionMode.Same) {

            int outD = (int) Math.ceil(inDepth / ((double) sD));
            int outH = (int) Math.ceil(inHeight / ((double) sH));
            int outW = (int) Math.ceil(inWidth / ((double) sW));

            return InputType.convolutional3D(dataFormat, outD, outH, outW, outputChannels);
        }

        long dOut = (inDepth - kD + 2 * padD) / sD + 1;
        long hOut = (inHeight - kH + 2 * padH) / sH + 1;
        long wOut = (inWidth - kW + 2 * padW) / sW + 1;
        return InputType.convolutional3D(dOut, hOut, wOut, outputChannels);
    }


    public static InputType getOutputTypeCnn1DLayers(InputType inputType, int kH, int sH, int padH, int dilation,
                    ConvolutionMode convolutionMode, long outputDepth, long layerIdx, String layerName,
                    Class<?> layerClass) {

        if (convolutionMode == null) {
            String name = layerName == null ? "(not named)" : layerName;
            throw new DL4JInvalidConfigException("Invalid configuration: convolution mode is null for layer (idx="
                            + layerIdx + ", name=" + name + ", type=" + layerClass.getName() + ")");
        }

        InputType.InputTypeRecurrent i = (InputType.InputTypeRecurrent) inputType;

        val inHeight = (int) i.getTimeSeriesLength();
        if (dilation != 1) {
            kH = kH + (kH - 1) * (dilation - 1);
        }

        if (sH <= 0) {
            throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, sH <= 0)
                            + " Invalid strides: strides must be > 0 (strideH = " + sH + ")" + "\n"
                            + getConfigErrorCommonLastLine1D(inputType, kH, sH, padH, outputDepth, convolutionMode));
        }

        if (kH <= 0 || kH > inHeight + 2 * padH) {
            throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, true)
                            + " Invalid input configuration for kernel height. Require 0 < kH <= inHeight + 2*padH; got (kH="
                            + kH + ", inHeight=" + inHeight + ", padH=" + padH + ")\n"
                            + getConfigErrorCommonLastLine1D(inputType, kH, sH, padH, outputDepth, convolutionMode));
        }


        //Strict mode: require exactly the right size...
        if (convolutionMode == ConvolutionMode.Strict) {
            if ((inHeight - kH + 2 * padH) % sH != 0) {
                double d = (inHeight - kH + 2 * padH) / ((double) sH) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int) d;
                int sameSize = (int) Math.ceil(inHeight / ((double) sH));
                throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, true)
                                + "\nCombination of kernel size, stride and padding are not valid for given input height, "
                                + "using ConvolutionMode.Strict\n"
                                + "ConvolutionMode.Strict requires: output height = (input height - kernelSize + "
                                + "2*padding)/stride + 1 in height dimension to be an integer. Got: (" + inHeight
                                + " - " + kH + " + 2*" + padH + ")/" + sH + " + 1 = " + str + "\n"
                                + "See ConvolutionType enumeration Javadoc and \"Constraints on strides\" at "
                                + "http://cs231n.github.io/convolutional-networks/\n"
                                + "To truncate/crop the input, such that output height = floor(" + str + ") = "
                                + truncated + ", use ConvolutionType.Truncate.\n"
                                + "Alternatively use ConvolutionType.Same, which will use padding to give an "
                                + "output height of ceil(" + inHeight + "/" + sH + ")=" + sameSize + "\n"
                                + getConfigErrorCommonLastLine1D(inputType, kH, sH, padH, outputDepth,
                                                convolutionMode));
            }

        } else if (convolutionMode == ConvolutionMode.Same) {

            int outH = (int) Math.ceil(inHeight / ((double) sH));

            return InputType.recurrent(outputDepth, outH);
        }

        int outH = (inHeight - kH + 2 * padH) / sH + 1;
        return InputType.recurrent(outputDepth, outH);
    }

    /**
     * @deprecated Use {@link #getOutputTypeCnnLayers(InputType, int[], int[], int[], int[], ConvolutionMode, long, long, String, CNN2DFormat, Class)}
     */
    @Deprecated
    public static InputType getOutputTypeCnnLayers(InputType inputType, int[] kernelSize, int[] stride, int[] padding,
                                                   int[] dilation, ConvolutionMode convolutionMode, long outputDepth, long layerIdx, String layerName,
                                                   Class<?> layerClass) {
        return getOutputTypeCnnLayers(inputType, kernelSize, stride, padding, dilation, convolutionMode, outputDepth,
                layerIdx, layerName, CNN2DFormat.NCHW, layerClass);
    }

    public static InputType getOutputTypeCnnLayers(InputType inputType, int[] kernelSize, int[] stride, int[] padding,
                    int[] dilation, ConvolutionMode convolutionMode, long outputDepth, long layerIdx, String layerName,
                    CNN2DFormat format, Class<?> layerClass) {

        if (convolutionMode == null) {
            String name = layerName == null ? "(not named)" : layerName;
            throw new DL4JInvalidConfigException("Invalid configuration: convolution mode is null for layer (idx="
                            + layerIdx + ", name=" + name + ", type=" + layerClass.getName() + ")");
        }

        InputType.InputTypeConvolutional i = (InputType.InputTypeConvolutional) inputType;

        long inHeight = i.getHeight();
        long inWidth = i.getWidth();
        int padH = (padding == null ? 0 : padding[0]); //May be null for ConvolutionMode.Same
        int padW = (padding == null ? 0 : padding[1]);
        int kH = kernelSize[0];
        int kW = kernelSize[1];
        if (dilation[0] != 1) {
            //Use *effective* kernel size, accounting for dilation
            kH = kH + (kH - 1) * (dilation[0] - 1);
        }
        if (dilation[1] != 1) {
            kW = kW + (kW - 1) * (dilation[1] - 1);
        }

        int sH = stride[0];
        int sW = stride[1];

        if (sH <= 0 || sW <= 0) {
            throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, sH <= 0)
                            + " Invalid strides: strides must be > 0 (strideH = " + sH + ", strideW = " + sW + ")"
                            + "\n" + getConfigErrorCommonLastLine(inputType, kernelSize, stride, padding, outputDepth,
                                            convolutionMode));
        }

        if (kH <= 0 || kH > inHeight + 2 * padH) {
            throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, true)
                            + " Invalid input configuration for kernel height. Require 0 < kH <= inHeight + 2*padH; got (kH="
                            + kH + ", inHeight=" + inHeight + ", padH=" + padH + ")\n" + getConfigErrorCommonLastLine(
                                            inputType, kernelSize, stride, padding, outputDepth, convolutionMode));
        }

        if (kW <= 0 || kW > inWidth + 2 * padW) {
            throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, false)
                            + " Invalid input configuration for kernel width. Require 0 < kW <= inWidth + 2*padW; got (kW="
                            + kW + ", inWidth=" + inWidth + ", padW=" + padW + ")\n" + getConfigErrorCommonLastLine(
                                            inputType, kernelSize, stride, padding, outputDepth, convolutionMode));
        }

        //Strict mode: require exactly the right size...
        if (convolutionMode == ConvolutionMode.Strict) {
            if ((inHeight - kH + 2 * padH) % sH != 0) {
                double d = (inHeight - kH + 2 * padH) / ((double) sH) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int) d;
                int sameSize = (int) Math.ceil(inHeight / ((double) stride[0]));
                throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, true)
                                + "\nCombination of kernel size, stride and padding are not valid for given input height, using ConvolutionMode.Strict\n"
                                + "ConvolutionMode.Strict requires: output height = (input height - kernelSize + 2*padding)/stride + 1 in height dimension to be an integer. Got: ("
                                + inHeight + " - " + kH + " + 2*" + padH + ")/" + sH + " + 1 = " + str + "\n"
                                + "See ConvolutionType enumeration Javadoc and \"Constraints on strides\" at http://cs231n.github.io/convolutional-networks/\n"
                                + "To truncate/crop the input, such that output height = floor(" + str + ") = "
                                + truncated + ", use ConvolutionType.Truncate.\n"
                                + "Alternatively use ConvolutionType.Same, which will use padding to give an output height of ceil("
                                + inHeight + "/" + stride[0] + ")=" + sameSize + "\n" + getConfigErrorCommonLastLine(
                                                inputType, kernelSize, stride, padding, outputDepth, convolutionMode));
            }


            if ((inWidth - kW + 2 * padW) % sW != 0) {
                double d = (inWidth - kW + 2 * padW) / ((double) sW) + 1.0;
                String str = String.format("%.2f", d);
                int truncated = (int) d;
                int sameSize = (int) Math.ceil(inWidth / ((double) stride[1]));
                throw new DL4JInvalidConfigException(getConfigErrorCommonLine(layerIdx, layerName, layerClass, false)
                                + "\nCombination of kernel size, stride and padding are not valid for given input width, using ConvolutionMode.Strict\n"
                                + "ConvolutionMode.Strict requires: output width = (input width - kernelSize + 2*padding)/stride + 1 in width dimension to be an integer. Got: ("
                                + inWidth + " - " + kW + " + 2*" + padW + ")/" + sW + " + 1 = " + str + "\n"
                                + "See \"Constraints on strides\" at http://cs231n.github.io/convolutional-networks/ and ConvolutionType enumeration Javadoc.\n"
                                + "To truncate/crop the input, such that output width = floor(" + str + ") = "
                                + truncated + ", use ConvolutionType.Truncate.\n"
                                + "Alternatively use ConvolutionType.Same, which will use padding to give an output width of ceil("
                                + inWidth + "/" + stride[1] + ")=" + sameSize + "\n" + getConfigErrorCommonLastLine(
                                                inputType, kernelSize, stride, padding, outputDepth, convolutionMode));
            }
        } else if (convolutionMode == ConvolutionMode.Same) {

            int outH = (int) Math.ceil(inHeight / ((double) stride[0]));
            int outW = (int) Math.ceil(inWidth / ((double) stride[1]));

            return InputType.convolutional(outH, outW, outputDepth, format);
        }

        long hOut = (inHeight - kH + 2 * padH) / sH + 1;
        long wOut = (inWidth - kW + 2 * padW) / sW + 1;
        return InputType.convolutional(hOut, wOut, outputDepth, format);
    }

    private static String getConfigErrorCommonLine(long layerIdx, String layerName, Class<?> layerClass,
                    boolean isHeight) {
        String name = layerName == null ? "(not named)" : layerName;
        String layerType = layerClass.getSimpleName();

        return "Invalid configuration for layer (idx=" + layerIdx + ", name=" + name + ", type=" + layerType + ") for "
                        + (isHeight ? "height" : "width") + " dimension: ";
    }

    private static String getConfigErrorCommonLastLine1D(InputType inputType, int kernelSize, int stride, int padding,
                    long outputDepth, ConvolutionMode convolutionMode) {
        return "Input type = " + inputType + ", kernel = " + kernelSize + ", strides = " + stride + ", padding = "
                        + padding + ", layer size (output channels) = " + outputDepth + ", convolution mode = "
                        + convolutionMode;
    }

    private static String getConfigErrorCommonLastLine(InputType inputType, int[] kernelSize, int[] stride,
                    int[] padding, long outputDepth, ConvolutionMode convolutionMode) {
        return "Input type = " + inputType + ", kernel = " + Arrays.toString(kernelSize) + ", strides = "
                        + Arrays.toString(stride) + ", padding = " + Arrays.toString(padding)
                        + ", layer size (output channels) = " + outputDepth + ", convolution mode = " + convolutionMode;
    }

    /**
     * Utility method for determining the appropriate preprocessor for CNN layers, such as {@link ConvolutionLayer} and
     * {@link SubsamplingLayer}
     *
     * @param inputType     Input type to get the preprocessor for
     * @return              Null if no preprocessor is required; otherwise the appropriate preprocessor for the given input type
     */
    public static InputPreProcessor getPreProcessorForInputTypeCnn3DLayers(InputType inputType, String layerName) {
        switch (inputType.getType()) {
            case FF:
                log.info("Automatic addition of FF -> CNN3D preprocessors: not yet implemented (layer name: \""
                                + layerName + "\")");
                return null;
            case RNN:
                log.warn("Automatic addition of RNN -> CNN3D preprocessors: not yet implemented (layer name: \""
                                + layerName + "\")");
                return null;
            // TODO: handle CNN to CNN3D
            case CNN3D:
                return null;
            default:
                throw new RuntimeException("Unknown input type: " + inputType);
        }
    }

    /**
     * Utility method for determining the appropriate preprocessor for CNN layers, such as {@link ConvolutionLayer} and
     * {@link SubsamplingLayer}
     *
     * @param inputType     Input type to get the preprocessor for
     * @return              Null if no preprocessor is required; otherwise the appropriate preprocessor for the given input type
     */
    public static InputPreProcessor getPreProcessorForInputTypeCnnLayers(InputType inputType, String layerName) {

        //To add x-to-CNN preprocessor: need to know image channels/width/height after reshaping
        //But this can't be inferred from the FF/RNN activations directly (could be anything)

        switch (inputType.getType()) {
            case FF:
                //FF -> CNN
                //                return new FeedForwardToCnnPreProcessor(inputSize[0], inputSize[1], inputDepth);
                log.info("Automatic addition of FF -> CNN preprocessors: not yet implemented (layer name: \""
                                + layerName + "\")");
                return null;
            case RNN:
                //RNN -> CNN
                //                return new RnnToCnnPreProcessor(inputSize[0], inputSize[1], inputDepth);
                log.warn("Automatic addition of RNN -> CNN preprocessors: not yet implemented (layer name: \""
                                + layerName + "\")");
                return null;
            case CNN:
                //CNN -> CNN: no preprocessor required
                return null;
            case CNNFlat:
                //CNN (flat) -> CNN
                InputType.InputTypeConvolutionalFlat f = (InputType.InputTypeConvolutionalFlat) inputType;
                return new FeedForwardToCnnPreProcessor(f.getHeight(), f.getWidth(), f.getDepth());
            default:
                throw new RuntimeException("Unknown input type: " + inputType);
        }
    }

    public static InputPreProcessor getPreprocessorForInputTypeRnnLayers(InputType inputType, RNNFormat rnnDataFormat, String layerName) {
        if (inputType == null) {
            throw new IllegalStateException(
                            "Invalid input for RNN layer (layer name = \"" + layerName + "\"): input type is null");
        }

        switch (inputType.getType()) {
            case CNNFlat:
                //FF -> RNN or CNNFlat -> RNN
                //In either case, input data format is a row vector per example
                return new FeedForwardToRnnPreProcessor(rnnDataFormat);
            case FF:
                //If time distributed format is defined, use that. Otherwise use the layer-defined rnnDataFormat, which may be default
                InputType.InputTypeFeedForward ff = (InputType.InputTypeFeedForward)inputType;
                if(ff.getTimeDistributedFormat() != null && ff.getTimeDistributedFormat() instanceof RNNFormat){
                    return new FeedForwardToRnnPreProcessor((RNNFormat) ff.getTimeDistributedFormat());
                }
                return new FeedForwardToRnnPreProcessor(rnnDataFormat);
            case RNN:
                //RNN -> RNN: No preprocessor necessary
                return null;
            case CNN:
                //CNN -> RNN
                InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
                return new CnnToRnnPreProcessor(c.getHeight(), c.getWidth(), c.getChannels(), rnnDataFormat);
            default:
                throw new RuntimeException("Unknown input type: " + inputType);
        }
    }

}
