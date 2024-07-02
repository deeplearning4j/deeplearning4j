/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.util;


import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.nd4j.common.base.Preconditions;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JArraySizeException;

import java.util.Arrays;

import static org.deeplearning4j.nn.conf.RNNFormat.NCW;
import static org.deeplearning4j.nn.conf.RNNFormat.NWC;

public class Convolution1DUtils {

    private static final int ONE = 1;


    private Convolution1DUtils() {
    }


    public static int getOutputSize(INDArray inputData, int kernel, int strides, int padding,
                                    ConvolutionMode convolutionMode) {
        return getOutputSize(inputData, kernel, strides, padding, convolutionMode, ONE);
    }

    /**
     * Returns true if the given layer has an
     * {@link RNNFormat}.
     * This is true for:
     * {@link Convolution1DLayer},
     * {@link Subsampling1DLayer}
     * {@link SimpleRnn}
     * {@link LSTM}
     * {@link EmbeddingSequenceLayer}
     * @param layer the layer to test
     * @return true if the input layer has an rnn format
     * false otherwise
     */
    public static boolean hasRnnDataFormat(Layer layer) {
        return layer instanceof Convolution1D ||
                layer instanceof Convolution1DLayer ||
                layer instanceof Subsampling1DLayer ||
                layer instanceof SimpleRnn ||
                layer instanceof LSTM ||
                layer instanceof EmbeddingSequenceLayer;
    }

    /**
     * Get the {@link RNNFormat} for the given layer.
     * Throws an {@link IllegalArgumentException}
     * if a layer doesn't have an rnn format
     * @param layer the layer to get the format for
     * @return the format for the layer
     */
    public static RNNFormat getRnnFormatFromLayer(Layer layer) {
        Preconditions.checkState(hasRnnDataFormat(layer),"Layer of type " + layer.getClass().getName() + " and name " + layer.getLayerName() + " does not have an RNNFormat");
        if(layer instanceof SimpleRnn) {
            SimpleRnn simpleRnn = (SimpleRnn) layer;
            return simpleRnn.getRnnDataFormat();
        } else if(layer instanceof Convolution1D) {
            Convolution1D convolution1D = (Convolution1D) layer;
            return convolution1D.getRnnDataFormat();
        } else if(layer instanceof Convolution1DLayer) {
            Convolution1DLayer convolution1DLayer = (Convolution1DLayer) layer;
            return convolution1DLayer.getRnnDataFormat();
        } else if(layer instanceof Subsampling1DLayer) {
            Subsampling1DLayer subsampling1DLayer = (Subsampling1DLayer) layer;
            return subsampling1DLayer.getCnn2dDataFormat() == CNN2DFormat.NCHW ? NCW : NWC;
        } else if(layer instanceof LSTM) {
            LSTM lstm = (LSTM) layer;
            return lstm.getRnnDataFormat();
        } else if(layer instanceof EmbeddingSequenceLayer) {
            EmbeddingSequenceLayer embeddingSequenceLayer = (EmbeddingSequenceLayer) layer;
            return embeddingSequenceLayer.getOutputFormat();
        }
        else {
            throw new IllegalArgumentException("Illegal layer type " + layer.getClass().getName() + " and name " + layer.getLayerName());
        }
    }

    /**
     * Reshapes the given weight array or weight gradient to work with the specified {@link RNNFormat}
     * @param w the weight array or gradient
     * @param wFormat the {@link RNNFormat} to use
     * @return the reshaped array
     */
    public static INDArray reshapeWeightArrayOrGradientForFormat(INDArray w, WeightsFormat wFormat) {
       if(w.rank() < 4)
           return w;
        switch(wFormat) {
            case OIYX:
                return w.reshape(w.size(0),w.size(1),w.size(3));
            case YXIO:
                return w.reshape(w.size(1),w.size(2),w.size(3));
            case OYXI:
                return w.reshape(w.size(0),w.size(2),w.size(3));
            default:
                throw new IllegalArgumentException("Illegal weights format " + wFormat);

        }
    }

    /**
     * Get the output size (height) for the given input data and CNN1D configuration
     *
     * @param inH             Input size (height, or channels).
     * @param kernel          Kernel size
     * @param strides         Stride
     * @param padding         Padding
     * @param convolutionMode Convolution mode (Same, Strict, Truncate)
     * @param dilation        Kernel dilation
     * @return Output size (width)
     */
    public static long getOutputSize(long inH, int kernel, int strides, int padding,
                                     ConvolutionMode convolutionMode, int dilation) {
        return getOutputSizeLong(inH, (long) kernel, (long) strides, (long) padding, convolutionMode, (long) dilation);
    }

    /**
     * Get the output size (height) for the given input data and CNN1D configuration
     *
     * @param inH             Input size (height, or channels).
     * @param kernel          Kernel size
     * @param strides         Stride
     * @param padding         Padding
     * @param convolutionMode Convolution mode (Same, Strict, Truncate)
     * @param dilation        Kernel dilation
     * @return Output size (width)
     */
    public static long getOutputSizeLong(long inH, long kernel, long strides, long padding,
                                         ConvolutionMode convolutionMode, long dilation) {
        long eKernel = effectiveKernelSize(kernel, dilation);
        if (convolutionMode == ConvolutionMode.Same || convolutionMode == ConvolutionMode.Causal) {
            return (long) Math.ceil(inH / ((double) strides));
        }
        return (inH - eKernel + 2 * padding) / strides + 1;
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
        return (int) getOutputSizeLong(inputData, (long) kernel, (long) strides, (long) padding, convolutionMode, (long) dilation);
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
    public static long getOutputSizeLong(INDArray inputData, long kernel, long strides, long padding,
                                         ConvolutionMode convolutionMode, long dilation) {
        long inH = inputData.size(2);
        long dilatedFilterSize = kernel + (kernel - 1) * (dilation - 1);
        long outputLength;
        if (convolutionMode == ConvolutionMode.Same) {
            outputLength = inH - dilatedFilterSize + 1;
        } else if (convolutionMode == ConvolutionMode.Causal) {
            outputLength = inH + dilatedFilterSize - 1;
        } else {
            throw new IllegalArgumentException("Unsupported convolution mode: " + convolutionMode);
        }

        return (outputLength + strides - 1) / strides;
    }

    public static void validateShapes(INDArray inputData, int eKernel, int strides, int padding,
                                      ConvolutionMode convolutionMode, int dilation, int inShape,
                                      boolean atrous) {

        int inH = inShape;
        boolean t = convolutionMode == ConvolutionMode.Truncate;

        if (t && (eKernel <= 0 || eKernel > inH + 2 * padding)) {
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

    /**
     * Calculates the effective kernel size, accounting for dilation.
     *
     * @param kernel   The kernel size.
     * @param dilation The dilation factor.
     * @return The effective kernel size.
     */
    private static long effectiveKernelSize(long kernel, long dilation) {
        return kernel + (kernel - 1) * (dilation - 1);
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

    public static int getSameModeBottomRightPadding(int outSize, int inSize, int kernel, int strides, int dilation) {
        int eKernel = effectiveKernelSize(kernel, dilation);
        int totalPad = ((outSize - 1) * strides + eKernel - inSize);
        int tlPad = totalPad / 2;
        int brPad = totalPad - tlPad;
        Preconditions.checkState(brPad >= 0, "Invalid padding values (right) calculated: %s - " +
                "layer configuration is invalid? Input size %s, output size %s, kernel %s, " +
                "strides %s, dilation %s", brPad, inSize, outSize, kernel, strides, dilation);
        return brPad;
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
