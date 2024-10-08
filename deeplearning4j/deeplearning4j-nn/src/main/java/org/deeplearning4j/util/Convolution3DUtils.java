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


import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.util.ArrayUtil;

import java.util.Arrays;

import static org.deeplearning4j.util.ConvolutionUtils.effectiveKernelSize;

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
    public static long[] get3DOutputSizeLong(INDArray inputData, long[] kernel, long[] strides, long[] padding,
                                        ConvolutionMode convolutionMode, long[] dilation, boolean isNCDHW) {

        // NCDHW vs. NDHWC
        long inD = (isNCDHW ? inputData.size(2) : inputData.size(1));
        long inH = (isNCDHW ? inputData.size(3) : inputData.size(2));
        long inW = (isNCDHW ? inputData.size(4) : inputData.size(3));

        long[] eKernel = effectiveKernelSize(kernel, dilation);
        boolean atrous = (eKernel == kernel);

        val inShape = new long[]{inD, inH, inW};
        validateShapesLong(inputData.shape(), eKernel, strides, padding, convolutionMode, dilation, inShape, atrous);

        if (convolutionMode == ConvolutionMode.Same) {
            int outD = (int) Math.ceil(inD / ((double) strides[0]));
            int outH = (int) Math.ceil(inH / ((double) strides[1]));
            int outW = (int) Math.ceil(inW / ((double) strides[2]));

            return new long[]{outD, outH, outW};
        }

        long outD = ((int)inD - eKernel[0] + 2 * padding[0]) / strides[0] + 1;
        long outH = ((int)inH - eKernel[1] + 2 * padding[1]) / strides[1] + 1;
        long outW = ((int)inW - eKernel[2] + 2 * padding[2]) / strides[2] + 1;

        return new long[]{outD, outH, outW};
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
        long inD = (isNCDHW ? inputData.size(2) : inputData.size(1));
        long inH = (isNCDHW ? inputData.size(3) : inputData.size(2));
        long inW = (isNCDHW ? inputData.size(4) : inputData.size(3));

        int[] eKernel = effectiveKernelSize(kernel, dilation);
        boolean atrous = (eKernel == kernel);

        val inShape = new long[]{inD, inH, inW};
        validateShapes(ArrayUtil.toInts(inputData.shape()), eKernel, strides, padding, convolutionMode, dilation, inShape, atrous);

        if (convolutionMode == ConvolutionMode.Same) {
            int outD = (int) Math.ceil(inD / ((double) strides[0]));
            int outH = (int) Math.ceil(inH / ((double) strides[1]));
            int outW = (int) Math.ceil(inW / ((double) strides[2]));

            return new int[]{outD, outH, outW};
        }

        int outD = ((int)inD - eKernel[0] + 2 * padding[0]) / strides[0] + 1;
        int outH = ((int)inH - eKernel[1] + 2 * padding[1]) / strides[1] + 1;
        int outW = ((int)inW - eKernel[2] + 2 * padding[2]) / strides[2] + 1;

        return new int[]{outD, outH, outW};
    }



    private static void validateShapesLong(long[] inputDataShape, long[] eKernel, long[] strides, long[] padding,
                                       ConvolutionMode convolutionMode, long[] dilation, long[] inShape,
                                       boolean atrous) {

        String[] dims = {"depth", "height", "width"};

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
                            .append(getCommonErrorMsgLong(inputDataShape, eKernel, strides, padding, dilation));

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
                            .append(getCommonErrorMsgLong(inputDataShape, eKernel, strides, padding, dilation));

                    throw new DL4JInvalidConfigException(sb.toString());
                }
            }
        }
    }

    private static void validateShapes(int[] inputDataShape, int[] eKernel, int[] strides, int[] padding,
                                       ConvolutionMode convolutionMode, int[] dilation, long[] inShape,
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


    private static String getCommonErrorMsgLong(long[] inputDatashape, long[] kernel, long[] strides, long[] padding, long[] dilation) {
        String s = "\nInput size: [numExamples, inputDepth, inputHeight, inputWidth]=" + Arrays.toString(inputDatashape)
                + ", inputKernel=" + Arrays.toString(kernel);
        if (dilation[0] != 1 || dilation[1] != 1) {
            long[] effectiveKernel = effectiveKernelSize(kernel, dilation);
            s += ", effectiveKernelGivenDilation=" + Arrays.toString(effectiveKernel);
        }
        return s + ", strides=" + Arrays.toString(strides) + ", padding="
                + Arrays.toString(padding) + ", dilation=" + Arrays.toString(dilation);
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
    public static long[] get3DSameModeTopLeftPaddingLong(long[] outSize, long[] inSize, long[] kernel, long[] strides,
                                                         long[] dilation) {
        long[] eKernel = effectiveKernelSize(kernel, dilation);
        long[] outPad = new long[3];
        outPad[0] = ((outSize[0] - 1) * strides[0] + eKernel[0] - inSize[0]) / 2;
        outPad[1] = ((outSize[1] - 1) * strides[1] + eKernel[1] - inSize[1]) / 2;
        outPad[2] = ((outSize[2] - 1) * strides[2] + eKernel[2] - inSize[2]) / 2;
        return outPad;
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
        validateCnn3DKernelStridePaddingLong(toLongArray(kernelSize), toLongArray(stride), toLongArray(padding));
    }

    /**
     * Perform validation on the CNN3D layer kernel/stride/padding. Expect 3d long[], with values > 0 for kernel size and
     * stride, and values >= 0 for padding.
     *
     * @param kernelSize Kernel size array to check
     * @param stride     Stride array to check
     * @param padding    Padding array to check
     */
    public static void validateCnn3DKernelStridePaddingLong(long[] kernelSize, long[] stride, long[] padding) {
        if (kernelSize == null || kernelSize.length != 3) {
            throw new IllegalStateException("Invalid kernel size: expected long[] of length 3, got "
                    + (kernelSize == null ? null : Arrays.toString(kernelSize)));
        }

        if (stride == null || stride.length != 3) {
            throw new IllegalStateException("Invalid stride configuration: expected long[] of length 3, got "
                    + (stride == null ? null : Arrays.toString(stride)));
        }

        if (padding == null || padding.length != 3) {
            throw new IllegalStateException("Invalid padding configuration: expected long[] of length 3, got "
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

    /**
     * Helper method to convert an int array to a long array.
     *
     * @param intArray The int array to convert.
     * @return The converted long array.
     */
    private static long[] toLongArray(int[] intArray) {
        if (intArray == null) {
            return null;
        }
        return Arrays.stream(intArray).asLongStream().toArray();
    }


    /**
     * Returns the {@link org.deeplearning4j.nn.conf.layers.Convolution3D.DataFormat}
     * for the associated layer.
     * Note that {@link #layerHasConvolution3DLayout(Layer)}
     * should return true on the given layer input or an {@link IllegalArgumentException}
     * will be thrown
     * @param inputLayer the input layer to get the format for
     * @return the associated {@link org.deeplearning4j.nn.conf.layers.Convolution3D.DataFormat} with
     * the layer
     */
    public static Convolution3D.DataFormat getFormatForLayer(Layer inputLayer) {
        if(inputLayer instanceof Convolution3D) {
            Convolution3D convolution3D = (Convolution3D) inputLayer;
            return convolution3D.getDataFormat();
        } else if(inputLayer instanceof Subsampling3DLayer) {
            Subsampling3DLayer subsampling3DLayer = (Subsampling3DLayer) inputLayer;
            return subsampling3DLayer.getDataFormat();
        } else if(inputLayer instanceof Upsampling3D) {
            Upsampling3D upsampling3D = (Upsampling3D) inputLayer;
            return upsampling3D.getDataFormat();
        } else if(inputLayer instanceof Deconvolution3D) {
            Deconvolution3D deconvolution3D = (Deconvolution3D) inputLayer;
            return deconvolution3D.getDataFormat();
        } else {
            throw new IllegalArgumentException("Illegal input format type " + inputLayer.getClass().getName() + " with layer name of " + inputLayer.getLayerName());
        }
    }

    /**
     * Returns true if any of the layers are 3d convolution, pooling, or upsampling layers including:
     * {@link Convolution3D}, {@link Deconvolution3D}, {@link Subsampling3DLayer}, {@link Upsampling3D}
     * @param layer the input layer to validate
     * @return true if the layer is any of the types specified.
     */
    public static boolean layerHasConvolution3DLayout(Layer layer) {
        return layer instanceof Convolution3D ||
                layer instanceof Deconvolution3D ||
                layer instanceof Subsampling3DLayer ||
                layer instanceof Upsampling3D;
    }


}
