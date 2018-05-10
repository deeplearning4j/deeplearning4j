/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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
package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Utility functionality for Keras convolution layers.
 *
 * @author Max Pumperla
 */
public class KerasConvolutionUtils {

    /**
     * Get (convolution) stride from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return Strides array from Keras configuration
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    public static int[] getStrideFromConfig(Map<String, Object> layerConfig, int dimension,
                                            KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        int[] strides;
        if (innerConfig.containsKey(conf.getLAYER_FIELD_CONVOLUTION_STRIDES()) && dimension >= 2) {
            /* 2D/3D Convolutional layers. */
            @SuppressWarnings("unchecked")
            List<Integer> stridesList = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_CONVOLUTION_STRIDES());
            strides = ArrayUtil.toArray(stridesList);
        } else if (innerConfig.containsKey(conf.getLAYER_FIELD_SUBSAMPLE_LENGTH()) && dimension == 1) {
           /* 1D Convolutional layers. */
            if ((int) layerConfig.get("keras_version") == 2) {
                @SuppressWarnings("unchecked")
                List<Integer> stridesList = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_SUBSAMPLE_LENGTH());
                strides = ArrayUtil.toArray(stridesList);
            } else {
                int subsampleLength = (int) innerConfig.get(conf.getLAYER_FIELD_SUBSAMPLE_LENGTH());
                strides = new int[]{subsampleLength};
            }
        } else if (innerConfig.containsKey(conf.getLAYER_FIELD_POOL_STRIDES()) && dimension >= 2) {
            /* 2D/3D Pooling layers. */
            @SuppressWarnings("unchecked")
            List<Integer> stridesList = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_POOL_STRIDES());
            strides = ArrayUtil.toArray(stridesList);
        } else if (innerConfig.containsKey(conf.getLAYER_FIELD_POOL_1D_STRIDES()) && dimension == 1) {
            /* 1D Pooling layers. */
            int stride = (int) innerConfig.get(conf.getLAYER_FIELD_POOL_1D_STRIDES());
            strides = new int[]{stride};
        } else
            throw new InvalidKerasConfigurationException("Could not determine layer stride: no "
                    + conf.getLAYER_FIELD_CONVOLUTION_STRIDES() + " or "
                    + conf.getLAYER_FIELD_POOL_STRIDES() + " field found");
        return strides;
    }

    static int getDepthMultiplier(Map<String, Object> layerConfig, KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        return  (int) innerConfig.get(conf.getLAYER_FIELD_DEPTH_MULTIPLIER());
    }

    /**
     * Get atrous / dilation rate from config
     *
     * @param layerConfig   dictionary containing Keras layer configuration
     * @param dimension     dimension of the convolution layer (1 or 2)
     * @param conf          Keras layer configuration
     * @param forceDilation boolean to indicate if dilation argument should be in config
     * @return list of integers with atrous rates
     *
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    static int[] getDilationRate(Map<String, Object> layerConfig, int dimension, KerasLayerConfiguration conf,
                                        boolean forceDilation)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        int[] atrousRate;
        if (innerConfig.containsKey(conf.getLAYER_FIELD_DILATION_RATE()) && dimension >= 2) {
            @SuppressWarnings("unchecked")
            List<Integer> atrousRateList = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_DILATION_RATE());
            atrousRate = ArrayUtil.toArray(atrousRateList);
        } else if (innerConfig.containsKey(conf.getLAYER_FIELD_DILATION_RATE()) && dimension == 1) {
            if ((int) layerConfig.get("keras_version") == 2) {
                @SuppressWarnings("unchecked")
                List<Integer> atrousRateList = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_DILATION_RATE());
                atrousRate = new int[]{atrousRateList.get(0), atrousRateList.get(0)};
            } else {
                int atrous = (int) innerConfig.get(conf.getLAYER_FIELD_DILATION_RATE());
                atrousRate = new int[]{atrous, atrous};
            }
        } else {
            // If we are using keras 1, for regular convolutions, there is no "atrous" argument, for keras
            // 2 there always is.
            if (forceDilation)
                throw new InvalidKerasConfigurationException("Could not determine dilation rate: no "
                        + conf.getLAYER_FIELD_DILATION_RATE() + " field found");
            else
                atrousRate = null;
        }
        return atrousRate;

    }

    /**
     * Get upsampling size from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     *
     * @return Upsampling integer array from Keras config
     * @throws InvalidKerasConfigurationException Invalid Keras configuration
     */
    static int[] getUpsamplingSizeFromConfig(Map<String, Object> layerConfig, int dimension,
                                                    KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        int[] size;
        if (innerConfig.containsKey(conf.getLAYER_FIELD_UPSAMPLING_2D_SIZE()) && dimension == 2) {
            @SuppressWarnings("unchecked")
            List<Integer> sizeList = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_UPSAMPLING_2D_SIZE());
            size = ArrayUtil.toArray(sizeList);
        } else if (innerConfig.containsKey(conf.getLAYER_FIELD_UPSAMPLING_1D_SIZE()) && dimension == 1) {
            int upsamplingSize1D = (int) innerConfig.get(conf.getLAYER_FIELD_UPSAMPLING_1D_SIZE());
            size = new int[]{upsamplingSize1D};
        } else {
            throw new InvalidKerasConfigurationException("Could not determine kernel size: no "
                    + conf.getLAYER_FIELD_UPSAMPLING_1D_SIZE() + ", "
                    + conf.getLAYER_FIELD_UPSAMPLING_2D_SIZE());
        }
        return size;
    }


    /**
     * Get (convolution) kernel size from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     *
     * @return Convolutional kernel sizes
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    public static int[] getKernelSizeFromConfig(Map<String, Object> layerConfig, int dimension,
                                                KerasLayerConfiguration conf, int kerasMajorVersion)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        int[] kernelSize;
        if (kerasMajorVersion != 2) {
            if (innerConfig.containsKey(conf.getLAYER_FIELD_NB_ROW()) && dimension == 2
                    && innerConfig.containsKey(conf.getLAYER_FIELD_NB_COL())) {
            /* 2D Convolutional layers. */
                List<Integer> kernelSizeList = new ArrayList<>();
                kernelSizeList.add((Integer) innerConfig.get(conf.getLAYER_FIELD_NB_ROW()));
                kernelSizeList.add((Integer) innerConfig.get(conf.getLAYER_FIELD_NB_COL()));
                kernelSize = ArrayUtil.toArray(kernelSizeList);
            } else if (innerConfig.containsKey(conf.getLAYER_FIELD_3D_KERNEL_1()) && dimension == 3
                    && innerConfig.containsKey(conf.getLAYER_FIELD_3D_KERNEL_2())
                    && innerConfig.containsKey(conf.getLAYER_FIELD_3D_KERNEL_3())) {
            /* 3D Convolutional layers. */
                List<Integer> kernelSizeList = new ArrayList<>();
                kernelSizeList.add((Integer) innerConfig.get(conf.getLAYER_FIELD_3D_KERNEL_1()));
                kernelSizeList.add((Integer) innerConfig.get(conf.getLAYER_FIELD_3D_KERNEL_2()));
                kernelSizeList.add((Integer) innerConfig.get(conf.getLAYER_FIELD_3D_KERNEL_3()));
                kernelSize = ArrayUtil.toArray(kernelSizeList);
            } else if (innerConfig.containsKey(conf.getLAYER_FIELD_FILTER_LENGTH()) && dimension == 1) {
            /* 1D Convolutional layers. */
                int filterLength = (int) innerConfig.get(conf.getLAYER_FIELD_FILTER_LENGTH());
                kernelSize = new int[]{filterLength};
            } else if (innerConfig.containsKey(conf.getLAYER_FIELD_POOL_SIZE()) && dimension >= 2) {
            /* 2D/3D Pooling layers. */
                @SuppressWarnings("unchecked")
                List<Integer> kernelSizeList = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_POOL_SIZE());
                kernelSize = ArrayUtil.toArray(kernelSizeList);
            } else if (innerConfig.containsKey(conf.getLAYER_FIELD_POOL_1D_SIZE()) && dimension == 1) {
            /* 1D Pooling layers. */
                int poolSize1D = (int) innerConfig.get(conf.getLAYER_FIELD_POOL_1D_SIZE());
                kernelSize = new int[]{poolSize1D};
            } else {
                throw new InvalidKerasConfigurationException("Could not determine kernel size: no "
                        + conf.getLAYER_FIELD_NB_ROW() + ", "
                        + conf.getLAYER_FIELD_NB_COL() + ", or "
                        + conf.getLAYER_FIELD_FILTER_LENGTH() + ", or "
                        + conf.getLAYER_FIELD_POOL_1D_SIZE() + ", or "
                        + conf.getLAYER_FIELD_POOL_SIZE() + " field found");
            }
        } else {
            /* 2D/3D Convolutional layers. */
            if (innerConfig.containsKey(conf.getLAYER_FIELD_KERNEL_SIZE()) && dimension >= 2) {
                @SuppressWarnings("unchecked")
                List<Integer> kernelSizeList = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_KERNEL_SIZE());
                kernelSize = ArrayUtil.toArray(kernelSizeList);
            } else if (innerConfig.containsKey(conf.getLAYER_FIELD_FILTER_LENGTH()) && dimension == 1) {
            /* 1D Convolutional layers. */
                @SuppressWarnings("unchecked")
                List<Integer> kernelSizeList = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_FILTER_LENGTH());
                kernelSize = ArrayUtil.toArray(kernelSizeList);
            } else if (innerConfig.containsKey(conf.getLAYER_FIELD_POOL_SIZE()) && dimension >= 2) {
            /* 2D Pooling layers. */
                @SuppressWarnings("unchecked")
                List<Integer> kernelSizeList = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_POOL_SIZE());
                kernelSize = ArrayUtil.toArray(kernelSizeList);
            } else if (innerConfig.containsKey(conf.getLAYER_FIELD_POOL_1D_SIZE()) && dimension == 1) {
            /* 1D Pooling layers. */
                @SuppressWarnings("unchecked")
                List<Integer> kernelSizeList = (List<Integer>) innerConfig.get(conf.getLAYER_FIELD_POOL_1D_SIZE());
                kernelSize = ArrayUtil.toArray(kernelSizeList);
            } else {
                throw new InvalidKerasConfigurationException("Could not determine kernel size: no "
                        + conf.getLAYER_FIELD_KERNEL_SIZE() + ", or "
                        + conf.getLAYER_FIELD_FILTER_LENGTH() + ", or "
                        + conf.getLAYER_FIELD_POOL_SIZE() + " field found");
            }
        }

        return kernelSize;
    }

    /**
     * Get convolution border mode from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return Border mode of convolutional layers
     * @throws InvalidKerasConfigurationException Invalid Keras configuration
     */
    public static ConvolutionMode getConvolutionModeFromConfig(Map<String, Object> layerConfig,
                                                               KerasLayerConfiguration conf)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(conf.getLAYER_FIELD_BORDER_MODE()))
            throw new InvalidKerasConfigurationException("Could not determine convolution border mode: no "
                    + conf.getLAYER_FIELD_BORDER_MODE() + " field found");
        String borderMode = (String) innerConfig.get(conf.getLAYER_FIELD_BORDER_MODE());
        ConvolutionMode convolutionMode;
        if (borderMode.equals(conf.getLAYER_BORDER_MODE_SAME())) {
            /* Keras relies upon the Theano and TensorFlow border mode definitions and operations:
             * TH: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
             * TF: https://www.tensorflow.org/api_docs/python/nn/convolution#conv2d
             */
            convolutionMode = ConvolutionMode.Same;

        } else if (borderMode.equals(conf.getLAYER_BORDER_MODE_VALID()) ||
                borderMode.equals(conf.getLAYER_BORDER_MODE_FULL())) {
            convolutionMode = ConvolutionMode.Truncate;

        } else {
            throw new UnsupportedKerasConfigurationException("Unsupported convolution border mode: " + borderMode);
        }
        return convolutionMode;
    }

    /**
     * Get (convolution) padding from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return Padding values derived from border mode
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    public static int[] getPaddingFromBorderModeConfig(Map<String, Object> layerConfig, int dimension,
                                                       KerasLayerConfiguration conf, int kerasMajorVersion)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        int[] padding = null;
        if (!innerConfig.containsKey(conf.getLAYER_FIELD_BORDER_MODE()))
            throw new InvalidKerasConfigurationException("Could not determine convolution border mode: no "
                    + conf.getLAYER_FIELD_BORDER_MODE() + " field found");
        String borderMode = (String) innerConfig.get(conf.getLAYER_FIELD_BORDER_MODE());
        if (borderMode.equals(conf.getLAYER_FIELD_BORDER_MODE())) {
            padding = getKernelSizeFromConfig(layerConfig, dimension, conf, kerasMajorVersion);
            for (int i = 0; i < padding.length; i++)
                padding[i]--;
        }
        return padding;
    }

    /**
     * Get padding and cropping configurations from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @param conf        KerasLayerConfiguration
     * @param layerField  String value of the layer config name to check for (e.g. "padding" or "cropping")
     * @param dimension   Dimension of the padding layer
     * @return padding list of integers
     * @throws InvalidKerasConfigurationException Invalid keras configuration
     */
    static int[] getPaddingFromConfig(Map<String, Object> layerConfig,
                                      KerasLayerConfiguration conf,
                                      String layerField,
                                      int dimension)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(layerField))
            throw new InvalidKerasConfigurationException(
                    "Field " + layerField + " not found in Keras cropping or padding layer");
        int[] padding;
        if (dimension >= 2) {
            List<Integer> paddingList;
            // For 2D layers, padding/cropping can either be a pair [[x_0, x_1].[y_0, y_1]] or a pair [x, y]
            // or a single integer x. Likewise for the 3D case.
            try {
                List paddingNoCast = (List) innerConfig.get(layerField);
                boolean isNested;
                try {
                    @SuppressWarnings("unchecked")
                    List<Integer> firstItem = (List<Integer>) paddingNoCast.get(0);
                    isNested = true;
                    paddingList = new ArrayList<>(2 * dimension);
                } catch (Exception e) {
                    int firstItem = (int) paddingNoCast.get(0);
                    isNested = false;
                    paddingList = new ArrayList<>(dimension);
                }

                if ((paddingNoCast.size() == dimension) && !isNested) {
                    for (int i=0; i < dimension; i++)
                        paddingList.add((int) paddingNoCast.get(i));
                    padding = ArrayUtil.toArray(paddingList);
                } else if ((paddingNoCast.size() == dimension) && isNested) {
                    for (int j=0; j < dimension; j++) {
                        @SuppressWarnings("unchecked")
                        List<Integer> item = (List<Integer>) paddingNoCast.get(0);
                        paddingList.add((item.get(0)));
                        paddingList.add((item.get(1)));
                    }
                    padding = ArrayUtil.toArray(paddingList);
                } else {
                    throw new InvalidKerasConfigurationException("Found Keras ZeroPadding" + dimension
                            + "D layer with invalid " + paddingList.size() + "D padding.");
                }
            } catch (Exception e) {
                int paddingInt = (int) innerConfig.get(layerField);
                if (dimension == 2) {
                    padding = new int[]{paddingInt, paddingInt, paddingInt, paddingInt};
                } else {
                    padding = new int[]{paddingInt, paddingInt, paddingInt, paddingInt, paddingInt, paddingInt};
                }
            }

        } else if (dimension == 1) {
            int paddingInt = (int) innerConfig.get(layerField);
            padding = new int[]{paddingInt};
        } else {
            throw new UnsupportedKerasConfigurationException(
                    "Keras padding layer not supported");
        }
        return padding;
    }
}
